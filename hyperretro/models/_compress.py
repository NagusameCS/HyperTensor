"""Unified model compression — works on any AbstractModel backend.

Routes to the appropriate backend-specific compression implementation
while providing a consistent API. The compression strategy adapts to
the model architecture:

- HuggingFace models: GRC + aware-SVD + int4 (existing pipeline)
- OpenMythos models: recurrent-block-aware GRC + MoE-aware SVD + int4
- The recurrent block gets rank amplification proportional to loop count
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from hyperretro.models import AbstractModel, CompressedModel


def _compress_abstract_model(
    model: AbstractModel,
    *,
    ffn_rank: int = 1024,
    attn_rank: int = 0,
    int4: bool = True,
    int4_block_size: int = 128,
    int4_awq: bool = True,
    activation_corpus: str | None = None,
    **kwargs,
) -> CompressedModel:
    """Compress any AbstractModel with HyperRetro.

    Dispatches to the appropriate backend-specific implementation.
    """
    backend = model.backend

    if backend == "huggingface":
        return _compress_hf(
            model, ffn_rank, attn_rank, int4, int4_block_size, int4_awq,
            activation_corpus, **kwargs,
        )
    elif backend == "openmythos":
        return _compress_om(
            model, ffn_rank, attn_rank, int4, int4_block_size, int4_awq,
            activation_corpus, **kwargs,
        )
    else:
        # Generic path: apply basic SVD + int4 to all 2D weight matrices
        return _compress_generic(
            model, ffn_rank, attn_rank, int4, int4_block_size, int4_awq,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# HuggingFace compression
# ---------------------------------------------------------------------------

def _compress_hf(
    model: AbstractModel,
    ffn_rank: int,
    attn_rank: int,
    int4: bool,
    block_size: int,
    awq: bool,
    corpus_path: str | None,
    **kwargs,
) -> CompressedModel:
    """Compress a HuggingFace model using the full HyperRetro pipeline."""
    import torch
    from hyperretro.hf.factored import factor_ffn_state_dict, factor_attn_state_dict, save_factored_checkpoint
    from hyperretro.hf.activation import collect_ffn_input_norms
    from hyperretro.hf.factor_int4 import save_int4_factored_checkpoint
    from hyperretro.hf.compress import _group_attn_by_layer, compress_state_dict_factored, CompressConfig
    from transformers import AutoTokenizer, AutoConfig
    from safetensors.torch import load_file
    import tempfile, shutil, json

    # Work directly on the model's state dict
    sd = {k: v.clone() for k, v in model.state_dict.items()}
    orig_config = model._config

    # Optional: AxiomGauge diagonal gauge optimization before SVD.
    # Finds the GL(d) gauge that minimizes joint tail energy across all
    # weight matrices. Zero runtime cost -- baked into the compressed factors.
    if kwargs.get("gauge_optimize", False):
        try:
            from hypercore import AxiomGauge
            d_model = model.get_hidden_size()
            if d_model > 0:
                # Collect read-side weight matrices (Q, K, V, gate, up)
                reads = {}
                for key, tensor in sd.items():
                    if tensor.ndim != 2:
                        continue
                    if tensor.shape[1] != d_model:
                        continue
                    if any(p in key.lower() for p in
                           ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
                            "query", "key", "value", "c_fc", "c_attn")):
                        reads[key] = tensor.float().cpu().numpy()
                    if len(reads) >= 16:
                        break
                if reads:
                    gauge_rank = kwargs.get("gauge_rank", min(ffn_rank, 1024))
                    gauge = AxiomGauge(d=d_model, rank=gauge_rank)
                    result = gauge.fit(reads, n_iter=kwargs.get("gauge_iters", 20),
                                       lr=0.01, verbose=False)
                    g = result.g.astype(np.float32)
                    g_inv = (1.0 / (g + 1e-10)).astype(np.float32)

                    # Bake gauge into read-side weights: W -> W * diag(1/g)
                    for key, tensor in list(sd.items()):
                        if tensor.ndim == 2 and tensor.shape[1] == d_model:
                            try:
                                sd[key] = tensor.float() * torch.from_numpy(g_inv).to(tensor.device)
                            except Exception:
                                pass
        except (ImportError, Exception):
            pass

    # Collect activation norms if corpus provided
    activation_norms = None
    if corpus_path and Path(corpus_path).exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                kwargs.get("model_id", "Qwen/Qwen2.5-1.5B")
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            activation_norms = collect_ffn_input_norms(
                model.model_obj, tokenizer,
                corpus_path=corpus_path,
                n_batches=4, seq_len=256, device="cpu",
            )
        except Exception:
            pass

    # GRC-compress attn if requested
    attn_entries = []
    if attn_rank > 0:
        layer_keys = _group_attn_by_layer(sd)
        sd, attn_entries = factor_attn_state_dict(
            sd, rank=attn_rank, layer_keys=layer_keys, rel_tol=1e-3,
        )

    # Factor FFN
    ffn_entries = []
    if ffn_rank > 0:
        max_rank = max(ffn_rank, 1)
        ffn_entries = factor_ffn_state_dict(
            sd, max_rank=max_rank, rel_tol=1e-4,
            activation_col_norms=activation_norms,
        )

    manifest = {"backend": "huggingface", "ffn": ffn_entries, "layers": attn_entries}

    # Int4 quantize
    if int4:
        from hyperretro.hf.factor_int4 import quantize_factored_state_dict
        sd, manifest = quantize_factored_state_dict(
            sd, manifest,
            n_bits=4, block_size=block_size,
            quantize_non_factored=True,
            activation_norms=activation_norms if awq else None,
        )

    # Auto-certificate if requested (default: True)
    cm = CompressedModel(sd, manifest, "huggingface", orig_config)
    cert = None
    if kwargs.get("certificate", True):
        cert = _compute_certificate(sd, manifest, orig_config,
                                     model_id=kwargs.get("model_id", "unknown"))
        if cert:
            cm.manifest["certificate"] = cert

    return cm


def _compute_certificate(sd, manifest, orig_config, model_id="unknown"):
    """Generate a quality certificate from compression results.

    Called automatically after compression when certificate=True.
    Returns a dict with trust_tier, PPL bounds, etc, or None if
    certificate dependencies are unavailable.
    """
    try:
        from hyperretro.certificates import certify_compression
        from hyperretro.hf.compress import CompressConfig

        cfg = CompressConfig(
            rank_k=manifest.get("ffn", [{}])[0].get("rank", 1024) if manifest.get("ffn") else 1024,
            sink_T=0, dtype="float32",
        )
        per_layer_stats = {}
        for entry in manifest.get("ffn", []):
            li = entry.get("layer_idx", 0)
            key = f"layer_{li}"
            per_layer_stats[key] = {
                "rank_k": entry.get("rank", 0),
                "frob_relerr_q": entry.get("frob_q", 0.0),
                "frob_relerr_k": entry.get("frob_k", 0.0),
                "frob_relerr_v": entry.get("frob_v", 0.0),
                "d": entry.get("in_features", 0),
            }

        cert = certify_compression(sd, cfg, per_layer_stats, model_id=model_id)
        return {
            "trust_tier": cert.trust_tier,
            "mean_spectral_efficiency": cert.mean_spectral_efficiency,
            "max_bp_ns_bound": cert.max_bp_ns,
            "mean_frob_q": cert.mean_frob_q,
            "mean_frob_k": cert.mean_frob_k,
            "mean_frob_v": cert.mean_frob_v,
            "has_jury_proof": cert.has_jury_proof,
            "ppl_multiplier_bound": cert.ppl_multiplier_bound,
            "ppl_multiplier_bound_typical": cert.ppl_multiplier_bound_typical,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenMythos compression (recurrent-block-aware)
# ---------------------------------------------------------------------------

def _compress_om(
    model: AbstractModel,
    ffn_rank: int,
    attn_rank: int,
    int4: bool,
    block_size: int,
    awq: bool,
    corpus_path: str | None,
    **kwargs,
) -> CompressedModel:
    """Compress an OpenMythos model.

    Key difference from HF: the recurrent block runs T times, so we
    amplify compression effort on recurrent weights.
    """
    from hyperretro.hf.openmythos import (
        compress_openmythos,
        OMCompressConfig,
        compress_openmythos_state_dict,
    )

    # Amplify rank for recurrent block weights (they run T times)
    n_loops = getattr(model, 'num_recurrent_loops', 16)
    # Use higher rank for recurrent block since compression benefit is T×
    recurrent_ffn_rank = min(ffn_rank * 2, 4096)  # up to 2× rank for recurrent

    cfg = OMCompressConfig(
        ffn_rank=ffn_rank,
        attn_rank=attn_rank,
        int4=int4,
        int4_block_size=block_size,
        int4_awq=awq,
    )

    sd, manifest, report = compress_openmythos(
        model.model_obj,
        ffn_rank=ffn_rank,
        attn_rank=attn_rank,
        int4=int4,
        activation_corpus_path=corpus_path,
    )

    report["recurrent_loops"] = n_loops
    report["recurrent_ffn_rank"] = recurrent_ffn_rank

    return CompressedModel(sd, manifest, "openmythos", model.config)


# ---------------------------------------------------------------------------
# Generic compression (any model)
# ---------------------------------------------------------------------------

def _compress_generic(
    model: AbstractModel,
    ffn_rank: int,
    attn_rank: int,
    int4: bool,
    block_size: int,
    awq: bool,
    **kwargs,
) -> CompressedModel:
    """Generic compression: SVD-factor all 2D weight matrices + int4."""
    import torch
    from hyperretro.hf.factored import _svd_factor
    from hyperretro.hf.factor_quantize import quantize_blockwise_int4
    from hyperretro.hf.factor_int4 import pack_int4_rows

    sd = {}
    manifest = {"backend": model.backend, "ffn": [], "layers": []}

    for key, tensor in model.state_dict.items():
        if not (hasattr(tensor, "dim") and tensor.dim() == 2):
            sd[key] = tensor
            continue

        m, n = tuple(tensor.shape)
        W = tensor.float().cpu().numpy()

        # Determine rank
        is_ffn = any(pat in key for pat in [".ffn.", ".mlp.", ".experts."])
        is_attn = any(pat in key for pat in [".attn.", ".self_attn."])
        rank = ffn_rank if is_ffn else (attn_rank if is_attn else 0)

        if rank > 0 and m * n > 10000:
            k = min(rank, m, n)
            A, B = _svd_factor(W, k)
            k_eff = A.shape[0]

            if int4:
                # Quantize A and B
                A_q, A_s = quantize_blockwise_int4(A, block_size=block_size)
                B_q, B_s = quantize_blockwise_int4(B, block_size=block_size)
                prefix = key[:-len(".weight")] if key.endswith(".weight") else key
                sd[f"{prefix}.factored_A.q"] = torch.from_numpy(pack_int4_rows(A_q)).to(torch.uint8)
                sd[f"{prefix}.factored_A.scales"] = torch.from_numpy(A_s).to(torch.float16)
                sd[f"{prefix}.factored_B.q"] = torch.from_numpy(pack_int4_rows(B_q)).to(torch.uint8)
                sd[f"{prefix}.factored_B.scales"] = torch.from_numpy(B_s).to(torch.float16)
            else:
                prefix = key[:-len(".weight")] if key.endswith(".weight") else key
                sd[f"{prefix}.factored_A"] = torch.from_numpy(A).to(torch.float16)
                sd[f"{prefix}.factored_B"] = torch.from_numpy(B).to(torch.float16)

            target_list = manifest["ffn"] if is_ffn else manifest["layers"]
            target_list.append({
                "weight_key": key, "rank": k_eff,
                "in_features": n, "out_features": m,
            })
        elif int4 and m * n > 10000:
            # Int4-quantize without factoring
            W_q, W_s = quantize_blockwise_int4(W, block_size=block_size)
            base = key[:-len(".weight")] if key.endswith(".weight") else key
            sd[f"{base}.q"] = torch.from_numpy(pack_int4_rows(W_q)).to(torch.uint8)
            sd[f"{base}.scales"] = torch.from_numpy(W_s).to(torch.float16)
        else:
            sd[key] = tensor

    return CompressedModel(sd, manifest, model.backend, model.config)
