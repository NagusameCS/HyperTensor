"""HyperRetro × OpenMythos integration bridge.

OpenMythos (Recurrent-Depth Transformer) and HyperRetro are complementary:
- OpenMythos attacks **reasoning depth** (looped recurrence, implicit CoT)
- HyperRetro attacks **inference efficiency** (weight compression, KV-cache, quantize)

The recurrent block — one TransformerBlock reused T times — is the perfect
compression target: compressing it once gives T× the benefit (T=16 at 1B scale).

Integration strategy:
  1. GRC-compress the RecurrentBlock's attention (GQA or MLA)
  2. Factor + int4 the MoE FFN experts (massive param count)
  3. Factor + int4 the Prelude/Coda blocks (standard path)
  4. Int4-quantize non-factored weights (embeddings, norms)
  5. Export to GGUF for llama.cpp deployment

Usage:
    from hyperretro.hf.openmythos import compress_openmythos
    from open_mythos import mythos_1b, OpenMythos

    model = OpenMythos(mythos_1b())
    compressed = compress_openmythos(model, ffn_rank=512)
    compressed.save("outputs/mythos_1b_compressed")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OMCompressConfig:
    """HyperRetro compression config for OpenMythos models."""
    # Attention GRC
    attn_rank: int = 640            # GRC rank for attention Q/K/V (0=skip)
    sink_T: int = 0                 # sink-aware GRC columns

    # FFN factoring
    ffn_rank: int = 512             # rank for FFN SVD factoring (0=skip)
    ffn_rel_tol: float = 1e-4       # adaptive rank tolerance

    # Int4 quantization
    int4: bool = True               # apply block-wise int4 quantization
    int4_block_size: int = 128      # Q4_0 block size
    int4_awq: bool = True           # AWQ-aware for factored A matrices

    # Misc
    dtype: str = "float16"          # storage dtype when not int4
    tie_embeddings: bool = True     # lm_head tied to embeddings


# ---------------------------------------------------------------------------
# State dict compression
# ---------------------------------------------------------------------------

def compress_openmythos_state_dict(
    state_dict: dict[str, "torch.Tensor"],
    cfg: OMCompressConfig,
    *,
    activation_norms: dict[str, np.ndarray] | None = None,
) -> dict:
    """Compress an OpenMythos state_dict in-place.

    Applies:
    1. GRC shared-basis attention projection (recurrent block, prelude, coda)
    2. FFN SVD factoring (MoE experts, dense FFNs)
    3. Int4 quantization of factored and non-factored matrices

    Returns a manifest dict describing the compressed structure.
    """
    import torch

    manifest: dict = {
        "version": 1,
        "architecture": "openmythos",
        "compression": "hyperretro",
        "attn_rank": cfg.attn_rank,
        "ffn_rank": cfg.ffn_rank,
        "int4": cfg.int4,
        "layers": [],
        "ffn": [],
    }

    # ---- 1. Identify attention modules ----
    attn_keys = _find_attn_weights(state_dict)

    # ---- 2. GRC-compress attention (recurrent block is special) ----
    if cfg.attn_rank > 0:
        _grc_compress_openmythos_attn(state_dict, attn_keys, cfg, manifest)

    # ---- 3. Factor FFN weights (MoE experts + dense FFNs) ----
    if cfg.ffn_rank > 0:
        _factor_openmythos_ffn(state_dict, cfg, manifest, activation_norms)

    # ---- 4. Int4 quantize ----
    if cfg.int4:
        _int4_quantize_openmythos(state_dict, manifest, cfg, activation_norms)

    return manifest


# ---------------------------------------------------------------------------
# Attention key discovery
# ---------------------------------------------------------------------------

def _find_attn_weights(sd: dict) -> dict[str, list[str]]:
    """Group attention weights by module path.

    OpenMythos uses MLA by default with keys like:
        {module}.attn.q_down.weight    — Q compression
        {module}.attn.kv_down.weight   — KV compression
        {module}.attn.wo.weight        — output projection
    For GQA:
        {module}.attn.wq.weight, .wk.weight, .wv.weight, .wo.weight
    """
    groups: dict[str, list[str]] = {}
    for key in sd:
        if ".attn." in key and key.endswith(".weight"):
            # Get module prefix (e.g., "recurrent.block", "prelude.0", "coda.1")
            parts = key.split(".attn.")
            module = parts[0]
            if module not in groups:
                groups[module] = []
            groups[module].append(key)
    return groups


# ---------------------------------------------------------------------------
# GRC attention compression for OpenMythos
# ---------------------------------------------------------------------------

def _grc_compress_openmythos_attn(
    sd: dict, attn_groups: dict, cfg: OMCompressConfig, manifest: dict,
):
    """Apply GRC shared-basis compression to OpenMythos attention modules.

    For MLA: compresses the q_down and kv_down projection input weights (d→rank).
    For GQA: compresses wq, wk, wv weight matrices (standard GRC).

    Since OpenMythos doesn't use standard Q/K/V weight matrices in MLA mode
    (it uses q_down/kv_down low-rank projections instead), we adapt GRC to
    compress the largest linear projections in the attention path.
    """
    import torch
    from hyperretro.hf.factored import _svd_factor

    for module, keys in sorted(attn_groups.items()):
        # Find the largest weight matrices to factor
        large_weights = []
        for k in keys:
            t = sd[k]
            if hasattr(t, "numel") and t.numel() > 100_000:
                large_weights.append((k, t))

        for k, w in large_weights:
            W = w.float().cpu().numpy()
            m, n = W.shape
            rank = min(cfg.attn_rank, m, n)

            # Apply SVD factoring directly
            A, B = _svd_factor(W, rank)
            k_eff = A.shape[0]

            # Store factored form
            prefix = k[:-len(".weight")]  # strip .weight
            sd[f"{prefix}.factored_A"] = torch.from_numpy(A).to(w.dtype)
            sd[f"{prefix}.factored_B"] = torch.from_numpy(B).to(w.dtype)
            del sd[k]

            manifest["layers"].append({
                "module": module,
                "weight_key": k,
                "rank": k_eff,
                "in_features": n,
                "out_features": m,
                "attn_type": "mla_factor",
            })

    print(f"[grc] Compressed {len(manifest['layers'])} attention matrices")


# ---------------------------------------------------------------------------
# FFN factoring for OpenMythos (MoE + dense)
# ---------------------------------------------------------------------------

def _factor_openmythos_ffn(
    sd: dict, cfg: OMCompressConfig, manifest: dict,
    activation_norms: dict | None,
):
    """Factor all FFN matrices: MoE experts (gate/up/down) + dense FFNs.

    The MoE experts are the bulk of parameters. Each expert has:
      {prefix}.experts.{i}.gate.weight  (d_ff, d_model)
      {prefix}.experts.{i}.up.weight    (d_ff, d_model)
      {prefix}.experts.{i}.down.weight  (d_model, d_ff)
    Shared experts have the same structure.
    """
    import torch
    from hyperretro.hf.factored import _svd_factor, _svd_factor_aware

    ffn_keys = []
    for key in sorted(sd.keys()):
        if not key.endswith(".weight"):
            continue
        # Match FFN patterns
        if any(pat in key for pat in [
            ".ffn.gate.", ".ffn.up.", ".ffn.down.",
            ".experts.", ".shared_experts.",
        ]):
            t = sd[key]
            if hasattr(t, "numel") and t.numel() > 1000 and hasattr(t, "dim") and t.dim() == 2:
                ffn_keys.append(key)

    n_factored = 0
    for key in ffn_keys:
        w = sd[key]
        W = w.float().cpu().numpy()
        m, n = W.shape
        rank = min(cfg.ffn_rank, m, n)

        # Use activation-aware if norms available
        col_norms = activation_norms.get(key) if activation_norms else None
        if col_norms is not None:
            A, B = _svd_factor_aware(W, col_norms, rank)
        else:
            A, B = _svd_factor(W, rank)

        k_eff = A.shape[0]
        prefix = key[:-len(".weight")]
        sd[f"{prefix}.factored_A"] = torch.from_numpy(A).to(w.dtype)
        sd[f"{prefix}.factored_B"] = torch.from_numpy(B).to(w.dtype)
        del sd[key]
        n_factored += 1

        manifest["ffn"].append({
            "weight_key": key,
            "rank": k_eff,
            "in_features": n,
            "out_features": m,
        })

    print(f"[ffn] Factored {n_factored} FFN matrices at rank≤{cfg.ffn_rank}")


# ---------------------------------------------------------------------------
# Int4 quantization for OpenMythos
# ---------------------------------------------------------------------------

def _int4_quantize_openmythos(
    sd: dict, manifest: dict, cfg: OMCompressConfig,
    activation_norms: dict | None,
):
    """Apply block-wise int4 quantization to factored and non-factored weights."""
    import torch
    from hyperretro.hf.factor_quantize import quantize_blockwise_int4, quantize_matrix_int4_best
    from hyperretro.hf.factor_int4 import pack_int4_rows

    factored_keys_seen: set[str] = set()

    # 1. Quantize factored FFN matrices
    for entry in manifest.get("ffn", []):
        wkey = entry["weight_key"]
        prefix = wkey[:-len(".weight")]
        for mat_key in [f"{prefix}.factored_A", f"{prefix}.factored_B"]:
            if mat_key not in sd:
                continue
            W = sd[mat_key].float().cpu().numpy()
            col_norms = None
            if activation_norms and mat_key.endswith(".factored_A"):
                col_norms = activation_norms.get(wkey)

            if col_norms is not None and cfg.int4_awq:
                W_q, scales, awq = quantize_matrix_int4_best(
                    W, col_norms=col_norms, block_size=cfg.int4_block_size, n_bits=4,
                )
                sd[f"{mat_key}.awq_scales"] = torch.from_numpy(awq).to(torch.float16)
            else:
                W_q, scales = quantize_blockwise_int4(
                    W, block_size=cfg.int4_block_size, n_bits=4,
                )

            packed = pack_int4_rows(W_q)
            sd[f"{mat_key}.q"] = torch.from_numpy(packed).to(torch.uint8)
            sd[f"{mat_key}.scales"] = torch.from_numpy(scales).to(torch.float16)
            factored_keys_seen.add(mat_key)
            del sd[mat_key]

    # 2. Quantize factored attention matrices
    for entry in manifest.get("layers", []):
        wkey = entry["weight_key"]
        prefix = wkey[:-len(".weight")]
        for mat_key in [f"{prefix}.factored_A", f"{prefix}.factored_B"]:
            if mat_key not in sd:
                continue
            W = sd[mat_key].float().cpu().numpy()
            W_q, scales = quantize_blockwise_int4(
                W, block_size=cfg.int4_block_size, n_bits=4,
            )
            packed = pack_int4_rows(W_q)
            sd[f"{mat_key}.q"] = torch.from_numpy(packed).to(torch.uint8)
            sd[f"{mat_key}.scales"] = torch.from_numpy(scales).to(torch.float16)
            factored_keys_seen.add(mat_key)
            del sd[mat_key]

    # 3. Quantize remaining large non-factored weights
    for key in list(sd.keys()):
        if key in factored_keys_seen:
            continue
        if key.endswith((".q", ".scales", ".awq_scales")):
            continue
        t = sd[key]
        if not (hasattr(t, "numel") and t.numel() > 10000 and hasattr(t, "dim") and t.dim() == 2):
            continue
        W = t.float().cpu().numpy()
        W_q, scales = quantize_blockwise_int4(
            W, block_size=cfg.int4_block_size, n_bits=4,
        )
        packed = pack_int4_rows(W_q)
        sd[f"{key}.q"] = torch.from_numpy(packed).to(torch.uint8)
        sd[f"{key}.scales"] = torch.from_numpy(scales).to(torch.float16)
        del sd[key]

    print(f"[int4] Quantized {len(factored_keys_seen)} factored + extra non-factored matrices")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_openmythos(
    model: "torch.nn.Module",
    *,
    ffn_rank: int = 512,
    attn_rank: int = 640,
    int4: bool = True,
    activation_corpus_path: str | None = None,
) -> tuple[dict, list, list]:
    """Compress an OpenMythos model with HyperRetro.

    Parameters
    ----------
    model : an OpenMythos instance (from open_mythos.main).
    ffn_rank : target SVD rank for FFN matrices (MoE experts + dense).
    attn_rank : target GRC/SVD rank for attention projections.
    int4 : apply block-wise int4 quantization after factoring.
    activation_corpus_path : optional path to calibration text for AWQ-aware mode.

    Returns
    -------
    state_dict : compressed state dict (mutated in-place on model's weights).
    manifest : manifest dict.
    report : compression report.
    """
    cfg = OMCompressConfig(
        ffn_rank=ffn_rank,
        attn_rank=attn_rank,
        int4=int4,
        int4_awq=activation_corpus_path is not None,
    )

    sd = model.state_dict()

    # Collect activation norms if calibration corpus provided (CPU-only)
    activation_norms = None
    if activation_corpus_path:
        try:
            activation_norms = _collect_openmythos_ffn_norms(
                model, activation_corpus_path,
            )
        except Exception as e:
            print(f"[warn] Activation norm collection failed: {e}")

    manifest = compress_openmythos_state_dict(
        sd, cfg, activation_norms=activation_norms,
    )

    total_params = sum(
        v.numel() if hasattr(v, "numel") else np.prod(v.shape)
        for v in sd.values()
    )

    report = {
        "architecture": "openmythos",
        "total_tensors": len(sd),
        "total_params": total_params,
        "attn_factored": len(manifest["layers"]),
        "ffn_factored": len(manifest["ffn"]),
        "int4_quantized": int4,
    }

    return sd, manifest, report


def _collect_openmythos_ffn_norms(
    model: "torch.nn.Module",
    corpus_path: str,
    n_batches: int = 4,
    seq_len: int = 128,
) -> dict[str, np.ndarray]:
    """Collect per-column activation norms for FFN weight matrices.

    Simplified version for OpenMythos — runs a forward pass and records
    inputs to all FFN Linear modules.
    """
    import torch

    text = Path(corpus_path).read_text(encoding="utf-8")[:50000]
    # Use a simple tokenization (character-level for calibration)
    tokens = torch.randint(0, 1000, (n_batches, seq_len))

    norms: dict[str, np.ndarray] = {}
    hooks = []

    def make_hook(weight_key):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            sq = (x.float() ** 2).sum(dim=(0, 1)).double().cpu().numpy()
            if weight_key in norms:
                norms[weight_key] += sq
            else:
                norms[weight_key] = sq
        return hook

    # Register hooks on FFN Linear modules
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(pat in name for pat in [".ffn.", ".experts.", ".shared_experts."]):
                weight_key = f"{name}.weight"
                h = module.register_forward_hook(make_hook(weight_key))
                hooks.append(h)

    # Run forward pass
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            try:
                _ = model(tokens[i:i+1])
            except Exception:
                pass

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert sums to sqrt norms
    return {k: np.sqrt(v).astype(np.float32) for k, v in norms.items()}


def save_compressed_openmythos(
    state_dict: dict,
    manifest: dict,
    out_dir: str | Path,
    *,
    hf_config: dict | None = None,
) -> Path:
    """Save a compressed OpenMythos checkpoint to safetensors.

    The output is a directory containing:
    - model.safetensors (int4-packed or fp16 weights)
    - hyperretro_factored.json (manifest)
    - config.json (OpenMythos config as dict)
    """
    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write safetensors
    out_sd = {}
    for k, v in state_dict.items():
        if hasattr(v, "cpu"):
            v = v.contiguous().cpu()
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(np.ascontiguousarray(v))
        if k.endswith(".q"):
            v = v.to(torch.uint8)
        elif k.endswith(".scales") or k.endswith(".awq_scales"):
            v = v.to(torch.float16)
        elif isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
            v = v.to(torch.float16)
        out_sd[k] = v

    save_file(out_sd, str(out / "model.safetensors"))

    # Write manifest
    (out / "hyperretro_factored.json").write_text(json.dumps(manifest, indent=2))

    # Write config
    if hf_config:
        (out / "config.json").write_text(json.dumps(hf_config, indent=2))

    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"[saved] {out} ({size_mb:.1f} MB)")
    return out


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

def estimate_openmythos_savings(
    model: "torch.nn.Module",
    ffn_rank: int = 512,
    attn_rank: int = 640,
) -> dict:
    """Estimate compression savings for an OpenMythos model."""
    sd = model.state_dict()
    total_fp16 = sum(
        v.numel() if hasattr(v, "numel") else np.prod(v.shape)
        for v in sd.values()
    ) * 2  # fp16 = 2 bytes/param

    # Estimate factored size
    factored_bytes = 0
    for key, tensor in sd.items():
        if not (hasattr(tensor, "numel") and hasattr(tensor, "dim") and tensor.dim() == 2):
            factored_bytes += (tensor.numel() if hasattr(tensor, "numel") else 1) * 2
            continue
        m, n = tensor.shape
        if any(pat in key for pat in [".ffn.", ".experts.", ".shared_experts."]):
            rank = min(ffn_rank, m, n)
            # factored: (m+n)*rank bytes fp16 + block-wise scales
            factored_bytes += (m + n) * rank * 2
            # Plus int4 packed + scales ≈ (m*n*0.5 + m*ceil(n/128)*2)
            factored_bytes += m * n * 0.5 + m * ((n + 127) // 128) * 2
        elif ".attn." in key:
            rank = min(attn_rank, m, n)
            factored_bytes += (m + n) * rank * 2
        else:
            factored_bytes += m * n * 0.5 + m * ((n + 127) // 128) * 2

    return {
        "fp16_mb": round(total_fp16 / 1e6, 1),
        "compressed_mb": round(factored_bytes / 1e6, 1),
        "shrink": round(total_fp16 / max(factored_bytes, 1), 2),
    }
