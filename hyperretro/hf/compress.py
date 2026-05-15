"""Offline HuggingFace model compression via GRC / sink-aware projection.

Pipeline:
  1. Load a standard HF / transformers model (or any directory with
     ``.safetensors`` + ``config.json``).
  2. For each transformer block, gather the (W_q, W_k, W_v) attention
     projection matrices.
  3. Build the GRC shared basis P_t from the joint Gram (mirrors
     ``scripts/grc_distill.build_shared_basis`` and
     ``runtime/nn/axiom_exploit.c``), optionally exempt the top-T
     sink channels (Sun et al. 2024).
  4. Project Q/K/V onto the top-k columns of P_t, leaving sink columns
     untouched.
  5. Write the compressed weights back as standard ``.safetensors``
     shards plus a copy of the original ``config.json`` and any
     tokenizer files, so the result is loadable with vanilla
     ``AutoModelForCausalLM.from_pretrained``.

The output stays 100% inside the HuggingFace ecosystem; no HyperTensor
runtime is required to use the compressed model.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# Reuse the geometry primitives that ship with the standalone HyperTensor
# scripts so we are provably consistent with Paper E / axiom_exploit.c.
try:
    from scripts.grc_distill import (  # type: ignore
        build_shared_basis,
        project,
        sink_indices,
        compute_rho,
    )
    _GRC_SRC = "scripts.grc_distill"
except Exception:
    # Standalone fallback (copy of the math from scripts/grc_distill.py).
    def build_shared_basis(Wq, Wk, Wv, n_iter: int = 3):  # type: ignore[no-redef]
        K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
        K = K / np.linalg.norm(K, "fro")
        A = K.copy()
        for _ in range(n_iter):
            A = A @ K
            A = A / np.linalg.norm(A, "fro")
        eigvals, eigvecs = np.linalg.eigh(A)
        order = np.argsort(eigvals)[::-1]
        return eigvecs[:, order]

    def project(W, P, k):  # type: ignore[no-redef]
        Pk = P[:, :k]
        return W @ Pk @ Pk.T

    def sink_indices(Wq, Wk, Wv, T: int):  # type: ignore[no-redef]
        if T <= 0:
            return np.array([], dtype=np.int64)
        mag = (np.linalg.norm(Wq, axis=0) ** 2
               + np.linalg.norm(Wk, axis=0) ** 2
               + np.linalg.norm(Wv, axis=0) ** 2)
        return np.argsort(mag)[::-1][:T]

    def compute_rho(*args, **kwargs):  # type: ignore[no-redef]
        return float("nan")

    _GRC_SRC = "hyperretro.hf.compress (vendored)"


@dataclass
class CompressConfig:
    rank_k: int = 1024            # GRC rank
    sink_T: int = 0               # 0 = vanilla GRC; >0 = sink-aware
    layers: list[int] = field(default_factory=list)  # [] = all
    dtype: str = "float32"        # storage dtype: float16 / bfloat16 / float32
    factored: bool = False        # emit (A, B_*) directly instead of B@A
    name_patterns: tuple[str, ...] = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    )


# ---------------------------------------------------------------------------
# Tensor-name grouping. Supports:
#
#   Llama / Qwen2 / Mistral:
#     model.layers.{i}.self_attn.{q,k,v}_proj.weight
#
#   GPT-2 (fused QKV):
#     transformer.h.{i}.attn.c_attn.weight   → split [d, 3d]
#     transformer.h.{i}.attn.c_proj.weight    → output projection (not GRC'd)
#
#   Gemma (fused QKV):
#     model.layers.{i}.self_attn.qkv_proj.weight  → split by head counts
#
#   Phi-3 (fused QKV):
#     model.layers.{i}.self_attn.qkv_proj.weight
#
#   Falcon (fused QKV):
#     transformer.h.{i}.self_attention.query_key_value.weight
#
#   BERT:
#     encoder.layer.{i}.attention.self.{query,key,value}.weight
#
#   GPT-NeoX / Pythia:
#     gpt_neox.layers.{i}.attention.{query,key,value}_proj.weight
#
# For fused architectures we split the fused tensor into Q/K/V slices,
# compress each slice independently via the shared basis, then recombine.
# ---------------------------------------------------------------------------

# Patterns: (regex_substring, layer_group, {"q": suffix, "k": suffix, "v": suffix})
# or for fused: (regex_substring, layer_group, {"qkv": suffix})
_ATTN_PATTERNS: list[tuple[str, int, dict[str, str]]] = [
    # Llama / Qwen2 / Mistral — separate Q/K/V
    (".layers.", 0, {"q": ".self_attn.q_proj.weight",
                      "k": ".self_attn.k_proj.weight",
                      "v": ".self_attn.v_proj.weight"}),
    # GPT-2 — fused QKV
    ("transformer.h.", 1, {"qkv": ".attn.c_attn.weight"}),
    # Gemma — fused QKV
    (".layers.", 0, {"qkv": ".self_attn.qkv_proj.weight"}),
    # Phi-3 — fused QKV (superseded by Gemma pattern above, same key)
    # Falcon — fused QKV
    ("transformer.h.", 1, {"qkv": ".self_attention.query_key_value.weight"}),
    # GPT-NeoX / Pythia — separate Q/K/V
    ("gpt_neox.layers.", 1, {"q": ".attention.query_proj.weight",
                               "k": ".attention.key_proj.weight",
                               "v": ".attention.value_proj.weight"}),
    # BERT
    ("encoder.layer.", 1, {"q": ".attention.self.query.weight",
                            "k": ".attention.self.key.weight",
                            "v": ".attention.self.value.weight"}),
]


def _parse_layer_idx(name: str, marker: str, group: int) -> int | None:
    """Extract layer index from name after marker."""
    idx = name.find(marker)
    if idx < 0:
        return None
    rest = name[idx + len(marker):]
    # Find first non-digit boundary after the layer number
    end = 0
    while end < len(rest) and rest[end].isdigit():
        end += 1
    if end == 0:
        return None
    try:
        return int(rest[:end])
    except ValueError:
        return None


def _group_attn_by_layer(state_dict: dict[str, "np.ndarray | object"]) -> dict[int, dict[str, str]]:
    """Discover attention Q/K/V (or fused QKV) weight keys by layer index.

    Returns:
        dict mapping layer_idx → {"q": key, "k": key, "v": key}
        or layer_idx → {"qkv": key} for fused architectures.
    """
    layers: dict[int, dict[str, str]] = {}

    for name in state_dict.keys():
        for marker, group, patterns in _ATTN_PATTERNS:
            li = _parse_layer_idx(name, marker, group)
            if li is None:
                continue
            for slot, suffix in patterns.items():
                if name.endswith(suffix):
                    # Don't overwrite a separate Q/K/V entry with a fused one
                    existing = layers.setdefault(li, {})
                    if slot not in existing:
                        existing[slot] = name
                    break
            else:
                continue
            break  # matched one pattern, stop trying others
    return layers


def _to_np(t) -> np.ndarray:
    if hasattr(t, "detach"):  # torch tensor
        return t.detach().to("cpu").float().numpy()
    return np.asarray(t, dtype=np.float32)


def _cast(arr: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return arr.astype(np.float32)
    if dtype == "float16":
        return arr.astype(np.float16)
    if dtype == "bfloat16":
        # safetensors writes bf16 by interpreting uint16 frames; defer to torch.
        import torch
        return arr  # caller will branch on bf16 via torch path
    raise ValueError(f"unknown dtype {dtype!r}")


def _split_fused_qkv(W_fused: np.ndarray, config: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a fused QKV weight into (Wq, Wk, Wv).

    Handles:
      - GPT-2: c_attn [d, 3*d] → three equal [d, d] slices
      - Gemma: qkv_proj [(n_q_heads+n_kv_heads+n_kv_heads)*head_dim, d]
                Split by head counts from config.
    """
    out_dim, in_dim = W_fused.shape

    # Try config-based split first (Gemma, Phi with GQA)
    if config:
        n_q_heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads")
        n_kv_heads = config.get("num_key_value_heads") or n_q_heads
        head_dim = config.get("head_dim") or (config.get("hidden_size", in_dim) // n_q_heads if n_q_heads else None)
        if n_q_heads and head_dim:
            q_dim = n_q_heads * head_dim
            kv_dim = n_kv_heads * head_dim
            if q_dim + 2 * kv_dim == out_dim:
                Wq = W_fused[:q_dim, :]
                Wk = W_fused[q_dim:q_dim + kv_dim, :]
                Wv = W_fused[q_dim + kv_dim:, :]
                return Wq, Wk, Wv

    # Default: assume equal 3-way split (GPT-2, standard transformers)
    if out_dim % 3 == 0:
        d_part = out_dim // 3
        Wq = W_fused[:d_part, :]
        Wk = W_fused[d_part:2*d_part, :]
        Wv = W_fused[2*d_part:, :]
        return Wq, Wk, Wv

    raise ValueError(
        f"Cannot split fused QKV of shape {W_fused.shape}. "
        f"out_dim={out_dim} is not divisible by 3 and no config provided."
    )


def compress_state_dict(state_dict: dict, cfg: CompressConfig) -> dict[str, dict]:
    """Apply GRC projection in place on a state-dict-like mapping.

    Handles both separate Q/K/V and fused QKV (GPT-2, Gemma, etc.).

    Returns a per-layer stats dict (rank, sink_T, frob_relerr).
    """
    stats: dict[str, dict] = {}
    layer_map = _group_attn_by_layer(state_dict)
    layers_to_do: Iterable[int] = sorted(layer_map.keys())
    if cfg.layers:
        layers_to_do = [i for i in layers_to_do if i in cfg.layers]

    # Try to discover model config for fused QKV splitting
    model_config = state_dict.get("_config", None)
    if model_config is None:
        # Try to extract from any available metadata
        for key in ("config", "_model_config", "__config__"):
            if key in state_dict:
                model_config = state_dict[key]
                break
    if isinstance(model_config, dict):
        pass  # use as-is
    elif hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()
    else:
        model_config = None

    for li in layers_to_do:
        names = layer_map[li]

        # --- Fused QKV path ---
        if "qkv" in names and "q" not in names:
            W_fused = _to_np(state_dict[names["qkv"]])
            Wq_orig, Wk_orig, Wv_orig = _split_fused_qkv(W_fused, model_config)
            d = Wq_orig.shape[1]
            k = min(cfg.rank_k, d)
            sink = sink_indices(Wq_orig, Wk_orig, Wv_orig, cfg.sink_T)

            if cfg.sink_T > 0 and len(sink) > 0:
                Wq_R = Wq_orig.copy(); Wq_R[:, sink] = 0.0
                Wk_R = Wk_orig.copy(); Wk_R[:, sink] = 0.0
                Wv_R = Wv_orig.copy(); Wv_R[:, sink] = 0.0
                P = build_shared_basis(Wq_R, Wk_R, Wv_R)
                Wq_new = project(Wq_R, P, k); Wq_new[:, sink] = Wq_orig[:, sink]
                Wk_new = project(Wk_R, P, k); Wk_new[:, sink] = Wk_orig[:, sink]
                Wv_new = project(Wv_R, P, k); Wv_new[:, sink] = Wv_orig[:, sink]
            else:
                P = build_shared_basis(Wq_orig, Wk_orig, Wv_orig)
                Wq_new = project(Wq_orig, P, k)
                Wk_new = project(Wk_orig, P, k)
                Wv_new = project(Wv_orig, P, k)

            W_fused_new = np.concatenate([Wq_new, Wk_new, Wv_new], axis=0)

            def _relerr(W, Wh):
                n = np.linalg.norm(W)
                return float(np.linalg.norm(W - Wh) / n) if n > 0 else 0.0

            stats[f"layer_{li}"] = {
                "rank_k": k, "sink_T": cfg.sink_T, "d": int(d),
                "frob_relerr_q": _relerr(Wq_orig, Wq_new),
                "frob_relerr_k": _relerr(Wk_orig, Wk_new),
                "frob_relerr_v": _relerr(Wv_orig, Wv_new),
                "fused": True,
            }

            # Write back fused
            orig = state_dict[names["qkv"]]
            if hasattr(orig, "detach"):
                import torch
                t = torch.from_numpy(W_fused_new).to(orig.dtype).to(orig.device)
                state_dict[names["qkv"]] = t
            else:
                state_dict[names["qkv"]] = W_fused_new.astype(getattr(orig, "dtype", np.float32))
            continue

        # --- Separate Q/K/V path ---
        if not all(k in names for k in ("q", "k", "v")):
            continue
        Wq = _to_np(state_dict[names["q"]])
        Wk = _to_np(state_dict[names["k"]])
        Wv = _to_np(state_dict[names["v"]])
        d = Wq.shape[1]
        k = min(cfg.rank_k, d)
        sink = sink_indices(Wq, Wk, Wv, cfg.sink_T)

        if cfg.sink_T > 0 and len(sink) > 0:
            Wq_R = Wq.copy(); Wq_R[:, sink] = 0.0
            Wk_R = Wk.copy(); Wk_R[:, sink] = 0.0
            Wv_R = Wv.copy(); Wv_R[:, sink] = 0.0
            P = build_shared_basis(Wq_R, Wk_R, Wv_R)
            Wq_new = project(Wq_R, P, k); Wq_new[:, sink] = Wq[:, sink]
            Wk_new = project(Wk_R, P, k); Wk_new[:, sink] = Wk[:, sink]
            Wv_new = project(Wv_R, P, k); Wv_new[:, sink] = Wv[:, sink]
        else:
            P = build_shared_basis(Wq, Wk, Wv)
            Wq_new = project(Wq, P, k)
            Wk_new = project(Wk, P, k)
            Wv_new = project(Wv, P, k)

        def _relerr(W, Wh):
            n = np.linalg.norm(W)
            return float(np.linalg.norm(W - Wh) / n) if n > 0 else 0.0

        stats[f"layer_{li}"] = {
            "rank_k": k,
            "sink_T": cfg.sink_T,
            "d": int(d),
            "frob_relerr_q": _relerr(Wq, Wq_new),
            "frob_relerr_k": _relerr(Wk, Wk_new),
            "frob_relerr_v": _relerr(Wv, Wv_new),
        }

        # Write back, preserving original dtype where possible.
        def _write(name: str, arr: np.ndarray):
            orig = state_dict[name]
            if hasattr(orig, "detach"):  # torch tensor in original
                import torch
                t = torch.from_numpy(arr).to(orig.dtype).to(orig.device)
                state_dict[name] = t
            else:
                state_dict[name] = arr.astype(getattr(orig, "dtype", np.float32))

        _write(names["q"], Wq_new)
        _write(names["k"], Wk_new)
        _write(names["v"], Wv_new)

    return stats


def compress_state_dict_factored(
    state_dict: dict,
    cfg: CompressConfig,
) -> tuple[dict[str, dict], list[dict]]:
    """GRC-compress and emit (A, B_*) factored keys directly.

    Unlike :func:`compress_state_dict` (which rematerialises to dense
    d×d weights), this writes::

        {prefix}.factored_A         (k+T, d_in)
        {prefix}.factored_Bq        (d_out_q, k+T)
        {prefix}.factored_Bk        (d_out_kv, k+T)
        {prefix}.factored_Bv        (d_out_kv, k+T)

    and removes the original ``q_proj.weight`` / ``k_proj.weight`` /
    ``v_proj.weight`` keys. Sink-T columns are folded in exactly via T
    extra rank-1 rows so the result equals the dense GRC + sink-restore
    output to machine precision (no SVD-retrofit loss).

    Returns ``(stats, ffn_entries)`` where ``ffn_entries`` is an empty list
    (FFN factoring is delegated to :func:`hyperretro.hf.factored.factor_ffn_state_dict`).
    """
    import torch

    stats: dict[str, dict] = {}
    factored_layers: list[dict] = []
    layer_map = _group_attn_by_layer(state_dict)
    layers_to_do: Iterable[int] = sorted(layer_map.keys())
    if cfg.layers:
        layers_to_do = [i for i in layers_to_do if i in cfg.layers]

    def _relerr(W, Wh):
        n = np.linalg.norm(W)
        return float(np.linalg.norm(W - Wh) / n) if n > 0 else 0.0

    for li in layers_to_do:
        names = layer_map[li]
        # v1: skip fused-QKV; supported in dense path only.
        if "qkv" in names and "q" not in names:
            continue
        if not all(s in names for s in ("q", "k", "v")):
            continue

        Wq = _to_np(state_dict[names["q"]])
        Wk = _to_np(state_dict[names["k"]])
        Wv = _to_np(state_dict[names["v"]])
        d_in = Wq.shape[1]
        if Wk.shape[1] != d_in or Wv.shape[1] != d_in:
            continue  # heterogeneous input dim (not standard transformer)
        k = min(cfg.rank_k, d_in)
        sink = sink_indices(Wq, Wk, Wv, cfg.sink_T)

        if cfg.sink_T > 0 and len(sink) > 0:
            Wq_R = Wq.copy(); Wq_R[:, sink] = 0.0
            Wk_R = Wk.copy(); Wk_R[:, sink] = 0.0
            Wv_R = Wv.copy(); Wv_R[:, sink] = 0.0
            P_full = build_shared_basis(Wq_R, Wk_R, Wv_R)
            Pk = P_full[:, :k]                       # (d_in, k)
            A_grc = Pk.T                              # (k, d_in)
            # B = W_R @ P_k  (sink cols zeroed in W_R, so they don't leak)
            Bq_grc = Wq_R @ Pk
            Bk_grc = Wk_R @ Pk
            Bv_grc = Wv_R @ Pk
            # Sink fold-in: append T extra rows to A (identity rows at sink cols)
            # and T extra columns to each B (the original sink columns).
            T = len(sink)
            sink_rows = np.zeros((T, d_in), dtype=A_grc.dtype)
            for i, idx in enumerate(sink):
                sink_rows[i, idx] = 1.0
            A = np.concatenate([A_grc, sink_rows], axis=0)             # (k+T, d_in)
            Bq = np.concatenate([Bq_grc, Wq[:, sink]], axis=1)         # (d_out_q, k+T)
            Bk = np.concatenate([Bk_grc, Wk[:, sink]], axis=1)
            Bv = np.concatenate([Bv_grc, Wv[:, sink]], axis=1)
            # Reconstruction (for relerr only)
            Wq_rec = Bq @ A
            Wk_rec = Bk @ A
            Wv_rec = Bv @ A
        else:
            P_full = build_shared_basis(Wq, Wk, Wv)
            Pk = P_full[:, :k]
            A = Pk.T
            Bq = Wq @ Pk
            Bk = Wk @ Pk
            Bv = Wv @ Pk
            T = 0
            Wq_rec = Bq @ A
            Wk_rec = Bk @ A
            Wv_rec = Bv @ A

        stats[f"layer_{li}"] = {
            "rank_k": k, "sink_T": int(T), "d": int(d_in),
            "frob_relerr_q": _relerr(Wq, Wq_rec),
            "frob_relerr_k": _relerr(Wk, Wk_rec),
            "frob_relerr_v": _relerr(Wv, Wv_rec),
            "factored": True,
        }

        # Write factored tensors back into state_dict, preserving original
        # dtype/device.
        orig = state_dict[names["q"]]
        dtype = orig.dtype if hasattr(orig, "dtype") else None
        device = orig.device if hasattr(orig, "device") else None

        def _save(name: str, arr: np.ndarray) -> None:
            t = torch.from_numpy(arr)
            if dtype is not None:
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            state_dict[name] = t

        prefix = names["q"].rsplit(".q_proj.weight", 1)[0]
        _save(f"{prefix}.factored_A", A)
        _save(f"{prefix}.factored_Bq", Bq)
        _save(f"{prefix}.factored_Bk", Bk)
        _save(f"{prefix}.factored_Bv", Bv)

        # Bias passthrough
        bias_keys: dict[str, str] = {}
        for slot in ("q", "k", "v"):
            bk = names[slot].replace(".weight", ".bias")
            if bk in state_dict:
                bias_keys[slot] = bk

        # Remove dense weight keys
        for slot in ("q", "k", "v"):
            state_dict.pop(names[slot], None)

        factored_layers.append({
            "layer_idx": li,
            "rank": int(k + T),
            "in_features": int(d_in),
            "out_features": int(Wq.shape[0]),  # for the Q side; K/V may differ
            "q_key": names["q"],
            "k_key": names["k"],
            "v_key": names["v"],
            "biases": bias_keys,
        })

    return stats, factored_layers


def compress_hf_model(
    model_id_or_path: str,
    out_dir: str | Path,
    *,
    rank_k: int = 1024,
    sink_T: int = 0,
    layers: list[int] | None = None,
    dtype: str = "float32",
    revision: str | None = None,
    certify: bool = False,
    ffn_rank_in: int = 0,
    ffn_rank_out: int = 0,
    ffn_mode: str = "svd",
    factored: bool = False,
    activation_aware: bool = False,
    activation_corpus_path: str | None = None,
    activation_n_batches: int = 16,
    activation_seq_len: int = 512,
    int4: bool = False,
) -> dict:
    """Load a HF model, apply GRC compression, save as standard safetensors.

    If ``certify=True``, also produces a ``hyperretro_certificate.json``
    with per-layer BP-NS bounds and a trust tier (PLATINUM/GOLD/SILVER/BRONZE).

    If ``factored=True``, emit attention projections in factored (A, B_*)
    form directly to disk. Output dir is loaded by
    :func:`hyperretro.hf.factored.load_factored_hf_model`, NOT by
    ``AutoModelForCausalLM.from_pretrained``. Use this for real on-disk
    byte savings (round-10 attack #2).

    Returns a dict with per-layer stats and the output directory.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "compress_hf_model requires the 'hf' extras: "
            "pip install hyperretro[hf]"
        ) from e

    cfg = CompressConfig(
        rank_k=rank_k, sink_T=sink_T, layers=layers or [], dtype=dtype,
        factored=factored,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, revision=revision, torch_dtype=torch.float32
    )
    sd = model.state_dict()

    # Activation-aware: collect input column 2-norms on the dense (pre-compression)
    # model before we touch the weights. This is the Wanda/AWQ signal we feed
    # into the SVD weighting for FFN factoring.
    activation_col_norms: dict | None = None
    if activation_aware and factored and (ffn_rank_in > 0 or ffn_rank_out > 0):
        if activation_corpus_path is None:
            raise ValueError(
                "activation_aware=True requires activation_corpus_path")
        from hyperretro.hf.activation import collect_ffn_input_norms
        try:
            tok_for_act = AutoTokenizer.from_pretrained(
                model_id_or_path, revision=revision)
        except Exception as e:
            raise RuntimeError(
                "activation_aware needs a tokenizer for the model") from e
        if tok_for_act.pad_token is None:
            tok_for_act.pad_token = tok_for_act.eos_token
        # Push to cuda for fast collection if available, then return to cpu.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prev_device = next(model.parameters()).device
        model_for_act = model.to(device).to(
            torch.bfloat16 if device == "cuda" else torch.float32)
        activation_col_norms = collect_ffn_input_norms(
            model_for_act, tok_for_act,
            corpus_path=activation_corpus_path,
            n_batches=activation_n_batches,
            seq_len=activation_seq_len,
            device=device,
        )
        model.to(prev_device).to(torch.float32)
        # state_dict was a reference before .to() — reacquire to be safe.
        sd = model.state_dict()

    factored_entries: list[dict] = []
    if factored:
        stats, factored_entries = compress_state_dict_factored(sd, cfg)
    else:
        stats = compress_state_dict(sd, cfg)

    # Optional FFN compression
    ffn_stats: dict = {}
    if ffn_rank_in > 0 or ffn_rank_out > 0:
        from hyperretro.hf.ffn_compress import (
            FFNCompressConfig, compress_ffn_state_dict,
        )
        ffn_cfg = FFNCompressConfig(
            rank_in=ffn_rank_in, rank_out=ffn_rank_out,
            layers=layers or [], mode=ffn_mode,
        )
        ffn_stats = compress_ffn_state_dict(sd, ffn_cfg)

    if factored:
        # Save raw state_dict (skipping dense q/k/v that were popped) +
        # factored manifest. We bypass model.save_pretrained because the
        # model object still has dense attn modules.
        from hyperretro.hf.factored import factor_ffn_state_dict, save_factored_checkpoint

        # If FFN compression was applied (rematerialised to dense), also
        # factor those Linears so we recover the bytes the rank reduction
        # bought us. Uses adaptive-rank SVD inside.
        ffn_factored_entries: list[dict] = []
        if ffn_rank_in > 0 or ffn_rank_out > 0:
            max_rank = max(ffn_rank_in, ffn_rank_out, 1)
            ffn_factored_entries = factor_ffn_state_dict(
                sd, max_rank=max_rank, rel_tol=1e-4,
                activation_col_norms=activation_col_norms,
            )

        try:
            tok = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision)
        except Exception:
            tok = None
        cfg_obj = AutoConfig.from_pretrained(model_id_or_path, revision=revision)
        if int4:
            from hyperretro.hf.factor_int4 import save_int4_factored_checkpoint
            save_int4_factored_checkpoint(
                sd, factored_entries, ffn_factored_entries,
                out_dir=out, hf_config=cfg_obj, tokenizer=tok,
                n_bits=4, block_size=128,
                quantize_non_factored=True,
                activation_norms=activation_col_norms,
            )
        else:
            save_factored_checkpoint(
                sd, factored_entries, ffn_factored_entries,
                out_dir=out, hf_config=cfg_obj, dtype=dtype, tokenizer=tok,
            )
    else:
        model.load_state_dict(sd)
        # Save in standard HF layout.
        model.save_pretrained(out, safe_serialization=True)
        try:
            tok = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision)
            tok.save_pretrained(out)
        except Exception:
            pass

    report = {
        "source": model_id_or_path,
        "out_dir": str(out),
        "config": {
            "rank_k": cfg.rank_k,
            "sink_T": cfg.sink_T,
            "layers": cfg.layers,
            "dtype": cfg.dtype,
            "grc_source": _GRC_SRC,
        },
        "per_layer": stats,
        "n_layers_compressed": len(stats),
        "ffn": {
            "rank_in": ffn_rank_in,
            "rank_out": ffn_rank_out,
            "per_layer": ffn_stats,
            "n_layers_compressed": len(ffn_stats),
        } if (ffn_rank_in > 0 or ffn_rank_out > 0) else None,
    }
    (out / "hyperretro_report.json").write_text(json.dumps(report, indent=2))

    # Certificate
    if certify:
        from hyperretro.certificates import certify_compression
        cert = certify_compression(sd, cfg, stats, model_id=model_id_or_path)
        (out / "hyperretro_certificate.json").write_text(
            json.dumps(cert.to_dict(), indent=2))
        report["certificate"] = cert.to_dict()
        print(f"\n{cert.summary()}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hyperretro-compress",
        description="Apply geometric compression to a HuggingFace model and "
                    "save the result as standard .safetensors.",
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model id or local path")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--rank", type=int, default=1024,
                   help="GRC rank k (default: 1024)")
    p.add_argument("--sink", type=int, default=0,
                   help="Number of sink channels to keep exact (default: 0 = vanilla GRC)")
    p.add_argument("--layers", type=str, default="",
                   help="Comma-separated layer indices to compress (default: all)")
    p.add_argument("--dtype",
                   choices=["float32", "float16", "bfloat16"],
                   default="float32",
                   help="Storage dtype of the projected weights")
    p.add_argument("--factored", action="store_true",
                   help="Emit attn projections as (A, B_*) factored safetensors "
                        "with on-disk byte savings. Load with "
                        "hyperretro.hf.factored.load_factored_hf_model.")
    p.add_argument("--revision", default=None, help="HF model revision/tag")
    p.add_argument("--certify", action="store_true",
                   help="Produce a hyperretro_certificate.json with per-layer BP-NS bounds")
    p.add_argument("--ffn-rank-in", type=int, default=0,
                   help="FFN gate/up shared-basis rank (hidden_size axis). 0 = no FFN compression on gate/up.")
    p.add_argument("--ffn-rank-out", type=int, default=0,
                   help="FFN down_proj SVD rank (intermediate axis). For GPT-2 style MLPs, applies to both c_fc and c_proj.")
    p.add_argument("--ffn-mode", choices=["svd", "shared"], default="svd",
                   help="FFN gate/up compression mode. 'svd' = independent SVD (default, safe). "
                        "'shared' = joint shared basis (known to break SwiGLU gating; ablation only).")
    p.add_argument("--activation-aware", action="store_true",
                   help="Use activation-aware SVD for FFN factoring (round 13)")
    p.add_argument("--activation-corpus", default=None,
                   help="Calibration corpus for activation norms")
    p.add_argument("--activation-n-batches", type=int, default=16)
    p.add_argument("--activation-seq-len", type=int, default=512)
    p.add_argument("--int4", action="store_true",
                   help="Store factored matrices as packed int4 (attack #7, 5-14x shrink)")
    args = p.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    report = compress_hf_model(
        args.model,
        args.out,
        rank_k=args.rank,
        sink_T=args.sink,
        layers=layers,
        dtype=args.dtype,
        revision=args.revision,
        certify=args.certify,
        ffn_rank_in=args.ffn_rank_in,
        ffn_rank_out=args.ffn_rank_out,
        ffn_mode=args.ffn_mode,
        factored=args.factored,
        activation_aware=args.activation_aware,
        activation_corpus_path=args.activation_corpus,
        activation_n_batches=args.activation_n_batches,
        activation_seq_len=args.activation_seq_len,
        int4=args.int4,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
