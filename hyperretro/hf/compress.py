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
    name_patterns: tuple[str, ...] = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    )


# ---------------------------------------------------------------------------
# Tensor-name grouping. Llama / Mistral / Qwen2 all use
#   model.layers.{i}.self_attn.{q,k,v}_proj.weight
# We discover layer indices from the keys themselves rather than asking the
# model's config so we don't need transformers loaded to merely inspect.
# ---------------------------------------------------------------------------

def _group_attn_by_layer(state_dict: dict[str, "np.ndarray | object"]) -> dict[int, dict[str, str]]:
    layers: dict[int, dict[str, str]] = {}
    for name in state_dict.keys():
        for tag, key in [("q", ".self_attn.q_proj.weight"),
                         ("k", ".self_attn.k_proj.weight"),
                         ("v", ".self_attn.v_proj.weight")]:
            if name.endswith(key):
                # parse layer idx from ".layers.<i>." substring
                marker = ".layers."
                if marker not in name:
                    continue
                start = name.index(marker) + len(marker)
                end = name.index(".", start)
                try:
                    li = int(name[start:end])
                except ValueError:
                    continue
                layers.setdefault(li, {})[tag] = name
                break
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


def compress_state_dict(state_dict: dict, cfg: CompressConfig) -> dict[str, dict]:
    """Apply GRC projection in place on a state-dict-like mapping.

    Returns a per-layer stats dict (rank, sink_T, frob_relerr).
    """
    stats: dict[str, dict] = {}
    layer_map = _group_attn_by_layer(state_dict)
    layers_to_do: Iterable[int] = sorted(layer_map.keys())
    if cfg.layers:
        layers_to_do = [i for i in layers_to_do if i in cfg.layers]

    for li in layers_to_do:
        names = layer_map[li]
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


def compress_hf_model(
    model_id_or_path: str,
    out_dir: str | Path,
    *,
    rank_k: int = 1024,
    sink_T: int = 0,
    layers: list[int] | None = None,
    dtype: str = "float32",
    revision: str | None = None,
) -> dict:
    """Load a HF model, apply GRC compression, save as standard safetensors.

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
        rank_k=rank_k, sink_T=sink_T, layers=layers or [], dtype=dtype
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, revision=revision, torch_dtype=torch.float32
    )
    sd = model.state_dict()
    stats = compress_state_dict(sd, cfg)
    model.load_state_dict(sd)

    # Save in standard HF layout.
    model.save_pretrained(out, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision)
        tok.save_pretrained(out)
    except Exception:
        pass  # tokenizer may not be available for some checkpoints

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
    }
    (out / "hyperretro_report.json").write_text(json.dumps(report, indent=2))
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
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="Storage dtype of the projected weights")
    p.add_argument("--revision", default=None, help="HF model revision/tag")
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
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
