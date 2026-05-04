"""
GRC light distillation: optional opt-in PPL recovery for low-rank attention.

Given a base model (uncompressed) and a target rank k, this tool:

  1. Computes the GRC shared-basis projection P_t per layer (matching the
     C runtime in runtime/nn/axiom_exploit.c) and applies it to obtain
     projected weights W_Q', W_K', W_V'.
  2. Optionally identifies sink channels (top-T columns by joint L2
     magnitude) and exempts them from projection (sink-aware GRC).
  3. Runs forward passes on a small calibration corpus with both the
     teacher (original weights) and the student (projected weights),
     and fits per-layer rank-r LoRA adapters on the residual to minimise
     teacher-student logit MSE.
  4. Exports the corrected weights (W_Q' + A_Q B_Q, etc.) as a .safetensors
     file that can be re-quantised to GGUF Q4_K_M and shipped to the
     runtime.

This is the "light distillation" protocol referenced as Paper E in the
HyperTensor research bundle. It is OPTIONAL; the calibration-free GRC
in Paper A is the default mode. Distillation breaks calibration-free
operation in exchange for measured PPL recovery.

This script is a runnable scaffold. The model load, forward-pass, and
LoRA-fit components are written against a transformers/torch interface
to allow GPU execution on EC2. PPL evaluation is delegated to llama.cpp
via subprocess after re-quantisation.

Status as of 2026-04: design pass complete; first end-to-end run
queued for EC2 (g5.xlarge or larger; needs at least one A100/L40S to
hold Llama-3.1-8B in FP16 with gradients). Results will populate
ARXIV_SUBMISSIONS/paper-E/grc-light-distillation.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------- shared GRC utilities (mirror of axiom_exploit.c) ----------

def build_shared_basis(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                       n_iter: int = 3) -> np.ndarray:
    """Top-d eigvecs of normalised joint Gram, sorted descending."""
    K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def project(W: np.ndarray, P: np.ndarray, k: int) -> np.ndarray:
    Pk = P[:, :k]
    return W @ Pk @ Pk.T


def sink_indices(Wq, Wk, Wv, T: int) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=np.int64)
    mag = (np.linalg.norm(Wq, axis=0) ** 2
           + np.linalg.norm(Wk, axis=0) ** 2
           + np.linalg.norm(Wv, axis=0) ** 2)
    return np.argsort(mag)[::-1][:T]


# ---------- distillation core (LoRA on projected residual) ----------

@dataclass
class DistillConfig:
    rank: int = 1024                # target GRC rank k
    sink_T: int = 0                 # 0 = vanilla GRC; >0 = sink-aware
    lora_rank: int = 8              # rank of correction adapters
    learning_rate: float = 1e-4
    steps: int = 500                # ~5-10k tokens at batch=8 seq=512
    batch_size: int = 8
    seq_len: int = 512
    layers: list[int] = field(default_factory=list)  # [] = all
    device: str = "cuda"
    dtype: str = "bfloat16"


def lora_fit_attention(layer_idx: int,
                       Wq_proj: np.ndarray,
                       Wk_proj: np.ndarray,
                       Wv_proj: np.ndarray,
                       teacher_acts_in: np.ndarray,
                       teacher_acts_qkv: tuple[np.ndarray, np.ndarray, np.ndarray],
                       cfg: DistillConfig):
    """
    Fit LoRA correction A_q B_q (similarly K, V) so that
        (Wq_proj + A_q B_q) @ x  approximates  Wq_teacher @ x
    on the calibration activations.

    This function is a torch-side stub. The actual implementation needs
    a torch tensor flow with optimizer; left here as a clear interface
    for the EC2 runner to fill in. Returns dict of LoRA factors.
    """
    raise NotImplementedError(
        "lora_fit_attention requires torch + a calibration corpus. "
        "Run scripts/distill_runner.py on a GPU host."
    )


# ---------- pipeline orchestration ----------

def compress_layer(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                   cfg: DistillConfig):
    """Apply GRC (optionally sink-aware) and return projected weights."""
    sink = sink_indices(Wq, Wk, Wv, cfg.sink_T)
    if cfg.sink_T > 0:
        Wq_R = Wq.copy(); Wq_R[:, sink] = 0.0
        Wk_R = Wk.copy(); Wk_R[:, sink] = 0.0
        Wv_R = Wv.copy(); Wv_R[:, sink] = 0.0
        P = build_shared_basis(Wq_R, Wk_R, Wv_R)
        Wq_proj = project(Wq_R, P, cfg.rank); Wq_proj[:, sink] = Wq[:, sink]
        Wk_proj = project(Wk_R, P, cfg.rank); Wk_proj[:, sink] = Wk[:, sink]
        Wv_proj = project(Wv_R, P, cfg.rank); Wv_proj[:, sink] = Wv[:, sink]
    else:
        P = build_shared_basis(Wq, Wk, Wv)
        Wq_proj = project(Wq, P, cfg.rank)
        Wk_proj = project(Wk, P, cfg.rank)
        Wv_proj = project(Wv, P, cfg.rank)
    return {"P": P, "sink": sink,
            "Wq_proj": Wq_proj, "Wk_proj": Wk_proj, "Wv_proj": Wv_proj}


def _load_attn_weights_gguf(gguf_path: str, layer_idx: int):
    """Return (Wq, Wk, Wv) as float32 numpy arrays for the given layer."""
    from gguf import GGUFReader, dequantize
    r = GGUFReader(gguf_path)
    tm = {t.name: t for t in r.tensors}
    prefix = f"blk.{layer_idx}"
    def _get(key):
        t = tm[key]
        return dequantize(t.data, t.tensor_type).astype(np.float32)
    Wq = _get(f"{prefix}.attn_q.weight")
    Wk = _get(f"{prefix}.attn_k.weight")
    Wv = _get(f"{prefix}.attn_v.weight")
    return Wq, Wk, Wv


def _n_layers_gguf(gguf_path: str) -> int:
    """Return number of transformer blocks in a GGUF file."""
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    for f in r.fields.values():
        if f.name.endswith(".block_count"):
            import struct
            return int(f.parts[-1][0])
    # fallback: count unique block indices
    indices = set()
    for t in r.tensors:
        parts = t.name.split(".")
        if parts[0] == "blk" and len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    return max(indices) + 1 if indices else 0


def _top_r_sq_eigvals(V: np.ndarray, r: int, n_iter: int = 60) -> float:
    """Sum of the top-r squared singular values of V (i.e. top-r eigenvalues
    of V^T V) using block power iteration with QR stabilisation.
    Returns a float (sum, not list).
    """
    rng = np.random.default_rng(0)
    d = V.shape[1]
    r = min(r, d)
    Z = rng.standard_normal((d, r)).astype(np.float32)
    for _ in range(n_iter):
        Z = V.T @ (V @ Z)           # V^T V Z
        Z, _ = np.linalg.qr(Z)     # keep orthonormal
    VZ = V @ Z                      # (m, r)
    lambdas = np.einsum("ij,ij->j", VZ, VZ)   # approx top-r eigenvalues
    return float(np.sum(np.maximum(lambdas, 0.0)))


def compute_rho(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                k: int, lora_rank: int = 8) -> float:
    """Per-layer recoverable energy ratio rho (Paper E §sec:gapbound).

    rho_l = sum_X eta_recov(l,X) / sum_X eta(l,X)

    where for each weight W_X in {Wq, Wk, Wv}:
      - P_perp = top-(d-k) GRC complement directions (d x (d-k))
      - V_X    = W_X @ P_perp  (the projection residual in a compact form)
      - eta(l,X)       = ||V_X||_F  (total residual Frobenius norm)
      - eta_recov(l,X) = sqrt(sum top-lora_rank eigenvalues of V_X^T V_X)
                       = the part of the residual a rank-lora_rank LoRA can cover

    rho ~ 0.42 at k=1024, lora_rank=8 for Llama-3.1-8B (Paper E estimate).
    """
    P = build_shared_basis(Wq, Wk, Wv)
    P_perp = P[:, k:]           # (d, d-k) --- discarded directions

    eta_total = 0.0
    eta_recov = 0.0
    for W in (Wq, Wk, Wv):
        V = W @ P_perp          # (m, d-k)
        eta = float(np.sqrt(np.sum(V ** 2)))   # ||V||_F
        if eta == 0.0:
            continue
        top_sq = _top_r_sq_eigvals(V, lora_rank)
        eta_total += eta
        eta_recov += float(np.sqrt(max(top_sq, 0.0)))

    if eta_total == 0.0:
        return 0.0
    return eta_recov / eta_total


def cmd_print_rho(model_path: str, rank: int, lora_rank: int, sink_T: int, out_dir: str | None) -> None:
    """Compute and print per-layer recoverable energy ratio rho."""
    from pathlib import Path as _Path

    model_path = model_path.strip()
    if model_path.lower().endswith(".gguf"):
        fmt = "gguf"
    else:
        print("[grc_distill] ERROR: --print-rho currently supports GGUF files only.",
              file=sys.stderr)
        sys.exit(2)

    n_layers = _n_layers_gguf(model_path)
    if n_layers == 0:
        print("[grc_distill] ERROR: could not determine layer count.", file=sys.stderr)
        sys.exit(2)
    print(f"[rho] model={model_path}  n_layers={n_layers}  k={rank}  lora_rank={lora_rank}  sink_T={sink_T}", flush=True)

    rhos = []
    for layer in range(n_layers):
        Wq, Wk, Wv = _load_attn_weights_gguf(model_path, layer)
        if sink_T > 0:
            sink = sink_indices(Wq, Wk, Wv, sink_T)
            Wq = Wq.copy(); Wq[:, sink] = 0.0
            Wk = Wk.copy(); Wk[:, sink] = 0.0
            Wv = Wv.copy(); Wv[:, sink] = 0.0
        rho = compute_rho(Wq, Wk, Wv, rank, lora_rank)
        rhos.append(rho)
        print(f"  layer {layer:2d}  rho={rho:.4f}", flush=True)

    mean_rho = float(np.mean(rhos))
    print(f"[rho] mean rho = {mean_rho:.4f}  (k={rank})", flush=True)

    if out_dir is not None:
        out = _Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = {"model": model_path, "rank": rank, "lora_rank": lora_rank, "sink_T": sink_T,
                  "n_layers": n_layers, "mean_rho": mean_rho,
                  "per_layer": rhos}
        import json as _json
        with open(out / "rho_summary.json", "w") as f:
            _json.dump(result, f, indent=2)
        print(f"[rho] wrote {out / 'rho_summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Path to base GGUF or HF dir")
    ap.add_argument("--out", default=None, help="Output dir for compressed/distilled artifacts")
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--sink-T", type=int, default=0)
    ap.add_argument("--distill", action="store_true",
                    help="Run light distillation (requires GPU + calibration corpus)")
    ap.add_argument("--corpus", default=None, help="Path to calibration .txt or .arrow")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--print-rho", action="store_true",
                    help="Compute and print per-layer recoverable energy ratio rho "
                         "(CPU-only numpy; requires --model pointing to a GGUF file)")
    args = ap.parse_args()

    if args.print_rho:
        if args.model is None:
            print("[grc_distill] ERROR: --print-rho requires --model", file=sys.stderr)
            sys.exit(2)
        cmd_print_rho(args.model, args.rank, args.lora_rank, args.sink_T, args.out)
        return

    if args.model is None or args.out is None:
        ap.error("--model and --out are required unless --print-rho is used")

    cfg = DistillConfig(
        rank=args.rank,
        sink_T=args.sink_T,
        lora_rank=args.lora_rank,
        steps=args.steps,
        device=args.device,
    )

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": cfg.__dict__,
        "stage": "scaffold",
        "notes": [
            "Phase 1 (GRC projection) is purely numpy; no GPU required.",
            "Phase 2 (LoRA fit) requires torch + GPU; gated behind --distill.",
            "Phase 3 (re-quantise + ship) calls out to llama.cpp's quantize tool.",
        ],
    }

    if args.distill:
        if args.corpus is None:
            print("[grc_distill] ERROR: --distill requires --corpus", file=sys.stderr)
            sys.exit(2)
        print(f"[grc_distill] distillation mode: rank={cfg.rank} "
              f"sink_T={cfg.sink_T} lora_rank={cfg.lora_rank} "
              f"steps={cfg.steps} corpus={args.corpus}")
        print("[grc_distill] NOT YET IMPLEMENTED end-to-end (torch path).")
        manifest["stage"] = "distill_requested"
    else:
        print(f"[grc_distill] calibration-free mode: rank={cfg.rank} sink_T={cfg.sink_T}")
        print("[grc_distill] (this script is the offline reference; the C runtime "
              "applies the same projection at model load time)")
        manifest["stage"] = "calibration_free"

    with open(out / "distill_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[grc_distill] wrote {out / 'distill_manifest.json'}")


if __name__ == "__main__":
    main()
