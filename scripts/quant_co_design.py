#!/usr/bin/env python3
"""
GRC + Quantization Co-Design Analysis (Tier 3).

The projected weights W_proj = W @ P_k @ P_k^T have different spectral
properties than the original weights W.  Specifically:
  - The projected weights have effective rank k, not d
  - Their singular values are truncated at k
  - The basis P_k encodes the dominant geometric directions

This suggests a co-designed quantization strategy:
  (a) Keep the basis P_k in FP16 (it's only dk, relatively small)
  (b) Quantize the projected weights more aggressively than normal,
      exploiting their reduced effective rank
  (c) Use per-cluster quantization on the projected subspace

This script analyzes the quantizability of projected vs original weights
by measuring:
  1. Weight distribution statistics (mean, std, kurtosis)
  2. Per-tensor quantization error at various bit widths
  3. Optimal quantization scale for projected weights
  4. Basis-aware quantization: quantize in the P_k basis, not the ambient basis

Usage:
  python scripts/quant_co_design.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/quant_co_design \
    --ranks 128,256,512 \
    --sample-layers 0,15,29
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from grc_distill import (
    build_shared_basis,
    project as grc_project,
    _load_attn_weights_gguf,
    _n_layers_gguf,
)


# ---------------------------------------------------------------------------
# Quantization simulation
# ---------------------------------------------------------------------------

def simulate_quantize(W: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """
    Simulate uniform quantization to `bits` bits per element.
    Returns (W_quantized, relative_frobenius_error).
    """
    if bits >= 32:
        return W.copy(), 0.0

    n_levels = 2 ** bits
    w_min, w_max = W.min(), W.max()
    if w_max - w_min < 1e-10:
        return W.copy(), 0.0

    scale = (w_max - w_min) / (n_levels - 1)
    W_int = np.round((W - w_min) / scale)
    W_int = np.clip(W_int, 0, n_levels - 1)
    W_q = W_int * scale + w_min

    err = np.linalg.norm(W - W_q, "fro") / max(np.linalg.norm(W, "fro"), 1e-10)
    return W_q, float(err)


def simulate_per_channel_quantize(W: np.ndarray, bits: int, axis: int = 0) -> tuple[np.ndarray, float]:
    """
    Per-channel quantization: each row (axis=0) or column (axis=1)
    gets its own scale. Returns (W_quantized, relative_frobenius_error).
    """
    if bits >= 32:
        return W.copy(), 0.0

    n_levels = 2 ** bits
    W_q = np.zeros_like(W)

    if axis == 0:
        for i in range(W.shape[0]):
            row = W[i, :]
            r_min, r_max = row.min(), row.max()
            if r_max - r_min < 1e-10:
                W_q[i, :] = row
                continue
            scale = (r_max - r_min) / (n_levels - 1)
            row_int = np.round((row - r_min) / scale)
            row_int = np.clip(row_int, 0, n_levels - 1)
            W_q[i, :] = row_int * scale + r_min
    else:
        for j in range(W.shape[1]):
            col = W[:, j]
            c_min, c_max = col.min(), col.max()
            if c_max - c_min < 1e-10:
                W_q[:, j] = col
                continue
            scale = (c_max - c_min) / (n_levels - 1)
            col_int = np.round((col - c_min) / scale)
            col_int = np.clip(col_int, 0, n_levels - 1)
            W_q[:, j] = col_int * scale + c_min

    err = np.linalg.norm(W - W_q, "fro") / max(np.linalg.norm(W, "fro"), 1e-10)
    return W_q, float(err)


def basis_aware_quantize(W: np.ndarray, P: np.ndarray, k: int, bits: int) -> tuple[np.ndarray, float]:
    """
    Basis-aware quantization: project W into P_k subspace, quantize the
    compressed representation W_tilde = P_k^T @ W, then reconstruct.
    """
    P_k = P[:, :k]  # (d, k)
    W_tilde = P_k.T @ W.T  # (k, d) @ (d, m) = (k, m)

    # Quantize the compressed representation
    W_tilde_q, _ = simulate_quantize(W_tilde, bits)

    # Reconstruct: W_recon = P_k @ W_tilde_q, then transpose back
    W_recon = (P_k @ W_tilde_q).T  # (d, k) @ (k, m) = (d, m) -> (m, d)

    err = np.linalg.norm(W - W_recon, "fro") / max(np.linalg.norm(W, "fro"), 1e-10)
    return W_recon, float(err)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def weight_stats(W: np.ndarray) -> dict:
    """Compute distribution statistics for a weight matrix."""
    flat = W.ravel()
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "kurtosis": float(np.mean((flat - np.mean(flat))**4) / max(np.var(flat)**2, 1e-10)),
        "frac_near_zero": float(np.mean(np.abs(flat) < 1e-6)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GRC + Quantization Co-Design Analysis")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/quant_co_design")
    ap.add_argument("--ranks", default="128,256,512,1024")
    ap.add_argument("--bits", default="2,3,4,8,16",
                    help="Comma-separated bit widths to simulate")
    ap.add_argument("--sample-layers", default="0,7,15,23,29")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = [int(x) for x in args.ranks.split(",")]
    bits_list = [int(x) for x in args.bits.split(",")]
    n_layers = _n_layers_gguf(args.model)
    layers = [int(x) for x in args.sample_layers.split(",")]
    layers = [l for l in layers if 0 <= l < n_layers]

    print(f"Model: {args.model}")
    print(f"Layers: {layers}")
    print(f"Ranks: {ranks}  Bits: {bits_list}")
    print()

    results = []

    for layer in layers:
        Wq, Wk, Wv = _load_attn_weights_gguf(args.model, layer)
        P = build_shared_basis(Wq, Wk, Wv)

        # Original weight stats
        for slot, W in [("Q", Wq), ("K", Wk), ("V", Wv)]:
            stats = weight_stats(W)
            stats["layer"] = layer
            stats["slot"] = slot
            stats["type"] = "original"

            # Quantization error at each bit width
            for bits in bits_list:
                _, err_tensor = simulate_quantize(W, bits)
                _, err_channel = simulate_per_channel_quantize(W, bits, axis=0)
                stats[f"q{bits}_err_tensor"] = round(err_tensor, 6)
                stats[f"q{bits}_err_channel"] = round(err_channel, 6)

            results.append(stats)

        # Projected weight stats at each rank
        for k in ranks:
            Wq_proj = grc_project(Wq, P, k)
            Wk_proj = grc_project(Wk, P, k)
            Wv_proj = grc_project(Wv, P, k)

            for slot, W, Wp in [("Q", Wq, Wq_proj), ("K", Wk, Wk_proj),
                                 ("V", Wv, Wv_proj)]:
                stats = weight_stats(Wp)
                stats["layer"] = layer
                stats["slot"] = slot
                stats["type"] = f"projected_k{k}"

                for bits in bits_list:
                    _, err_tensor = simulate_quantize(Wp, bits)
                    _, err_channel = simulate_per_channel_quantize(Wp, bits, axis=0)
                    _, err_basis = basis_aware_quantize(W, P, k, bits)
                    stats[f"q{bits}_err_tensor"] = round(err_tensor, 6)
                    stats[f"q{bits}_err_channel"] = round(err_channel, 6)
                    stats[f"q{bits}_err_basis"] = round(err_basis, 6)

                results.append(stats)

    # Summarize: compare quantization error of original vs projected at same bit width
    print("\n=== Quantization Error: Original vs Projected ===")
    print(f"{'bits':>5s}  {'orig_tensor':>12s}  {'proj_tensor':>12s}  "
          f"{'basis_aware':>12s}  {'basis_win':>10s}")

    for bits in bits_list:
        orig_errs = [r[f"q{bits}_err_tensor"] for r in results
                     if r["type"] == "original"]
        proj_errs = [r[f"q{bits}_err_tensor"] for r in results
                     if r["type"].startswith("projected_k")]
        basis_errs = [r[f"q{bits}_err_basis"] for r in results
                      if r["type"].startswith("projected_k") and f"q{bits}_err_basis" in r]

        orig_mean = np.mean(orig_errs) if orig_errs else 0
        proj_mean = np.mean(proj_errs) if proj_errs else 0
        basis_mean = np.mean(basis_errs) if basis_errs else 0
        win = (proj_mean - basis_mean) / proj_mean * 100 if proj_mean > 0 else 0

        print(f"{bits:5d}  {orig_mean:12.6f}  {proj_mean:12.6f}  "
              f"{basis_mean:12.6f}  {win:+9.1f}%")

    # Per-rank quantization sensitivity
    print("\n=== Per-Rank Quantization Error (4-bit, tensor) ===")
    for k in ranks:
        errs = [r["q4_err_tensor"] for r in results
                if r["type"] == f"projected_k{k}"]
        mean_err = np.mean(errs) if errs else 0
        print(f"  k={k:5d}: q4_err={mean_err:.6f}")

    with open(out_dir / "quant_co_design_summary.json", "w") as f:
        json.dump({"ranks": ranks, "bits": bits_list, "results": results}, f, indent=2)

    print(f"\n[done] {out_dir / 'quant_co_design_summary.json'}")


if __name__ == "__main__":
    main()
