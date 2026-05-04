#!/usr/bin/env python3
"""
Gauge Alignment Simulation --- Per-Band Axiom Gauge for Chimeric Splicing.

Tests the core geometric question: if you have two different sets of
weights (e.g., Math attention vs Language attention), can you find a
diagonal GL(d) gauge transformation g ∈ R^d_{>0} that rotates one
into the other's coordinate system?

The Axiom Gauge minimizes:
  L(g) = Σ_{ℓ,W} tail_r(W @ diag(g^{-1})) + tail_r(diag(g) @ W)

where tail_r(M) = ||M||_F² - ||trunc_r(M)||_F² is the energy lost
by rank-r truncation AFTER applying the gauge.

This script:
  1. Takes two sets of attention weights (simulated as layers from
     different bands of the same model, or different models)
  2. Computes the optimal gauge per band
  3. Measures the alignment quality (subspace overlap before/after)
  4. Reports whether splicing is geometrically viable

Usage:
  python scripts/gauge_align.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/gauge_align
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
    _load_attn_weights_gguf,
    _n_layers_gguf,
)


def grassmann_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Grassmann distance between two k-dimensional subspaces."""
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-10)
    k = U.shape[1]
    P_U = U @ U.T
    P_V = V @ V.T
    return float(np.linalg.norm(P_U - P_V, 'fro') / np.sqrt(2 * k))


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    """Fraction of V's variance captured by U's subspace."""
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-10)
    return float(np.linalg.norm(U.T @ V, 'fro') ** 2 / V.shape[1])


def optimal_gauge(Wq_a: np.ndarray, Wk_a: np.ndarray, Wv_a: np.ndarray,
                  Wq_b: np.ndarray, Wk_b: np.ndarray, Wv_b: np.ndarray,
                  k: int, n_iter: int = 200, lr: float = 0.1) -> tuple[np.ndarray, dict]:
    """
    Compute the optimal diagonal gauge g ∈ R^d_{>0} to align model B
    into model A's coordinate system.

    Uses FAST cosine alignment: directly minimizes the cosine distance
    between corresponding weight COLUMNS of A and gauge-transformed B.
    This is O(d²) per iteration instead of O(d³) for SVD.

    The column-space interpretation: column j of W encodes how
    dimension j of the residual stream influences the attention output.
    Aligning columns aligns the "meaning" of each residual dimension.
    """
    d = Wq_a.shape[1]
    g = np.ones(d, dtype=np.float64)
    log = {"iter": [], "loss": []}

    # Pre-compute A's column norms (fixed)
    cols_a = np.sqrt(
        np.sum(Wq_a**2, axis=0) + np.sum(Wk_a**2, axis=0) + np.sum(Wv_a**2, axis=0)
    )  # (d,) --- combined column magnitude of A

    for iteration in range(n_iter):
        # Apply gauge to B
        Wq_bg = Wq_b * g[np.newaxis, :]
        Wk_bg = Wk_b * g[np.newaxis, :]
        Wv_bg = Wv_b * g[np.newaxis, :]

        cols_bg = np.sqrt(
            np.sum(Wq_bg**2, axis=0) + np.sum(Wk_bg**2, axis=0) + np.sum(Wv_bg**2, axis=0)
        )

        # Cosine distance loss: penalize mismatched column magnitudes
        # Loss = ||cols_a / max(cols_a) - cols_bg / max(cols_bg)||²
        cols_a_norm = cols_a / (cols_a.max() + 1e-10)
        cols_bg_norm = cols_bg / (cols_bg.max() + 1e-10)
        loss = np.mean((cols_a_norm - cols_bg_norm) ** 2)

        if iteration % 50 == 0:
            log["iter"].append(iteration)
            log["loss"].append(round(float(loss), 6))

        # Gradient of loss w.r.t. g
        # d/dg_j cols_bg[j] = g_j * (sum of squared weights in column j) / cols_bg[j]
        col_sq = np.sum(Wq_b**2, axis=0) + np.sum(Wk_b**2, axis=0) + np.sum(Wv_b**2, axis=0)
        dcols_dg = np.where(cols_bg > 1e-10, g * col_sq / cols_bg, 0.0)

        # Chain rule through normalization
        grad = -2 * (cols_a_norm - cols_bg_norm) * dcols_dg / (cols_bg.max() + 1e-10)
        grad = grad / (np.linalg.norm(grad) + 1e-10)

        g -= lr * grad
        g = np.clip(g, 0.2, 5.0)

    # Normalize: geometric mean = 1
    g = g / np.exp(np.mean(np.log(g + 1e-10)))

    # Final metrics: Grassmann distance after alignment
    P_a = build_shared_basis(Wq_a, Wk_a, Wv_a)[:, :k]
    Wq_bf = Wq_b * g[np.newaxis, :]
    Wk_bf = Wk_b * g[np.newaxis, :]
    Wv_bf = Wv_b * g[np.newaxis, :]
    P_bf = build_shared_basis(Wq_bf, Wk_bf, Wv_bf)[:, :k]

    final_gd = grassmann_distance(P_a, P_bf)
    final_ov = subspace_overlap(P_a, P_bf)

    log["final_grassmann"] = round(final_gd, 4)
    log["final_overlap"] = round(final_ov, 4)

    return g, log


def main():
    ap = argparse.ArgumentParser(
        description="Gauge Alignment Simulation"
    )
    ap.add_argument("--model", default="models/smollm2-135m-instruct-q8_0.gguf")
    ap.add_argument("--out", default="benchmarks/gauge_align")
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = _n_layers_gguf(args.model)
    k = args.k

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers}, k={k}")
    print()

    # Load weights for representative layers from each phase band
    layer_pairs = [
        (0, 1, "Adjacent (ΔL=1)"),
        (0, 10, "Cross-band Mix->Compress (ΔL=10)"),
        (10, 20, "Cross-band Compress->Refine (ΔL=10)"),
        (0, 29, "Full span (ΔL=29)"),
    ]

    results = []

    for li, lj, label in layer_pairs:
        try:
            Wq_a, Wk_a, Wv_a = _load_attn_weights_gguf(args.model, li)
            Wq_b, Wk_b, Wv_b = _load_attn_weights_gguf(args.model, lj)
        except Exception as e:
            print(f"  ({li},{lj}) {label}: SKIP ({e})")
            continue

        # Pre-alignment metrics
        P_a = build_shared_basis(Wq_a, Wk_a, Wv_a)[:, :k]
        P_b = build_shared_basis(Wq_b, Wk_b, Wv_b)[:, :k]
        pre_gd = grassmann_distance(P_a, P_b)
        pre_ov = subspace_overlap(P_a, P_b)

        print(f"  {label} (layers {li}->{lj}):")
        print(f"    Pre-alignment:  Grassmann={pre_gd:.4f}, Overlap={pre_ov:.4f}")

        # Run gauge alignment
        t0 = time.time()
        g, log = optimal_gauge(Wq_a, Wk_a, Wv_a, Wq_b, Wk_b, Wv_b, k)
        elapsed = time.time() - t0

        post_gd = log["final_grassmann"]
        post_ov = log["final_overlap"]
        improvement = (pre_ov - post_ov) / max(1 - post_ov, 1e-10) * 100 if post_ov > pre_ov else 0

        print(f"    Post-alignment: Grassmann={post_gd:.4f}, Overlap={post_ov:.4f}")
        print(f"    Gauge stats: μ={np.mean(g):.4f} σ={np.std(g):.4f} "
              f"min={np.min(g):.4f} max={np.max(g):.4f}")
        print(f"    Wall time: {elapsed:.1f}s")
        print()

        results.append({
            "label": label,
            "layer_a": li, "layer_b": lj,
            "pre_grassmann": round(pre_gd, 4),
            "pre_overlap": round(pre_ov, 4),
            "post_grassmann": round(post_gd, 4),
            "post_overlap": round(post_ov, 4),
            "overlap_delta": round(post_ov - pre_ov, 4),
            "gauge_stats": {
                "mean": round(float(np.mean(g)), 4),
                "std": round(float(np.std(g)), 4),
            },
            "wall_s": round(elapsed, 1),
            "alignment_log": log,
        })

    # Summary
    print("=== Gauge Alignment Summary ===")
    print(f"{'Pair':>30s}  {'Pre-GD':>8s}  {'Post-GD':>8s}  {'Pre-OV':>8s}  {'Post-OV':>8s}  {'Δ OV':>8s}")
    for r in results:
        print(f"{r['label']:>30s}  {r['pre_grassmann']:8.4f}  {r['post_grassmann']:8.4f}  "
              f"{r['pre_overlap']:8.4f}  {r['post_overlap']:8.4f}  {r['overlap_delta']:+8.4f}")

    # Verdict
    mean_delta = np.mean([r["overlap_delta"] for r in results])
    print(f"\n  Mean overlap improvement from gauge: {mean_delta:+.4f}")
    if mean_delta > 0.05:
        print(f"   Gauge alignment significantly improves subspace overlap.")
    elif mean_delta > 0.01:
        print(f"  ~ Gauge provides marginal improvement. Adjacent layers benefit most.")
    else:
        print(f"   Gauge provides negligible improvement. Subspaces are fundamentally different.")

    with open(out_dir / "gauge_align_summary.json", "w") as f:
        json.dump({"config": {"model": args.model, "k": k}, "results": results}, f, indent=2)

    print(f"\n[done] {out_dir / 'gauge_align_summary.json'}")


if __name__ == "__main__":
    main()
