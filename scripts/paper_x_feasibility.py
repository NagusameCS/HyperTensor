#!/usr/bin/env python3
"""
Paper X Feasibility: Gauge-Aligned Subspace Overlap Analysis.

Tests whether the intrinsic k-dimensional subspaces of different
transformer layers overlap — the prerequisite for chimeric model splicing.
Uses layers as proxies for "specialized models" since we can't yet
train dedicated Math/Language models.

Key question: If layer 0 (early, syntactic) and layer 15 (middle, semantic)
have different "skills," do their top-k subspaces overlap enough for
a gauge transformation to align them?

Metric: Grassmann distance (principal angles) between subspaces.
d_G(U, V) = ||U U^T - V V^T||_F / sqrt(2k)
0 = identical subspaces, 1 = orthogonal subspaces.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    """Grassmann distance between two k-dimensional subspaces.
    U, V: (d, k) orthonormal basis matrices.
    Returns value in [0, 1] where 0 = identical, 1 = orthogonal.
    """
    # Normalize
    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    k = U.shape[1]
    P_U = U @ U.T  # projector
    P_V = V @ V.T
    return float(np.linalg.norm(P_U - P_V, 'fro') / np.sqrt(2 * k))


def subspace_overlap_fraction(U: np.ndarray, V: np.ndarray) -> float:
    """Fraction of V's variance captured by U's subspace.
    Returns value in [0, 1] where 1 = V entirely in span(U).
    """
    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    # Project V onto U: ||U^T V||_F^2 / k
    overlap = np.linalg.norm(U.T @ V, 'fro') ** 2 / V.shape[1]
    return float(min(overlap, 1.0))


def gauge_alignment_residual(U: np.ndarray, V: np.ndarray) -> float:
    """
    Minimal Frobenius residual after optimal orthogonal alignment.
    The Procrustes problem: min_R ||U @ R - V||_F subject to R^T R = I.
    Solution: R = U_W @ V_W^T where U_W S V_W^T = V^T U (SVD).
    Residual = ||U @ R - V||_F / sqrt(k).
    Returns value in [0, 1].
    """
    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    M = V.T @ U  # (k, k)
    Uw, _, Vwt = np.linalg.svd(M, full_matrices=False)
    R = Uw @ Vwt  # optimal rotation
    residual = np.linalg.norm(U @ R - V, 'fro') / np.sqrt(U.shape[1])
    return float(residual)


def main():
    ap = argparse.ArgumentParser(
        description="Paper X Feasibility: Subspace Overlap Analysis"
    )
    ap.add_argument("--model", default="models/smollm2-135m-instruct-q8_0.gguf",
                    help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/paper_x_feasibility")
    ap.add_argument("--k", type=int, default=32,
                    help="Intrinsic dimension for subspace comparison")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = _n_layers_gguf(args.model)
    k = args.k

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers}")
    print(f"Intrinsic dimension k = {k}")
    print()

    # Compute subspaces for each layer
    print("Computing per-layer subspaces...")
    subspaces = {}
    for layer in range(min(n_layers, 16)):  # First 16 layers
        try:
            Wq, Wk, Wv = _load_attn_weights_gguf(args.model, layer)
            P = build_shared_basis(Wq, Wk, Wv)
            subspaces[layer] = P[:, :k]  # Top-k eigenvectors
            if layer % 4 == 0:
                print(f"  layer {layer}: done")
        except Exception as e:
            print(f"  layer {layer}: SKIP ({e})")

    print(f"  Computed {len(subspaces)} subspaces")
    print()

    # Cross-layer subspace comparison
    print("=== Cross-Layer Subspace Overlap ===")
    print(f"{'L_i':>4s} {'L_j':>4s}  {'Grassmann':>10s}  {'Overlap':>8s}  {'AlignRes':>10s}")
    print("-" * 50)

    results = []
    layers = sorted(subspaces.keys())

    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            li, lj = layers[i], layers[j]
            Ui, Uj = subspaces[li], subspaces[lj]

            gd = grassmann_distance(Ui, Uj)
            ov = subspace_overlap_fraction(Ui, Uj)
            ar = gauge_alignment_residual(Ui, Uj)

            marker = ""
            if ov > 0.5:
                marker = "  HIGH"
            elif ov > 0.3:
                marker = " ~ MEDIUM"

            print(f"{li:4d} {lj:4d}  {gd:10.4f}  {ov:8.4f}  {ar:10.4f}{marker}")

            results.append({
                "layer_i": li, "layer_j": lj,
                "grassmann": round(gd, 4),
                "overlap": round(ov, 4),
                "alignment_residual": round(ar, 4),
                "splice_viable": ov > 0.3,
            })

    # Summary statistics
    overlaps = [r["overlap"] for r in results]
    grassmanns = [r["grassmann"] for r in results]
    align_residuals = [r["alignment_residual"] for r in results]

    print()
    print("=== Summary ===")
    print(f"  Mean overlap:        {np.mean(overlaps):.4f}")
    print(f"  Median overlap:      {np.median(overlaps):.4f}")
    print(f"  Mean Grassmann:      {np.mean(grassmanns):.4f}")
    print(f"  Mean align residual: {np.mean(align_residuals):.4f}")
    print(f"  Viable pairs (>30%): {sum(1 for o in overlaps if o > 0.3)}/{len(overlaps)} "
          f"({sum(1 for o in overlaps if o > 0.3)/len(overlaps)*100:.1f}%)")

    # Layer distance effect
    print()
    print("=== Overlap vs Layer Distance ===")
    print(f"{'ΔL':>4s}  {'Mean Overlap':>12s}  {'Mean Grassmann':>14s}")
    for dl in [1, 2, 4, 8, 12]:
        dl_results = [r for r in results if abs(r["layer_i"] - r["layer_j"]) == dl]
        if dl_results:
            mean_ov = np.mean([r["overlap"] for r in dl_results])
            mean_gd = np.mean([r["grassmann"] for r in dl_results])
            print(f"{dl:4d}  {mean_ov:12.4f}  {mean_gd:14.4f}")

    # Verdict
    viable_pct = sum(1 for o in overlaps if o > 0.3) / max(len(overlaps), 1) * 100
    print()
    print("=== Feasibility Verdict ===")
    if viable_pct > 50:
        print(f"   PROMISING: {viable_pct:.0f}% of layer pairs have >30% subspace overlap.")
        print(f"  Chimeric splicing is geometrically feasible.")
        print(f"  The gauge transformation can align subspaces with mean residual {np.mean(align_residuals):.4f}.")
    elif viable_pct > 20:
        print(f"   MARGINAL: {viable_pct:.0f}% overlap >30%.")
        print(f"  Splicing may work for nearby layers but not across the full model.")
    else:
        print(f"   CHALLENGING: Only {viable_pct:.0f}% overlap >30%.")
        print(f"  Subspaces are too different for direct splicing.")
        print(f"  Consider per-cluster alignment instead of global gauge.")

    with open(out_dir / "paper_x_feasibility.json", "w") as f:
        json.dump({
            "config": {"model": args.model, "k": k, "n_layers": len(subspaces)},
            "summary": {
                "mean_overlap": round(float(np.mean(overlaps)), 4),
                "median_overlap": round(float(np.median(overlaps)), 4),
                "mean_grassmann": round(float(np.mean(grassmanns)), 4),
                "mean_align_residual": round(float(np.mean(align_residuals)), 4),
                "viable_pct": round(viable_pct, 1),
            },
            "cross_layer": results,
        }, f, indent=2)

    print(f"\n[done] {out_dir / 'paper_x_feasibility.json'}")


if __name__ == "__main__":
    main()
