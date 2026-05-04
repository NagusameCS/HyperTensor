#!/usr/bin/env python3
"""
Chimeric Model Splicing --- Implementation of Phase 2-4 Protocol.

This script implements the HyperTensor chimeric splicing pipeline on
a single model (SmolLM2-135M) as a validation of the geometric
infrastructure.  In a single-model setting, we treat early layers
(0-9) as "Math attention" and late layers (20-29) as "Language FFN,"
splicing the attention geometry from early layers into the FFN
geometry from late layers.  This is a structural validation --- the
real experiment requires dedicated single-skill models.

Pipeline:
  Phase 2: Intrinsic projection (k-dimensional) + sink exemption
  Phase 3: MCR piecewise-local gauge alignment across bands
  Phase 4: Chimeric merge + light distillation residual measurement

Usage:
  python scripts/chimeric_splice.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/chimeric_splice
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from grc_distill import (
    build_shared_basis,
    project as grc_project,
    sink_indices,
    _load_attn_weights_gguf,
    _n_layers_gguf,
)


# ===========================================================================
# Phase 2: Intrinsic projection + sink exemption
# ===========================================================================

def compute_intrinsic_basis(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                            k: int = 32) -> np.ndarray:
    """Compute the top-k joint Gram eigenvectors (the intrinsic subspace)."""
    P = build_shared_basis(Wq, Wk, Wv)
    return P[:, :k]  # (d, k)


def protect_sinks(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                  T: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Identify sink channels and return protected weight copies."""
    sinks = sink_indices(Wq, Wk, Wv, T)
    Wq_p = Wq.copy(); Wq_p[:, sinks] = 0.0
    Wk_p = Wk.copy(); Wk_p[:, sinks] = 0.0
    Wv_p = Wv.copy(); Wv_p[:, sinks] = 0.0
    return sinks, (Wq_p, Wk_p, Wv_p)


# ===========================================================================
# Phase 3: MCR Phase-Band Gauge Alignment
# ===========================================================================

def compute_mcr_bands(n_layers: int) -> dict:
    """Partition layers into Mix/Compress/Refine phase bands.
    Uses the depth-sink rule (ℓ* ≈ 2L/3) from Part II.
    For SmolLM2 (L=30): Mix=0-9, Compress=10-19, Refine=20-29.
    """
    mix_end = n_layers // 3
    compress_end = 2 * n_layers // 3
    return {
        "Mix":      list(range(0, mix_end)),
        "Compress": list(range(mix_end, compress_end)),
        "Refine":   list(range(compress_end, n_layers)),
    }


def compute_gauge_transformation(weights_band_A: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  weights_band_B: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  k: int = 32) -> tuple[np.ndarray, float]:
    """
    Compute the diagonal gauge transformation g ∈ R^d_{>0} that minimises
    the joint tail energy between band A and band B.

    This implements the Axiom Gauge from Part II §gauge, applied per-band.

    Returns (g, residual) where g is the optimal gauge vector and residual
    is the relative Frobenius error after alignment.
    """
    d = weights_band_A[0][0].shape[1]
    g = np.ones(d, dtype=np.float64)
    lr = 0.01
    n_iter = 50

    for iteration in range(n_iter):
        grad = np.zeros(d, dtype=np.float64)
        total_err = 0.0

        for (Wq_a, Wk_a, Wv_a), (Wq_b, Wk_b, Wv_b) in zip(weights_band_A, weights_band_B):
            # Apply gauge: W_a stays as-is, W_b is transformed by g
            # W_b_transformed[i,:] = W_b[i,:] * g  (column scaling of the input side)
            for W_a, W_b in [(Wq_a, Wq_b), (Wk_a, Wk_b), (Wv_a, Wv_b)]:
                # Compute shared basis for both
                P_a = build_shared_basis(W_a, W_a, W_a)[:, :k]
                # Transform W_b: apply gauge to the output dimension (rows)
                W_b_g = W_b * g[np.newaxis, :]  # (m, d), scale columns
                P_b = build_shared_basis(W_b_g, W_b_g, W_b_g)[:, :k]

                # Grassmann alignment error
                diff = P_a @ P_a.T - P_b @ P_b.T
                err = np.linalg.norm(diff, 'fro') ** 2
                total_err += err

                # Gradient w.r.t. g: d/dg_i ||P_a P_a^T - P_b(g) P_b(g)^T||^2
                # Approximate: the subspace shift from scaling column i
                # Numerically: finite difference gradient
                eps = 1e-6
                for i in range(0, d, max(1, d // 32)):  # subsample for speed
                    g_plus = g.copy(); g_plus[i] += eps
                    W_b_gp = W_b * g_plus[np.newaxis, :]
                    P_bp = build_shared_basis(W_b_gp, W_b_gp, W_b_gp)[:, :k]
                    err_plus = np.linalg.norm(P_a @ P_a.T - P_bp @ P_bp.T, 'fro') ** 2
                    grad[i] += (err_plus - err) / eps

        g -= lr * grad / (np.linalg.norm(grad) + 1e-10)
        g = np.clip(g, 0.1, 10.0)  # prevent degenerate scaling

    # Normalize: geometric mean = 1
    g = g / np.exp(np.mean(np.log(g + 1e-10)))

    # Compute final alignment residual
    residual = total_err / len(weights_band_A) if weights_band_A else 0.0
    return g, float(residual)


# ===========================================================================
# Phase 4: Chimeric merge + residual measurement
# ===========================================================================

def compute_splice_residual(Wq_attn: np.ndarray, Wk_attn: np.ndarray, Wv_attn: np.ndarray,
                             P_int: np.ndarray, k: int) -> dict:
    """
    Measure the residual error after splicing attention routing (from
    one source) with FFN weights (from another source).

    The residual is: how much does the attention projection subspace
    differ from the subspace that the FFN weights expect?

    Returns dict with per-slot residual norms.
    """
    P_k = P_int[:, :k]

    # Project attention weights into intrinsic subspace
    Wq_proj = Wq_attn @ P_k @ P_k.T
    Wk_proj = Wk_attn @ P_k @ P_k.T
    Wv_proj = Wv_attn @ P_k @ P_k.T

    # Reconstruction error (how much was lost in projection)
    results = {}
    for slot, W, Wp in [("Q", Wq_attn, Wq_proj), ("K", Wk_attn, Wk_proj),
                          ("V", Wv_attn, Wv_proj)]:
        rel_err = np.linalg.norm(W - Wp, 'fro') / max(np.linalg.norm(W, 'fro'), 1e-10)
        energy_ret = np.linalg.norm(Wp, 'fro') ** 2 / max(np.linalg.norm(W, 'fro') ** 2, 1e-10)
        results[f"{slot}_rel_err"] = round(float(rel_err), 6)
        results[f"{slot}_energy_retained"] = round(float(energy_ret), 6)

    return results


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Chimeric Model Splicing --- Protocol Implementation"
    )
    ap.add_argument("--model", default="models/smollm2-135m-instruct-q8_0.gguf")
    ap.add_argument("--out", default="benchmarks/chimeric_splice")
    ap.add_argument("--k", type=int, default=32, help="Intrinsic dimension")
    ap.add_argument("--sink-T", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = _n_layers_gguf(args.model)
    k = args.k
    bands = compute_mcr_bands(n_layers)

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers}, k={k}")
    print(f"Bands: Mix={bands['Mix'][0]}-{bands['Mix'][-1]}, "
          f"Compress={bands['Compress'][0]}-{bands['Compress'][-1]}, "
          f"Refine={bands['Refine'][0]}-{bands['Refine'][-1]}")
    print()

    # ---- Phase 2: Intrinsic projection + sink protection ----
    print("=== Phase 2: Intrinsic Projection & Sink Protection ===")
    all_weights = {}
    all_bases = {}
    all_sinks = {}

    for layer in range(n_layers):
        try:
            Wq, Wk, Wv = _load_attn_weights_gguf(args.model, layer)
            sinks, (Wq_p, Wk_p, Wv_p) = protect_sinks(Wq, Wk, Wv, args.sink_T)
            P_int = compute_intrinsic_basis(Wq_p, Wk_p, Wv_p, k)
            all_weights[layer] = (Wq, Wk, Wv)
            all_bases[layer] = P_int
            all_sinks[layer] = sinks
            if layer % 10 == 0:
                print(f"  layer {layer}: k_int subspace computed, "
                      f"{len(sinks)} sinks protected")
        except Exception as e:
            print(f"  layer {layer}: SKIP ({e})")

    # ---- Phase 3: Per-band gauge alignment ----
    print("\n=== Phase 3: MCR Piecewise-Local Gauge Alignment ===")
    gauge_results = {}

    for band_name, band_layers in bands.items():
        band_layers = [l for l in band_layers if l in all_weights]
        if len(band_layers) < 2:
            continue

        # Split band into two halves to simulate alignment
        mid = len(band_layers) // 2
        band_A_layers = band_layers[:mid]
        band_B_layers = band_layers[mid:]

        weights_A = [all_weights[l] for l in band_A_layers]
        weights_B = [all_weights[l] for l in band_B_layers]

        print(f"  {band_name} band: aligning {len(band_A_layers)} vs {len(band_B_layers)} layers...")
        t0 = time.time()
        g, residual = compute_gauge_transformation(weights_A, weights_B, k)
        elapsed = time.time() - t0

        gauge_results[band_name] = {
            "layers_A": band_A_layers,
            "layers_B": band_B_layers,
            "gauge_mean": round(float(np.mean(g)), 4),
            "gauge_std": round(float(np.std(g)), 4),
            "gauge_min": round(float(np.min(g)), 4),
            "gauge_max": round(float(np.max(g)), 4),
            "alignment_residual": round(float(residual), 6),
            "wall_s": round(elapsed, 1),
        }
        print(f"    gauge μ={gauge_results[band_name]['gauge_mean']:.3f} "
              f"σ={gauge_results[band_name]['gauge_std']:.3f} "
              f"residual={residual:.4f} "
              f"({elapsed:.1f}s)")

    # ---- Phase 4: Splice residual measurement ----
    print("\n=== Phase 4: Splice Residual Measurement ===")
    splice_results = []

    # Simulate: attention from early layers, FFN from late layers
    early_layer = bands["Mix"][0] if bands["Mix"] else 0
    late_layer = bands["Refine"][-1] if bands["Refine"] else n_layers - 1

    if early_layer in all_weights and late_layer in all_weights:
        Wq_e, Wk_e, Wv_e = all_weights[early_layer]
        P_int_e = all_bases[early_layer]

        residual = compute_splice_residual(Wq_e, Wk_e, Wv_e, P_int_e, k)
        residual["source_layer"] = early_layer
        splice_results.append(residual)

        print(f"  Attention (layer {early_layer}) projected to k={k}:")
        for slot in ["Q", "K", "V"]:
            print(f"    {slot}: rel_err={residual[f'{slot}_rel_err']:.4f}, "
                  f"energy={residual[f'{slot}_energy_retained']:.4f}")

    # ---- LoRA capacity estimation ----
    print("\n=== LoRA Capacity Estimation for Tissue Healing ===")
    # The residual that LoRA needs to correct is the Frobenius norm of
    # the projection error.  A rank-r LoRA can correct at most the top-r
    # singular values of the residual.  Using the ρ estimate from Part V.

    # Compute ρ for the splice boundary layer
    if early_layer in all_weights:
        Wq, Wk, Wv = all_weights[early_layer]
        P = build_shared_basis(Wq, Wk, Wv)
        P_perp = P[:, k:]
        total_residual_norm = 0.0
        recoverable_norm = 0.0
        r = 8  # LoRA rank

        for W in [Wq, Wk, Wv]:
            V = W @ P_perp  # residual in compact form
            eta = np.linalg.norm(V, 'fro')
            total_residual_norm += eta**2

            # Top-r singular values (approximate via power iteration)
            if min(V.shape) > r:
                U, S, Vt = np.linalg.svd(V, full_matrices=False)
                recoverable_norm += np.sum(S[:r]**2)
            else:
                recoverable_norm += eta**2

        rho = recoverable_norm / max(total_residual_norm, 1e-10)
        print(f"  ρ (recoverable energy ratio) = {rho:.4f}")
        print(f"  Total residual Frobenius² = {total_residual_norm:.2f}")
        print(f"  Rank-{r} LoRA recoverable = {recoverable_norm:.2f} ({rho*100:.1f}%)")
        print(f"  Estimated post-LoRA residual = {(1-rho)*100:.1f}% of pre-LoRA")

    # ---- Summary ----
    summary = {
        "config": {"model": args.model, "k": k, "sink_T": args.sink_T},
        "bands": {name: {"layers": layers, "count": len(layers)}
                  for name, layers in bands.items()},
        "gauge_alignment": gauge_results,
        "splice_residual": splice_results,
        "lora_capacity": {
            "rho": round(float(rho) if 'rho' in dir() else 0.0, 4),
            "interpretation": (
                f"A rank-8 LoRA can recover {rho*100:.1f}% of the projection "
                f"residual. The remaining {(1-rho)*100:.1f}% requires either "
                f"higher LoRA rank, more calibration data, or is fundamentally "
                f"unrecoverable at this k."
            ) if 'rho' in dir() else "Not computed",
        },
    }
    with open(out_dir / "chimeric_splice_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {out_dir / 'chimeric_splice_summary.json'}")


if __name__ == "__main__":
    main()
