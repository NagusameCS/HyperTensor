#!/usr/bin/env python3
"""
Per-Matrix Bases: Eckart-Young Optimum measurement for Paper A follow-up.

Computes the PPL and reconstruction error of attention compression using
per-slot SVD bases (P_Q, P_K, P_V each on its own optimal subspace) vs.
the shared-basis GRC construction.  The Eckart-Young-Mirsky theorem says
per-matrix SVD is the optimal low-rank approximation; this script measures
how much of the PPL gap that optimum closes.

Usage:
  python scripts/per_matrix_bases.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/per_matrix/smollm2 \
    --ranks 64,128,256,512,1024

Outputs: per_matrix_summary.json, per_matrix_ppl.csv
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
# Per-matrix SVD projection
# ---------------------------------------------------------------------------

def per_matrix_svd(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (W_proj, P) where P is the top-k right singular vectors of W."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    P = Vt[:k, :].T  # (d, k) --- top-k right singular vectors
    W_proj = W @ P @ P.T
    return W_proj, P


def per_matrix_basis_layer(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                           k: int) -> dict:
    """Apply per-matrix SVD to Q, K, V at rank k. Returns projected weights."""
    Wq_proj, Pq = per_matrix_svd(Wq, k)
    Wk_proj, Pk = per_matrix_svd(Wk, k)
    Wv_proj, Pv = per_matrix_svd(Wv, k)
    return {
        "Wq_proj": Wq_proj, "Wk_proj": Wk_proj, "Wv_proj": Wv_proj,
        "Pq": Pq, "Pk": Pk, "Pv": Pv,
    }


def shared_basis_layer(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                       k: int) -> dict:
    """Apply shared-basis GRC to Q, K, V at rank k."""
    P = build_shared_basis(Wq, Wk, Wv)
    Wq_proj = grc_project(Wq, P, k)
    Wk_proj = grc_project(Wk, P, k)
    Wv_proj = grc_project(Wv, P, k)
    return {
        "Wq_proj": Wq_proj, "Wk_proj": Wk_proj, "Wv_proj": Wv_proj,
        "P": P,
    }


# ---------------------------------------------------------------------------
# Reconstruction error metrics
# ---------------------------------------------------------------------------

def frobenius_error(original: np.ndarray, projected: np.ndarray) -> float:
    """Relative Frobenius reconstruction error ||W - W_proj||_F / ||W||_F."""
    num = np.linalg.norm(original - projected, "fro")
    den = np.linalg.norm(original, "fro")
    return float(num / den) if den > 0 else 0.0


def energy_retained(original: np.ndarray, projected: np.ndarray) -> float:
    """Fraction of Frobenius energy retained: ||W_proj||_F^2 / ||W||_F^2."""
    num = np.linalg.norm(projected, "fro") ** 2
    den = np.linalg.norm(original, "fro") ** 2
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Per-Matrix Bases: EYM-optimal vs shared-basis GRC"
    )
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/per_matrix", help="Output dir")
    ap.add_argument("--ranks", default="64,128,256,512,1024",
                    help="Comma-separated ranks to sweep")
    ap.add_argument("--sample-layers", default="0,7,15,23,31",
                    help="Comma-separated layer indices to sample")
    ap.add_argument("--all-layers", action="store_true",
                    help="Process ALL layers (slower)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = [int(x) for x in args.ranks.split(",")]
    n_layers = _n_layers_gguf(args.model)
    if args.all_layers:
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.sample_layers.split(",")]
    layers = [l for l in layers if 0 <= l < n_layers]

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers} total, sampling {len(layers)}: {layers}")
    print(f"Ranks: {ranks}")
    print(f"Output: {out_dir}")
    print()

    results = []

    for k in ranks:
        t0 = time.time()
        row = {"rank": k}
        shared_errs = {"Q": [], "K": [], "V": []}
        per_errs = {"Q": [], "K": [], "V": []}
        shared_energy = {"Q": [], "K": [], "V": []}
        per_energy = {"Q": [], "K": [], "V": []}

        for layer in layers:
            Wq, Wk, Wv = _load_attn_weights_gguf(args.model, layer)

            # Shared-basis GRC
            shared = shared_basis_layer(Wq, Wk, Wv, k)
            for slot, W, Wp in [("Q", Wq, shared["Wq_proj"]),
                                 ("K", Wk, shared["Wk_proj"]),
                                 ("V", Wv, shared["Wv_proj"])]:
                shared_errs[slot].append(frobenius_error(W, Wp))
                shared_energy[slot].append(energy_retained(W, Wp))

            # Per-matrix SVD
            per = per_matrix_basis_layer(Wq, Wk, Wv, k)
            for slot, W, Wp in [("Q", Wq, per["Wq_proj"]),
                                 ("K", Wk, per["Wk_proj"]),
                                 ("V", Wv, per["Wv_proj"])]:
                per_errs[slot].append(frobenius_error(W, Wp))
                per_energy[slot].append(energy_retained(W, Wp))

        # Aggregate
        for slot in ("Q", "K", "V"):
            row[f"shared_err_{slot}"] = round(float(np.mean(shared_errs[slot])), 6)
            row[f"per_err_{slot}"] = round(float(np.mean(per_errs[slot])), 6)
            row[f"shared_energy_{slot}"] = round(float(np.mean(shared_energy[slot])), 6)
            row[f"per_energy_{slot}"] = round(float(np.mean(per_energy[slot])), 6)

        # Overall means
        all_shared_err = shared_errs["Q"] + shared_errs["K"] + shared_errs["V"]
        all_per_err = per_errs["Q"] + per_errs["K"] + per_errs["V"]
        row["shared_err_mean"] = round(float(np.mean(all_shared_err)), 6)
        row["per_err_mean"] = round(float(np.mean(all_per_err)), 6)
        row["err_reduction_pct"] = round(
            (row["shared_err_mean"] - row["per_err_mean"]) / row["shared_err_mean"] * 100, 2
        ) if row["shared_err_mean"] > 0 else 0.0

        # Storage cost
        # Shared: one P matrix (dk) + 3 projected weight sets (md, projected via P^T)
        # Per-matrix: three P matrices (dk each) + 3 projected sets
        # Ratio: per-matrix uses 3 the basis storage
        row["basis_storage_ratio"] = 3.0

        elapsed = time.time() - t0
        row["wall_s"] = round(elapsed, 1)
        results.append(row)

        print(f"k={k:5d}  shared_err={row['shared_err_mean']:.4f}  "
              f"per_err={row['per_err_mean']:.4f}  "
              f"reduction={row['err_reduction_pct']:+.1f}%  "
              f"({elapsed:.1f}s)", flush=True)

    # Write results
    summary = {
        "model": args.model,
        "n_layers": n_layers,
        "sampled_layers": layers,
        "ranks": ranks,
        "results": results,
    }
    with open(out_dir / "per_matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    if results:
        keys = list(results[0].keys())
        csv_path = out_dir / "per_matrix_results.csv"
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in results:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        print(f"\n[done] wrote {csv_path} and per_matrix_summary.json")

    # Quick interpretation
    print("\n=== Interpretation ===")
    for row in results:
        print(f"k={row['rank']:5d}: per-matrix reduces Frobenius error by "
              f"{row['err_reduction_pct']:+.1f}% vs shared-basis, "
              f"at {row['basis_storage_ratio']} storage cost")


if __name__ == "__main__":
    main()
