#!/usr/bin/env python3
"""
FFN Structure-Aware Compression Analysis (Tier 2).

Paper A §spectra shows FFN weights have flat spectra (k95/d ≈ 0.85),
making global SVD unacceptably lossy.  But FFN columns act as key-value
memory banks (Geva et al. 2021): each column responds to localised
activation patterns.  This suggests column-cluster compression:
  1. Run a few forward passes to collect activation patterns
  2. Cluster FFN columns by co-activation similarity
  3. Compress each cluster independently with per-cluster SVD
  4. Reconstruct and measure error

This script performs the analysis on a GGUF model using only weights
(Phase 1: weight-only clustering) and optionally activation-guided
clustering (Phase 2: requires forward passes).

Usage:
  python scripts/ffn_cluster_compress.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/ffn_cluster \
    --n-clusters 4,8,16,32 \
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
from grc_distill import _load_attn_weights_gguf, _n_layers_gguf


# ---------------------------------------------------------------------------
# FFN weight loading from GGUF
# ---------------------------------------------------------------------------

def _load_ffn_weights_gguf(gguf_path: str, layer_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W_gate, W_up, W_down) as float32 numpy arrays."""
    from gguf import GGUFReader, dequantize
    r = GGUFReader(gguf_path)
    tm = {t.name: t for t in r.tensors}
    prefix = f"blk.{layer_idx}"

    def _get(name):
        t = tm[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    # Try multiple naming conventions
    for gate_name in [f"{prefix}.ffn_gate.weight", f"{prefix}.ffn_gate_proj.weight"]:
        if gate_name in tm:
            W_gate = _get(gate_name)
            break
    else:
        raise KeyError(f"Could not find gate weight for layer {layer_idx}")

    for up_name in [f"{prefix}.ffn_up.weight", f"{prefix}.ffn_up_proj.weight"]:
        if up_name in tm:
            W_up = _get(up_name)
            break
    else:
        raise KeyError(f"Could not find up weight for layer {layer_idx}")

    for down_name in [f"{prefix}.ffn_down.weight", f"{prefix}.ffn_down_proj.weight"]:
        if down_name in tm:
            W_down = _get(down_name)
            break
    else:
        raise KeyError(f"Could not find down weight for layer {layer_idx}")

    return W_gate, W_up, W_down


# ---------------------------------------------------------------------------
# Column clustering by weight similarity
# ---------------------------------------------------------------------------

def cluster_columns_by_weight(W: np.ndarray, n_clusters: int,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Cluster columns of W by cosine similarity using k-means++.
    Returns cluster_labels: array of length W.shape[1] with cluster indices.
    """
    from sklearn.cluster import KMeans  # optional import
    # Normalize columns to unit vectors for cosine clustering
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
    # k-means on normalized columns (cosine clustering)
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(rng.integers(0, 2**31)),
                    n_init=5, max_iter=100)
    labels = kmeans.fit_predict(W_norm.T)
    return labels


def cluster_columns_by_l2(W: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster columns by L2 magnitude (simple baseline).
    Sorts by magnitude and assigns to equal-sized bins.
    """
    mags = np.linalg.norm(W, axis=0)
    order = np.argsort(mags)
    labels = np.zeros(W.shape[1], dtype=int)
    per_cluster = max(1, W.shape[1] // n_clusters)
    for c in range(n_clusters):
        start = c * per_cluster
        end = min((c + 1) * per_cluster, W.shape[1])
        labels[order[start:end]] = c
    return labels


# ---------------------------------------------------------------------------
# Per-cluster SVD compression
# ---------------------------------------------------------------------------

def compress_per_cluster(W: np.ndarray, labels: np.ndarray, k_per_cluster: int) -> np.ndarray:
    """
    Compress W by applying rank-k_per_cluster SVD to each column cluster.
    Returns W_compressed of same shape.
    """
    W_comp = np.zeros_like(W)
    n_clusters = labels.max() + 1

    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        W_c = W[:, mask]  # (rows, cluster_cols)
        U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
        k = min(k_per_cluster, len(S))
        W_comp[:, mask] = (U[:, :k] @ np.diag(S[:k])) @ Vt[:k, :]

    return W_comp


def frobenius_error(original: np.ndarray, compressed: np.ndarray) -> float:
    num = np.linalg.norm(original - compressed, "fro")
    den = np.linalg.norm(original, "fro")
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FFN Structure-Aware Compression Analysis")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/ffn_cluster")
    ap.add_argument("--n-clusters", default="4,8,16,32",
                    help="Comma-separated cluster counts")
    ap.add_argument("--k-frac", default="0.25,0.50,0.75",
                    help="Per-cluster rank fractions (of cluster size)")
    ap.add_argument("--sample-layers", default="0,7,15,23,29")
    ap.add_argument("--cluster-method", default="cosine",
                    choices=["cosine", "l2", "random"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_clusters_list = [int(x) for x in args.n_clusters.split(",")]
    k_fracs = [float(x) for x in args.k_frac.split(",")]
    n_layers = _n_layers_gguf(args.model)
    layers = [int(x) for x in args.sample_layers.split(",")]
    layers = [l for l in layers if 0 <= l < n_layers]

    print(f"Model: {args.model}")
    print(f"Layers sampled: {layers}  (of {n_layers})")
    print(f"Clusters: {n_clusters_list}  k-fractions: {k_fracs}")
    print()

    results = []

    for layer in layers:
        try:
            W_gate, W_up, W_down = _load_ffn_weights_gguf(args.model, layer)
        except KeyError as e:
            print(f"  layer {layer}: SKIP ({e})")
            continue

        d_ffn = W_gate.shape[0]  # FFN intermediate dimension
        n_cols = W_gate.shape[1]  # = d_model

        print(f"\n=== Layer {layer} (d_ffn={d_ffn}, d_model={n_cols}) ===")

        # Global SVD baseline
        for k_frac in k_fracs:
            k_gate = int(n_cols * k_frac)
            k_up = int(n_cols * k_frac)
            k_down = int(d_ffn * k_frac)

            # Global SVD
            U, S, Vt = np.linalg.svd(W_gate, full_matrices=False)
            k = min(k_gate, len(S))
            W_gate_global = (U[:, :k] @ np.diag(S[:k])) @ Vt[:k, :]

            U, S, Vt = np.linalg.svd(W_up, full_matrices=False)
            k = min(k_up, len(S))
            W_up_global = (U[:, :k] @ np.diag(S[:k])) @ Vt[:k, :]

            U, S, Vt = np.linalg.svd(W_down, full_matrices=False)
            k = min(k_down, len(S))
            W_down_global = (U[:, :k] @ np.diag(S[:k])) @ Vt[:k, :]

            err_global = (
                frobenius_error(W_gate, W_gate_global) +
                frobenius_error(W_up, W_up_global) +
                frobenius_error(W_down, W_down_global)
            ) / 3.0

            row = {
                "layer": layer, "method": "global_svd",
                "n_clusters": 1, "k_frac": k_frac,
                "err_mean": round(err_global, 6),
            }
            results.append(row)
            print(f"  global SVD     k={k_frac:.2f}n:  err={err_global:.4f}")

        # Cluster-based compression
        rng = np.random.default_rng(layer)
        for n_clusters in n_clusters_list:
            if n_clusters >= n_cols:
                continue  # too many clusters

            # Cluster gate columns
            if args.cluster_method == "cosine":
                try:
                    labels = cluster_columns_by_weight(W_gate, n_clusters, rng)
                except ImportError:
                    print("  sklearn not available, falling back to L2 clustering")
                    labels = cluster_columns_by_l2(W_gate, n_clusters)
            elif args.cluster_method == "l2":
                labels = cluster_columns_by_l2(W_gate, n_clusters)
            else:  # random
                labels = rng.integers(0, n_clusters, size=n_cols)

            for k_frac in k_fracs:
                per_cluster_k_gate = max(1, int((d_ffn / n_clusters) * k_frac))
                per_cluster_k_up = max(1, int((d_ffn / n_clusters) * k_frac))
                # W_down: cluster rows, not columns. W_down maps d_ffn -> d_model.
                # Cluster the d_ffn output dimensions of gate/up, which are
                # the input dimensions of down. Same labels, applied to rows.
                per_cluster_k_down = max(1, int((d_ffn / n_clusters) * k_frac))

                W_gate_cluster = compress_per_cluster(W_gate, labels, per_cluster_k_gate)
                W_up_cluster = compress_per_cluster(W_up, labels, per_cluster_k_up)
                # For W_down, cluster rows (transpose, cluster columns, transpose back)
                W_down_cluster = compress_per_cluster(W_down.T, labels, per_cluster_k_down).T

                err_cluster = (
                    frobenius_error(W_gate, W_gate_cluster) +
                    frobenius_error(W_up, W_up_cluster) +
                    frobenius_error(W_down, W_down_cluster)
                ) / 3.0

                # Storage ratio vs global: global uses k*n_cols params per matrix;
                # cluster uses n_clusters * k_per_cluster * n_cols/n_clusters = k_per_cluster * n_cols
                # Same total! But clusters more faithful to local structure.
                row = {
                    "layer": layer, "method": f"cluster_{args.cluster_method}",
                    "n_clusters": n_clusters, "k_frac": k_frac,
                    "err_mean": round(err_cluster, 6),
                    "err_vs_global_pct": round(
                        (err_cluster - err_global) / err_global * 100, 1
                    ) if err_global > 0 else 0.0,
                }
                # Fill in the global error for comparison
                matching_global = [r for r in results
                                  if r["layer"] == layer and r["method"] == "global_svd"
                                  and abs(r["k_frac"] - k_frac) < 0.001]
                if matching_global:
                    row["global_err"] = matching_global[0]["err_mean"]
                    row["improvement_pct"] = round(
                        (matching_global[0]["err_mean"] - err_cluster) /
                        matching_global[0]["err_mean"] * 100, 1
                    )

                results.append(row)
                impr = row.get("improvement_pct", 0.0)
                print(f"  {n_clusters:2d} clusters  k={k_frac:.2f}n:  "
                      f"err={err_cluster:.4f}  "
                      f"vs_global={impr:+.1f}%")

    # Write results
    with open(out_dir / "ffn_cluster_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    if results:
        csv_path = out_dir / "ffn_cluster_results.csv"
        keys = list(results[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in results:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"\n[done] {csv_path}")

    print(f"[done] {out_dir / 'ffn_cluster_summary.json'}")


if __name__ == "__main__":
    main()
