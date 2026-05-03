#!/usr/bin/env python3
"""
FFN Language Memory Extraction via Column Clustering (Phase 2, Step 3).

The FFN block acts as a key-value memory bank (Geva et al. 2021).
Different columns respond to different input patterns.  This script
clusters FFN columns by their activation patterns to identify:

  1. "Language memory" — columns that activate on syntactic/lexical patterns
  2. "Math memory" — columns that activate on numerical/symbolic patterns
  3. "Shared memory" — columns that respond to both

In a real experiment, this would separate the Language model's FFN into
extractable language-memory clusters and non-language (shared/general) clusters.

For validation on a single model, we use the L2-norm distribution of FFN
columns as a proxy (the massive-activation phenomenon separates high-impact
columns from low-impact ones).

Usage:
  python scripts/ffn_language_extract.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/ffn_language_extract
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
    _load_attn_weights_gguf,
    _n_layers_gguf,
)


def load_ffn_weights_gguf(gguf_path: str, layer_idx: int):
    """Return (W_gate, W_up, W_down) as float32 numpy arrays."""
    from gguf import GGUFReader, dequantize
    r = GGUFReader(gguf_path)
    tm = {t.name: t for t in r.tensors}
    prefix = f"blk.{layer_idx}"

    def _get(name):
        t = tm[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    for gn in [f"{prefix}.ffn_gate.weight", f"{prefix}.ffn_gate_proj.weight"]:
        if gn in tm:
            W_gate = _get(gn); break
    else:
        raise KeyError(f"No gate weight for layer {layer_idx}")
    for un in [f"{prefix}.ffn_up.weight", f"{prefix}.ffn_up_proj.weight"]:
        if un in tm:
            W_up = _get(un); break
    else:
        raise KeyError(f"No up weight for layer {layer_idx}")
    for dn in [f"{prefix}.ffn_down.weight", f"{prefix}.ffn_down_proj.weight"]:
        if dn in tm:
            W_down = _get(dn); break
    else:
        raise KeyError(f"No down weight for layer {layer_idx}")
    return W_gate, W_up, W_down


def cluster_ffn_columns(W_gate: np.ndarray, W_up: np.ndarray,
                        n_clusters: int = 4) -> dict:
    """
    Cluster FFN columns by their combined gate+up weight patterns.

    Uses L2-magnitude sorting (simplest) — the massive-activation
    phenomenon means a few columns have outlier norms and carry
    disproportionate importance.  These are the "sink-like" columns
    that encode high-impact knowledge (likely domain-specific).

    Returns:
      clusters: dict mapping cluster_id -> column indices
      stats: per-cluster statistics
    """
    d_ffn, d_model = W_gate.shape

    # Combined magnitude from gate and up
    mags = np.sqrt(np.sum(W_gate**2, axis=0) + np.sum(W_up**2, axis=0))

    # Sort by magnitude and partition into equal-sized clusters
    order = np.argsort(mags)[::-1]
    clusters = {}
    per_cluster = max(1, d_model // n_clusters)

    stats = {}
    for c in range(n_clusters):
        start = c * per_cluster
        end = min((c + 1) * per_cluster, d_model)
        if start >= d_model:
            break
        idx = order[start:end]
        clusters[c] = idx.tolist()
        stats[c] = {
            "n_columns": len(idx),
            "mean_magnitude": round(float(np.mean(mags[idx])), 4),
            "frac_total_energy": round(
                float(np.sum(mags[idx]**2) / max(np.sum(mags**2), 1e-10)), 4
            ),
            "label": (
                "HIGH-IMPACT (likely domain-specific knowledge)"
                if c == 0 else
                "MEDIUM-HIGH" if c == 1 else
                "MEDIUM-LOW" if c == 2 else
                "LOW-IMPACT (likely general/shared)"
            ),
        }

    return clusters, stats


def estimate_extractable_memory(W_gate: np.ndarray, W_up: np.ndarray,
                                W_down: np.ndarray,
                                n_language_clusters: int = 1) -> dict:
    """
    Estimate how much of the FFN is extractable as "language memory."

    The top cluster (highest L2 norm columns) is the candidate for
    domain-specific knowledge extraction.  The remaining clusters
    are the "shared" or "general" memory.

    Returns extraction metrics.
    """
    d_ffn, d_model = W_gate.shape
    clusters, stats = cluster_ffn_columns(W_gate, W_up, n_clusters=4)

    # Language cluster = top cluster
    lang_cluster_idx = clusters[0]
    lang_energy = stats[0]["frac_total_energy"]

    # Shared clusters = rest
    shared_energy = sum(stats[c]["frac_total_energy"]
                        for c in range(1, 4) if c in stats)

    return {
        "language_cluster_size": len(lang_cluster_idx),
        "language_energy_fraction": round(lang_energy, 4),
        "shared_energy_fraction": round(shared_energy, 4),
        "extractable_params": len(lang_cluster_idx) * d_ffn,  # gate+up columns
        "extractable_params_pct": round(
            len(lang_cluster_idx) / d_model * 100, 1
        ),
        "interpretation": (
            f"The top cluster ({len(lang_cluster_idx)} columns, "
            f"{len(lang_cluster_idx)/d_model*100:.1f}% of FFN) carries "
            f"{lang_energy*100:.1f}% of total gate+up energy. "
            f"These are the candidate 'language memory' columns for extraction."
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="FFN Language Memory Extraction"
    )
    ap.add_argument("--model", default="models/smollm2-135m-instruct-q8_0.gguf")
    ap.add_argument("--out", default="benchmarks/ffn_language_extract")
    ap.add_argument("--sample-layers", default="0,7,15,23,29")
    ap.add_argument("--n-clusters", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = _n_layers_gguf(args.model)
    layers = [int(x) for x in args.sample_layers.split(",")]
    layers = [l for l in layers if 0 <= l < n_layers]

    print(f"Model: {args.model}")
    print(f"Layers sampled: {layers}")
    print(f"Clusters: {args.n_clusters}")
    print()

    all_extractions = []
    all_stats = []

    for layer in layers:
        try:
            W_gate, W_up, W_down = load_ffn_weights_gguf(args.model, layer)
        except KeyError as e:
            print(f"  layer {layer}: SKIP ({e})")
            continue

        clusters, stats = cluster_ffn_columns(W_gate, W_up, args.n_clusters)
        extraction = estimate_extractable_memory(W_gate, W_up, W_down)

        all_extractions.append({"layer": layer, **extraction})
        all_stats.append({"layer": layer, "clusters": {
            str(c): s for c, s in stats.items()
        }})

        print(f"  Layer {layer}:")
        print(f"    Top cluster: {stats[0]['n_columns']} cols "
              f"({stats[0]['frac_total_energy']*100:.1f}% energy)")
        for c in range(1, args.n_clusters):
            if c in stats:
                print(f"    Cluster {c}:     {stats[c]['n_columns']} cols "
                      f"({stats[c]['frac_total_energy']*100:.1f}% energy)")
        print(f"    Extractable: {extraction['extractable_params_pct']:.1f}% of FFN columns")
        print()

    # Cross-layer summary
    mean_extractable = np.mean([e["language_energy_fraction"] for e in all_extractions])
    print(f"=== Cross-Layer Summary ===")
    print(f"  Mean extractable energy fraction: {mean_extractable*100:.1f}%")
    print(f"  Interpretation: Across {len(layers)} sampled layers, the top "
          f"L2-magnitude cluster carries ~{mean_extractable*100:.0f}% of FFN energy.")
    print(f"  This is the fraction that could be isolated as 'domain-specific memory' "
          f"in a dedicated language model for the chimeric splice.")

    summary = {
        "config": {"model": args.model, "n_clusters": args.n_clusters},
        "extractions": all_extractions,
        "cluster_stats": all_stats,
        "cross_layer_mean_extractable_energy": round(float(mean_extractable), 4),
    }
    with open(out_dir / "ffn_language_extract.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {out_dir / 'ffn_language_extract.json'}")


if __name__ == "__main__":
    main()
