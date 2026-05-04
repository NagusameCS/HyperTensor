#!/usr/bin/env python3
"""
MoE  GRC: Why GQA Attention Is Harder to Compress (Tier 3).

Paper A Table 5 reports k_int/d ratios:
  - MHA models: 0.52--0.65 (SmolLM2, Gemma3/4)
  - GQA models: 0.68--0.71 (Qwen3.5, Gemma4-27B/31B)

The 0.70 ratio means rank-128 GRC captures only ~3% of joint energy
on Gemma4-27B --- producing incoherent output.  This script explains WHY.

In GQA (Grouped-Query Attention), n_q > n_kv:
  - W_Q has shape (n_q  h)  d  --- tall and rectangular
  - W_K, W_V have shape (n_kv  h)  d --- shorter
  - The joint Gram W_Q^T W_Q dominates because Q has more rows
  - Q's Gram is closer to full-rank because each head learns its own subspace

The fix: per-head compression instead of joint QKV compression.
Each head's Q/K/V projections are compressed independently, then
reassembled.  This preserves per-head specialization at the cost of
more basis storage.

This script:
  1. Analyzes GQA vs MHA joint Gram spectra from real GGUF models
  2. Computes per-head k_int vs joint k_int
  3. Estimates the rank needed for 95% energy retention under GQA
  4. Proposes per-head compression strategy

Usage:
  python scripts/moe_gqa_analysis.py \
    --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --out benchmarks/moe_gqa
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


# ---------------------------------------------------------------------------
# GQA structure detection
# ---------------------------------------------------------------------------

def detect_attention_structure(gguf_path: str) -> dict:
    """Detect MHA vs GQA structure from GGUF metadata."""
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)

    # Try to read metadata
    info = {"type": "unknown", "n_heads": 0, "n_kv_heads": 0, "head_dim": 0, "d_model": 0}

    # Check field names (GGUF metadata varies by model)
    for field in r.fields.values():
        name = field.name
        if "attention.head_count" in name or name.endswith(".attention.head_count"):
            info["n_heads"] = int(field.parts[-1][0])
        if "attention.head_count_kv" in name or name.endswith(".attention.head_count_kv"):
            info["n_kv_heads"] = int(field.parts[-1][0])
        if "embedding_length" in name or name.endswith(".embedding_length"):
            info["d_model"] = int(field.parts[-1][0])

    # Detect from weight shapes
    for t in r.tensors:
        if t.name.endswith(".attn_q.weight"):
            info["d_model"] = t.shape[1]
            info["n_heads"] = t.shape[0] // (t.shape[0] // info.get("n_heads") or 64)
            # head_dim = total_q_dim / n_heads
        if t.name.endswith(".attn_k.weight"):
            kv_dim = t.shape[0]
            if info.get("n_heads", 0) > 0:
                info["n_kv_heads"] = kv_dim // (info["n_heads"] // info.get("n_kv_heads", 1) or 64)

    # Heuristic: if n_kv_heads unknown, infer from weight shapes
    if info["n_kv_heads"] == 0:
        # Get first layer's Q and K shapes
        for t in r.tensors:
            if t.name == "blk.0.attn_q.weight":
                q_dim = t.shape[0]
            if t.name == "blk.0.attn_k.weight":
                k_dim = t.shape[0]
        if 'q_dim' in locals() and 'k_dim' in locals():
            ratio = q_dim / k_dim
            if ratio > 2:
                info["type"] = "GQA"
                info["n_heads"] = q_dim // 128  # assume head_dim=128
                info["n_kv_heads"] = k_dim // 128
            else:
                info["type"] = "MHA"
        else:
            # Fallback: check ratio from first layer load
            try:
                Wq, Wk, Wv = _load_attn_weights_gguf(gguf_path, 0)
                ratio = Wq.shape[0] / Wk.shape[0]
                info["type"] = "GQA" if ratio > 2 else "MHA"
                info["n_heads"] = Wq.shape[0] // info.get("head_dim", 128)
                info["n_kv_heads"] = Wk.shape[0] // info.get("head_dim", 128)
            except Exception:
                pass

    if info["type"] == "unknown":
        # One more try: load first layer
        try:
            Wq, Wk, Wv = _load_attn_weights_gguf(gguf_path, 0)
            ratio = Wq.shape[0] / max(Wk.shape[0], 1)
            info["type"] = "GQA" if ratio > 2 else "MHA"
            info["n_heads"] = int(Wq.shape[0] / (Wk.shape[0] / max(info.get("n_kv_heads", 1), 1)))
            info["n_kv_heads"] = max(info.get("n_kv_heads", 1), int(Wk.shape[0] / 128))
            info["head_dim"] = Wk.shape[0] // max(info["n_kv_heads"], 1)
            info["d_model"] = Wq.shape[1]
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Joint vs per-head Gram analysis
# ---------------------------------------------------------------------------

def analyze_joint_gram(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> dict:
    """Analyze the joint Gram spectrum and return k_int at various thresholds."""
    K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[::-1]  # descending
    eigvals = np.maximum(eigvals, 0)

    total = eigvals.sum()
    cumsum = np.cumsum(eigvals)

    result = {"d": len(eigvals)}
    for threshold in [0.50, 0.75, 0.90, 0.95, 0.99]:
        k = int(np.searchsorted(cumsum / total, threshold))
        result[f"k_{int(threshold*100)}"] = k
        result[f"k_{int(threshold*100)}_d"] = round(k / len(eigvals), 4)

    # Eigenvalue concentration (Herfindahl index of top 10%)
    top_10pct = max(1, len(eigvals) // 10)
    herf = float(np.sum(eigvals[:top_10pct] ** 2) / max(np.sum(eigvals ** 2), 1e-30))
    result["eigval_concentration_top10pct"] = round(herf, 4)

    return result


def analyze_per_head(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                     n_heads: int, n_kv_heads: int) -> dict:
    """
    Analyze Gram spectrum per KV head.
    For GQA, each KV head is shared by (n_heads / n_kv_heads) Q heads.
    """
    head_dim_q = Wq.shape[0] // n_heads
    head_dim_k = Wk.shape[0] // n_kv_heads

    per_head_results = []
    for h in range(n_kv_heads):
        # Get the KV head slice
        k_start = h * head_dim_k
        k_end = (h + 1) * head_dim_k
        Wk_h = Wk[k_start:k_end, :]
        Wv_h = Wv[k_start:k_end, :]

        # Get the corresponding Q heads
        q_heads_per_kv = n_heads // n_kv_heads
        q_start = h * q_heads_per_kv * head_dim_q
        q_end = (h + 1) * q_heads_per_kv * head_dim_q
        Wq_h = Wq[q_start:q_end, :]

        # Per-head joint Gram
        K_h = Wq_h.T @ Wq_h + Wk_h.T @ Wk_h + Wv_h.T @ Wv_h
        K_h = K_h / max(np.linalg.norm(K_h, "fro"), 1e-10)
        eigvals_h = np.linalg.eigvalsh(K_h)
        eigvals_h = np.maximum(eigvals_h[::-1], 0)
        total_h = eigvals_h.sum()
        cumsum_h = np.cumsum(eigvals_h)

        k95_h = int(np.searchsorted(cumsum_h / max(total_h, 1e-10), 0.95))
        per_head_results.append({
            "head": h,
            "k95_per_head": k95_h,
            "k95_per_head_d": round(k95_h / len(eigvals_h), 4),
            "head_dim_q": head_dim_q,
            "head_dim_k": head_dim_k,
        })

    # Compare: joint k95 vs sum of per-head k95
    sum_per_head_k95 = sum(r["k95_per_head"] for r in per_head_results)
    return {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim_q": head_dim_q,
        "head_dim_k": head_dim_k,
        "per_head_k95_sum": sum_per_head_k95,
        "per_head_details": per_head_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MoE  GRC: GQA Attention Analysis")
    ap.add_argument("--model", required=True, help="Path to GGUF model (GQA preferred)")
    ap.add_argument("--out", default="benchmarks/moe_gqa")
    ap.add_argument("--sample-layers", default="0,7,15")
    ap.add_argument("--compare-mha", default=None,
                    help="Optional MHA model for comparison")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_path in [args.model] + ([args.compare_mha] if args.compare_mha else []):
        if not Path(model_path).exists():
            print(f"SKIP: {model_path} not found")
            continue

        info = detect_attention_structure(model_path)
        print(f"\n{'='*60}")
        print(f"Model: {Path(model_path).name}")
        print(f"Type: {info['type']}  "
              f"n_heads={info.get('n_heads','?')}  "
              f"n_kv_heads={info.get('n_kv_heads','?')}  "
              f"d_model={info.get('d_model','?')}")
        print(f"{'='*60}")

        n_layers = _n_layers_gguf(model_path)
        layers = [int(x) for x in args.sample_layers.split(",")]
        layers = [l for l in layers if 0 <= l < n_layers]

        all_joint = []
        all_per_head = []

        for layer in layers:
            Wq, Wk, Wv = _load_attn_weights_gguf(model_path, layer)
            joint = analyze_joint_gram(Wq, Wk, Wv)
            joint["layer"] = layer
            all_joint.append(joint)

            print(f"\n  Layer {layer}:")
            print(f"    Joint Gram k95 = {joint['k_95']} "
                  f"({joint['k_95_d']:.3f} d)")

            if info["type"] == "GQA" and info.get("n_heads", 0) > 0:
                per_head = analyze_per_head(
                    Wq, Wk, Wv, info["n_heads"], info["n_kv_heads"]
                )
                per_head["layer"] = layer
                all_per_head.append(per_head)
                print(f"    Per-head k95 sum = {per_head['per_head_k95_sum']}")
                print(f"    Avg per-head k95 = {per_head['per_head_k95_sum']/per_head['n_kv_heads']:.0f}")
                ratio = per_head['per_head_k95_sum'] / max(joint['k_95'], 1)
                print(f"    Per-head / Joint ratio = {ratio:.2f}")
                for hd in per_head["per_head_details"]:
                    print(f"      Head {hd['head']}: k95={hd['k95_per_head']} "
                          f"({hd['k95_per_head_d']:.3f}d)")

        # Summary
        mean_k95 = np.mean([j["k_95"] for j in all_joint])
        mean_k95_d = np.mean([j["k_95_d"] for j in all_joint])
        print(f"\n  Mean joint k95 = {mean_k95:.0f} ({mean_k95_d:.3f}d)")

        if info["type"] == "GQA" and all_per_head:
            mean_ph_sum = np.mean([p["per_head_k95_sum"] for p in all_per_head])
            print(f"  Mean per-head k95 sum = {mean_ph_sum:.0f}")
            print(f"  Per-head overhead = {mean_ph_sum / max(mean_k95, 1):.2f} joint")
            print(f"  Interpretation: per-head compression needs "
                  f"{mean_ph_sum/mean_k95:.1f} the rank budget but preserves "
                  f"per-head specialization, avoiding the GQA collapse at low k.")

    # Write results
    result = {
        "model": str(args.model),
        "model_info": info,
        "joint_analysis": all_joint,
        "per_head_analysis": all_per_head,
    }
    with open(out_dir / "moe_gqa_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[done] {out_dir / 'moe_gqa_summary.json'}")


if __name__ == "__main__":
    main()
