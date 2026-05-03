#!/usr/bin/env python3
"""
The 106% Anomaly as a General GPU Phenomenon (Tier 3).

Paper A discovered that at k=1024, GRC-compressed decode runs at 106.27%
of baseline throughput on an RTX 4070 Laptop.  The mechanism is kernel
fusion + cache-fit: replacing three independent GEMV launches with a
fused trio operating on k-dimensional intermediates.

This raises a broader question: under what conditions does a bandwidth-bound
kernel with a dimensionality-reduction step exhibit a "super-baseline" rank?

This script generalizes the analysis:
  1. Derives the condition for super-baseline throughput: T_compressed / T_baseline > 1
  2. Computes the optimal rank k* for arbitrary (GPU, kernel-shape) pairs
  3. Predicts which real-world GPU workloads might exhibit the effect

The core inequality (from Paper A, simplified):
  T(k) / T(∞)  =  (bytes_baseline / bytes_compressed(k))  (BW_eff(k) / BW_eff(∞))
                  (FLOP_overhead_factor)

  Super-baseline condition: T(k) / T(∞) > 1
    bytes_baseline / bytes_compressed(k) > BW_eff(∞) / BW_eff(k)  FLOP_overhead

  The transition from HBM bandwidth (256 GB/s) to L2 bandwidth (~2-4 TB/s)
  provides a ~10 bandwidth multiplier that can overcome the projection FLOP
  overhead when the working set fits in L2.

Usage:
  python scripts/super_baseline_general.py --gpu 4090
  python scripts/super_baseline_general.py --all-gpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# GPU database: {name: {L2_MB, BW_GBps, peak_TFLOPS, SM_count, arch}}
GPU_DB = {
    "RTX 4070 Laptop": {"l2_mb": 32,  "bw_gbps": 256,  "tflops": 40,   "sm": 36,  "arch": "Ada"},
    "RTX 4090":         {"l2_mb": 72,  "bw_gbps": 1008, "tflops": 82.6, "sm": 128, "arch": "Ada"},
    "A100":             {"l2_mb": 40,  "bw_gbps": 1555, "tflops": 312,  "sm": 108, "arch": "Ampere"},
    "H100":             {"l2_mb": 50,  "bw_gbps": 3350, "tflops": 989,  "sm": 132, "arch": "Hopper"},
    "L40S":             {"l2_mb": 96,  "bw_gbps": 864,  "tflops": 91.6, "sm": 142, "arch": "Ada"},
    "A10G":             {"l2_mb": 6,   "bw_gbps": 600,  "tflops": 35,   "sm": 80,  "arch": "Ampere"},
    "L4":               {"l2_mb": 48,  "bw_gbps": 300,  "tflops": 30.3, "sm": 58,  "arch": "Ada"},
    "RTX 3080":         {"l2_mb": 6,   "bw_gbps": 760,  "tflops": 29.8, "sm": 68,  "arch": "Ampere"},
    "RTX 4080":         {"l2_mb": 64,  "bw_gbps": 717,  "tflops": 48.7, "sm": 76,  "arch": "Ada"},
    "RTX 5060":         {"l2_mb": 32,  "bw_gbps": 448,  "tflops": 25,   "sm": 36,  "arch": "Blackwell"},
}


def predict_super_baseline(gpu_name: str, d_model: int = 4096,
                           d_kv: int = 1024, n_q_heads: int = 32) -> dict:
    """
    Predict whether a GPU can exhibit the super-baseline effect for
    attention compression at model dimension d_model.
    """
    gpu = GPU_DB.get(gpu_name)
    if not gpu:
        return {"error": f"Unknown GPU: {gpu_name}"}

    l2_mb = gpu["l2_mb"]
    bw_gbps = gpu["bw_gbps"]
    l2_bw_est = bw_gbps * 10  # rough: L2 ~10 HBM bandwidth

    results = []
    for k in [384, 512, 768, 1024, 1280, 1536, 2048]:
        # Working set per attention launch (Paper A §workingset-math)
        # S(k) = weight (dk fp16) + basis (dk fp16) = 4dk bytes
        S_k_bytes = 4 * d_model * k
        S_k_mb = S_k_bytes / (1024 * 1024)

        # Baseline bytes: full Q/K/V at dd each, in Q8 (~1 byte/elem)
        # Plus attention math overhead
        bytes_baseline = 3 * d_model * d_model  # 3 matrices  d² elem  1 byte

        # Compressed bytes: projected weights + basis traffic
        bytes_compressed = 3 * d_model * k + d_model * k  # 3 proj + basis
        # Plus fusion overhead (additional ~10%)
        bytes_compressed *= 1.1

        byte_ratio = bytes_baseline / bytes_compressed

        # Effective bandwidth: L2 if S(k) fits, HBM otherwise
        if S_k_mb <= l2_mb * 0.75:  # 25% headroom
            bw_eff = l2_bw_est
            regime = "L2"
        elif S_k_mb <= l2_mb:
            bw_eff = (l2_bw_est + bw_gbps) / 2  # transition
            regime = "transition"
        else:
            bw_eff = bw_gbps
            regime = "HBM"

        bw_ratio = bw_eff / bw_gbps

        # FLOP overhead: GRC adds projection FLOPs
        # Cost is roughly (k/d) extra matmul per attention step
        flop_overhead = 1.0 + (k / d_model) * 0.3  # ~30% of the k/d ratio

        # Net throughput ratio
        t_ratio = byte_ratio * bw_ratio / flop_overhead

        results.append({
            "k": k,
            "S_MB": round(S_k_mb, 1),
            "regime": regime,
            "byte_ratio": round(byte_ratio, 3),
            "bw_ratio": round(bw_ratio, 3),
            "flop_overhead": round(flop_overhead, 3),
            "t_ratio": round(t_ratio, 4),
            "super_baseline": t_ratio > 1.01,
        })

    # Find optimal k*
    best = max(results, key=lambda r: r["t_ratio"])
    return {
        "gpu": gpu_name,
        "l2_mb": l2_mb,
        "bw_gbps": bw_gbps,
        "d_model": d_model,
        "predicted_kstar": best["k"],
        "predicted_ratio": best["t_ratio"],
        "super_baseline_possible": best["super_baseline"],
        "rank_sweep": results,
    }


# ---------------------------------------------------------------------------
# Generalization: which real-world kernels might have a super-baseline rank?
# ---------------------------------------------------------------------------

def analyze_general_kernels():
    """Identify classes of GPU workloads that might exhibit the effect."""
    kernels = [
        {
            "name": "Attention QKV projection",
            "bytes_per_token": "3d² (full) → 3dk + dk (proj+basis)",
            "dimensionality_reduction": "k/d",
            "condition": "k < d/2, dk < L2",
            "likelihood": "HIGH (Paper A confirmed)",
        },
        {
            "name": "LoRA-augmented FFN",
            "bytes_per_token": "dd_ffn (full) → dr (LoRA)",
            "dimensionality_reduction": "r/d_ffn",
            "condition": "r << d_ffn, dr < L2",
            "likelihood": "MEDIUM (LoRA already exists, but not fused)",
        },
        {
            "name": "KV-cache projection",
            "bytes_per_token": "2d_kvctx (full) → 2kctx (proj)",
            "dimensionality_reduction": "k/d_kv",
            "condition": "long context, k < d_kv",
            "likelihood": "HIGH (Paper C §kv-cache, unmeasured)",
        },
        {
            "name": "Embedding table lookup",
            "bytes_per_token": "Vd (full) → Vk (proj)",
            "dimensionality_reduction": "k/d",
            "condition": "V large, k/d < 0.5",
            "likelihood": "LOW (embedding is rarely bottlenecked by BW)",
        },
        {
            "name": "Mixture-of-Experts routing",
            "bytes_per_token": "n_expertsd (full) → k (top-1)",
            "dimensionality_reduction": "1/n_experts",
            "condition": "sparse gating, expert weights in L2",
            "likelihood": "MEDIUM (MoE routing is bandwidth-heavy)",
        },
        {
            "name": "Speculative draft verification",
            "bytes_per_token": "T_V + γT_D (full) → T_V + γT_D(k)",
            "dimensionality_reduction": "k/d (on drafter only)",
            "condition": "drafter cost dominates, k < d/2",
            "likelihood": "HIGH (Paper C, tier-asymmetric deployment)",
        },
    ]
    return kernels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="106% Anomaly Generalization — Super-Baseline Predictor"
    )
    ap.add_argument("--gpu", default=None,
                    help="GPU name to analyze (e.g. 'RTX 4090', 'A100')")
    ap.add_argument("--d-model", type=int, default=4096,
                    help="Model dimension")
    ap.add_argument("--all-gpus", action="store_true",
                    help="Analyze all GPUs in the database")
    ap.add_argument("--kernels", action="store_true",
                    help="Show general kernel applicability analysis")
    ap.add_argument("--out", default="benchmarks/super_baseline")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kernels:
        print("=== GPU Kernels with Potential Super-Baseline Ranks ===\n")
        for k in analyze_general_kernels():
            print(f"  {k['name']}:")
            print(f"    Bytes: {k['bytes_per_token']}")
            print(f"    Reduction: {k['dimensionality_reduction']}")
            print(f"    Condition: {k['condition']}")
            print(f"    Likelihood: {k['likelihood']}")
            print()
        return

    # GPU analysis
    gpus_to_analyze = list(GPU_DB.keys()) if args.all_gpus else [args.gpu] if args.gpu else ["RTX 4070 Laptop"]

    all_results = {}
    for gpu_name in gpus_to_analyze:
        result = predict_super_baseline(gpu_name, args.d_model)
        all_results[gpu_name] = result

        if result.get("error"):
            print(f"{gpu_name}: {result['error']}")
            continue

        print(f"\n=== {gpu_name} (L2={result['l2_mb']}MB, BW={result['bw_gbps']}GB/s) ===")
        print(f"  d_model={result['d_model']}")
        print(f"  Predicted k*: {result['predicted_kstar']}")
        print(f"  Predicted T/T_baseline: {result['predicted_ratio']:.4f}")
        print(f"  Super-baseline possible: {result['super_baseline_possible']}")
        print()
        print(f"  {'k':>6s}  {'S(MB)':>7s}  {'regime':>12s}  "
              f"{'byte_ratio':>10s}  {'bw_ratio':>10s}  {'flop_ovh':>10s}  {'T_ratio':>10s}")
        print(f"  {'-'*72}")
        for r in result["rank_sweep"]:
            marker = " ← k*" if r["k"] == result["predicted_kstar"] else ""
            print(f"  {r['k']:6d}  {r['S_MB']:7.1f}  {r['regime']:>12s}  "
                  f"{r['byte_ratio']:10.3f}  {r['bw_ratio']:10.3f}  "
                  f"{r['flop_overhead']:10.3f}  {r['t_ratio']:10.4f}{marker}")

    # Cross-GPU comparison
    if args.all_gpus:
        print(f"\n=== Cross-GPU Super-Baseline Summary ===")
        print(f"{'GPU':>20s}  {'L2':>5s}  {'k*':>5s}  {'T_ratio':>8s}  {'Super?':>6s}")
        print("-" * 58)
        for gpu_name in gpus_to_analyze:
            r = all_results.get(gpu_name, {})
            if r.get("error"):
                continue
            print(f"{gpu_name:>20s}  {r['l2_mb']:5d}  {r['predicted_kstar']:5d}  "
                  f"{r['predicted_ratio']:8.4f}  {'YES' if r['super_baseline_possible'] else 'NO':>6s}")

    with open(out_dir / "super_baseline_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[done] {out_dir / 'super_baseline_analysis.json'}")


if __name__ == "__main__":
    main()
