#!/usr/bin/env python3
"""
Heterogeneous Drafter Simulation (Tier 3).

Paper C uses a single compression rank k for all γ draft slots.
But different draft slots have different acceptance probabilities:
  - Early slots (1, 2): high acceptance, benefit from accurate drafter
  - Late slots (3, 4): low acceptance, can tolerate aggressive compression

This script simulates multi-level drafting:
  For γ=4, use k₁, k₂ for slots 1-2 and k₃, k₄ for slots 3-4
  where k₁₂ > k₃₄ (more rank for early slots, less for late).

The expected throughput is:
  T(k₁:k₄) = E[accepted] / (Σᵢ T_D(kᵢ) + T_V)

where acceptance probability α(k) depends on compression rank
(higher rank -> higher α, from Paper C measurements).

Usage:
  python scripts/heterogeneous_drafters.py --gamma 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Acceptance-rate model (calibrated from Paper C data)
# ---------------------------------------------------------------------------

def alpha_of_k(k: int, d_model: int = 4096,
               alpha_inf: float = 0.469,  # α at k=∞ (uncompressed)
               k_critical: int = 768,      # k where α drops sharply
               sharpness: float = 0.01,    # sharpness of the drop
               ) -> float:
    """
    Model acceptance rate as a function of compression rank.
    Uses a sigmoid centered at k_critical:
      α(k) = α_inf  σ((k - k_critical) / (sharpness  d_model))
    where σ is the logistic function.

    Calibrated from Paper C:
      - α(∞) = 46.9% (Llama-8B)
      - α(128) = 0.0% (acceptance collapse)
      - α(256) varies per-prompt, one at 56.2%
      - α(1024) = 46.9% (headline)
    """
    if k is None:
        return alpha_inf
    z = (k - k_critical) / (sharpness * d_model)
    # Clip for numerical stability
    z = np.clip(z, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return alpha_inf * sigmoid


def drafter_cost(k: int, T_D_full: float = 28.1,  # ms for Llama-8B full decode
                 d_model: int = 4096) -> float:
    """Drafter step cost as function of compression rank (ms)."""
    if k is None:
        return T_D_full
    # Linear interpolation between compressed and full
    rho = k / d_model  # fraction of full dimension
    return T_D_full * (0.3 + 0.7 * rho)  # floor at 30% of full cost


def expected_accepted(ks: list[int], gamma: int,
                      d_model: int = 4096) -> float:
    """
    Expected number of accepted tokens for a heterogeneous drafter.
    ks[i] is the compression rank for draft slot i.

    Acceptance is sequential: token i is accepted iff all tokens 1..i are accepted.
    E[accepted] = Σᵢ₌₁ᵍᵃᵐᵐᵃ Π₌₁ⁱ α(k)
    """
    total = 0.0
    cum_alpha = 1.0
    for i in range(gamma):
        a = alpha_of_k(ks[i], d_model)
        cum_alpha *= a
        total += cum_alpha
    return total


def throughput(ks: list[int], gamma: int, T_V: float = 90.0,  # ms verifier
               d_model: int = 4096) -> float:
    """Expected tokens per millisecond for heterogeneous drafter."""
    T_D_total = sum(drafter_cost(ks[i], d_model=d_model) for i in range(gamma))
    E_acc = expected_accepted(ks, gamma, d_model)
    return E_acc / (T_D_total + T_V)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_grid(gamma: int = 4, d_model: int = 4096) -> list[dict]:
    """Simulate across a grid of heterogeneous rank configurations."""
    # Rank options for early slots (more accurate) and late slots (more aggressive)
    rank_options = [None, 2048, 1536, 1024, 768, 512, 256, 128]

    results = []
    # Baseline: uniform rank
    for k in rank_options:
        ks = [k] * gamma
        t = throughput(ks, gamma, d_model=d_model)
        a = alpha_of_k(k, d_model)
        results.append({
            "config": f"uniform k={k}",
            "ks": [str(k)] * gamma,
            "throughput_tok_per_ms": round(t, 4),
            "E_accepted": round(expected_accepted(ks, gamma, d_model), 2),
            "alpha_per_slot": round(a, 4),
        })

    # Heterogeneous: decreasing rank across slots
    for k_early in [1536, 1024, 768]:
        for k_late in [512, 256, 128]:
            if k_late >= (k_early or 9999):
                continue
            ks = [k_early, k_early, k_late, k_late][:gamma]
            t = throughput(ks, gamma, d_model=d_model)
            results.append({
                "config": f"k₁₂={k_early} k₃₄={k_late}",
                "ks": [str(k) for k in ks],
                "throughput_tok_per_ms": round(t, 4),
                "E_accepted": round(expected_accepted(ks, gamma, d_model), 2),
            })

    # Sort by throughput
    results.sort(key=lambda r: r["throughput_tok_per_ms"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Heterogeneous Drafter Simulation")
    ap.add_argument("--gamma", type=int, default=4, help="Draft length")
    ap.add_argument("--d-model", type=int, default=4096, help="Model dimension")
    ap.add_argument("--out", default="benchmarks/heterogeneous_drafters")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Heterogeneous Drafter Simulation (γ={args.gamma}, d={args.d_model})")
    print()

    # Show acceptance-rate curve
    print("=== Acceptance Rate vs Rank ===")
    ks_demo = [None, 2048, 1536, 1024, 768, 512, 384, 256, 128]
    for k in ks_demo:
        a = alpha_of_k(k, args.d_model)
        print(f"  k={str(k):>5s}: α={a:.4f}")

    # Show drafter cost
    print("\n=== Drafter Cost vs Rank ===")
    for k in ks_demo:
        c = drafter_cost(k, d_model=args.d_model)
        print(f"  k={str(k):>5s}: T_D={c:.2f} ms")

    # Simulate
    print(f"\n=== Heterogeneous Drafter Configurations (γ={args.gamma}) ===")
    results = simulate_grid(args.gamma, args.d_model)

    # Show top configurations
    print(f"\n{'Rank':>6s}  {'Config':<25s}  {'T (tok/ms)':>10s}  {'E[acc]':>8s}")
    print("-" * 58)
    for r in results[:15]:
        print(f"{'':>6s}  {r['config']:<25s}  {r['throughput_tok_per_ms']:10.4f}  "
              f"{r['E_accepted']:8.2f}")

    # Best
    best = results[0]
    best_uniform = [r for r in results if r["config"].startswith("uniform")][0]
    print(f"\n  Best heterogeneous: {best['config']} "
          f"-> {best['throughput_tok_per_ms']:.4f} tok/ms")
    print(f"  Best uniform:       {best_uniform['config']} "
          f"-> {best_uniform['throughput_tok_per_ms']:.4f} tok/ms")
    if best["throughput_tok_per_ms"] > best_uniform["throughput_tok_per_ms"]:
        gain = (best["throughput_tok_per_ms"] / best_uniform["throughput_tok_per_ms"] - 1) * 100
        print(f"  Heterogeneous win: +{gain:.1f}% over best uniform")
    else:
        print(f"  Heterogeneous does not beat uniform at this γ")

    summary = {
        "gamma": args.gamma,
        "d_model": args.d_model,
        "top_configs": results[:20],
        "best_heterogeneous": best,
        "best_uniform": best_uniform,
    }
    with open(out_dir / "heterogeneous_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {out_dir / 'heterogeneous_summary.json'}")


if __name__ == "__main__":
    main()
