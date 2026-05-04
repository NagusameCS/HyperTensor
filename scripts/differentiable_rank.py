#!/usr/bin/env python3
"""
Differentiable Compression Rank (Tier 3).

Paper B's MCR heuristic assigns per-layer rank using a closed-form
curvature-ratio proxy.  A more powerful approach: treat per-layer rank
k_ℓ as a learnable parameter optimized via gradient descent during
fine-tuning, with an L0/L1 sparsity penalty on the rank budget.

This script prototypes the approach:
  1. Soft rank selection via Gumbel-Softmax over discrete k values
  2. L1 penalty on total rank budget
  3. Forward pass: for each layer, sample k_ℓ ~ softmax(logits_ℓ)
     and apply GRC projection at that rank
  4. Training signal: task loss + λ  Σ k_ℓ

The prototype is NumPy-only (no autograd) and demonstrates the
mathematical construction.  A full PyTorch implementation would
replace the Gumbel-Softmax sampling with torch.nn.functional.gumbel_softmax
and optimize logits_ℓ via standard training loops.

Usage:
  python scripts/differentiable_rank.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/differentiable_rank
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
# Soft rank selection
# ---------------------------------------------------------------------------

def gumbel_softmax_sample(logits: np.ndarray, temperature: float = 1.0,
                          hard: bool = False) -> np.ndarray:
    """
    Sample from Gumbel-Softmax distribution over discrete k values.

    logits: (n_options,) --- unnormalized log-probabilities for each k
    temperature: controls sharpness (τ->0 gives argmax)
    hard: if True, return one-hot; if False, return soft probabilities

    Returns: (n_options,) probability vector
    """
    # Gumbel noise
    gumbel = -np.log(-np.log(np.random.uniform(1e-10, 1.0, logits.shape)))
    y = (logits + gumbel) / max(temperature, 1e-10)
    y_soft = np.exp(y - y.max()) / np.exp(y - y.max()).sum()

    if hard:
        # Straight-through: forward is one-hot, backward would use soft
        y_hard = np.zeros_like(y_soft)
        y_hard[y_soft.argmax()] = 1.0
        return y_hard
    return y_soft


def compute_expected_rank(rank_logits: np.ndarray,
                          k_options: np.ndarray) -> float:
    """
    Expected rank under the current softmax distribution.
    E[k] = Σᵢ p(kᵢ)  kᵢ
    """
    probs = np.exp(rank_logits - rank_logits.max())
    probs /= probs.sum()
    return float(np.dot(probs, k_options))


def compute_rank_penalty(rank_logits: np.ndarray,
                         k_options: np.ndarray,
                         penalty_type: str = "l1") -> float:
    """
    Penalty on rank budget.
    L1: penalty = E[k] (encourages smaller ranks)
    L0-ish: penalty = Σᵢ softmax(logits)ᵢ  indicator(kᵢ > 0)
    """
    probs = np.exp(rank_logits - rank_logits.max())
    probs /= probs.sum()
    if penalty_type == "l1":
        return float(np.dot(probs, k_options))
    elif penalty_type == "l0":
        # Encourage choosing k=0 (no compression = full rank)
        # Penalize choosing small k
        return float(np.dot(probs, 1.0 / np.maximum(k_options, 1)))
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Layer-wise reconstruction-quality proxy
# ---------------------------------------------------------------------------

def reconstruction_quality(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                           k: int) -> float:
    """
    Estimate reconstruction quality at rank k.
    Returns fraction of Frobenius energy retained.
    Higher is better --- this is what the optimizer maximizes.
    """
    P = build_shared_basis(Wq, Wk, Wv)
    P_k = P[:, :k]

    def energy_frac(W):
        W_proj = W @ P_k @ P_k.T
        return np.linalg.norm(W_proj, "fro") ** 2 / max(np.linalg.norm(W, "fro") ** 2, 1e-10)

    return float((energy_frac(Wq) + energy_frac(Wk) + energy_frac(Wv)) / 3.0)


# ---------------------------------------------------------------------------
# Optimization loop (gradient-free, for demonstration)
# ---------------------------------------------------------------------------

def optimize_ranks(model_path: str, n_layers: int,
                   k_options: np.ndarray,
                   lambda_penalty: float = 0.01,
                   n_iterations: int = 500,
                   learning_rate: float = 0.1,
                   temperature: float = 1.0) -> dict:
    """
    Gradient-free optimization of per-layer rank logits via REINFORCE.
    Weights are cached in memory after first load (not re-read from GGUF).
    """
    # ---- Cache all weights upfront ----
    print("  [cache] loading all layer weights into memory...", flush=True)
    weight_cache = {}
    for layer in range(n_layers):
        try:
            Wq, Wk, Wv = _load_attn_weights_gguf(model_path, layer)
            weight_cache[layer] = (Wq, Wk, Wv)
        except Exception:
            pass
    print(f"  [cache] loaded {len(weight_cache)}/{n_layers} layers", flush=True)

    # Initialize logits: bias toward middle ranks
    rank_logits = np.zeros((n_layers, len(k_options)))
    for layer in range(n_layers):
        rank_logits[layer, len(k_options)//2] = 1.0

    history = []

    for iteration in range(n_iterations):
        total_reward = 0.0

        for layer in range(n_layers):
            if layer not in weight_cache:
                continue
            Wq, Wk, Wv = weight_cache[layer]

            # Sample k
            probs = gumbel_softmax_sample(rank_logits[layer], temperature)
            k_idx = probs.argmax()
            k = int(k_options[k_idx])

            # Reconstruction quality
            Q = reconstruction_quality(Wq, Wk, Wv, k)

            # Penalty
            penalty = compute_rank_penalty(rank_logits[layer], k_options)
            reward = Q - lambda_penalty * penalty

            # REINFORCE update with baseline
            # Compute baseline: expected reward under current softmax
            rewards_for_baseline = []
            for ki, ko in enumerate(k_options):
                Q_ko = reconstruction_quality(Wq, Wk, Wv, int(ko))
                pen_ko = compute_rank_penalty(rank_logits[layer], k_options)
                rewards_for_baseline.append(Q_ko - lambda_penalty * pen_ko)
            softmax_probs = np.exp(rank_logits[layer] - rank_logits[layer].max())
            softmax_probs /= softmax_probs.sum()
            baseline = float(np.dot(softmax_probs, rewards_for_baseline))

            advantage = reward - baseline
            one_hot = np.zeros_like(rank_logits[layer])
            one_hot[k_idx] = 1.0

            rank_logits[layer] += learning_rate * advantage * (one_hot - softmax_probs)

            total_reward += reward

            if layer == 0 and iteration % 50 == 0:
                exp_k = compute_expected_rank(rank_logits[layer], k_options)
                print(f"  iter {iteration:4d}  layer 0: E[k]={exp_k:.0f}  "
                      f"sampled k={k}  Q={Q:.4f}  reward={reward:.4f}",
                      flush=True)

        history.append({
            "iteration": iteration,
            "total_reward": float(total_reward),
            "mean_expected_k": float(np.mean([
                compute_expected_rank(rank_logits[l], k_options)
                for l in range(n_layers) if l in weight_cache
            ])),
        })

        # Anneal temperature
        temperature = max(0.1, temperature * 0.995)

    # Final ranks
    final_ranks = {}
    for layer in range(n_layers):
        if layer in weight_cache:
            exp_k = compute_expected_rank(rank_logits[layer], k_options)
            final_ranks[layer] = round(exp_k)

    return {
        "final_ranks": final_ranks,
        "mean_rank": float(np.mean(list(final_ranks.values()))),
        "k_options": k_options.tolist(),
        "lambda_penalty": lambda_penalty,
        "history": history[-10:],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Differentiable Compression Rank Prototype")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/differentiable_rank")
    ap.add_argument("--k-options", default="64,128,256,384,512,768,1024,1536",
                    help="Discrete rank options")
    ap.add_argument("--lambda", type=float, default=0.01, dest="lambda_penalty",
                    help="L1 penalty weight on rank budget")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--sample-layers", type=int, default=0,
                    help="Number of layers to optimize (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_options = np.array([int(x) for x in args.k_options.split(",")])
    n_layers_total = _n_layers_gguf(args.model)
    n_layers = args.sample_layers if args.sample_layers > 0 else n_layers_total
    n_layers = min(n_layers, n_layers_total)

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers} (of {n_layers_total})")
    print(f"k options: {k_options}")
    print(f"λ = {args.lambda_penalty}")
    print(f"Iterations: {args.iterations}")
    print()

    result = optimize_ranks(
        args.model, n_layers, k_options,
        lambda_penalty=args.lambda_penalty,
        n_iterations=args.iterations,
        learning_rate=args.lr,
    )

    print(f"\n=== Optimized Per-Layer Ranks ===")
    print(f"  Mean rank: {result['mean_rank']:.0f}")
    print(f"  λ = {result['lambda_penalty']}")
    print()

    # Show per-layer distribution
    ranks = list(result["final_ranks"].values())
    print(f"  Layer rank distribution:")
    print(f"    Min:     {min(ranks)}")
    print(f"    Median:  {np.median(ranks):.0f}")
    print(f"    Max:     {max(ranks)}")
    print(f"    Std:     {np.std(ranks):.0f}")

    # Compare with MCR (Paper B): does it discover Mix/Compress/Refine phases?
    early_layers = [result["final_ranks"].get(l, 0) for l in range(min(5, n_layers))]
    mid_layers = [result["final_ranks"].get(l, 0) for l in range(n_layers//3, 2*n_layers//3)]
    late_layers = [result["final_ranks"].get(l, 0) for l in range(max(0, n_layers-5), n_layers)]
    print(f"\n  Phase analysis (compare with MCR):")
    print(f"    Early layers (0-4):     mean rank = {np.mean(early_layers):.0f}")
    print(f"    Middle layers (1/3-2/3): mean rank = {np.mean(mid_layers):.0f}")
    print(f"    Late layers (L-5:L):    mean rank = {np.mean(late_layers):.0f}")

    with open(out_dir / "differentiable_rank_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[done] {out_dir / 'differentiable_rank_summary.json'}")


if __name__ == "__main__":
    main()
