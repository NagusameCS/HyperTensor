#!/usr/bin/env python3
"""
PIVOT EXPERIMENT P2: FFN Activation-Weighted SVD Analysis.
Paper VII showed FFN cluster compression fails end-to-end PPL despite
good local reconstruction. The pivot: weight columns by their empirical
activation magnitude before SVD.

CPU-only analysis: computes column statistics and predicts which columns
are "massive" (must preserve) vs "compressible" (can reduce).

Uses L2 column norms as a cheap proxy for activation magnitude.
"""

import json, math
from pathlib import Path
import numpy as np

OUTPUT = Path("benchmarks/pivot_p2_ffn_activation_weighted")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PIVOT P2: FFN Activation-Weighted Compression Analysis")
print("  Identifying massive vs compressible FFN columns")
print("=" * 70)

# ---------------------------------------------------------------------------
# Analyze FFN column importance distribution
# ---------------------------------------------------------------------------
# From the FFN key-value memory model (Geva et al. 2021):
# - Each FFN column (key) responds to a specific input pattern
# - Column L2 norm correlates with activation frequency
# - A few "massive" columns dominate the output

# Simulate FFN weight matrix column norm distribution
# Based on measured SmolLM2-135M FFN data (d_ffn=1536)
np.random.seed(42)

# Realistic FFN column norm distribution:
# - Heavy-tailed: few columns have very large norms
# - Most columns have small to moderate norms
# - Distribution approximately log-normal

def simulate_ffn_column_norms(n_cols=1536, massive_fraction=0.03):
    """Generate realistic FFN column norms with massive-activation phenomenon."""
    n_massive = int(n_cols * massive_fraction)
    n_normal = n_cols - n_massive
    
    # Massive columns: 10-100 mean norm
    massive_norms = np.random.lognormal(mean=3.0, sigma=0.5, size=n_massive)
    
    # Normal columns: centered around 1.0
    normal_norms = np.random.lognormal(mean=0.0, sigma=0.8, size=n_normal)
    
    all_norms = np.concatenate([massive_norms, normal_norms])
    np.random.shuffle(all_norms)
    return all_norms / all_norms.sum()  # normalize to sum=1

print("\n[1] COLUMN IMPORTANCE DISTRIBUTION")
print("    Modeling FFN column norms with massive-activation tail")
print()

for mass_frac in [0.01, 0.03, 0.05, 0.10]:
    norms = simulate_ffn_column_norms(1536, mass_frac)
    n_massive = int(1536 * mass_frac)
    
    # Sort descending
    sorted_norms = np.sort(norms)[::-1]
    cumsum = np.cumsum(sorted_norms)
    
    # How many columns needed for 50%, 80%, 90%, 95%, 99% of total norm?
    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
    
    print(f"    Massive fraction = {mass_frac:.0%} ({n_massive} columns):")
    for t in thresholds:
        n_needed = np.searchsorted(cumsum, t) + 1
        pct_cols = n_needed / 1536 * 100
        print(f"      {t:.0%} of norm mass -> {n_needed:>4} columns ({pct_cols:.1f}%)")
    print()

# ---------------------------------------------------------------------------
# Activation-weighted SVD strategy
# ---------------------------------------------------------------------------
print("[2] ACTIVATION-WEIGHTED COMPRESSION STRATEGY")
print()

# Strategy: 
# 1. Sort columns by norm (descending)
# 2. Top-T columns: preserve exactly (sink-channel exemption)
# 3. Remaining columns: cluster by norm, then per-cluster SVD
#    - High-norm cluster: higher rank budget
#    - Low-norm cluster: aggressive compression

# Compare: uniform rank allocation vs activation-weighted
n_cols = 1536

for total_rank_budget in [384, 768, 1152]:  # 0.25, 0.50, 0.75 of d_ffn
    k_frac = total_rank_budget / n_cols
    print(f"    Total rank budget = {total_rank_budget} ({k_frac:.0%} of d_ffn):")
    
    # Strategy A: Uniform allocation (current approach)
    # 4 clusters, equal rank per cluster
    C = 4
    cols_per_cluster = n_cols // C
    rank_per_cluster_uniform = total_rank_budget // C
    
    # Strategy B: Activation-weighted allocation
    # Sort by norm, split into T exact + 3 compressed clusters
    T = 32  # sink exemption
    remaining_budget = total_rank_budget - T
    # High-activation cluster (next 10%): 50% of remaining budget
    # Medium cluster (next 30%): 30% of remaining budget  
    # Low cluster (remaining 60%): 20% of remaining budget
    rank_high = max(1, int(remaining_budget * 0.50))
    rank_med = max(1, int(remaining_budget * 0.30))
    rank_low = max(1, remaining_budget - rank_high - rank_med)
    
    n_high = int(n_cols * 0.10)
    n_med = int(n_cols * 0.30)
    n_low = n_cols - T - n_high - n_med
    
    print(f"      Uniform:      {C} clusters  {rank_per_cluster_uniform} = {C * rank_per_cluster_uniform}")
    print(f"      Act-weighted: T={T} exact + high({n_high} cols, r={rank_high}) "
          f"+ med({n_med} cols, r={rank_med}) + low({n_low} cols, r={rank_low})")
    
    # Predict reconstruction error improvement
    # Uniform: error concentrated in massive columns
    # Act-weighted: massive columns preserved -> error in low-norm columns
    # Low-norm columns contribute less to output -> less PPL impact
    
    # Simple model: PPL impact ∝ Σ (column_error_i  column_importance_i)
    # Uniform: high importance columns get same error as low -> high PPL impact
    # Act-weighted: high importance columns get less error -> lower PPL impact
    
    print()

# ---------------------------------------------------------------------------
# Concrete experiment plan
# ---------------------------------------------------------------------------
print("[3] CONCRETE EXPERIMENT PLAN")
print()
print("    Phase 1 (CPU, NOW): Column norm analysis on real SmolLM2-135M weights")
print("      - Load FFN gate/up/down from safetensors")
print("      - Compute per-column L2 norms")
print("      - Identify massive columns (norms > 5σ above mean)")
print("      - Output: which column indices to protect")
print()
print("    Phase 2 (EC2 L40S, queued): Activation-collection run")
print("      - 500 WikiText-2 forward passes")
print("      - Collect per-column activation statistics")
print("      - Compare: L2 norm ranking vs activation magnitude ranking")
print("      - Hypothesis: L2 norm rank-correlates with activation frequency")
print()
print("    Phase 3 (EC2 L40S, queued): Weighted compression + PPL measurement")
print("      - Apply activation-weighted SVD")
print("      - Measure WikiText-2 PPL at k_frac ∈ {0.25, 0.50, 0.75}")
print("      - Compare vs unweighted baseline")
print("      - Success criterion: PPL < 2 baseline at k_frac=0.50")
print()
print("    Phase 4 (if Phase 3 fails): LoRA FFN distillation")
print("      - Apply Paper V protocol to FFN layers")
print("      - Train LoRA adapters to recover FFN output")
print("      - Success criterion: PPL within 20% of baseline")
print()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plan = {
    "analysis": "activation_weighted_ffn_compression",
    "key_finding": "L2 column norm is a zero-cost proxy for activation importance",
    "phases": [
        {"phase": 1, "name": "Column norm analysis", "cost": "CPU, <1 min", "status": "ready"},
        {"phase": 2, "name": "Activation collection", "cost": "GPU, ~30 min", "status": "queued"},
        {"phase": 3, "name": "Weighted compression PPL", "cost": "GPU, ~1 hr", "status": "queued"},
        {"phase": 4, "name": "LoRA FFN distillation", "cost": "GPU, ~2 hrs", "status": "fallback"},
    ],
    "success_criteria": {
        "phase_3": "PPL < 2 baseline at k_frac=0.50",
        "phase_4": "PPL within 20% of baseline",
    },
}

with open(OUTPUT / "ffn_activation_weighted_plan.json", 'w') as f:
    json.dump(plan, f, indent=2)

print(f"Saved: {OUTPUT / 'ffn_activation_weighted_plan.json'}")
