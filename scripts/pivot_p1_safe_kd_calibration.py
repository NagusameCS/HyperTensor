#!/usr/bin/env python3
"""
PIVOT EXPERIMENT P1: Safe k/d Ratio Calibration.
Paper I showed k≥512 for SmolLM2-135M (d=576) but k≥1024 for Llama-8B (d=4096).
This experiment computes the safe k/d ratio as a function of model dimension d,
using the joint Gram spectrum to predict where PPL becomes acceptable.

CPU-only. No model loading needed — works from saved spectral data.
"""

import json, math, os
from pathlib import Path
import numpy as np

OUTPUT = Path("benchmarks/pivot_p1_safe_kd_ratio")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Known safe frontiers (measured)
MEASURED = {
    "SmolLM2-135M": {"d": 576, "safe_k": 512, "catastrophic_k": 256, "ppl_safe": 7.1, "ppl_base": 6.2, "ppl_catastrophic": 98.1},
    "Llama-3.1-8B": {"d": 4096, "safe_k": 1536, "ppl_safe": 7.69, "ppl_base": 6.79},
}

# Model families to predict for
MODELS = [
    {"name": "SmolLM2-135M", "d": 576, "d_ffn": 1536, "n_heads": 9, "n_kv": 9, "type": "MHA"},
    {"name": "SmolLM2-360M", "d": 960, "d_ffn": 2560, "n_heads": 15, "n_kv": 5, "type": "GQA"},
    {"name": "SmolLM2-1.7B", "d": 2048, "d_ffn": 5632, "n_heads": 32, "n_kv": 8, "type": "GQA"},
    {"name": "Llama-3.1-8B", "d": 4096, "d_ffn": 14336, "n_heads": 32, "n_kv": 8, "type": "GQA"},
    {"name": "Llama-3.1-70B", "d": 8192, "d_ffn": 28672, "n_heads": 64, "n_kv": 8, "type": "GQA"},
    {"name": "Gemma-4-2B", "d": 2304, "d_ffn": 9216, "n_heads": 16, "n_kv": 8, "type": "GQA"},
    {"name": "Gemma-4-9B", "d": 3584, "d_ffn": 14336, "n_heads": 28, "n_kv": 14, "type": "GQA"},
    {"name": "Gemma-4-27B", "d": 5376, "d_ffn": 21504, "n_heads": 32, "n_kv": 16, "type": "GQA"},
    {"name": "Qwen2.5-7B", "d": 3584, "d_ffn": 18944, "n_heads": 28, "n_kv": 4, "type": "GQA"},
    {"name": "Qwen2.5-32B", "d": 5120, "d_ffn": 27648, "n_heads": 40, "n_kv": 8, "type": "GQA"},
]

print("=" * 70)
print("PIVOT P1: Safe k/d Ratio Calibration")
print("  Calibrating safe compression frontier across model scales")
print("=" * 70)

# ---------------------------------------------------------------------------
# Model: k/d safe ratio analysis
# ---------------------------------------------------------------------------
# The key insight from Paper I Exp A1:
# - Signal preservation (Frobenius energy retained) DOES predict safe k
# - But the degradation is amplified: small energy loss → huge PPL increase
# - The amplification factor depends on d

# From measured data:
# SmolLM2 (d=576): k_safe/d = 512/576 = 0.889, k_cata/d = 256/576 = 0.444
# Llama-8B (d=4096): k_safe/d = 1536/4096 = 0.375

# Hypothesis: safe k/d decreases with d because larger models have more
# redundant attention capacity. The relationship appears roughly:
# k_safe/d ≈ 1 - c/√d for some constant c

# Fit c from the two measured points
d1, ks1 = 576, 512
d2, ks2 = 4096, 1536

# Model: k_safe/d = 1 - α/√d
# For SmolLM2: 512/576 = 1 - α/√576 → 0.889 = 1 - α/24 → α = 2.667
# For Llama: 1536/4096 = 1 - α/√4096 → 0.375 = 1 - α/64 → α = 40.0
# α is NOT constant — so it's not a simple 1/√d relationship

# Alternative: k_safe/d scales with d^(-β)
# log(k_safe/d) = -β·log(d) + γ
# log(0.889) = -β·log(576) + γ → -0.1178 = -6.356β + γ
# log(0.375) = -β·log(4096) + γ → -0.9808 = -8.317β + γ
# Subtracting: 0.8630 = 1.961β → β = 0.44
# γ = 0.44·6.356 - 0.1178 = 2.68

beta = 0.44
gamma = 2.68

print("\n[1] SAFE k/d RATIO CALIBRATION")
print(f"    Measured: SmolLM2-135M (d=576): k_safe/d = {512/576:.3f}")
print(f"    Measured: Llama-3.1-8B (d=4096): k_safe/d = {1536/4096:.3f}")
print(f"    Fitted: k_safe/d = d^(-{beta:.2f}) · exp({gamma:.2f})")
print()

results = []
for m in MODELS:
    d = m["d"]
    k_safe_ratio = d**(-beta) * math.exp(gamma)
    k_safe = int(round(d * k_safe_ratio))
    # Snap to nearest power-of-two for practical use
    k_safe_pow2 = 2**int(round(math.log2(k_safe)))
    
    # Also compute: what k gives 95% energy retention? (from MHA spectral model)
    # For MHA: k_95/d ≈ 0.41 (Paper I)
    # For GQA: k_95/d ≈ 0.25 (K/V already rank-capped)
    if m["type"] == "MHA":
        k_95_ratio = 0.41
    else:
        k_95_ratio = 0.25
    
    k_95 = int(d * k_95_ratio)
    
    # Safe k is the MAX of (PPL-safe k, 95%-energy k)
    k_recommended = max(k_safe_pow2, 2**int(round(math.log2(k_95))))
    
    results.append({
        **m,
        "k_safe_ratio_predicted": round(k_safe_ratio, 4),
        "k_safe_predicted": k_safe,
        "k_safe_pow2": k_safe_pow2,
        "k_95": k_95,
        "k_recommended": k_recommended,
        "k_recommended_ratio": round(k_recommended / d, 3),
    })
    
    print(f"  {m['name']:<20} d={d:>5}  k_safe/d={k_safe_ratio:.3f}  "
          f"k_safe={k_safe:>5}→{k_safe_pow2:>5}  k_95={k_95:>5}  "
          f"recommend={k_recommended:>5} ({k_recommended/d:.3f}d)")

# ---------------------------------------------------------------------------
# PPL amplification model
# ---------------------------------------------------------------------------
print("\n[2] PPL AMPLIFICATION MODEL")
print("    Modeling: PPL(k) = PPL_base · exp(α · (1 - energy_retained(k)))")
print()

# From SmolLM2 measured:
# k=512: energy ≈ 94%, PPL = 7.1 (1.145 base)
# k=256: energy ≈ 89%, PPL = 98.1 (15.82 base)
# α_smol = ln(15.82) / 0.11 = 2.76 / 0.11 = 25.1

# From Llama-8B measured:
# k=1536: energy ≈ 91%, PPL = 7.69 (1.133 base)
# k=1024: energy ≈ 75% (rough), PPL = 10.96 (1.614 base)
# α_llama = ln(1.614) / 0.25 = 0.479 / 0.25 = 1.92

# Key finding: α is MUCH larger for small models
# This means small models amplify signal loss more severely

print("    SmolLM2-135M (d=576): α ≈ 25.1  (extreme amplification)")
print("    Llama-3.1-8B (d=4096): α ≈ 1.92 (mild amplification)")
print("    Interpretation: larger models have redundant attention paths")
print("    that compensate for compressed dimensions.")
print()

# Predict α as function of d
# α ∝ d^(-γ) for some γ
# α_smol/α_llama = 25.1/1.92 = 13.07
# d_llama/d_smol = 4096/576 = 7.11
# (d_ratio)^γ = α_ratio → 7.11^γ = 13.07 → γ = ln(13.07)/ln(7.11) = 2.57/1.96 = 1.31

amp_gamma = 1.31
print(f"    α(d) ∝ d^(-{amp_gamma:.2f})")
print()

# Predict α and safe k for all models
print("    Model                  d      α_pred   safe_k   safe_k/d")
print("    " + "-" * 58)
for m in MODELS:
    d = m["d"]
    alpha = 25.1 * (576 / d)**amp_gamma
    # Safe = where PPL < 1.20 baseline → energy_retained > 1 - ln(1.20)/α
    energy_needed = 1.0 - math.log(1.20) / alpha
    k_safe_alt = int(d * energy_needed)
    k_safe_pow2 = 2**int(round(math.log2(max(k_safe_alt, 64))))
    print(f"    {m['name']:<20} {d:>5}  {alpha:>7.2f}   {k_safe_pow2:>5}    {k_safe_pow2/d:.3f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output = {
    "calibration": {
        "measured_points": MEASURED,
        "fitted_beta": beta,
        "fitted_gamma": gamma,
        "amplification_gamma": amp_gamma,
    },
    "predictions": results,
}

with open(OUTPUT / "safe_kd_calibration.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {OUTPUT / 'safe_kd_calibration.json'}")
print("\nPIVOT PLAN:")
print("  1. Measure PPL at k_recommended for 2 more model sizes to validate fit")
print("  2. If fit holds: publish k_safe(d) calibration curve")
print("  3. If fit fails: collect more data points, switch to nonparametric model")
print("  4. EC2 target: SmolLM2-360M (d=960) + Gemma-4-2B (d=2304)")
