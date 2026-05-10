#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
SHF Full Build — Real Training Integration
============================================
May 7, 2026

Integrates SHF (Spectral Hamiltonian Flow) loss into an actual
fine-tuning loop with LoRA adapters on Qwen2.5-0.5B.

Compares:
  A. Baseline: LM loss only (standard fine-tuning)
  B. SHF:      LM loss + λ · SHF geodicity penalty

Metrics:
  - Geodicity of residual-stream trajectories (lower = more geodesic)
  - LM loss (should not degrade significantly)
  - Per-layer curvature κ_ℓ

Output: benchmarks/shf_full_build/results.json
"""

import torch, numpy as np, json, os, math, time, warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

ROOT = 'c:/Users/legom/HyperTensor'
os.chdir(ROOT)
os.makedirs('benchmarks/shf_full_build', exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# 
# SETUP
# 
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("SHF Full Build — Real Training Integration")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Device: {DEVICE}")

print("\nLoading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
print(f"  d={d_model}, layers={n_layers}")

# 
# TRAINING DATA
# 
train_texts = [
    "Water boils at 100 degrees Celsius at sea level. This is because the vapor pressure equals atmospheric pressure.",
    "DNA is a double helix structure with hydrogen bonds between complementary base pairs.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared in a right triangle.",
    "A prime number has exactly two positive integer divisors: one and itself.",
    "Shakespeare's Hamlet explores themes of mortality, madness, and revenge in medieval Denmark.",
    "The French Revolution of 1789 established principles of liberty, equality, and fraternity.",
    "A for loop iterates over elements of an array sequentially, executing the body for each element.",
    "Recursion solves problems by having functions call themselves with smaller subproblems.",
    "The derivative of x cubed is 3x squared by the power rule of differentiation.",
    "Binary search splits a sorted array in half each iteration, achieving logarithmic time complexity.",
    "A hash table provides constant time average case lookup using a hash function to map keys to buckets.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
]

# Tokenize
print("\nTokenizing training data...")
encodings = tokenizer(train_texts, return_tensors='pt', padding=True, 
                       truncation=True, max_length=64)
input_ids = encodings['input_ids'].to(DEVICE)
attention_mask = encodings['attention_mask'].to(DEVICE)
labels = input_ids.clone()

print(f"  {len(train_texts)} sequences, max length={input_ids.shape[1]}")

# 
# LoRA CONFIG
# 
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# 
# HELPER: Measure geodicity of hidden states
# 
def measure_geodicity(hidden_states):
    """||Δ²s|| averaged over layers. Lower = more geodesic."""
    # hidden_states: tuple of (layer_count) tensors, each [B, T, d]
    # Take last token, all layers
    L = len(hidden_states)
    if L < 3:
        return 0.0
    
    # Stack all layers for the last token of first batch item
    traj = torch.stack([h[0, -1, :].float().cpu() for h in hidden_states])  # [L, d]
    
    total = 0.0
    for ell in range(1, L - 1):
        d2s = traj[ell+1] - 2*traj[ell] + traj[ell-1]
        total += (d2s * d2s).sum().item()
    return total / (L - 2)

def measure_per_layer_geodicity(hidden_states):
    """Per-layer ||Δ²s_ℓ||."""
    L = len(hidden_states)
    if L < 3:
        return []
    traj = torch.stack([h[0, -1, :].float().cpu() for h in hidden_states])
    geos = []
    for ell in range(1, L - 1):
        d2s = traj[ell+1] - 2*traj[ell] + traj[ell-1]
        geos.append((d2s * d2s).sum().item())
    return geos

# 
# PRE-TRAINING MEASUREMENT
# 
print("\n" + "=" * 70)
print("Pre-training baseline measurement")
print("=" * 70)

model.eval()
with torch.no_grad():
    out_base = model(input_ids[:1], attention_mask=attention_mask[:1], 
                      output_hidden_states=True)
geo_baseline = measure_geodicity(out_base.hidden_states)
geo_per_layer_baseline = measure_per_layer_geodicity(out_base.hidden_states)
# Also measure LM loss
with torch.no_grad():
    out_lm = model(input_ids[:1], attention_mask=attention_mask[:1], labels=labels[:1])
lm_loss_baseline = out_lm.loss.item()

print(f"  Baseline LM loss:     {lm_loss_baseline:.4f}")
print(f"  Baseline geodicity:   {geo_baseline:.2f}")
print(f"  Per-layer geo range:  [{min(geo_per_layer_baseline):.1f}, {max(geo_per_layer_baseline):.1f}]")

# 
# EXPERIMENT A: BASELINE FINE-TUNING (LM loss only)
# 
print("\n" + "=" * 70)
print("EXPERIMENT A: Baseline Fine-Tuning (LM loss only)")
print("=" * 70)

model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
model_a = get_peft_model(model_a, lora_config)
model_a.train()
model_a.config.output_hidden_states = True

optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
lm_losses_a = []; geo_hist_a = []

print("  Training (200 steps)...")
t0 = time.time()
for step in range(200):
    optimizer_a.zero_grad()
    
    # Use a cycling subset for variety
    idx = step % len(input_ids)
    out = model_a(input_ids[idx:idx+1], attention_mask=attention_mask[idx:idx+1], 
                  labels=labels[idx:idx+1], output_hidden_states=True)
    loss = out.loss
    
    loss.backward()
    optimizer_a.step()
    
    lm_losses_a.append(loss.item())
    if step % 20 == 0:
        with torch.no_grad():
            geo = measure_geodicity(out.hidden_states)
        geo_hist_a.append((step, geo))
        print(f"    Step {step:3d}: LM={loss.item():.4f}, geo={geo:.1f}")

elapsed_a = time.time() - t0

# Final measurement
model_a.eval()
with torch.no_grad():
    out_a = model_a(input_ids[:1], attention_mask=attention_mask[:1],
                     output_hidden_states=True, labels=labels[:1])
geo_final_a = measure_geodicity(out_a.hidden_states)
lm_final_a = out_a.loss.item()
geo_per_layer_a = measure_per_layer_geodicity(out_a.hidden_states)

print(f"\n  Final LM loss:     {lm_final_a:.4f} (baseline: {lm_loss_baseline:.4f})")
print(f"  Final geodicity:   {geo_final_a:.2f} (baseline: {geo_baseline:.2f})")
print(f"  Δ geodicity:       {geo_final_a - geo_baseline:+.2f} ({(geo_final_a/geo_baseline - 1)*100:+.1f}%)")
print(f"  Time: {elapsed_a:.1f}s")

# Cleanup
del model_a; torch.cuda.empty_cache()

# 
# EXPERIMENT B: SHF FINE-TUNING (LM + SHF loss)
# 
print("\n" + "=" * 70)
print("EXPERIMENT B: SHF Fine-Tuning (LM + λ·SHF loss)")
print("=" * 70)

model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
model_b = get_peft_model(model_b, lora_config)
model_b.train()
model_b.config.output_hidden_states = True

optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)

# SHF hyperparameters
LAMBDA_SHF = 0.01   # SHF loss weight
KAPPA = 1.0          # curvature (κ=1.0 is reasonable per v4 findings)

def shf_penalty(hidden_states, kappa=1.0):
    """Compute SHF Jacobi residual over all layers.
    
    L_SHF = (1/(L-2)) Σ_ℓ ||Δ²s_ℓ + κ·Δs_ℓ||²
    """
    # hidden_states: tuple of tensors, each [B, T, d]
    L = len(hidden_states)
    if L < 3:
        return torch.tensor(0.0, device=hidden_states[0].device)
    
    # Take last token position for batch item 0
    traj = torch.stack([h[0, -1, :].float() for h in hidden_states])  # [L, d]
    
    shf = 0.0
    for ell in range(1, L - 1):
        s_prev = traj[ell-1]
        s_curr = traj[ell]
        s_next = traj[ell+1]
        
        ds = s_next - s_curr         # forward difference
        d2s = s_next - 2*s_curr + s_prev  # second difference
        
        residual = d2s + kappa * ds
        shf += (residual * residual).sum()
    
    return shf / (L - 2)

lm_losses_b = []; geo_hist_b = []; shf_losses_b = []

print("  Training (200 steps)...")
t0 = time.time()
for step in range(200):
    optimizer_b.zero_grad()
    
    idx = step % len(input_ids)
    out = model_b(input_ids[idx:idx+1], attention_mask=attention_mask[idx:idx+1],
                  labels=labels[idx:idx+1], output_hidden_states=True)
    
    lm_loss = out.loss
    shf_l = shf_penalty(out.hidden_states, kappa=KAPPA)
    total_loss = lm_loss + LAMBDA_SHF * shf_l
    
    total_loss.backward()
    optimizer_b.step()
    
    lm_losses_b.append(lm_loss.item())
    shf_losses_b.append(shf_l.item())
    if step % 20 == 0:
        with torch.no_grad():
            geo = measure_geodicity(out.hidden_states)
        geo_hist_b.append((step, geo))
        print(f"    Step {step:3d}: LM={lm_loss.item():.4f}, SHF={shf_l.item():.4f}, geo={geo:.1f}")

elapsed_b = time.time() - t0

# Final measurement
model_b.eval()
with torch.no_grad():
    out_b = model_b(input_ids[:1], attention_mask=attention_mask[:1],
                     output_hidden_states=True, labels=labels[:1])
geo_final_b = measure_geodicity(out_b.hidden_states)
lm_final_b = out_b.loss.item()
geo_per_layer_b = measure_per_layer_geodicity(out_b.hidden_states)

geo_delta_b = geo_final_b - geo_baseline
geo_pct_b = (geo_final_b / geo_baseline - 1) * 100

print(f"\n  Final LM loss:     {lm_final_b:.4f} (baseline: {lm_loss_baseline:.4f})")
print(f"  Final geodicity:   {geo_final_b:.2f} (baseline: {geo_baseline:.2f})")
print(f"  Δ geodicity:       {geo_delta_b:+.2f} ({geo_pct_b:+.1f}%)")
print(f"  Time: {elapsed_b:.1f}s")

# 
# COMPARISON & RESULTS
# 
print("\n" + "=" * 70)
print("RESULTS: Baseline vs SHF Fine-Tuning")
print("=" * 70)

geo_delta_a = geo_final_a - geo_baseline
geo_pct_a = (geo_final_a / geo_baseline - 1) * 100

print(f"  {'Metric':<30} {'Baseline':>10} {'LM-only':>10} {'SHF':>10}")
print(f"  {'-'*60}")
print(f"  {'LM loss':<30} {lm_loss_baseline:>10.4f} {lm_final_a:>10.4f} {lm_final_b:>10.4f}")
print(f"  {'Geodicity':<30} {geo_baseline:>10.2f} {geo_final_a:>10.2f} {geo_final_b:>10.2f}")
print(f"  {'Δ Geodicity':<30} {'—':>10} {geo_delta_a:>+10.2f} {geo_delta_b:>+10.2f}")
print(f"  {'Δ Geodicity %':<30} {'—':>10} {geo_pct_a:>+9.1f}% {geo_pct_b:>+9.1f}%")

# Judge
shf_improvement = geo_final_a - geo_final_b  # positive = SHF better (lower geodicity)
shf_lm_cost = lm_final_b - lm_final_a  # positive = SHF has higher LM loss (cost)

print(f"\n  SHF geodicity improvement: {shf_improvement:+.2f} " +
      f"({'SHF is MORE geodesic' if shf_improvement > 0 else 'SHF is LESS geodesic'})")
print(f"  SHF LM cost: {shf_lm_cost:+.4f} " +
      f"({'acceptable tradeoff' if shf_lm_cost < 0.1 else 'significant degradation'})")

if shf_improvement > 0 and shf_lm_cost < 0.1:
    verdict = " SHF WORKS — reduces geodicity at negligible LM cost"
elif shf_improvement > 0:
    verdict = " SHF WORKS — reduces geodicity but with LM tradeoff"
elif shf_improvement < 0:
    verdict = f" SHF DID NOT HELP — geodicity increased by {-shf_improvement:.2f}"
else:
    verdict = " SHF NEUTRAL — no significant difference"

print(f"\n  VERDICT: {verdict}")

# 
# Per-layer comparison
# 
print(f"\n  Per-layer geodicity (first 10 layers):")
print(f"  {'Layer':<8} {'Baseline':>10} {'LM-only':>10} {'SHF':>10} {'SHF Δ':>10}")
for ell in range(min(10, len(geo_per_layer_baseline))):
    bl = geo_per_layer_baseline[ell]
    la = geo_per_layer_a[ell] if ell < len(geo_per_layer_a) else 0
    lb = geo_per_layer_b[ell] if ell < len(geo_per_layer_b) else 0
    delta = lb - bl
    print(f"  {ell:<8} {bl:>10.1f} {la:>10.1f} {lb:>10.1f} {delta:>+10.1f}")

# 
# SAVE RESULTS
# 
results = {
    'model': MODEL_ID,
    'd_model': d_model,
    'n_layers': n_layers,
    'lora_r': 8,
    'lora_alpha': 16,
    'lambda_shf': LAMBDA_SHF,
    'kappa': KAPPA,
    'training_steps': 200,
    'learning_rate': 1e-4,
    'baseline': {
        'lm_loss': lm_loss_baseline,
        'geodicity': geo_baseline,
        'per_layer_geodicity': geo_per_layer_baseline,
    },
    'lm_only': {
        'final_lm_loss': lm_final_a,
        'final_geodicity': geo_final_a,
        'delta_geodicity': geo_delta_a,
        'delta_geodicity_pct': geo_pct_a,
        'per_layer_geodicity': geo_per_layer_a,
        'training_time_s': elapsed_a,
    },
    'shf': {
        'final_lm_loss': lm_final_b,
        'final_geodicity': geo_final_b,
        'delta_geodicity': geo_delta_b,
        'delta_geodicity_pct': geo_pct_b,
        'per_layer_geodicity': geo_per_layer_b,
        'shf_improvement_vs_lm_only': shf_improvement,
        'shf_lm_cost': shf_lm_cost,
        'training_time_s': elapsed_b,
    },
    'verdict': verdict,
}

out_path = 'benchmarks/shf_full_build/results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

# Cleanup
del model, model_b
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("SHF Full Build Complete")
print("=" * 70)
