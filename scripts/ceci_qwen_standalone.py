#!/usr/bin/env python3
import os, json, time, numpy as np
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
MATH = os.path.abspath("outputs/pure_models/qwen2.5-1.5b-math-pure/final")
LANG = os.path.abspath("outputs/pure_models/qwen2.5-1.5b-language-pure/final")
K = 768

print("Loading math...")
bm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
math_model = PeftModel.from_pretrained(bm, MATH)
math_model = math_model.merge_and_unload()

print("Loading lang...")
bl = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
lang_model = PeftModel.from_pretrained(bl, LANG)
lang_model = lang_model.merge_and_unload()

n = len(math_model.model.layers)
print(f"{n} layers. Running CECI k={K}...")

def grassmann(P1, P2):
    _, S, _ = np.linalg.svd(P1.T @ P2)
    return float(np.sqrt(1 - min(np.clip(S, 0, 1))**2))

def overlap(P1, P2):
    _, S, _ = np.linalg.svd(P1.T @ P2)
    return float(np.mean(S))

viable = 0
all_gd, all_rho = [], []
for i in range(n):
    Wm = math_model.model.layers[i].self_attn.q_proj.weight.float().cpu().numpy()
    Wl = lang_model.model.layers[i].self_attn.q_proj.weight.float().cpu().numpy()
    Um, Sm, Vtm = np.linalg.svd(Wm, full_matrices=False)
    Ul, Sl, Vtl = np.linalg.svd(Wl, full_matrices=False)
    ke = min(K, len(Sm))
    Pm, Pl = Vtm[:ke, :].T, Vtl[:ke, :].T
    g = grassmann(Pm, Pl)
    o = overlap(Pm, Pl)
    # Splice residual
    Iproj = Pm @ Pm.T
    residual = Wl - (Wm + (Wl - Wm) @ Iproj)
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    r = min(8, len(S))
    rho = float(np.sum(S[:r]**2) / max(np.sum(S**2), 1e-10))
    v = g < 0.90 and rho > 0.20
    if v: viable += 1
    all_gd.append(g); all_rho.append(rho)
    if i % 4 == 0:
        print(f"  L{i:>2}: GD={g:.4f} ov={o:.3f} rho={rho:.4f} {'OK' if v else '--'}")

gd_mean = float(np.mean(all_gd))
print(f"\nQwen2.5-1.5B CECI (k={K}, k/d={K/1536:.2f}):")
print(f"  GD: mean={gd_mean:.4f}")
print(f"  VIABLE: {viable}/{n} ({viable/n*100:.1f}%)")
print(f"  SmolLM2 k=512: GD=0.014, 13/30 (43.3%)")
if gd_mean < 0.05:
    print("  SHARED SCAFFOLD CONFIRMED at 3x scale!")
else:
    print(f"  GD={gd_mean:.4f} --- scaffold may not hold at this scale")
