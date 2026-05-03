#!/usr/bin/env python3
"""
CECI CROSS-MODEL SPLICE: Qwen2.5-1.5B Math + Language.
k=768 (k/d=0.50, REAL compression), d=1536, 28 layers, GQA 16:4.
Upload to EC2 and run: python3 ceci_qwen.py
"""

import json, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MATH_MODEL = "outputs/pure_models/qwen2.5-1.5b-math-pure/final"
LANG_MODEL = "outputs/pure_models/qwen2.5-1.5b-language-pure/final"
BASE = "Qwen/Qwen2.5-1.5B"
K = 768  # k/d = 0.50 — super-baseline compression!
SINK_T = 32
OUT = Path("benchmarks/ceci_qwen2.5_1.5b_k768")
OUT.mkdir(parents=True, exist_ok=True)

def build_basis(Wq, k):
    U, S, Vt = np.linalg.svd(Wq, full_matrices=False)
    return Vt[:min(k, len(S)), :].T

def grassmann(P1, P2):
    _, S, _ = np.linalg.svd(P1.T @ P2, full_matrices=False)
    S = np.clip(S, 0, 1)
    return float(np.sqrt(1 - np.min(S)**2))

def overlap(P1, P2):
    _, S, _ = np.linalg.svd(P1.T @ P2, full_matrices=False)
    return float(np.mean(S))

def main():
    print("=" * 60)
    print("CECI: Qwen2.5-1.5B Cross-Model")
    print(f"  k={K} (k/d={K/1536:.2f}), d=1536, GQA 16:4, 28 layers")
    print("=" * 60)
    
    print("[1] Loading models...")
    base_m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    math_path = os.path.abspath(MATH_MODEL)
    math_model = PeftModel.from_pretrained(base_m, math_path)
    math_model = math_model.merge_and_unload()
    
    base_l = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    lang_path = os.path.abspath(LANG_MODEL)
    lang_model = PeftModel.from_pretrained(base_l, lang_path)
    lang_model = lang_model.merge_and_unload()
    
    n_layers = len(math_model.model.layers)
    print(f"  {n_layers} layers loaded")
    
    print(f"\n[2] CECI splice (k={K})...")
    layer_results = {}
    all_gd, all_overlap, all_rho, all_qerr = [], [], [], []
    
    for layer_idx in range(n_layers):
        ml = math_model.model.layers[layer_idx]
        ll = lang_model.model.layers[layer_idx]
        
        Wq_m = ml.self_attn.q_proj.weight.data.float().numpy()
        Wq_l = ll.self_attn.q_proj.weight.data.float().numpy()
        
        Pm = build_basis(Wq_m, K)
        Pl = build_basis(Wq_l, K)
        k_eff = Pm.shape[1]
        
        gd = grassmann(Pm, Pl)
        ov = overlap(Pm, Pl)
        
        # Splice residual
        I_proj = Pm @ Pm.T
        delta = Wq_l - Wq_m
        delta_proj = delta @ I_proj
        residual = Wq_l - (Wq_m + delta_proj)
        q_err = float(np.linalg.norm(residual, 'fro') / max(np.linalg.norm(Wq_l, 'fro'), 1e-10))
        
        # LoRA recoverability (r=8)
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        r = 8
        rho = float(np.sum(S[:r]**2) / max(np.sum(S**2), 1e-10)) if len(S) >= r else 1.0
        
        viable = gd < 0.90 and rho > 0.20
        
        layer_results[str(layer_idx)] = {
            "layer": layer_idx, "gd": round(gd, 4), "overlap": round(ov, 4),
            "q_err": round(q_err, 4), "rho": round(rho, 4), "viable": viable,
        }
        all_gd.append(gd); all_overlap.append(ov); all_rho.append(rho); all_qerr.append(q_err)
        
        if layer_idx % 4 == 0:
            print(f"  L{layer_idx:>2}: GD={gd:.4f} ov={ov:.2%} ρ={rho:.4f} {'OK' if viable else '--'}")

    # Aggregate
    gd_mean, gd_std = float(np.mean(all_gd)), float(np.std(all_gd))
    ov_mean = float(np.mean(all_overlap))
    rho_mean = float(np.mean(all_rho))
    n_viable = sum(1 for v in layer_results.values() if v['viable'])
    
    print(f"\n[3] Qwen2.5-1.5B CECI RESULTS (k={K}, k/d={K/1536:.2f}):")
    print(f"  GD: μ={gd_mean:.4f} σ={gd_std:.4f}")
    print(f"  Overlap: μ={ov_mean:.2%}")
    print(f"  ρ (LoRA r=8): μ={rho_mean:.4f}")
    print(f"  Q rel error: μ={float(np.mean(all_qerr)):.4f}")
    print(f"  VIABLE: {n_viable}/{n_layers} ({n_viable/n_layers*100:.1f}%)")
    
    if gd_mean < 0.05 and ov_mean > 0.99:
        print(f"\n  SHARED SCAFFOLD CONFIRMED at Qwen2.5-1.5B scale!")
        print(f"  GD={gd_mean:.4f} — cross-model geometry is essentially perfect.")
        print(f"  SmolLM2-135M (d=576): GD=0.014, 13/30 viable at k=512")
        print(f"  Qwen2.5-1.5B (d=1536): GD={gd_mean:.4f}, {n_viable}/{n_layers} viable at k={K}")
    
    summary = {
        "config": {"k": K, "d": 1536, "k_d_ratio": round(K/1536, 3),
                   "n_layers": n_layers, "model": BASE, "gqa": "16:4"},
        "aggregate": {
            "gd_mean": gd_mean, "gd_std": gd_std,
            "overlap_mean": ov_mean, "rho_mean": rho_mean,
            "q_err_mean": float(np.mean(all_qerr)),
            "n_viable": n_viable, "n_total": n_layers,
            "viable_pct": round(n_viable/n_layers*100, 1),
        },
        "layers": layer_results,
    }
    
    with open(OUT / "ceci_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT / 'ceci_results.json'}")
    
    # Comparison with SmolLM2
    print(f"\n[4] CROSS-SCALE COMPARISON:")
    print(f"  {'Metric':<25} {'SmolLM2-135M':<20} {'Qwen2.5-1.5B':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'Model dim d':<25} {'576':<20} {'1536':<20}")
    print(f"  {'CECI k':<25} {'512':<20} {str(K):<20}")
    print(f"  {'k/d ratio':<25} {'0.89':<20} {str(round(K/1536, 2)):<20}")
    print(f"  {'GD (cross-model)':<25} {'0.014':<20} {f'{gd_mean:.4f}':<20}")
    print(f"  {'Viable layers':<25} {'13/30 (43%)':<20} {f'{n_viable}/{n_layers} ({n_viable/n_layers*100:.0f}%)':<20}")

if __name__ == "__main__":
    main()
