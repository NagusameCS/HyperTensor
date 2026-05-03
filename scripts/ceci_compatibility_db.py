#!/usr/bin/env python3
"""
CECI Compatibility Database — systematically test model pairs for splice viability.
Builds a structured database of compatible pairs for Paper X.

Tests: SmolLM2 base+instruct, Qwen base+instruct, cross-family pairs.
"""

import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "ceci_compatibility"
OUT.mkdir(parents=True, exist_ok=True)


def build_shared_basis(Wq, Wk, Wv):
    """Top-d eigvecs of normalized joint Gram."""
    K = Wq.T @ Wq
    if Wk.shape[1] == Wq.shape[1]: K += Wk.T @ Wk
    if Wv.shape[1] == Wq.shape[1]: K += Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(3):
        A = A @ K; A = A / np.linalg.norm(A, "fro")
    e, v = np.linalg.eigh(A)
    return v[:, np.argsort(e)[::-1]]


def ceci_pair_score(model_id_a, model_id_b, k_values=[256, 512, 768]):
    """Score CECI compatibility between two models."""
    print(f"\n  {model_id_a.split('/')[-1][:25]} X {model_id_b.split('/')[-1][:25]}")
    
    # Check config compatibility
    cfg_a = AutoConfig.from_pretrained(model_id_a)
    cfg_b = AutoConfig.from_pretrained(model_id_b)
    
    if cfg_a.hidden_size != cfg_b.hidden_size:
        return {"compatible": False, "reason": f"d mismatch ({cfg_a.hidden_size} vs {cfg_b.hidden_size})"}
    if cfg_a.num_hidden_layers != cfg_b.num_hidden_layers:
        return {"compatible": False, "reason": f"layer count mismatch"}
    if cfg_a.model_type != cfg_b.model_type:
        return {"compatible": False, "reason": f"architecture mismatch ({cfg_a.model_type} vs {cfg_b.model_type})"}
    
    d = cfg_a.hidden_size
    n_layers = cfg_a.num_hidden_layers
    
    # Load models
    print(f"    Loading A...")
    ma = AutoModelForCausalLM.from_pretrained(model_id_a, dtype=torch.float16, device_map="auto")
    print(f"    Loading B...")
    mb = AutoModelForCausalLM.from_pretrained(model_id_b, dtype=torch.float16, device_map="auto")
    
    results = {"compatible": True, "d": d, "n_layers": n_layers, "model_type": cfg_a.model_type,
               "model_a": model_id_a, "model_b": model_id_b, "k_results": {}}
    
    for k in k_values:
        k_eff = min(k, d)
        total_gd, total_rho, viable = 0.0, 0.0, 0
        layer_data = {}
        
        for layer_idx in range(n_layers):
            att_a = ma.model.layers[layer_idx].self_attn
            att_b = mb.model.layers[layer_idx].self_attn
            
            Wq_a = att_a.q_proj.weight.data.float().cpu().numpy()
            Wq_b = att_b.q_proj.weight.data.float().cpu().numpy()
            Wk_a = att_a.k_proj.weight.data.float().cpu().numpy()
            Wk_b = att_b.k_proj.weight.data.float().cpu().numpy()
            Wv_a = att_a.v_proj.weight.data.float().cpu().numpy()
            Wv_b = att_b.v_proj.weight.data.float().cpu().numpy()
            
            Pa = build_shared_basis(Wq_a, Wk_a, Wv_a)
            Pb = build_shared_basis(Wq_b, Wk_b, Wv_b)
            
            M = Pa[:, :k_eff].T @ Pb[:, :k_eff]
            _, S, _ = np.linalg.svd(M, full_matrices=False)
            S = np.clip(S, 0, 1)
            gd = float(np.sqrt(np.sum(np.arccos(S)**2)))
            rho = float(np.mean(np.abs(np.diag(M))))
            
            total_gd += gd; total_rho += rho
            ok = gd < 0.90 and rho > 0.30
            if ok: viable += 1
            
            layer_data[str(layer_idx)] = {"gd": round(gd, 4), "rho": round(rho, 4), "viable": ok}
        
        results["k_results"][str(k)] = {
            "k_eff": k_eff,
            "mean_gd": round(total_gd / n_layers, 4),
            "mean_rho": round(total_rho / n_layers, 4),
            "viable_layers": viable,
            "viable_pct": round(viable / n_layers * 100, 1),
            "layer_data": layer_data,
        }
        print(f"    k={k}: GD={total_gd/n_layers:.4f}, rho={total_rho/n_layers:.4f}, viable={viable}/{n_layers}")
    
    del ma, mb; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    # Candidate pairs (all available without auth)
    pairs = [
        # Same architecture, different training stage (EXPECTED: HIGH compatibility)
        ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct", "same_arch_diff_stage"),
        ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", "same_arch_diff_stage"),
        
        # Same family, different sized (NOT compatible — different d/layers)
        # ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "diff_size_same_family"),
        
        # Within-model (same model vs itself) as control
        ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "self_control"),
    ]
    
    print("=" * 70)
    print("CECI COMPATIBILITY DATABASE")
    print("=" * 70)
    
    database = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "viability_criteria": "GD < 0.90 AND rho > 0.30",
        "pairs": []
    }
    
    for model_a, model_b, category in pairs:
        try:
            result = ceci_pair_score(model_a, model_b, k_values=[256, 512, 768])
            result["category"] = category
            database["pairs"].append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            database["pairs"].append({
                "model_a": model_a, "model_b": model_b,
                "compatible": False, "reason": str(e), "category": category,
            })
        gc.collect(); torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("CECI COMPATIBILITY SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<50s} {'k=256':>10s} {'k=512':>10s} {'k=768':>10s}")
    print("-" * 80)
    
    for pair in database["pairs"]:
        if not pair.get("compatible"):
            name = f"{pair['model_a'].split('/')[-1][:20]} x {pair['model_b'].split('/')[-1][:20]}"
            print(f"{name:<50s} {'INCOMPATIBLE':>10s}: {pair.get('reason','')}")
            continue
        
        name = f"{pair['model_a'].split('/')[-1][:20]} x {pair['model_b'].split('/')[-1][:20]}"
        row = name
        for k in ["256","512","768"]:
            kr = pair["k_results"].get(k, {})
            vp = kr.get("viable_pct", 0)
            row += f" {vp:>6.1f}%  "
        print(row)
    
    # Save
    out_file = OUT / "compatibility_db.json"
    with open(out_file, "w") as f:
        json.dump(database, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
