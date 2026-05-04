#!/usr/bin/env python3
"""
CECI Cross-Model Splicing: Qwen2.5-1.5B-Instruct × DeepSeek-R1-Distill-Qwen-1.5B

Two real pre-trained instruct models, identical architecture (Qwen2, d=1536, 28 layers).
Tests whether CECI can produce a functional chimeric model from PROPER pre-trained models.

Usage:
  python scripts/ceci_qwen_deepseek.py --k 768 --test
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "ceci_qwen_deepseek"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"  # General chat
MODEL_B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Math reasoning

# 
# CECI protocol (mirrored from ceci_cross_model.py)
# 

def build_shared_basis(Wq, Wk, Wv, n_iter=3):
    """Top-d eigvecs of normalized joint Gram, sorted descending."""
    # Handle GQA: K,V may have fewer heads than Q
    d_q = Wq.shape[1]
    K = Wq.T @ Wq
    if Wk.shape[1] == d_q:
        K += Wk.T @ Wk
    if Wv.shape[1] == d_q:
        K += Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order], eigvals[order]


def geodesic_distance(Pa, Pb, k):
    """Subspace angle between two k-dim subspaces."""
    Pak, Pbk = Pa[:, :k], Pb[:, :k]
    # Principal angles via SVD of Pa^T Pb
    M = Pak.T @ Pbk
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    S = np.clip(S, 0, 1)
    angles = np.arccos(S)
    return float(np.sqrt(np.sum(angles ** 2)))


def subspace_correlation(Pa, Pb, k):
    """Average cosine similarity between basis vectors."""
    Pak, Pbk = Pa[:, :k], Pb[:, :k]
    M = Pak.T @ Pbk
    corrs = np.abs(np.diag(M))
    return float(np.mean(corrs))


def splice_attention(model_body, model_attn, viable_layers, k):
    """Replace attention Q/K/V weights in model_body with projected weights from model_attn."""
    layers_body = model_body.model.layers
    layers_attn = model_attn.model.layers
    
    for layer_idx in range(len(layers_body)):
        if not viable_layers.get(layer_idx, False):
            continue
        
        ml_body = layers_body[layer_idx]
        ml_attn = layers_attn[layer_idx]
        
        device = ml_body.self_attn.q_proj.weight.device
        dtype = ml_body.self_attn.q_proj.weight.dtype
        
        Wq_a = ml_attn.self_attn.q_proj.weight.data.float().cpu().numpy()
        Wk_a = ml_attn.self_attn.k_proj.weight.data.float().cpu().numpy()
        Wv_a = ml_attn.self_attn.v_proj.weight.data.float().cpu().numpy()
        
        Wq_b = ml_body.self_attn.q_proj.weight.data.float().cpu().numpy()
        Wk_b = ml_body.self_attn.k_proj.weight.data.float().cpu().numpy()
        Wv_b = ml_body.self_attn.v_proj.weight.data.float().cpu().numpy()
        
        # Build basis from attn model, project INTO body model's space
        Pa, _ = build_shared_basis(Wq_a, Wk_a, Wv_a)
        Pk_a = Pa[:, :k]
        
        # Project Qwen-Instruct attention onto DeepSeek's attention basis
        # Then replace in body model
        Wq_new = (Wq_b @ Pk_a @ Pk_a.T).astype(np.float32)
        Wk_new = (Wk_b @ Pk_a @ Pk_a.T).astype(np.float32)
        Wv_new = (Wv_b @ Pk_a @ Pk_a.T).astype(np.float32)
        
        ml_body.self_attn.q_proj.weight.data = torch.from_numpy(Wq_new).to(dtype=dtype, device=device)
        ml_body.self_attn.k_proj.weight.data = torch.from_numpy(Wk_new).to(dtype=dtype, device=device)
        ml_body.self_attn.v_proj.weight.data = torch.from_numpy(Wv_new).to(dtype=dtype, device=device)


def compute_viability(model_a, model_b, k):
    """Compute CECI viability: GD, rho, and per-layer verdict."""
    layers_a = model_a.model.layers
    layers_b = model_b.model.layers
    n_layers = len(layers_a)
    
    results = {"n_layers": n_layers, "k": k, "layers": {}}
    viable_count = 0
    
    for layer_idx in range(n_layers):
        ml_a = layers_a[layer_idx]
        ml_b = layers_b[layer_idx]
        
        Wq_a = ml_a.self_attn.q_proj.weight.data.float().cpu().numpy()
        Wk_a = ml_a.self_attn.k_proj.weight.data.float().cpu().numpy()
        Wv_a = ml_a.self_attn.v_proj.weight.data.float().cpu().numpy()
        
        Wq_b = ml_b.self_attn.q_proj.weight.data.float().cpu().numpy()
        Wk_b = ml_b.self_attn.k_proj.weight.data.float().cpu().numpy()
        Wv_b = ml_b.self_attn.v_proj.weight.data.float().cpu().numpy()
        
        Pa, _ = build_shared_basis(Wq_a, Wk_a, Wv_a)
        Pb, _ = build_shared_basis(Wq_b, Wk_b, Wv_b)
        
        k_eff = min(k, Pa.shape[1])
        gd = geodesic_distance(Pa, Pb, k_eff)
        rho = subspace_correlation(Pa, Pb, k_eff)
        
        # Viability criteria: GD < 0.90 AND rho > 0.30
        viable = bool(gd < 0.90 and rho > 0.30)
        if viable:
            viable_count += 1
        
        results["layers"][str(layer_idx)] = {
            "gd": round(gd, 6),
            "rho": round(rho, 6),
            "viable": viable,
        }
        
        if layer_idx % 5 == 0:
            print(f"    Layer {layer_idx}: GD={gd:.4f}, ρ={rho:.4f}, viable={viable}")
    
    results["viable_count"] = viable_count
    results["viable_pct"] = round(viable_count / n_layers * 100, 1)
    
    print(f"\n  Viability: {viable_count}/{n_layers} ({results['viable_pct']}%)")
    return results


# 
# Functional testing
# 

TEST_PROMPTS = [
    # Knowledge (tests FFN memory)
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the chemical symbol for water?", "H2O"),
    # Math (tests attention routing + reasoning)
    ("What is 12 * 7?", "84"),
    ("If a train travels 60 miles in 2 hours, what is its speed in mph?", "30"),
    ("What is the square root of 144?", "12"),
    # Mixed
    ("Summarize the theory of relativity in one sentence.", None),
    ("Write a Python function to compute factorial.", "def factorial"),
]


def test_model(model, tokenizer, name):
    """Test a model on a battery of prompts."""
    device = next(model.parameters()).device
    results = []
    
    for prompt, expected in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        response_clean = response.strip()[:200]
        
        # Check for gibberish: high ratio of non-alphanumeric chars
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in response_clean) / max(len(response_clean), 1)
        is_gibberish = alpha_ratio < 0.5
        
        # Check expected content
        match = expected and expected.lower() in response_clean.lower() if expected else None
        
        results.append({
            "prompt": prompt[:80],
            "response": response_clean[:200],
            "expected_match": match,
            "gibberish": is_gibberish,
        })
        
        status = " GIBBERISH" if is_gibberish else (" MATCH" if match else " OK")
        print(f"  [{status}] {prompt[:60]}...")
        print(f"    -> {response_clean[:120]}")
    
    return results


# 
# Main
# 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=768, help="CECI rank")
    parser.add_argument("--test", action="store_true", help="Run functional test after splicing")
    parser.add_argument("--viability-only", action="store_true", help="Only compute viability, no splice")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CECI: Qwen2.5-1.5B-Instruct × DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"  k={args.k}")
    print("=" * 60)
    
    # Load models
    print("\n[1] Loading Model A (Qwen Instruct)...")
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A, dtype=torch.float16, device_map="auto")
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
    print(f"    Loaded: {sum(p.numel() for p in model_a.parameters())/1e6:.0f}M params")
    
    print("\n[2] Loading Model B (DeepSeek-R1)...")
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B, dtype=torch.float16, device_map="auto")
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
    print(f"    Loaded: {sum(p.numel() for p in model_b.parameters())/1e6:.0f}M params")
    
    # CECI viability analysis
    print(f"\n[3] CECI Viability Analysis (k={args.k})...")
    viability = compute_viability(model_a, model_b, args.k)
    
    if args.viability_only:
        out_file = OUTPUT / f"viability_k{args.k}.json"
        with open(out_file, "w") as f:
            json.dump(viability, f, indent=2)
        print(f"\nSaved: {out_file}")
        return
    
    # Test baselines
    if args.test:
        print("\n[4] BASELINE: Testing Model A (Qwen Instruct)...")
        results_a = test_model(model_a, tokenizer_a, "Qwen-Instruct")
        
        print("\n[5] BASELINE: Testing Model B (DeepSeek-R1)...")
        results_b = test_model(model_b, tokenizer_b, "DeepSeek-R1")
    
    # Build viable layers dict
    viable_layers = {}
    for layer_str, info in viability["layers"].items():
        viable_layers[int(layer_str)] = info["viable"]
    
    if viability["viable_count"] == 0:
        print("\n NO VIABLE LAYERS --- CECI not possible at this rank")
        return
    
    # Splice: DeepSeek attention -> Qwen body
    print(f"\n[6] Splicing: DeepSeek attention -> Qwen body ({viability['viable_count']} layers)...")
    splice_attention(model_a, model_b, viable_layers, args.k)
    
    # Test chimeric model
    if args.test:
        print("\n[7] CHIMERIC MODEL: Testing (DeepSeek-attn + Qwen-FFN)...")
        results_chimera = test_model(model_a, tokenizer_a, "CHIMERA")
    
    # Save results
    all_results = {
        "config": {"model_a": MODEL_A, "model_b": MODEL_B, "k": args.k},
        "viability": viability,
    }
    if args.test:
        all_results["baseline_a"] = results_a
        all_results["baseline_b"] = results_b
        all_results["chimera"] = results_chimera
        
        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for name, res in [("Qwen-Instruct", results_a), ("DeepSeek-R1", results_b), ("CHIMERA", results_chimera)]:
            ok = sum(1 for r in res if r["expected_match"]) if res else 0
            gib = sum(1 for r in res if r.get("gibberish", False)) if res else 0
            print(f"  {name:<20} matches={ok}/{len(TEST_PROMPTS)}, gibberish={gib}")
    
    out_file = OUTPUT / f"results_k{args.k}.json"
    with open(out_file, "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
