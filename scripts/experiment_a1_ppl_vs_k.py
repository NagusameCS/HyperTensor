#!/usr/bin/env python3
"""
EXPERIMENT A1: End-to-End PPL vs GRC Compression Rank.
Proves Paper A's core claim: GRC at k≥256 preserves ≥95% of attention signal.

Measures WikiText-2 perplexity of SmolLM2-135M at compression ranks:
  full (576), k=1536, k=1024, k=512, k=256, k=128, k=64, k=32

Uses HuggingFace evaluate + GRC projection applied via weight modification.
"""

import json, os, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
OUTPUT = Path("benchmarks/experiment_a1_ppl_vs_k")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# GRC Projection
# ===========================================================================

def grc_project_layer(layer, k):
    """Project attention Q,K,V weights to rank-k via GRC (joint SVD)."""
    Wq = layer.self_attn.q_proj.weight.data  # (d, d)
    Wk = layer.self_attn.k_proj.weight.data  # (d, d_kv)
    Wv = layer.self_attn.v_proj.weight.data  # (d, d_kv)
    
    device = Wq.device
    dtype = Wq.dtype
    
    # Handle GQA: expand K,V if needed
    d = Wq.shape[1]
    d_kv = Wk.shape[1]
    
    if d_kv < d:
        num_heads = layer.self_attn.num_heads
        num_kv_heads = layer.self_attn.num_key_value_heads
        n_rep = num_heads // num_kv_heads
        head_dim = d // num_heads
        
        Wk_exp = torch.zeros(d, d, device=device, dtype=dtype)
        Wv_exp = torch.zeros(d, d, device=device, dtype=dtype)
        for h in range(num_heads):
            kv_idx = h // n_rep
            Wk_exp[:, h*head_dim:(h+1)*head_dim] = Wk[:, kv_idx*head_dim:(kv_idx+1)*head_dim]
            Wv_exp[:, h*head_dim:(h+1)*head_dim] = Wv[:, kv_idx*head_dim:(kv_idx+1)*head_dim]
        Wk = Wk_exp
        Wv = Wv_exp
    
    # Concatenate and SVD
    M = torch.cat([Wq.float(), Wk.float(), Wv.float()], dim=0)  # (3d, d)
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    
    # Project to rank k
    k_eff = min(k, S.shape[0])
    P = Vt[:k_eff, :].T  # (d, k_eff)
    
    # Apply projection
    Wq_proj = (Wq.float() @ P @ P.T).to(dtype)
    Wk_proj_orig = (Wk.float() @ P @ P.T).to(dtype)
    Wv_proj_orig = (Wv.float() @ P @ P.T).to(dtype)
    
    # Restore original shapes for K,V if GQA
    if d_kv < d:
        Wk_proj = torch.zeros_like(layer.self_attn.k_proj.weight.data)
        Wv_proj = torch.zeros_like(layer.self_attn.v_proj.weight.data)
        for h in range(num_heads):
            kv_idx = h // n_rep
            Wk_proj[:, kv_idx*head_dim:(kv_idx+1)*head_dim] = Wk_proj_orig[:, h*head_dim:(h+1)*head_dim]
            Wv_proj[:, kv_idx*head_dim:(kv_idx+1)*head_dim] = Wv_proj_orig[:, h*head_dim:(h+1)*head_dim]
    else:
        Wk_proj = Wk_proj_orig
        Wv_proj = Wv_proj_orig
    
    # Update weights in-place
    layer.self_attn.q_proj.weight.data.copy_(Wq_proj)
    layer.self_attn.k_proj.weight.data.copy_(Wk_proj)
    layer.self_attn.v_proj.weight.data.copy_(Wv_proj)
    
    # Compute signal preservation
    signal_preserved = float(torch.sum(S[:k_eff]**2) / torch.sum(S**2))
    
    return signal_preserved


def apply_grc_to_model(model, k):
    """Apply GRC projection to all attention layers."""
    n_layers = model.config.num_hidden_layers
    signals = []
    for i in range(n_layers):
        sig = grc_project_layer(model.model.layers[i], k)
        signals.append(sig)
    return np.mean(signals)


# ===========================================================================
# PPL Measurement
# ===========================================================================

def measure_ppl(model, tokenizer, max_samples=200):
    """Measure perplexity on WikiText-2 test set."""
    model.eval()
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t and len(t) > 100]
    
    total_loss = 0.0
    total_tokens = 0
    
    device = next(model.parameters()).device
    
    for i, text in enumerate(texts[:max_samples]):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = tokens["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        if loss is not None and not torch.isnan(loss):
            n_tokens = input_ids.shape[1]
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        
        if (i + 1) % 50 == 0:
            ppl_sofar = np.exp(total_loss / max(total_tokens, 1))
            print(f"  ... {i+1}/{min(max_samples, len(texts))}, PPL={ppl_sofar:.2f}")
    
    ppl = np.exp(total_loss / max(total_tokens, 1))
    return ppl


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT A1: End-to-End PPL vs GRC Compression Rank")
    print("Model: SmolLM2-135M, Data: WikiText-2")
    print("=" * 70)
    
    # K values to test
    k_values = [576, 1536, 1024, 512, 256, 128, 64, 32]
    # 576 = full dimension (no compression), 1536 > d (full)
    
    results = {
        'model': BASE_MODEL,
        'dataset': 'wikitext-2-raw-v1',
        'k_values': k_values,
        'measurements': {},
    }
    
    # Load model once, clone for each k
    print("\n[1] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    base_model.eval()
    
    # Measure uncompressed baseline
    print("\n[2] Baseline PPL (uncompressed, k=576)...")
    baseline_ppl = measure_ppl(base_model, tokenizer)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    results['baseline_ppl'] = round(baseline_ppl, 2)
    
    # Measure each compression level
    for k in k_values:
        if k >= 576:  # >= full dim = no compression
            results['measurements'][str(k)] = {
                'ppl': baseline_ppl,
                'signal_preserved': 1.0,
                'ppl_increase_pct': 0.0,
            }
            continue
        
        print(f"\n[3] k={k} — loading fresh model and compressing...")
        
        # Fresh model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        model.eval()
        
        # Apply GRC
        t0 = time.perf_counter()
        signal = apply_grc_to_model(model, k)
        compress_time = time.perf_counter() - t0
        
        # Measure PPL
        ppl = measure_ppl(model, tokenizer)
        ppl_increase = 100 * (ppl - baseline_ppl) / baseline_ppl
        
        results['measurements'][str(k)] = {
            'ppl': round(float(ppl), 2),
            'signal_preserved': round(float(signal), 4),
            'ppl_increase_pct': round(float(ppl_increase), 2),
            'compress_time_s': round(compress_time, 2),
        }
        
        print(f"  k={k}: PPL={ppl:.2f} (+{ppl_increase:.1f}%), signal={signal:.2%}, time={compress_time:.1f}s")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Save results
    out_path = OUTPUT / "ppl_vs_k_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print(f"\n{'='*70}")
    print("RESULTS: PPL vs Compression Rank")
    print(f"{'='*70}")
    print(f"{'k':>6}  {'PPL':>8}  {'ΔPPL%':>8}  {'Signal':>8}  {'Verdict'}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
    
    for k in sorted([int(x) for x in results['measurements'].keys()]):
        m = results['measurements'][str(k)]
        ppl = m['ppl']
        delta = m['ppl_increase_pct']
        sig = m['signal_preserved']
        
        if delta <= 5:
            verdict = " SAFE (<5%)"
        elif delta <= 10:
            verdict = " MARGINAL (5-10%)"
        else:
            verdict = " DEGRADED (>10%)"
        
        print(f"{k:>6}  {ppl:>8.2f}  {delta:>+7.1f}%  {sig:>7.1%}  {verdict}")
    
    print(f"\nSaved: {out_path}")
    
    # Check paper A prediction
    k256 = results['measurements'].get('256', {})
    if k256.get('ppl_increase_pct', 999) <= 5:
        print("\n PAPER A VERIFIED: k=256 preserves ≥95% signal (≤5% PPL increase)")
    else:
        print(f"\n PAPER A FALSIFIED: k=256 has {k256.get('ppl_increase_pct', 'N/A')}% PPL increase")
    
    return results


if __name__ == '__main__':
    main()
