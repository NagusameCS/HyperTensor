#!/usr/bin/env python3
"""
P2 PHASE 4: FFN LoRA Distillation --- Paper VII gap closure.
Applies Paper V's LoRA distillation protocol to FFN layers.
Trains rank-8 LoRA adapters to recover FFN output after cluster compression.

Usage: python scripts/p2_phase4_ffn_distill.py --k-frac 0.50 --steps 500
GPU: RTX 4070 (8GB) sufficient for SmolLM2-135M
Runtime: ~30 min
"""

import json, os, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

OUTPUT = Path("benchmarks/p2_ffn_distill")
OUTPUT.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"

def _get_layers(model):
    """Get transformer layers handling PeftModel wrapping."""
    # Try to find layers through common paths
    for candidate in [model, getattr(model, 'base_model', None), 
                      getattr(getattr(model, 'base_model', None), 'model', None)]:
        if candidate is None:
            continue
        for attr in ['model.layers', 'model.decoder.layers', 'transformer.h']:
            parts = attr.split('.')
            obj = candidate
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
    raise RuntimeError("Cannot find transformer layers in model")

def apply_cluster_compression(model, k_frac=0.50, C=4, T_sink=32):
    """Apply activation-weighted FFN cluster compression to all layers."""
    layers = _get_layers(model)
    n_layers = len(layers)
    d_ffn = model.config.intermediate_size
    
    for layer_idx in range(n_layers):
        ml = layers[layer_idx]
        device = next(ml.parameters()).device
        dtype = next(ml.parameters()).dtype
        
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            W = getattr(ml.mlp, name).weight.data.float().cpu().numpy()
            n_rows, n_cols = W.shape
            
            # Column importance via L2 norm
            col_norms = np.linalg.norm(W, axis=0)
            sorted_idx = np.argsort(col_norms)[::-1]
            
            # Sink protection
            sink_idx = sorted_idx[:T_sink]
            remaining_idx = sorted_idx[T_sink:]
            
            # Group columns by importance tier
            n_rem = len(remaining_idx)
            n_high = int(n_rem * 0.15)
            n_med = int(n_rem * 0.35)
            
            k_total = int(n_cols * k_frac)
            k_comp = k_total - T_sink
            k_hi = max(1, int(k_comp * 0.50))
            k_md = max(1, int(k_comp * 0.30))
            k_lo = max(1, k_comp - k_hi - k_md)
            
            W_new = np.zeros_like(W)
            W_new[:, sink_idx] = W[:, sink_idx]
            
            for group_name, cols, k in [("high", remaining_idx[:n_high], k_hi),
                                        ("med", remaining_idx[n_high:n_high+n_med], k_md),
                                        ("low", remaining_idx[n_high+n_med:], k_lo)]:
                if len(cols) == 0:
                    continue
                Wc = W[:, cols]
                # Use randomized SVD for speed: scipy is faster than numpy for large matrices
                try:
                    from scipy.linalg import svd
                    U, S, Vt = svd(Wc, full_matrices=False, lapack_driver='gesdd')
                except ImportError:
                    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
                ke = min(k, len(S))
                W_new[:, cols] = (U[:, :ke] @ np.diag(S[:ke])) @ Vt[:ke, :]
            
            getattr(ml.mlp, name).weight.data = torch.from_numpy(W_new).to(dtype=dtype, device=device)
        
        if layer_idx % 5 == 0 or layer_idx == n_layers - 1:
            print(f"    Layer {layer_idx}/{n_layers} compressed")
    
    return model

def compute_ppl(model, tokenizer, texts, max_length=256):
    """Compute perplexity."""
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for text in texts[:30]:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            if enc['input_ids'].shape[1] < 2:
                continue
            out = model(**enc, labels=enc['input_ids'])
            total_loss += out.loss.item() * enc['input_ids'].shape[1]
            total_tokens += enc['input_ids'].shape[1]
    return float(np.exp(total_loss / max(total_tokens, 1)))

def train_lora_distill(model, tokenizer, texts, k_frac=0.50, steps=500, lr=5e-4, r=8):
    """Train LoRA adapters on FFN layers to recover compressed output."""
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    
    losses = []
    for step in range(steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc['input_ids'].shape[1] < 4:
            continue
        
        out = model(**enc, labels=enc['input_ids'])
        loss = out.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(float(loss))
        if step % 100 == 0:
            print(f"    Step {step}/{steps}: loss={np.mean(losses[-20:]):.4f}")
    
    return float(np.mean(losses[-50:])) if losses else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-frac', type=float, default=0.50)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lora-r', type=int, default=8)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"P2 PHASE 4: FFN LoRA Distillation")
    print(f"  k_frac={args.k_frac}, distill_steps={args.steps}, LoRA r={args.lora_r}")
    print("=" * 60)
    
    # Load base model
    print("[1] Loading SmolLM2-135M...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load calibration texts
    print("[2] Loading WikiText-2...")
    try:
        wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        train_texts = [t for t in wiki['text'] if len(t.strip()) > 50][:200]
        ppl_texts = [t for t in wiki['text'] if len(t.strip()) > 50][200:230]
    except:
        train_texts = ["The quick brown fox jumps over the lazy dog."] * 50
        ppl_texts = train_texts[:20]
    
    # Baseline PPL
    print("[3] Baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, ppl_texts)
    print(f"  Baseline: {baseline_ppl:.2f}")
    
    # Apply LoRA to FFN layers
    print("[4] Adding LoRA adapters to FFN...")
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Compress FFN
    print(f"[5] Compressing FFN (k_frac={args.k_frac})...")
    model = apply_cluster_compression(model, k_frac=args.k_frac)
    
    # Compressed PPL (before distillation)
    print("[6] Compressed PPL (no distillation)...")
    model.eval()
    compressed_ppl = compute_ppl(model, tokenizer, ppl_texts)
    print(f"  Compressed: {compressed_ppl:.2f} ({compressed_ppl/baseline_ppl:.2f}× baseline)")
    
    # Train LoRA distillation
    print(f"[7] LoRA distillation ({args.steps} steps)...")
    final_loss = train_lora_distill(model, tokenizer, train_texts, 
                                     k_frac=args.k_frac, steps=args.steps, 
                                     lr=5e-4, r=args.lora_r)
    
    # Distilled PPL
    print("[8] Distilled PPL...")
    model.eval()
    distilled_ppl = compute_ppl(model, tokenizer, ppl_texts)
    
    recovery = (compressed_ppl - distilled_ppl) / max(compressed_ppl - baseline_ppl, 1e-10) * 100
    print(f"  Distilled: {distilled_ppl:.2f}")
    print(f"  Recovery: {recovery:.1f}% of PPL gap")
    print(f"  Final ratio: {distilled_ppl/baseline_ppl:.2f}× baseline")
    
    result = {
        "k_frac": args.k_frac,
        "distill_steps": args.steps,
        "lora_r": args.lora_r,
        "baseline_ppl": round(baseline_ppl, 2),
        "compressed_ppl": round(compressed_ppl, 2),
        "distilled_ppl": round(distilled_ppl, 2),
        "recovery_pct": round(recovery, 1),
        "final_ratio": round(distilled_ppl / baseline_ppl, 2),
    }
    
    out_file = OUTPUT / f"distill_results_k{int(args.k_frac*100)}_r{args.lora_r}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved: {out_file}")

if __name__ == '__main__':
    main()
