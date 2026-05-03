#!/usr/bin/env python3
"""
REAL FFN Activation Collection (fixes P2 Phase 2 failure).
Uses PyTorch forward hooks to capture actual FFN intermediate activations.
Previous approach (weight column norms) FAILED — PPL 1230 vs 27 baseline.

This collects the real per-column activation statistics needed for
activation-weighted FFN compression.

Usage: python scripts/ffn_real_activations.py
Runtime: ~30 min on RTX 4070 (500 WikiText-2 samples, SmolLM2-135M)
"""

import json, os, time, argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

OUTPUT = Path("benchmarks/p2_ffn_actweighted")

class FFNActivationCollector:
    """Hook-based collector for FFN intermediate activations."""
    
    def __init__(self, model, n_layers, d_ffn, d_model):
        self.n_layers = n_layers
        self.d_ffn = d_ffn
        self.d_model = d_model
        self.gate_acts = [np.zeros(d_ffn) for _ in range(n_layers)]
        self.up_acts = [np.zeros(d_ffn) for _ in range(n_layers)]
        # down_proj: hook INPUT (d_ffn), not output (d_model)
        self.down_acts = [np.zeros(d_ffn) for _ in range(n_layers)]
        self.counts = np.zeros(n_layers)
        self.hooks = []
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            # Hook after gate projection (before activation)
            def make_gate_hook(idx):
                def hook(module, input, output):
                    # output shape: (batch, seq, d_ffn)
                    acts = output.detach().float().abs().mean(dim=(0,1)).numpy()
                    self.gate_acts[idx] += acts
                    self.counts[idx] += 1
                return hook
            
            # Hook after up projection
            def make_up_hook(idx):
                def hook(module, input, output):
                    acts = output.detach().float().abs().mean(dim=(0,1)).numpy()
                    self.up_acts[idx] += acts
                    self.counts[idx] += 1
                return hook
            
            # Hook down_proj INPUT (d_ffn dim), not output (d_model dim)
            def make_down_hook(idx):
                def hook(module, input, output):
                    # input[0] shape: (batch, seq, d_ffn)
                    acts = input[0].detach().float().abs().mean(dim=(0,1)).numpy()
                    self.down_acts[idx] += acts
                    self.counts[idx] += 1
                return hook
            
            h1 = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx))
            h2 = layer.mlp.up_proj.register_forward_hook(make_up_hook(layer_idx))
            h3 = layer.mlp.down_proj.register_forward_hook(make_down_hook(layer_idx))
            self.hooks.extend([h1, h2, h3])
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
    
    def get_stats(self):
        """Return averaged activation statistics."""
        result = {'gate': [], 'up': [], 'down': []}
        for i in range(self.n_layers):
            c = max(self.counts[i], 1)
            result['gate'].append((self.gate_acts[i] / c).tolist())
            result['up'].append((self.up_acts[i] / c).tolist())
            result['down'].append((self.down_acts[i] / c).tolist())
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    parser.add_argument('--n-samples', type=int, default=500)
    args = parser.parse_args()
    
    print("=" * 60)
    print("REAL FFN Activation Collection (Hook-Based)")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print("=" * 60)
    
    print("[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    n_layers = len(model.model.layers)
    d_ffn = model.config.intermediate_size
    d_model = model.config.hidden_size
    print(f"  {n_layers} layers, d_ffn={d_ffn}, d_model={d_model}")
    
    print("[2] Loading WikiText-2...")
    try:
        wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        texts = [t for t in wiki['text'] if len(t.strip()) > 50][:args.n_samples]
    except:
        texts = ["The quick brown fox jumps over the lazy dog."] * args.n_samples
    
    print(f"[3] Registering hooks and collecting activations...")
    collector = FFNActivationCollector(model, n_layers, d_ffn, d_model)
    
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            if enc.input_ids.shape[1] < 4:
                continue
            _ = model(**enc)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(texts)} samples processed")
    
    collector.remove_hooks()
    stats = collector.get_stats()
    
    # Analyze: find massive columns
    print("\n[4] Analyzing activation patterns...")
    for layer_idx in [0, 7, 15, 23, 29]:
        gate_acts = np.array(stats['gate'][layer_idx])
        mean_a = np.mean(gate_acts)
        std_a = np.std(gate_acts)
        n_massive = np.sum(gate_acts > mean_a + 5 * std_a)
        top5 = np.argsort(gate_acts)[-5:][::-1]
        print(f"  Layer {layer_idx:>2}: mean={mean_a:.4f}, massive cols={n_massive}, "
              f"top5={top5.tolist()}, top5_frac={gate_acts[top5].sum()/gate_acts.sum()*100:.1f}%")
    
    # Save
    result = {
        'method': 'real_forward_hooks',
        'model': args.model,
        'n_samples': args.n_samples,
        'n_layers': n_layers,
        'd_ffn': d_ffn,
        'activations': stats,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / 'real_activation_stats.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved: {OUTPUT / 'real_activation_stats.json'}")
    print("Next: Use these stats for weighted SVD compression.")
    print("  python scripts/p2_ffn_actweighted.py --phase 3 --use-real-acts")

if __name__ == '__main__':
    main()
