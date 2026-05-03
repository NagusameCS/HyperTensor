#!/usr/bin/env python3
"""
PAPER XV PRODUCT: Organic Generation
=====================================
Parallel geodesic relaxation without autoregressive decoding.
Evolves hidden states via noise-scheduled Christoffel drift.
Also supports AR-guided hybrid mode to prevent collapse.

Subcommands:
  relax    — Pure geodesic relaxation (no AR)
  hybrid   — AR-guided relaxation (prevents collapse)
  compare  — Compare pure vs hybrid vs standard AR generation

Examples:
  python scripts/organic_generation.py relax --model X --seed "The cat" --n-tokens 20 --iter 100
  python scripts/organic_generation.py hybrid --model X --seed "The cat" --n-tokens 20 --iter 50 --ar-interval 5
  python scripts/organic_generation.py compare --model X --seed "Paris is"
"""

import json, sys, time, math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def token_embed_sphere(model) -> torch.Tensor:
    """Normalized token embeddings (unit sphere)."""
    embed = model.model.embed_tokens.weight.data.float()
    norms = torch.norm(embed, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    return embed / norms


def geodesic_drift(h_current: torch.Tensor, embed_normed: torch.Tensor,
                   gamma: torch.Tensor, dim_eff: int) -> torch.Tensor:
    """Compute drift toward nearest tokens on geodesic manifold."""
    d = h_current.shape[-1]
    keff = min(dim_eff, d)
    
    # Cosine similarity to token embeddings
    h_norm = h_current / (torch.norm(h_current, dim=-1, keepdim=True) + 1e-10)
    scores = h_norm @ embed_normed.T
    
    # Top-k tokens as drift targets
    topk = torch.topk(scores, k=min(20, scores.shape[-1]), dim=-1)
    
    drift = torch.zeros_like(h_current)
    for k_idx in range(topk.values.shape[-1]):
        weight = torch.softmax(topk.values[:, k_idx:k_idx+1] * 2.0, dim=0)
        drift_k = embed_normed[topk.indices[:, k_idx]] * h_norm.norm(dim=-1, keepdim=True)
        drift = drift + drift_k * weight
    
    return drift / max(topk.values.shape[-1], 1)


# ══════════════════════════════════════════════════════════════════════
# Pure geodesic relaxation (no AR)
# ══════════════════════════════════════════════════════════════════════

def pure_relaxation(
    model, tokenizer, seed_text: str, n_tokens: int,
    n_iter: int = 100, noise_scale: float = 0.05,
    temperature: float = 0.8, verbose: bool = False,
) -> Dict:
    """Pure geodesic relaxation — no autoregressive guidance."""
    device = next(model.parameters()).device
    embed_normed = token_embed_sphere(model)
    gamma = torch.eye(model.config.hidden_size, device=device) * 0.001
    
    # Encode seed
    inputs = tokenizer(seed_text, return_tensors="pt").to(device)
    n_seed = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_seed = out.hidden_states[-1][0]
    
    d = h_seed.shape[-1]
    
    # Initialize continuation as noisy interpolation
    h_cont = torch.zeros(n_tokens, d, device=device, dtype=torch.float32)
    for i in range(n_tokens):
        alpha = (i + 1) / (n_tokens + 1)
        h_cont[i] = h_seed[-1].float() * (1 - alpha) + h_seed[0].float() * alpha
    
    noise = torch.randn(n_tokens, d, device=device) * noise_scale
    h_all = torch.cat([h_seed.float(), h_cont + noise], dim=0)
    
    deltas = []
    for it in range(n_iter):
        alpha_t = 0.5 * (1 + math.cos(math.pi * it / n_iter))
        curr_noise = noise_scale * alpha_t
        
        noise_vec = torch.randn(n_tokens, d, device=device) * curr_noise
        h_work = h_all.clone()
        h_work[n_seed:] = h_work[n_seed:] + noise_vec
        
        drift = geodesic_drift(h_work[n_seed:], embed_normed, gamma, d)
        
        lr = 0.01 * (1 - alpha_t * 0.5)
        h_all[n_seed:] = h_all[n_seed:] + drift * lr
        
        delta = torch.norm(drift).item()
        deltas.append(delta)
        
        if verbose and it % max(1, n_iter // 5) == 0:
            h_norm = h_all / (torch.norm(h_all, dim=-1, keepdim=True) + 1e-10)
            logits = h_norm @ embed_normed.T / max(temperature, 0.01)
            top_ids = torch.argmax(logits, dim=-1)
            decoded = tokenizer.decode(top_ids, skip_special_tokens=True)
            print(f"    iter {it:4d}: delta={delta:.4f} | \"{decoded[:60]}\"")
    
    # Final decode
    h_norm = h_all / (torch.norm(h_all, dim=-1, keepdim=True) + 1e-10)
    logits = h_norm @ embed_normed.T / max(temperature, 0.01)
    top_ids = torch.argmax(logits, dim=-1)
    full_decoded = tokenizer.decode(top_ids, skip_special_tokens=True)
    
    deltas_arr = np.array(deltas)
    return {
        'method': 'pure_relaxation',
        'seed': seed_text,
        'generated': full_decoded,
        'n_tokens': n_tokens,
        'n_iter': n_iter,
        'initial_delta': round(float(deltas_arr[0]), 4),
        'final_delta': round(float(deltas_arr[-1]), 4),
        'convergence_ratio': round(float(deltas_arr[0] / max(deltas_arr[-1], 1e-10)), 1),
        'delta_trace': [round(float(d), 4) for d in deltas_arr[-10:]],
        'converged': bool(deltas_arr[-1] < deltas_arr[0] * 0.1),
    }


# ══════════════════════════════════════════════════════════════════════
# AR-guided hybrid relaxation
# ══════════════════════════════════════════════════════════════════════

def hybrid_relaxation(
    model, tokenizer, seed_text: str, n_tokens: int,
    n_iter: int = 50, noise_scale: float = 0.03,
    temperature: float = 0.8, ar_interval: int = 5,
    verbose: bool = False,
) -> Dict:
    """
    AR-guided geodesic relaxation: periodically injects AR token
    to prevent collapse into degenerate states.
    """
    device = next(model.parameters()).device
    embed_normed = token_embed_sphere(model)
    gamma = torch.eye(model.config.hidden_size, device=device) * 0.001
    
    inputs = tokenizer(seed_text, return_tensors="pt").to(device)
    all_ids = list(inputs.input_ids[0].tolist())
    
    deltas = []
    for t in range(n_tokens):
        # Run a few relaxation iterations for this token position
        n_seed = len(all_ids)
        embeds = model.model.embed_tokens(torch.tensor([all_ids], device=device))
        h_current = embeds[0].float()
        
        for it in range(n_iter):
            alpha_t = 0.5 * (1 + math.cos(math.pi * it / n_iter))
            curr_noise = noise_scale * alpha_t
            noise = torch.randn(1, h_current.shape[-1], device=device) * curr_noise
            h_work = h_current[-1:] + noise
            
            drift = geodesic_drift(h_work, embed_normed, gamma, model.config.hidden_size)
            h_current = torch.cat([h_current[:-1], h_current[-1:] + drift * 0.01], dim=0)
            
            # AR injection every ar_interval iterations
            if it % ar_interval == 0 and it > 0:
                h_norm = h_current[-1:] / (torch.norm(h_current[-1:], dim=-1, keepdim=True) + 1e-10)
                logits = h_norm @ embed_normed.T / max(temperature, 0.01)
                ar_id = torch.argmax(logits, dim=-1).item()
                all_ids.append(ar_id)
                embeds = model.model.embed_tokens(torch.tensor([all_ids], device=device))
                h_current = embeds[0].float()
                break
        
        delta = torch.norm(drift).item() if 'drift' in dir() else 0.0
        deltas.append(delta)
        
        if verbose and t % max(1, n_tokens // 5) == 0:
            partial = tokenizer.decode(all_ids, skip_special_tokens=True)
            print(f"    token {t:3d}: delta={delta:.4f} | \"{partial[-60:]}\"")
    
    full_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    gen_only = tokenizer.decode(all_ids[len(tokenizer.encode(seed_text)):], skip_special_tokens=True)
    
    deltas_arr = np.array(deltas)
    return {
        'method': 'hybrid',
        'seed': seed_text,
        'generated': gen_only,
        'full_text': full_text,
        'n_tokens': n_tokens,
        'n_iter': n_iter,
        'initial_delta': round(float(deltas_arr[0]), 4),
        'final_delta': round(float(deltas_arr[-1]), 4),
        'mean_delta': round(float(np.mean(deltas_arr)), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# Standard AR baseline
# ══════════════════════════════════════════════════════════════════════

def standard_ar(model, tokenizer, seed_text: str, n_tokens: int,
                temperature: float = 0.8, do_sample: bool = True) -> Dict:
    """Standard autoregressive generation as baseline."""
    device = next(model.parameters()).device
    inputs = tokenizer(seed_text, return_tensors="pt").to(device)
    n_seed = inputs.input_ids.shape[1]
    
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens, do_sample=do_sample,
            temperature=temperature, pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    
    gen_text = tokenizer.decode(out[0][n_seed:], skip_special_tokens=True)
    return {
        'method': 'standard_ar',
        'seed': seed_text,
        'generated': gen_text,
        'n_tokens': n_tokens,
        'tokens_per_sec': round(n_tokens / max(elapsed, 0.01), 1),
        'elapsed_s': round(elapsed, 2),
    }


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def cmd_relax(args):
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    print(f"Pure geodesic relaxation: \"{args.seed[:60]}\"")
    result = pure_relaxation(model, tok, args.seed, args.n_tokens,
                              args.iter, args.noise_scale, args.temperature, args.verbose)
    
    print(f"\n  Generated: \"{result['generated'][:120]}\"")
    print(f"  Delta: {result['initial_delta']:.4f} → {result['final_delta']:.4f}")
    print(f"  Converged: {result['converged']}")
    
    if args.json_out:
        with open(args.json_out, 'w') as f: json.dump(result, f, indent=2)
    
    return result


def cmd_hybrid(args):
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    print(f"AR-guided relaxation: \"{args.seed[:60]}\"")
    result = hybrid_relaxation(model, tok, args.seed, args.n_tokens,
                                args.iter, args.noise_scale, args.temperature,
                                args.ar_interval, args.verbose)
    
    print(f"\n  Generated: \"{result['generated'][:120]}\"")
    print(f"  Delta: {result['initial_delta']:.4f} → {result['final_delta']:.4f}")
    
    if args.json_out:
        with open(args.json_out, 'w') as f: json.dump(result, f, indent=2)
    
    return result


def cmd_compare(args):
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    print(f"\n{'='*60}")
    print("METHOD 1: Standard Autoregressive")
    ar = standard_ar(model, tok, args.seed, args.n_tokens)
    print(f"  \"{ar['generated'][:120]}\"")
    print(f"  {ar['tokens_per_sec']} tok/s")
    
    print(f"\n{'='*60}")
    print("METHOD 2: Pure Geodesic Relaxation")
    pure = pure_relaxation(model, tok, args.seed, args.n_tokens,
                           args.iter, args.noise_scale, args.temperature, verbose=False)
    print(f"  \"{pure['generated'][:120]}\"")
    print(f"  Converged: {pure['converged']}")
    
    print(f"\n{'='*60}")
    print("METHOD 3: AR-Guided Hybrid Relaxation")
    hybrid = hybrid_relaxation(model, tok, args.seed, args.n_tokens,
                                args.hybrid_iter, args.noise_scale, args.temperature,
                                args.ar_interval, verbose=False)
    print(f"  \"{hybrid['generated'][:120]}\"")
    
    comparison = {'seed': args.seed, 'standard_ar': ar, 'pure_relaxation': pure, 'hybrid': hybrid}
    
    if args.json_out:
        with open(args.json_out, 'w') as f: json.dump(comparison, f, indent=2)
    
    return comparison


def main():
    import argparse
    p = argparse.ArgumentParser(description="Organic Generation — geodesic relaxation")
    sub = p.add_subparsers(dest='cmd')
    
    rp = sub.add_parser('relax', help='Pure geodesic relaxation')
    rp.add_argument('--model', required=True); rp.add_argument('--seed', required=True)
    rp.add_argument('--n-tokens', type=int, default=20); rp.add_argument('--iter', type=int, default=100)
    rp.add_argument('--noise-scale', type=float, default=0.05); rp.add_argument('--temperature', type=float, default=0.8)
    rp.add_argument('--verbose', action='store_true'); rp.add_argument('--json-out')
    
    hp = sub.add_parser('hybrid', help='AR-guided hybrid relaxation')
    hp.add_argument('--model', required=True); hp.add_argument('--seed', required=True)
    hp.add_argument('--n-tokens', type=int, default=20); hp.add_argument('--iter', type=int, default=50)
    hp.add_argument('--noise-scale', type=float, default=0.03); hp.add_argument('--temperature', type=float, default=0.8)
    hp.add_argument('--ar-interval', type=int, default=5); hp.add_argument('--verbose', action='store_true')
    hp.add_argument('--json-out')
    
    cp = sub.add_parser('compare', help='Compare all three methods')
    cp.add_argument('--model', required=True); cp.add_argument('--seed', required=True)
    cp.add_argument('--n-tokens', type=int, default=20); cp.add_argument('--iter', type=int, default=80)
    cp.add_argument('--hybrid-iter', type=int, default=40); cp.add_argument('--noise-scale', type=float, default=0.05)
    cp.add_argument('--temperature', type=float, default=0.8); cp.add_argument('--ar-interval', type=int, default=5)
    cp.add_argument('--json-out')
    
    args = p.parse_args()
    
    if args.cmd == 'relax': cmd_relax(args)
    elif args.cmd == 'hybrid': cmd_hybrid(args)
    elif args.cmd == 'compare': cmd_compare(args)
    else: p.print_help()

if __name__ == "__main__":
    main()
