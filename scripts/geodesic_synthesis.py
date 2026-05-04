#!/usr/bin/env python3
"""
PAPER XIII PRODUCT: Geodesic Synthesis
=======================================
Token generation via Christoffel-guided manifold walk.
Estimates intrinsic dimension and curvature field from prompt context,
then samples tokens by following the geodesic flow.

Subcommands:
  generate  --- Generate text from a prompt
  analyze   --- Analyze manifold geometry of a prompt
  batch     --- Generate from a file of prompts
  sweep     --- Temperature sweep to find optimal generation params

Examples:
  python scripts/geodesic_synthesis.py generate --model HuggingFaceTB/SmolLM2-135M-Instruct --prompt "Paris is" --n-tokens 30
  python scripts/geodesic_synthesis.py analyze --model HuggingFaceTB/SmolLM2-135M-Instruct --prompt "The meaning of life"
  python scripts/geodesic_synthesis.py batch --model X --prompts-file prompts.txt --out results.json
"""

import json, sys, time, math, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

# ======================================================================
# Manifold geometry estimation
# ======================================================================

def estimate_intrinsic_dim(hidden_states: torch.Tensor, n_angles: int = 200) -> int:
    """Estimate intrinsic dimension via angle distribution entropy of trajectory differences."""
    B, T, D = hidden_states.shape
    if T < 3: return D
    
    diffs = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    n = T - 1
    max_pairs = min(n_angles, n * (n - 1) // 2)
    
    angles = []
    for _ in range(max_pairs):
        i, j = np.random.randint(0, n, 2)
        if i == j: continue
        a, b = diffs[0, i].float(), diffs[0, j].float()
        cos_sim = float((a @ b) / (torch.norm(a) * torch.norm(b) + 1e-10))
        angles.append(math.acos(max(-1.0, min(1.0, cos_sim))))
    
    if not angles: return D
    
    hist, _ = np.histogram(np.array(angles), bins=20, range=(0, math.pi))
    hist = hist / max(hist.sum(), 1)
    entropy = -sum(p * math.log(max(p, 1e-10)) for p in hist if p > 0)
    return int(D * math.exp(entropy) / math.exp(math.log(20)))


def estimate_christoffel(hidden_states: torch.Tensor) -> np.ndarray:
    """Estimate Christoffel field from trajectory curvature (second differences)."""
    B, T, D = hidden_states.shape
    if T < 3: return np.eye(D, dtype=np.float32) * 0.001
    
    # Acceleration = second difference
    accel = hidden_states[:, 2:, :] - 2 * hidden_states[:, 1:-1, :] + hidden_states[:, :-2, :]
    
    # Christoffel proxy: curvature tensor from acceleration × position
    gamma = torch.einsum('ti,tj->ij', accel[0].float(), hidden_states[0, 1:-1].float())
    gamma = gamma / (torch.norm(gamma) + 1e-10)
    
    return gamma.cpu().numpy()


def manifold_curvature_at(h: torch.Tensor, gamma: np.ndarray) -> float:
    """Compute scalar curvature at a point on the manifold."""
    h_np = h.float().cpu().numpy()
    d = h_np.shape[-1]
    g = gamma[:d, :d] if gamma.shape[0] >= d else gamma
    
    # Riemannian curvature scalar: h^T Γ h
    curv = float(np.einsum('i,ij,j->', h_np, g, h_np))
    return curv


# ======================================================================
# Geodesic sampling
# ======================================================================

def geodesic_step(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    gamma: np.ndarray,
    dim_eff: int,
    temperature: float = 0.7,
    top_k: int = 50,
) -> Tuple[int, float, Dict]:
    """
    Take one geodesic step: curvature-corrected top-k sampling.
    Returns (token_id, curvature_factor, diagnostics).
    """
    d = hidden.shape[-1]
    keff = min(dim_eff, d)
    
    # Curvature-aware temperature modulation
    curv = manifold_curvature_at(hidden[0, -1], gamma)
    # High curvature -> lower effective temperature (more deterministic)
    curvature_factor = float(np.clip(1.0 / (1.0 + abs(curv)), 0.7, 1.0))
    effective_temp = temperature * curvature_factor
    
    # Top-k sampling
    k = min(top_k, logits.shape[-1])
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(topk_vals / max(effective_temp, 0.01), dim=-1)
    
    # Sample
    chosen = torch.multinomial(probs, 1).item()
    token_id = topk_idx[chosen].item()
    
    diagnostics = {
        'curvature': round(curv, 6),
        'curvature_factor': round(curvature_factor, 4),
        'effective_temp': round(effective_temp, 4),
        'topk_entropy': round(float(-(probs * torch.log(probs + 1e-10)).sum()), 4),
    }
    
    return token_id, curvature_factor, diagnostics


# ======================================================================
# Generation
# ======================================================================

def generate_geodesic(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int = 30,
    temperature: float = 0.7,
    top_k: int = 50,
    verbose: bool = False,
) -> Dict:
    """Generate text via geodesic walk. Returns full generation report."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = inputs.input_ids.shape[1]
    
    # Estimate manifold geometry from prompt
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)
    
    # Use last layer for geometry estimation
    h_last = hidden[-1]
    dim_eff = estimate_intrinsic_dim(h_last)
    gamma = estimate_christoffel(h_last)
    
    # Generate tokens
    generated_ids = []
    cur_inputs = inputs
    step_diagnostics = []
    
    start_time = time.time()
    with torch.no_grad():
        for step in range(n_tokens):
            out = model(**cur_inputs, output_hidden_states=True)
            logits = out.logits[0, -1]
            h_cur = out.hidden_states[-1]
            
            token_id, curv_factor, diag = geodesic_step(
                logits, h_cur, gamma, dim_eff, temperature, top_k
            )
            generated_ids.append(token_id)
            step_diagnostics.append(diag)
            
            # Extend context
            token_tensor = torch.tensor([[token_id]], device=device)
            cur_inputs = {
                'input_ids': torch.cat([cur_inputs['input_ids'], token_tensor], dim=1),
                'attention_mask': torch.ones(1, cur_inputs['input_ids'].shape[1] + 1,
                                            dtype=torch.long, device=device),
            }
            
            if verbose and step % max(1, n_tokens // 10) == 0:
                partial = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"    [{step:3d}/{n_tokens}] curv={diag['curvature']:.4f} | \"{partial[:60]}\"")
    
    elapsed = time.time() - start_time
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return {
        'prompt': prompt,
        'generated': gen_text,
        'full_text': prompt + gen_text,
        'n_prompt_tokens': n_prompt,
        'n_generated': n_tokens,
        'temperature': temperature,
        'elapsed_s': round(elapsed, 2),
        'tokens_per_sec': round(n_tokens / max(elapsed, 0.01), 1),
        'intrinsic_dim': dim_eff,
        'hidden_dim': model.config.hidden_size,
        'mean_curvature': round(float(np.mean([d['curvature'] for d in step_diagnostics])), 4),
        'mean_curvature_factor': round(float(np.mean([d['curvature_factor'] for d in step_diagnostics])), 4),
        'steps': step_diagnostics,
    }


def analyze_prompt(model, tokenizer, prompt: str) -> Dict:
    """Analyze manifold geometry of a prompt without generating."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)
    
    n_layers = hidden.shape[0]
    results = {'prompt': prompt, 'n_tokens': inputs.input_ids.shape[1],
               'hidden_dim': model.config.hidden_size, 'n_layers': n_layers}
    
    # Per-layer intrinsic dimension
    dims = []
    for l in range(n_layers):
        d = estimate_intrinsic_dim(hidden[l])
        dims.append(d)
    results['intrinsic_dims'] = dims
    results['mean_intrinsic_dim'] = round(float(np.mean(dims)), 1)
    
    # Last layer Christoffel
    gamma = estimate_christoffel(hidden[-1])
    results['christoffel_norm'] = round(float(np.linalg.norm(gamma)), 4)
    results['christoffel_rank'] = int(np.linalg.matrix_rank(gamma, tol=0.01))
    
    # Trajectory smoothness (mean cosine similarity between adjacent diffs)
    h = hidden[-1][0].float()
    diffs = h[1:] - h[:-1]
    if diffs.shape[0] >= 2:
        sims = []
        for i in range(diffs.shape[0] - 1):
            cos = float((diffs[i] @ diffs[i+1]) / (torch.norm(diffs[i]) * torch.norm(diffs[i+1]) + 1e-10))
            sims.append(cos)
        results['trajectory_smoothness'] = round(float(np.mean(sims)), 4)
    
    return results


# ======================================================================
# CLI
# ======================================================================

def cmd_generate(args):
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    print(f"Generating {args.n_tokens} tokens from: \"{args.prompt[:60]}...\"")
    result = generate_geodesic(model, tok, args.prompt, args.n_tokens,
                                args.temperature, args.top_k, args.verbose)
    
    print(f"\n  Generated ({result['tokens_per_sec']} tok/s):")
    print(f"  \"{result['generated']}\"")
    print(f"\n  Full: \"{result['full_text']}\"")
    print(f"\n  Geometry:")
    print(f"    Intrinsic dim: {result['intrinsic_dim']}/{result['hidden_dim']}")
    print(f"    Mean curvature: {result['mean_curvature']}")
    print(f"    Curvature factor: {result['mean_curvature_factor']}")
    
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Report: {args.json_out}")
    
    return result


def cmd_analyze(args):
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    result = analyze_prompt(model, tok, args.prompt)
    
    print(f"\n  Prompt: \"{args.prompt[:80]}\"")
    print(f"  Tokens: {result['n_tokens']}")
    print(f"  Hidden dim: {result['hidden_dim']}")
    print(f"  Layers: {result['n_layers']}")
    print(f"  Mean intrinsic dim: {result['mean_intrinsic_dim']}")
    print(f"  Christoffel norm: {result['christoffel_norm']}")
    print(f"  Christoffel rank: {result['christoffel_rank']}")
    if 'trajectory_smoothness' in result:
        print(f"  Trajectory smoothness: {result['trajectory_smoothness']}")
    print(f"  Per-layer intrinsic dims: {result['intrinsic_dims']}")
    
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2)
    
    return result


def cmd_batch(args):
    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] \"{prompt[:60]}...\"")
        r = generate_geodesic(model, tok, prompt, args.n_tokens,
                              args.temperature, args.top_k, verbose=False)
        results.append(r)
        print(f"  -> \"{r['generated'][:80]}\"")
    
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {args.out}")
    
    return results


def cmd_sweep(args):
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    temps = np.linspace(args.t_min, args.t_max, args.t_steps)
    results = []
    
    for temp in temps:
        r = generate_geodesic(model, tok, args.prompt, args.n_tokens, temp, args.top_k)
        results.append({'temperature': round(temp, 2), 'generated': r['generated'],
                        'curvature_factor': r['mean_curvature_factor'],
                        'tokens_per_sec': r['tokens_per_sec']})
        print(f"  T={temp:.2f}: \"{r['generated'][:80]}\"")
    
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    import argparse
    p = argparse.ArgumentParser(description="Geodesic Synthesis --- Christoffel-guided generation")
    sub = p.add_subparsers(dest='cmd')
    
    gp = sub.add_parser('generate', help='Generate text from a prompt')
    gp.add_argument('--model', required=True); gp.add_argument('--prompt', required=True)
    gp.add_argument('--n-tokens', type=int, default=30); gp.add_argument('--temperature', type=float, default=0.7)
    gp.add_argument('--top-k', type=int, default=50); gp.add_argument('--verbose', action='store_true')
    gp.add_argument('--json-out', help='Save report to JSON')
    
    ap = sub.add_parser('analyze', help='Analyze manifold geometry')
    ap.add_argument('--model', required=True); ap.add_argument('--prompt', required=True)
    ap.add_argument('--json-out');
    
    bp = sub.add_parser('batch', help='Generate from a file of prompts')
    bp.add_argument('--model', required=True); bp.add_argument('--prompts-file', required=True)
    bp.add_argument('--n-tokens', type=int, default=30); bp.add_argument('--temperature', type=float, default=0.7)
    bp.add_argument('--top-k', type=int, default=50); bp.add_argument('--out')
    
    sp = sub.add_parser('sweep', help='Temperature sweep')
    sp.add_argument('--model', required=True); sp.add_argument('--prompt', required=True)
    sp.add_argument('--n-tokens', type=int, default=20); sp.add_argument('--top-k', type=int, default=50)
    sp.add_argument('--t-min', type=float, default=0.3); sp.add_argument('--t-max', type=float, default=1.5)
    sp.add_argument('--t-steps', type=int, default=7); sp.add_argument('--json-out')
    
    args = p.parse_args()
    
    if args.cmd == 'generate': cmd_generate(args)
    elif args.cmd == 'analyze': cmd_analyze(args)
    elif args.cmd == 'batch': cmd_batch(args)
    elif args.cmd == 'sweep': cmd_sweep(args)
    else: p.print_help()

if __name__ == "__main__":
    main()
