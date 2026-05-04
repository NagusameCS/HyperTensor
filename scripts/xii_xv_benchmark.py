#!/usr/bin/env python3
"""
XII-XV INTEGRATION BENCHMARK
Tests full pipeline: Compile -> Synthesize -> Snipe -> Organic Gen

Usage:
  python scripts/xii_xv_benchmark.py --model HuggingFaceTB/SmolLM2-135M-Instruct --k 64 --snip-frac 0.01 --prompt "The capital of France is" --n-tokens 10 --iter 40
"""

import sys, json, time, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import geodesic_compile, geodesic_synthesis, geodesic_sniping, organic_generation

def main():
    import argparse
    p = argparse.ArgumentParser(description="XII-XV Integration Benchmark")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--snip-frac", type=float, default=0.01)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n-tokens", type=int, default=10)
    p.add_argument("--iter", type=int, default=40)
    args = p.parse_args()
    
    out_dir = ROOT / "benchmarks" / "xii_xv_integration"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"model": args.model, "k": args.k, "snip_frac": args.snip_frac, "prompt": args.prompt}
    
    print(f"Loading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    n_layers = len(model.model.layers); d = model.config.hidden_size
    
    # ==== XII: Geodesic Compiler ====
    print("\n" + "=" * 50)
    print("PAPER XII: Geodesic Compiler")
    print("=" * 50)
    
    SLOTS = {'Q': ('self_attn', 'q_proj'), 'K': ('self_attn', 'k_proj'), 'V': ('self_attn', 'v_proj')}
    total_orig = 0; total_comp = 0
    
    for li in range(n_layers):
        ml = model.model.layers[li]
        matrices = []
        for sn, (mn, attr) in SLOTS.items():
            W = getattr(getattr(ml, mn), attr).weight.data.float().cpu().numpy()
            matrices.append((sn, W))
        P = geodesic_compile.build_shared_basis(matrices)
        if P is None: continue
        ke = min(args.k, P.shape[1])
        for sn, W in matrices:
            _, Wp, Pk = geodesic_compile.compress(W, P, ke)
            total_orig += W.size * 2; total_comp += Wp.size * 2 + Pk.size * 2
    
    ratio = total_comp / max(total_orig, 1)
    results['xii_compression'] = round(1/ratio, 1)
    print(f"  {total_orig/1e6:.1f}MB -> {total_comp/1e6:.1f}MB ({results['xii_compression']}x smaller)")
    
    # ==== XIII: Geodesic Synthesis ====
    print("\n" + "=" * 50)
    print("PAPER XIII: Geodesic Synthesis")
    print("=" * 50)
    
    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)
    h = hidden[-1]
    
    dim_eff = geodesic_synthesis.estimate_intrinsic_dim(h)
    gamma_est = geodesic_synthesis.christoffel_field(h)
    results['xiii_intrinsic_dim'] = dim_eff
    print(f"  Intrinsic dimension: {dim_eff}/{d}")
    
    generated = []; cur_inputs = inputs; curvatures = []
    start = time.time()
    with torch.no_grad():
        for _ in range(args.n_tokens):
            out = model(**cur_inputs, output_hidden_states=True)
            tid, curv = geodesic_synthesis.geodesic_sample(out.logits[0,-1], out.hidden_states[-1], gamma_est, 0.7, dim_eff)
            generated.append(tid); curvatures.append(curv)
            cur_inputs = {
                'input_ids': torch.cat([cur_inputs['input_ids'], torch.tensor([[tid]], device=device)], dim=1),
                'attention_mask': torch.ones(1, cur_inputs['input_ids'].shape[1], dtype=torch.long, device=device)
            }
    
    gen_text = tok.decode(generated, skip_special_tokens=True)
    results['xiii_generated'] = gen_text
    results['xiii_tok_per_sec'] = round(args.n_tokens / max(time.time()-start, 0.01), 1)
    print(f"  Generated: \"{gen_text}\"")
    print(f"  Speed: {results['xiii_tok_per_sec']} tok/s")
    
    # ==== XIV: Geodesic Sniping ====
    print("\n" + "=" * 50)
    print("PAPER XIV: Geodesic Sniping")
    print("=" * 50)
    
    all_scores = {}
    for sn, (mn, attr) in geodesic_sniping.SLOT_ATTR.items():
        scores = []
        for li in range(n_layers):
            W = getattr(getattr(model.model.layers[li], mn), attr).weight.data.float().cpu()
            scores.append(torch.norm(W, dim=1))
        all_scores[sn] = torch.stack(scores)
    
    flat = []
    for sn in all_scores:
        for li in range(all_scores[sn].shape[0]):
            for ci in range(all_scores[sn].shape[1]):
                flat.append((float(all_scores[sn][li, ci]), sn, li, ci))
    flat.sort(key=lambda x: x[0])
    n_snip = max(1, int(len(flat) * args.snip_frac))
    total_cols = sum(s.shape[0]*s.shape[1] for s in all_scores.values())
    
    results['xiv_total_columns'] = total_cols
    results['xiv_n_snipped'] = n_snip
    results['xiv_snip_slots'] = {sn: sum(1 for _, s, _, _ in flat[:n_snip] if s==sn) for sn in geodesic_sniping.SLOT_ATTR}
    print(f"  Total: {total_cols} columns, snipped: {n_snip}")
    print(f"  By slot: {results['xiv_snip_slots']}")
    
    # ==== XV: Organic Generation ====
    print("\n" + "=" * 50)
    print("PAPER XV: Organic Generation")
    print("=" * 50)
    
    with torch.no_grad():
        gen_full, deltas = organic_generation.organic_relax(
            model, args.prompt, args.n_tokens, args.iter, 0.05, 0.8, tok, device
        )
    
    results['xv_init_delta'] = round(float(deltas[0]), 4)
    results['xv_final_delta'] = round(float(deltas[-1]), 4)
    results['xv_converged'] = bool(deltas[-1] < deltas[0] * 0.1)
    results['xv_generated'] = gen_full[:100]
    print(f"  Delta: {deltas[0]:.4f} -> {deltas[-1]:.4f} (converged: {results['xv_converged']})")
    print(f"  Generated: \"{gen_full[:80]}\"")
    
    # ==== SUMMARY ====
    print("\n" + "=" * 50)
    print("INTEGRATION SUMMARY")
    print("=" * 50)
    print(f"  XII:  {results['xii_compression']}x compression")
    print(f"  XIII: {results['xiii_tok_per_sec']} tok/s, dim={dim_eff}")
    print(f"  XIV:  {n_snip}/{total_cols} snipped")
    print(f"  XV:   {'CONVERGED' if results['xv_converged'] else 'NO CONVERGENCE (need AR)'}")
    
    out_file = out_dir / f"integration_k{args.k}.json"
    with open(out_file, "w") as f: json.dump(results, f, indent=2)
    print(f"\n  Report: {out_file}")
    return results

if __name__ == "__main__":
    main()
