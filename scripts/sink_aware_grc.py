#!/usr/bin/env python3
"""
Paper V gap closure: Sink-aware GRC measurement.
Protects top-T sink channels from compression, measures PPL and text quality.

Usage: python scripts/sink_aware_grc.py --k 512 --sink-T 32
"""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "sink_aware_grc"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_shared_basis(Wq, Wk, Wv, n_iter=3):
    K = Wq.T @ Wq
    if Wk.shape[1] == Wq.shape[1]: K += Wk.T @ Wk
    if Wv.shape[1] == Wq.shape[1]: K += Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K; A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs[:, np.argsort(eigvals)[::-1]]


def sink_indices(Wq, Wk, Wv, T):
    if T <= 0: return np.array([], dtype=np.int64)
    mag = (np.linalg.norm(Wq, axis=0)**2 + np.linalg.norm(Wk, axis=0)**2 + np.linalg.norm(Wv, axis=0)**2)
    return np.argsort(mag)[::-1][:T]


def apply_grc_sink(model, k, T):
    """GRC compress attention at rank k, protecting top-T sink columns."""
    n_layers = len(model.model.layers)
    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        device = attn.q_proj.weight.device; dtype = attn.q_proj.weight.dtype
        
        Wq = attn.q_proj.weight.data.float().cpu().numpy()
        Wk = attn.k_proj.weight.data.float().cpu().numpy()
        Wv = attn.v_proj.weight.data.float().cpu().numpy()
        
        sinks = sink_indices(Wq, Wk, Wv, T)
        P = build_shared_basis(Wq, Wk, Wv)
        Pk = P[:, :k]
        PkPkT = Pk @ Pk.T
        
        # Compress non-sink cols, preserve sink cols
        Wq_new = Wq @ PkPkT
        Wk_new = Wk @ PkPkT
        Wv_new = Wv @ PkPkT
        
        # Restore sink columns from original
        for s in sinks:
            Wq_new[:, s] = Wq[:, s]
            Wk_new[:, s] = Wk[:, s]
            Wv_new[:, s] = Wv[:, s]
        
        attn.q_proj.weight.data = torch.from_numpy(Wq_new.astype(np.float32)).to(dtype=dtype, device=device)
        attn.k_proj.weight.data = torch.from_numpy(Wk_new.astype(np.float32)).to(dtype=dtype, device=device)
        attn.v_proj.weight.data = torch.from_numpy(Wv_new.astype(np.float32)).to(dtype=dtype, device=device)
        
        if layer_idx % 10 == 0:
            print(f"    Layer {layer_idx}: {len(sinks)} sinks protected")


def apply_grc_vanilla(model, k):
    """Vanilla GRC (no sink protection)."""
    for layer_idx in range(len(model.model.layers)):
        attn = model.model.layers[layer_idx].self_attn
        device = attn.q_proj.weight.device; dtype = attn.q_proj.weight.dtype
        
        Wq = attn.q_proj.weight.data.float().cpu().numpy()
        Wk = attn.k_proj.weight.data.float().cpu().numpy()
        Wv = attn.v_proj.weight.data.float().cpu().numpy()
        
        P = build_shared_basis(Wq, Wk, Wv); Pk = P[:, :k]; PkPkT = Pk @ Pk.T
        
        attn.q_proj.weight.data = torch.from_numpy((Wq @ PkPkT).astype(np.float32)).to(dtype=dtype, device=device)
        attn.k_proj.weight.data = torch.from_numpy((Wk @ PkPkT).astype(np.float32)).to(dtype=dtype, device=device)
        attn.v_proj.weight.data = torch.from_numpy((Wv @ PkPkT).astype(np.float32)).to(dtype=dtype, device=device)


def compute_ppl(model, tokenizer, texts, max_length=256):
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for text in texts[:30]:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            if enc['input_ids'].shape[1] < 2: continue
            out = model(**enc, labels=enc['input_ids'])
            total_loss += out.loss.item() * enc['input_ids'].shape[1]
            total_tokens += enc['input_ids'].shape[1]
    return float(np.exp(total_loss / max(total_tokens, 1)))


# Quick text quality test
PROMPTS = [
    ("What is the capital of France?", "Paris"),
    ("What is 12 * 7?", "84"),
    ("Name three planets.", "Earth"),
]


def test_text(model, tokenizer):
    device = next(model.parameters()).device
    results = []
    for prompt, expected in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = expected.lower() in response.lower()
        results.append({"prompt": prompt, "response": response[:150], "match": match})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--sink-T", type=int, default=32)
    p.add_argument("--compare", action="store_true", help="Compare sink vs vanilla")
    args = p.parse_args()
    
    print("=" * 60)
    print(f"SINK-AWARE GRC: k={args.k}, T={args.sink_T}")
    print("=" * 60)
    
    # Load texts
    print("[1] Loading WikiText-2...")
    try:
        wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        ppl_texts = [t for t in wiki['text'] if len(t.strip()) > 50][:30]
    except:
        ppl_texts = ["The quick brown fox jumps over the lazy dog."] * 30
    
    results = {}
    
    # Baseline
    print("[2] Baseline...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    baseline_ppl = compute_ppl(model, tokenizer, ppl_texts)
    baseline_text = test_text(model, tokenizer)
    print(f"  PPL: {baseline_ppl:.2f}")
    for r in baseline_text:
        print(f"  [{('' if r['match'] else '')}] {r['prompt'][:40]}: {r['response'][:80]}")
    results["baseline"] = {"ppl": round(baseline_ppl, 2)}
    del model; torch.cuda.empty_cache()
    
    # Sink-aware GRC
    print(f"\n[3] Sink-aware GRC (k={args.k}, T={args.sink_T})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto", local_files_only=True)
    apply_grc_sink(model, args.k, args.sink_T)
    
    sink_ppl = compute_ppl(model, tokenizer, ppl_texts)
    sink_text = test_text(model, tokenizer)
    print(f"  PPL: {sink_ppl:.2f} ({sink_ppl/baseline_ppl:.2f}× baseline)")
    for r in sink_text:
        print(f"  [{('' if r['match'] else '')}] {r['prompt'][:40]}: {r['response'][:80]}")
    results["sink_aware"] = {"ppl": round(sink_ppl, 2), "ratio": round(sink_ppl/baseline_ppl, 2)}
    del model; torch.cuda.empty_cache()
    
    # Vanilla GRC for comparison
    if args.compare:
        print(f"\n[4] Vanilla GRC (k={args.k})...")
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto", local_files_only=True)
        apply_grc_vanilla(model, args.k)
        
        vanilla_ppl = compute_ppl(model, tokenizer, ppl_texts)
        vanilla_text = test_text(model, tokenizer)
        print(f"  PPL: {vanilla_ppl:.2f} ({vanilla_ppl/baseline_ppl:.2f}× baseline)")
        for r in vanilla_text:
            print(f"  [{('' if r['match'] else '')}] {r['prompt'][:40]}: {r['response'][:80]}")
        results["vanilla"] = {"ppl": round(vanilla_ppl, 2), "ratio": round(vanilla_ppl/baseline_ppl, 2)}
        
        # Delta
        delta = vanilla_ppl - sink_ppl
        print(f"\n  Delta (vanilla - sink): {delta:+.2f} PPL")
        print(f"  Sink-aware {'better' if delta > 0 else 'worse'} by {abs(delta/baseline_ppl*100):.1f}% of baseline")
        results["delta"] = round(delta, 2)
        results["delta_pct"] = round(delta/baseline_ppl*100, 1)
        del model; torch.cuda.empty_cache()
    
    # Save
    out_file = OUT / f"results_k{args.k}_T{args.sink_T}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
