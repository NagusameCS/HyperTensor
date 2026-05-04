#!/usr/bin/env python3
"""
FUNCTIONAL SANITY CHECKER --- Tests whether geometric/proxy metrics correlate with actual model output quality.

Runs across all paper claims that are currently "measured by proxy":
- Paper I:  GRC PPL vs actual text generation at each k
- Paper V:  LoRA-distilled PPL recovery vs actual text
- Paper VII: FFN cluster compression Frobenius vs actual text  
- Paper X:  CECI geometric viability vs functional splicing

Each test: generate text on 5 prompts, check for (a) gibberish, (b) coherence, (c) basic factuality.

Usage: python scripts/functional_sanity.py --paper all
       python scripts/functional_sanity.py --paper I --ranks 256,512,full
"""

import argparse, json, os, re, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "functional_sanity"
OUT.mkdir(parents=True, exist_ok=True)

# 
# Test prompts
# 

PROMPTS = [
    ("What is the capital of France?", "Paris", "factual"),
    ("Explain what photosynthesis is in one sentence.", None, "coherence"),
    ("What is 12 * 7?", "84", "math"),
    ("Write a haiku about a cat.", None, "creativity"),
    ("What color is the sky on a clear day?", "blue", "factual"),
    ("Name three planets in our solar system.", "Earth", "list"),
]

# 
# Scoring
# 

def score_output(text):
    """Score generated text: 0=gibberish, 1=broken, 2=coherent, 3=good."""
    if not text or len(text.strip()) < 2:
        return 0, "EMPTY"
    
    text = text.strip()
    
    # Gibberish detection: high ratio of non-alphanumeric, or repeating patterns
    alpha = sum(c.isalnum() or c.isspace() or c in ".,!?-'\"" for c in text) / max(len(text), 1)
    if alpha < 0.35:
        return 0, "GIBBERISH"
    
    # Extreme repetition
    words = text.split()
    if len(words) > 5:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.3:
            return 0, "REPETITIVE"
    
    # Word salad: too many rare characters
    weird_chars = sum(1 for c in text if ord(c) > 127 and ord(c) < 0x4E00)
    if weird_chars > len(text) * 0.1:
        return 0, "WEIRD_CHARS"
    
    # Coherence heuristics
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) == 0 and len(text) < 15:
        return 1, "TOO_SHORT"
    
    if len(sentences) == 0:
        return 1, "NO_SENTENCES"
    
    # Check for common English function words (coherence signal)
    function_words = {'the','is','a','an','in','of','to','and','that','it','for','was','on','are','be','has','have','had','not','but','or','from','with','this','by','at','they','we','can','all'}
    found = sum(1 for fw in function_words if fw in text.lower().split())
    
    if found >= 5:
        return 3, "COHERENT"
    elif found >= 3:
        return 2, "PARTIAL"
    else:
        return 1, "WEAK"


def check_expected(text, expected):
    """Check if expected content is present."""
    if expected is None:
        return None
    return expected.lower() in text.lower()


# 
# GRC compression (mirror)
# 

def build_shared_basis(Wq, Wk, Wv, n_iter=3):
    K = Wq.T @ Wq
    if Wk.shape[1] == Wq.shape[1]:
        K += Wk.T @ Wk
    if Wv.shape[1] == Wq.shape[1]:
        K += Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs[:, np.argsort(eigvals)[::-1]]


def apply_grc(model, k):
    """GRC compress attention Q/K/V at rank k."""
    for layer_idx in range(len(model.model.layers)):
        attn = model.model.layers[layer_idx].self_attn
        device = attn.q_proj.weight.device
        dtype = attn.q_proj.weight.dtype
        
        Wq = attn.q_proj.weight.data.float().cpu().numpy()
        Wk = attn.k_proj.weight.data.float().cpu().numpy()
        Wv = attn.v_proj.weight.data.float().cpu().numpy()
        
        P = build_shared_basis(Wq, Wk, Wv)
        Pk = P[:, :k]
        
        attn.q_proj.weight.data = torch.from_numpy(
            (Wq @ Pk @ Pk.T).astype(np.float32)).to(dtype=dtype, device=device)
        attn.k_proj.weight.data = torch.from_numpy(
            (Wk @ Pk @ Pk.T).astype(np.float32)).to(dtype=dtype, device=device)
        attn.v_proj.weight.data = torch.from_numpy(
            (Wv @ Pk @ Pk.T).astype(np.float32)).to(dtype=dtype, device=device)


# 
# Test runner
# 

def test_model_output(model, tokenizer, model_name, test_prompts=None):
    """Generate and score outputs for a model."""
    if test_prompts is None:
        test_prompts = PROMPTS
    device = next(model.parameters()).device
    
    results = []
    for prompt, expected, category in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            formatted = prompt
        
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        dt = time.time() - t0
        
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        score, label = score_output(response)
        match = check_expected(response, expected)
        n_tokens = out.shape[1] - enc["input_ids"].shape[1]
        
        results.append({
            "prompt": prompt[:80],
            "category": category,
            "response": response.strip()[:200],
            "score": score,
            "label": label,
            "expected_match": match,
            "n_tokens": n_tokens,
            "time_s": round(dt, 2),
        })
    
    return results


def summarize(results, name):
    """Print summary of test results."""
    scores = [r["score"] for r in results]
    matches = [r["expected_match"] for r in results if r["expected_match"] is not None]
    gibberish = sum(1 for r in results if r["score"] == 0)
    coherent = sum(1 for r in results if r["score"] >= 2)
    
    avg_score = np.mean(scores) if scores else 0
    match_rate = np.mean(matches) if matches else 0
    
    verdict = " FUNCTIONAL" if gibberish == 0 and coherent >= len(results) * 0.5 else \
              " DEGRADED" if coherent >= len(results) * 0.25 else \
              " BROKEN" if coherent == 0 else " MIXED"
    
    print(f"  {name:30s} | score={avg_score:.1f} | coherent={coherent}/{len(results)} | "
          f"gibberish={gibberish} | matches={match_rate:.0%} | {verdict}")
    
    return {"avg_score": avg_score, "coherent": coherent, "gibberish": gibberish,
            "match_rate": match_rate, "verdict": verdict, "details": results}


# 
# Main
# 

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=str, default="I",
                       help="Which paper to check: I, V, VII, X, all")
    parser.add_argument("--ranks", type=str, default="256,512,full",
                       help="Compression ranks to test")
    parser.add_argument("--ceci-model-a", type=str, help="CECI model A path")
    parser.add_argument("--ceci-model-b", type=str, help="CECI model B path")
    args = parser.parse_args()
    
    papers = ["I","V","VII","X"] if args.paper == "all" else [args.paper]
    ranks = []
    for r in args.ranks.split(","):
        r = r.strip()
        ranks.append(None if r.lower() == "full" else int(r))
    
    all_summary = {}
    
    print("=" * 70)
    print("FUNCTIONAL SANITY CHECKER --- Geometric vs Actual Output Quality")
    print("=" * 70)
    
    # 
    # Paper I: GRC compression vs text quality
    # 
    if "I" in papers:
        print("\n" + "=" * 70)
        print("PAPER I: GRC --- PPL (geometric proxy) vs Text Quality (functional)")
        print("=" * 70)
        
        for rank in ranks:
            rank_label = "full" if rank is None else str(rank)
            print(f"\n  Loading {MODEL_ID}...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if rank is not None:
                print(f"  Applying GRC k={rank}...")
                apply_grc(model, rank)
            
            print(f"  Testing at k={rank_label}...")
            results = test_model_output(model, tokenizer, f"GRC_k{rank_label}")
            
            for r in results:
                status = "" if r["score"]>=2 else ("" if r["score"]==1 else "")
                print(f"    [{status} {r['label']:12s}] {r['category']:10s}: {r['response'][:100]}")
            
            summary = summarize(results, f"Paper I, k={rank_label}")
            all_summary[f"I_k{rank_label}"] = summary
            
            del model; gc.collect(); torch.cuda.empty_cache()
    
    # 
    # Paper V: LoRA distillation --- PPL recovery vs text
    # 
    if "V" in papers:
        print("\n" + "=" * 70)
        print("PAPER V: GRC Light Distillation --- PPL 107% recovery vs Text Quality")
        print("=" * 70)
        
        for rank in ranks:
            if rank is None: continue
            label = str(rank)
            
            print(f"\n  Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Apply GRC
            print(f"  GRC k={rank}...")
            apply_grc(model, rank)
            
            # Apply LoRA distillation (train on-the-fly on WikiText-2)
            print(f"  Training LoRA r=8 on WikiText-2 (200 steps)...")
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import load_dataset
            
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj"],
                                    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
            model = get_peft_model(model, lora_config)
            
            try:
                wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                texts = [t for t in wiki['text'] if len(t.strip()) > 50][:50]
            except:
                texts = ["The quick brown fox jumps over the lazy dog."] * 50
            
            device = next(model.parameters()).device
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
            model.train()
            
            for step in range(200):
                text = texts[step % len(texts)]
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                if enc["input_ids"].shape[1] < 4:
                    continue
                out = model(**enc, labels=enc["input_ids"])
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            print(f"  Testing LoRA-distilled at k={label}...")
            model.eval()
            results = test_model_output(model, tokenizer, f"LoRA_k{label}")
            
            for r in results:
                status = "" if r["score"]>=2 else ("" if r["score"]==1 else "")
                print(f"    [{status} {r['label']:12s}] {r['category']:10s}: {r['response'][:100]}")
            
            summary = summarize(results, f"Paper V (LoRA), k={label}")
            all_summary[f"V_k{label}"] = summary
            
            del model; gc.collect(); torch.cuda.empty_cache()
    
    # 
    # Paper VII: FFN cluster compression --- Frobenius vs text
    # 
    if "VII" in papers:
        print("\n" + "=" * 70)
        print("PAPER VII: FFN Cluster Compression --- Frobenius vs Text")
        print("=" * 70)
        
        for k_frac in [0.75]:  # Conservative FFN compression
            print(f"\n  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Conservative FFN cluster compression at k_frac=0.75
            print(f"  FFN compression k_frac={k_frac}...")
            for layer_idx in range(len(model.model.layers)):
                mlp = model.model.layers[layer_idx].mlp
                device = mlp.gate_proj.weight.device
                dtype = mlp.gate_proj.weight.dtype
                
                for name in ['gate_proj','up_proj','down_proj']:
                    W = getattr(mlp, name).weight.data.float().cpu().numpy()
                    n_rows, n_cols = W.shape
                    
                    # Simple SVD compression
                    U, S, Vt = np.linalg.svd(W, full_matrices=False)
                    ke = int(n_cols * k_frac)
                    W_new = (U[:, :ke] @ np.diag(S[:ke])) @ Vt[:ke, :]
                    
                    getattr(mlp, name).weight.data = torch.from_numpy(
                        W_new.astype(np.float32)).to(dtype=dtype, device=device)
            
            print(f"  Testing FFN-compressed at k_frac={k_frac}...")
            results = test_model_output(model, tokenizer, f"FFN_{k_frac}")
            
            for r in results:
                status = "" if r["score"]>=2 else ("" if r["score"]==1 else "")
                print(f"    [{status} {r['label']:12s}] {r['category']:10s}: {r['response'][:100]}")
            
            summary = summarize(results, f"Paper VII FFN, k_frac={k_frac}")
            all_summary[f"VII_{k_frac}"] = summary
            
            del model; gc.collect(); torch.cuda.empty_cache()
    
    # 
    # Paper X: CECI --- geometric viability vs functional splicing
    # (SmolLM2 base + SmolLM2 instruct = same architecture different training stage)
    # 
    if "X" in papers:
        print("\n" + "=" * 70)
        print("PAPER X: CECI --- Geometric Viability vs Functional Splicing")
        print("  Pair: SmolLM2-135M (base) × SmolLM2-135M-Instruct")
        print("=" * 70)
        
        CECI_A = "HuggingFaceTB/SmolLM2-135M"  # Base pre-trained
        CECI_B = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Instruct fine-tuned
        
        for k in [576, 256]:  # full rank + partial
            print(f"\n  === k={k} ===")
            
            print(f"  Loading Model A (base)...")
            ma = AutoModelForCausalLM.from_pretrained(
                CECI_A, dtype=torch.float16, device_map="auto", local_files_only=True)
            ta = AutoTokenizer.from_pretrained(CECI_A, local_files_only=True)
            if ta.pad_token is None: ta.pad_token = ta.eos_token
            
            print(f"  Loading Model B (instruct)...")
            mb = AutoModelForCausalLM.from_pretrained(
                CECI_B, dtype=torch.float16, device_map="auto", local_files_only=True)
            tb = AutoTokenizer.from_pretrained(CECI_B, local_files_only=True)
            if tb.pad_token is None: tb.pad_token = tb.eos_token
            
            # Test baselines
            print(f"\n  BASELINE A (SmolLM2 base, no chat template):")
            ra = test_model_output(ma, ta, "base", PROMPTS[:3])
            for r in ra:
                status = "" if r["score"]>=2 else ("" if r["score"]==1 else "")
                print(f"    [{status}] {r['response'][:120]}")
            sa = summarize(ra, "Baseline A (base)")
            
            print(f"\n  BASELINE B (SmolLM2 instruct):")
            rb = test_model_output(mb, tb, "instruct", PROMPTS[:3])
            for r in rb:
                status = "" if r["score"]>=2 else ("" if r["score"]==1 else "")
                print(f"    [{status}] {r['response'][:120]}")
            sb = summarize(rb, "Baseline B (instruct)")
            
            # CECI viability
            print(f"\n  CECI viability (k={k}):")
            n_layers = len(ma.model.layers)
            viable = 0
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
                
                k_eff = min(k, Pa.shape[1])
                M = Pa[:,:k_eff].T @ Pb[:,:k_eff]
                _, S, _ = np.linalg.svd(M, full_matrices=False)
                S = np.clip(S, 0, 1)
                gd = float(np.sqrt(np.sum(np.arccos(S)**2)))
                rho = float(np.mean(np.abs(np.diag(M))))
                
                ok = gd < 0.90 and rho > 0.30
                if ok: viable += 1
                if layer_idx % 10 == 0:
                    print(f"    L{layer_idx}: GD={gd:.4f}, ρ={rho:.4f}, viable={ok}")
            
            print(f"  Viable: {viable}/{n_layers}")
            
            if viable == 0 and k != 576:
                print(f"   NO VIABLE LAYERS --- skipping splice")
                del ma, mb; gc.collect(); torch.cuda.empty_cache()
                continue
            
            # Splice: base attention -> instruct body
            print(f"  Splicing ({viable} layers)...")
            for layer_idx in range(n_layers):
                att_a = ma.model.layers[layer_idx].self_attn
                att_b = mb.model.layers[layer_idx].self_attn
                device = att_b.q_proj.weight.device
                dtype = att_b.q_proj.weight.dtype
                
                Wq_a = att_a.q_proj.weight.data.float().cpu().numpy()
                Wk_a = att_a.k_proj.weight.data.float().cpu().numpy()
                Wv_a = att_a.v_proj.weight.data.float().cpu().numpy()
                Wq_b = att_b.q_proj.weight.data.float().cpu().numpy()
                
                Pa = build_shared_basis(Wq_a, Wk_a, Wv_a)
                Pk = Pa[:, :min(k, Pa.shape[1])]
                
                Wq_new = (Wq_b @ Pk @ Pk.T).astype(np.float32)
                Wk_new = (att_b.k_proj.weight.data.float().cpu().numpy() @ Pk @ Pk.T).astype(np.float32)
                Wv_new = (att_b.v_proj.weight.data.float().cpu().numpy() @ Pk @ Pk.T).astype(np.float32)
                
                att_b.q_proj.weight.data = torch.from_numpy(Wq_new).to(dtype=dtype, device=device)
                att_b.k_proj.weight.data = torch.from_numpy(Wk_new).to(dtype=dtype, device=device)
                att_b.v_proj.weight.data = torch.from_numpy(Wv_new).to(dtype=dtype, device=device)
            
            print(f"\n  CHIMERIC MODEL (SmolLM2-base attn + SmolLM2-instruct FFN, k={k}):")
            rc = test_model_output(mb, tb, f"CHIMERA_k{k}", PROMPTS[:5])
            for r in rc:
                status = "" if r["score"]>=2 else ("" if r["score"]==1 else "")
                print(f"    [{status}] {r['response'][:120]}")
            sc = summarize(rc, f"CECI chimera k={k}")
            
            all_summary[f"X_baseline_A"] = sa
            all_summary[f"X_baseline_B"] = sb
            all_summary[f"X_chimera_k{k}"] = sc
            
            del ma, mb; gc.collect(); torch.cuda.empty_cache()
    
    # 
    # Final verdict
    # 
    print("\n" + "=" * 70)
    print("FINAL VERDICT --- Geometric metrics vs Functional output")
    print("=" * 70)
    print(f"{'Test':<45s} {'Score':>6s} {'Coherent':>9s} {'Gibberish':>10s} {'Verdict':>15s}")
    print("-" * 85)
    for name, s in all_summary.items():
        print(f"{name:<45s} {s['avg_score']:>5.1f}  {s['coherent']:>4d}/{len(PROMPTS):<4d}  {s['gibberish']:>5d}     {s['verdict']:>15s}")
    
    # Save
    out_file = OUT / "results.json"
    with open(out_file, "w") as f:
        # Convert for JSON
        clean = {}
        for k, v in all_summary.items():
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "details"}
        json.dump(clean, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
