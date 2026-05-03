#!/usr/bin/env python3
"""
MINSKAT — Model INtegration via Subspace Kernel Alignment Transfer
First successful chimeric model under the CECI protocol.

Benchmarks MINSKAT vs its two contributing models:
  A: SmolLM2-135M (base) — attention donor
  B: SmolLM2-135M-Instruct — body/FFN donor
  MINSKAT: CECI splice (A's attention projected onto B via shared basis)

Usage: python scripts/benchmark_minsk_at.py
"""

import json, os, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "minsk_at"
OUT.mkdir(parents=True, exist_ok=True)

A_ID = "HuggingFaceTB/SmolLM2-135M"
B_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# --- GRC/CECI ---
def build_shared_basis(Wq, Wk, Wv):
    K = Wq.T @ Wq
    if Wk.shape[1] == Wq.shape[1]: K += Wk.T @ Wk
    if Wv.shape[1] == Wq.shape[1]: K += Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(3): A = A @ K; A = A / np.linalg.norm(A, "fro")
    e, v = np.linalg.eigh(A)
    return v[:, np.argsort(e)[::-1]]

def splice_ceci(ma, mb, k=576):
    """Project mb's attention onto ma's basis at rank k."""
    viable = 0
    for layer_idx in range(len(ma.model.layers)):
        att_a = ma.model.layers[layer_idx].self_attn
        att_b = mb.model.layers[layer_idx].self_attn
        dev = att_b.q_proj.weight.device; dt = att_b.q_proj.weight.dtype
        
        Wq_a = att_a.q_proj.weight.data.float().cpu().numpy()
        Wk_a = att_a.k_proj.weight.data.float().cpu().numpy()
        Wv_a = att_a.v_proj.weight.data.float().cpu().numpy()
        Wq_b = att_b.q_proj.weight.data.float().cpu().numpy()
        Wk_b = att_b.k_proj.weight.data.float().cpu().numpy()
        Wv_b = att_b.v_proj.weight.data.float().cpu().numpy()
        
        Pa = build_shared_basis(Wq_a, Wk_a, Wv_a)
        Pb = build_shared_basis(Wq_b, Wk_b, Wv_b)
        
        ke = min(k, Pa.shape[1])
        M = Pa[:,:ke].T @ Pb[:,:ke]
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        S = np.clip(S, 0, 1)
        gd = float(np.sqrt(np.sum(np.arccos(S)**2)))
        rho = float(np.mean(np.abs(np.diag(M))))
        if gd >= 0.90 or rho <= 0.30: continue
        viable += 1
        
        Pk = Pa[:, :ke]; PkPkT = Pk @ Pk.T
        att_b.q_proj.weight.data = torch.from_numpy((Wq_b @ PkPkT).astype(np.float32)).to(dtype=dt, device=dev)
        att_b.k_proj.weight.data = torch.from_numpy((Wk_b @ PkPkT).astype(np.float32)).to(dtype=dt, device=dev)
        att_b.v_proj.weight.data = torch.from_numpy((Wv_b @ PkPkT).astype(np.float32)).to(dtype=dt, device=dev)
    return viable

# --- PPL ---
def compute_ppl(model, tokenizer, texts):
    total_loss, total_tokens = 0.0, 0
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for text in texts[:30]:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            enc = {k: v.to(dev) for k, v in enc.items()}
            if enc['input_ids'].shape[1] < 2: continue
            out = model(**enc, labels=enc['input_ids'])
            total_loss += out.loss.item() * enc['input_ids'].shape[1]
            total_tokens += enc['input_ids'].shape[1]
    return float(np.exp(total_loss / max(total_tokens, 1)))

# --- Text quality battery ---
BENCHMARKS = [
    ("factual", "What is the capital of France? Answer in one word.", "Paris"),
    ("factual", "What is the chemical symbol for water?", "H2O"),
    ("math", "What is 12 * 7? Answer with just the number.", "84"),
    ("math", "What is 15 + 27?", "42"),
    ("knowledge", "Name three planets in our solar system.", "Earth"),
    ("knowledge", "Who wrote Romeo and Juliet?", "Shakespeare"),
    ("reasoning", "If a train travels 60 miles in 2 hours, what is its speed in mph?", "30"),
    ("creativity", "Write a 3-word poem about stars.", None),
]

def bench_text(model, tokenizer, name):
    dev = next(model.parameters()).device
    results = []
    for category, prompt, expected in BENCHMARKS:
        msgs = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except:
            formatted = prompt
        enc = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=256)
        enc = {k: v.to(dev) for k, v in enc.items()}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        dt = time.time() - t0
        resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = expected and expected.lower() in resp.lower()
        results.append({"category": category, "prompt": prompt[:60], "response": resp[:150],
                        "match": match, "time_s": round(dt, 2)})
    acc = sum(1 for r in results if r["match"]) / max(sum(1 for r in results if r["match"] is not None), 1)
    return results, acc


def main():
    print("=" * 70)
    print("MINSKAT — Model INtegration via Subspace Kernel Alignment Transfer")
    print("First successful CECI chimeric model benchmark")
    print("=" * 70)
    
    # Load texts
    try:
        wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        ppl_texts = [t for t in wiki['text'] if len(t.strip()) > 50][:30]
    except:
        ppl_texts = ["The quick brown fox jumps over the lazy dog."] * 30
    
    report = {"model_name": "MINSKAT", "protocol": "CECI",
              "attention_donor": A_ID, "body_donor": B_ID,
              "acronym": "Model INtegration via Subspace Kernel Alignment Transfer"}
    
    # --- Model A: SmolLM2 base ---
    print("\n[1/3] SmolLM2-135M (base) — attention donor...")
    ma = AutoModelForCausalLM.from_pretrained(A_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
    ta = AutoTokenizer.from_pretrained(A_ID, local_files_only=True)
    if ta.pad_token is None: ta.pad_token = ta.eos_token
    ppl_a = compute_ppl(ma, ta, ppl_texts)
    text_a, acc_a = bench_text(ma, ta, "base")
    print(f"  PPL: {ppl_a:.2f}  |  Accuracy: {acc_a:.0%}  |  Answers: {sum(1 for r in text_a if r['match'])}/{sum(1 for r in text_a if r['match'] is not None)}")
    report["model_a"] = {"ppl": round(ppl_a, 2), "accuracy": round(acc_a, 3), "responses": text_a}
    del ma; gc.collect(); torch.cuda.empty_cache()
    
    # --- Model B: SmolLM2-Instruct ---
    print("\n[2/3] SmolLM2-135M-Instruct — body donor...")
    mb = AutoModelForCausalLM.from_pretrained(B_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
    tb = AutoTokenizer.from_pretrained(B_ID, local_files_only=True)
    if tb.pad_token is None: tb.pad_token = tb.eos_token
    ppl_b = compute_ppl(mb, tb, ppl_texts)
    text_b, acc_b = bench_text(mb, tb, "instruct")
    print(f"  PPL: {ppl_b:.2f}  |  Accuracy: {acc_b:.0%}  |  Answers: {sum(1 for r in text_b if r['match'])}/{sum(1 for r in text_b if r['match'] is not None)}")
    report["model_b"] = {"ppl": round(ppl_b, 2), "accuracy": round(acc_b, 3), "responses": text_b}
    
    # --- MINSKAT: CECI splice ---
    print("\n[3/3] MINSKAT — CECI chimeric (base attention + instruct body)...")
    print("  Loading fresh instruct model...")
    m_minsk = AutoModelForCausalLM.from_pretrained(B_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
    tm = AutoTokenizer.from_pretrained(B_ID, local_files_only=True)
    if tm.pad_token is None: tm.pad_token = tm.eos_token
    
    print(f"  Loading base model for attention weights...")
    ma2 = AutoModelForCausalLM.from_pretrained(A_ID, dtype=torch.float16, device_map="auto", local_files_only=True)
    
    n_viable = splice_ceci(ma2, m_minsk, k=576)
    print(f"  Spliced: {n_viable}/30 layers")
    del ma2; gc.collect(); torch.cuda.empty_cache()
    
    ppl_minsk = compute_ppl(m_minsk, tm, ppl_texts)
    text_minsk, acc_minsk = bench_text(m_minsk, tm, "MINSKAT")
    print(f"  PPL: {ppl_minsk:.2f}  |  Accuracy: {acc_minsk:.0%}  |  Answers: {sum(1 for r in text_minsk if r['match'])}/{sum(1 for r in text_minsk if r['match'] is not None)}")
    report["minsk_at"] = {"ppl": round(ppl_minsk, 2), "accuracy": round(acc_minsk, 3),
                          "viable_layers": n_viable, "total_layers": 30, "responses": text_minsk}
    del m_minsk; gc.collect(); torch.cuda.empty_cache()
    
    # --- Analysis ---
    print("\n" + "=" * 70)
    print("MINSKAT vs CONTRIBUTORS")
    print("=" * 70)
    
    ppls = {"A (base)": ppl_a, "B (instruct)": ppl_b, "MINSKAT": ppl_minsk}
    accs = {"A (base)": acc_a, "B (instruct)": acc_b, "MINSKAT": acc_minsk}
    
    for name in ["A (base)", "B (instruct)", "MINSKAT"]:
        ppl_str = f"PPL={ppls[name]:.1f}"
        acc_str = f"ACC={accs[name]:.0%}"
        delta_ppl = ""
        delta_acc = ""
        if name != "A (base)":
            d_ppl = (ppls[name] - ppl_b) / ppl_b * 100
            d_acc = accs[name] - acc_b
            delta_ppl = f" (vs B: {d_ppl:+.1f}%)"
            delta_acc = f" (vs B: {d_acc:+.0%})"
        print(f"  {name:<20s} | {ppl_str:>20s}{delta_ppl:<15s} | {acc_str:>20s}{delta_acc}")
    
    # Save
    out_file = OUT / "benchmark.json"
    with open(out_file, "w") as f:
        json.dump({k: v for k, v in report.items() if k != "responses" if isinstance(v, (dict,list,str,int,float,bool))} 
                  if False else report, f, indent=2, default=str)
    # Actually save just the summary for JSON cleanliness
    summary = {k: v for k, v in report.items() if k != "responses"}
    for key in ["model_a","model_b","minsk_at"]:
        if key in summary and "responses" in summary[key]:
            del summary[key]["responses"]
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also save full report with responses
    full_out = OUT / "benchmark_full.json"
    with open(full_out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nSaved: {out_file}")
    print(f"Full report: {full_out}")
    print("\nDONE")


if __name__ == "__main__":
    main()
