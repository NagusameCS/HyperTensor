#!/usr/bin/env python3
"""
EC2 Llama-8B GRC Light Distillation --- Paper V gap closure.
Runs on EC2 g6e.xlarge (L40S 24GB).
Target: Llama-3.1-8B-Instruct at k=1024, r=8 LoRA.

Usage: python ec2_distill_llama8b.py --k 1024 --steps 500 --lora-r 8
"""

import argparse, json, os, sys, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

OUT = Path("benchmarks/llama8b_distill")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# ---- GRC (from grc_distill.py) ----
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

def apply_grc(model, k):
    for layer_idx in range(len(model.model.layers)):
        attn = model.model.layers[layer_idx].self_attn
        d, dev = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        Wq = attn.q_proj.weight.data.float().cpu().numpy()
        Wk = attn.k_proj.weight.data.float().cpu().numpy()
        Wv = attn.v_proj.weight.data.float().cpu().numpy()
        P = build_shared_basis(Wq, Wk, Wv); Pk = P[:, :k]; PkPkT = Pk @ Pk.T
        attn.q_proj.weight.data = torch.from_numpy((Wq @ PkPkT).astype(np.float32)).to(dtype=d, device=dev)
        attn.k_proj.weight.data = torch.from_numpy((Wk @ PkPkT).astype(np.float32)).to(dtype=d, device=dev)
        attn.v_proj.weight.data = torch.from_numpy((Wv @ PkPkT).astype(np.float32)).to(dtype=d, device=dev)

def compute_ppl(model, tokenizer, texts, max_length=256):
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for text in texts[:20]:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            if enc['input_ids'].shape[1] < 2: continue
            out = model(**enc, labels=enc['input_ids'])
            total_loss += out.loss.item() * enc['input_ids'].shape[1]
            total_tokens += enc['input_ids'].shape[1]
    return float(np.exp(total_loss / max(total_tokens, 1)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lora-r", type=int, default=8)
    args = p.parse_args()
    
    print("=" * 60)
    print(f"LLAMA-8B GRC DISTILLATION: k={args.k}, steps={args.steps}, r={args.lora_r}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")
    print("=" * 60)
    
    # Load model
    print("[1] Loading Llama-3.1-8B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"    Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
          f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Load texts
    print("[2] Loading WikiText-2...")
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    train_texts = [t for t in wiki['text'] if len(t.strip()) > 50][:100]
    ppl_texts = [t for t in wiki['text'] if len(t.strip()) > 50][100:120]
    
    # Baseline
    print("[3] Baseline PPL...")
    b_ppl = compute_ppl(model, tokenizer, ppl_texts)
    print(f"    Baseline: {b_ppl:.2f}")
    
    # GRC compress
    print(f"[4] GRC k={args.k}...")
    t0 = time.time()
    apply_grc(model, args.k)
    print(f"    Done in {time.time()-t0:.1f}s")
    
    # Compressed PPL
    print("[5] Compressed PPL...")
    c_ppl = compute_ppl(model, tokenizer, ppl_texts)
    print(f"    Compressed: {c_ppl:.2f} ({c_ppl/b_ppl:.2f}× baseline)")
    
    # LoRA distill
    print(f"[6] LoRA distillation (r={args.lora_r}, {args.steps} steps)...")
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r*2,
        target_modules=["q_proj","k_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    losses = []
    model.train()
    
    t0 = time.time()
    for step in range(args.steps):
        text = train_texts[step % len(train_texts)]
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc['input_ids'].shape[1] < 4: continue
        out = model(**enc, labels=enc['input_ids'])
        loss = out.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 100 == 0:
            print(f"    Step {step}/{args.steps}: loss={np.mean(losses[-20:]):.4f}")
    
    print(f"    Training: {time.time()-t0:.0f}s")
    
    # Distilled PPL
    print("[7] Distilled PPL...")
    model.eval()
    d_ppl = compute_ppl(model, tokenizer, ppl_texts)
    recovery = (c_ppl - d_ppl) / max(c_ppl - b_ppl, 1e-10) * 100
    
    print(f"    Distilled: {d_ppl:.2f}")
    print(f"    Recovery: {recovery:.1f}%")
    print(f"    Final: {d_ppl/b_ppl:.2f}× baseline")
    
    # Save
    result = {
        "model": "Llama-3.1-8B-Instruct",
        "k": args.k, "lora_r": args.lora_r, "steps": args.steps,
        "gpu": torch.cuda.get_device_name(0),
        "baseline_ppl": round(b_ppl, 2),
        "compressed_ppl": round(c_ppl, 2),
        "distilled_ppl": round(d_ppl, 2),
        "recovery_pct": round(recovery, 1),
        "final_ratio": round(d_ppl/b_ppl, 2),
    }
    with open(OUT / f"llama8b_k{args.k}_r{args.lora_r}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: benchmarks/llama8b_distill/llama8b_k{args.k}_r{args.lora_r}.json")
    print("DONE")

if __name__ == "__main__":
    main()
