#!/usr/bin/env python3
"""
Python-only Task Benchmark Harness — bypasses geodessical2 ChatML blocker.

Evaluates GRC-compressed SmolLM2-135M-Instruct on MMLU and GSM8K
using transformers + proper ChatML template.

Usage:
  python scripts/task_bench_python.py --ranks 256,512,full --benchmarks mmlu,gsm8k
  python scripts/task_bench_python.py --ranks 512 --benchmarks gsm8k --gsm8k-samples 50

GPU: RTX 4070 (8GB) sufficient for SmolLM2-135M
Runtime: ~2-4 hours for full MMLU+GSM8K sweep at 3 ranks
"""

import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "task_bench_python"
OUTPUT.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


# 
# GRC compression (mirrored from grc_distill.py)
# 

def build_shared_basis(Wq, Wk, Wv, n_iter=3):
    K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def apply_grc(model, k):
    """Apply GRC compression to attention Q, K, V at rank k. In-place."""
    n_layers = len(model.model.layers)
    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        device = attn.q_proj.weight.device
        dtype = attn.q_proj.weight.dtype
        
        Wq = attn.q_proj.weight.data.float().cpu().numpy()
        Wk = attn.k_proj.weight.data.float().cpu().numpy()
        Wv = attn.v_proj.weight.data.float().cpu().numpy()
        
        P = build_shared_basis(Wq, Wk, Wv)
        Pk = P[:, :k]
        
        # Project: W' = W @ P_k @ P_k^T
        attn.q_proj.weight.data = torch.from_numpy(
            (Wq @ Pk @ Pk.T).astype(np.float32)).to(dtype=dtype, device=device)
        attn.k_proj.weight.data = torch.from_numpy(
            (Wk @ Pk @ Pk.T).astype(np.float32)).to(dtype=dtype, device=device)
        attn.v_proj.weight.data = torch.from_numpy(
            (Wv @ Pk @ Pk.T).astype(np.float32)).to(dtype=dtype, device=device)


# 
# MMLU
# 

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous",
    "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
    "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def load_mmlu_questions():
    """Load MMLU from HF datasets. Returns {subject: [{question, choices, answer}]}."""
    try:
        from datasets import load_dataset as ld
        ds = ld("cais/mmlu", "all", split="test")
        questions = {}
        for row in ds:
            subj = row["subject"]
            if subj not in questions:
                questions[subj] = []
            questions[subj].append({
                "question": row["question"],
                "choices": [row["choices"][i] for i in range(min(4, len(row["choices"])))],
                "answer": row["answer"],  # 0-3 index or A-D
            })
        return questions
    except Exception:
        print("  WARNING: Could not load MMLU from HF. Using placeholder.")
        return {}


def format_mmlu_chat(few_shot, question_dict, tokenizer):
    """Format MMLU as ChatML messages."""
    messages = []
    for ex in few_shot:
        q_text = f"Question: {ex['question']}\n"
        for i, choice in enumerate(ex["choices"]):
            q_text += f"{chr(65+i)}. {choice}\n"
        q_text += "Answer with the letter of the correct choice."
        messages.append({"role": "user", "content": q_text})
        ans_idx = ex["answer"]
        if isinstance(ans_idx, int):
            ans_letter = chr(65 + ans_idx)
        else:
            ans_letter = str(ans_idx).strip().upper()
        messages.append({"role": "assistant", "content": ans_letter})
    
    q_text = f"Question: {question_dict['question']}\n"
    for i, choice in enumerate(question_dict["choices"]):
        q_text += f"{chr(65+i)}. {choice}\n"
    q_text += "Answer with the letter of the correct choice."
    messages.append({"role": "user", "content": q_text})
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def eval_mmlu(model, tokenizer, subjects=None, n_shot=2, max_per_subject=50,
              max_total=None):
    """Evaluate MMLU with 5-shot ChatML prompting. Returns accuracy dict."""
    all_qs = load_mmlu_questions()
    if not all_qs:
        return {"error": "MMLU dataset not available"}
    
    if subjects is None:
        subjects = list(all_qs.keys())
    
    # Build few-shot from first subject with enough questions
    few_shot = []
    for subj in subjects:
        if subj in all_qs and len(all_qs[subj]) >= n_shot:
            few_shot = all_qs[subj][:n_shot]
            break
    
    results = {}
    total_correct, total_q = 0, 0
    
    for subj in subjects:
        if subj not in all_qs:
            continue
        qs = all_qs[subj][n_shot:n_shot + max_per_subject]
        if max_total and total_q >= max_total:
            break
        
        correct = 0
        for q in qs:
            if max_total and total_q >= max_total:
                break
            
            prompt = format_mmlu_chat(few_shot, q, tokenizer)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            
            response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            response = response.strip().upper()
            
            ans = q["answer"]
            if isinstance(ans, int):
                ans_letter = chr(65 + ans)
            else:
                ans_letter = str(ans).strip().upper()
            
            # Match first A/B/C/D
            m = re.search(r'\b([A-D])\b', response)
            pred = m.group(1) if m else "?"
            
            if pred == ans_letter:
                correct += 1
            total_q += 1
        
        acc = correct / len(qs) if qs else 0
        results[subj] = {"correct": correct, "total": len(qs), "accuracy": acc}
        total_correct += correct
        
        if len(results) % 5 == 0:
            print(f"    MMLU: {len(results)}/{len(subjects)} subjects, "
                  f"running acc={total_correct/max(total_q,1):.1%}")
    
    results["_overall"] = {"correct": total_correct, "total": total_q,
                           "accuracy": total_correct / max(total_q, 1)}
    return results


# 
# GSM8K
# 

def load_gsm8k(n_samples=None):
    """Load GSM8K test set. Returns [{question, answer}]."""
    try:
        ds = load_dataset("gsm8k", "main", split="test")
        qs = []
        for row in ds:
            qs.append({"question": row["question"], "answer": row["answer"]})
            if n_samples and len(qs) >= n_samples:
                break
        return qs
    except Exception:
        # Fallback: local file
        local = ROOT / "data" / "gsm8k_test.jsonl"
        if local.exists():
            import json as jmod
            qs = []
            with open(local) as f:
                for line in f:
                    qs.append(jmod.loads(line))
                    if n_samples and len(qs) >= n_samples:
                        break
            return qs
        print("  WARNING: Could not load GSM8K. Using placeholder.")
        return []


def format_gsm8k_chat(few_shot, question_dict, tokenizer):
    """Format GSM8K as ChatML with chain-of-thought examples."""
    messages = []
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["question"]})
        # Extract final answer for few-shot
        ans = ex["answer"]
        # Keep the full chain-of-thought answer
        messages.append({"role": "assistant", "content": ans})
    
    messages.append({"role": "user", "content": question_dict["question"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_gsm8k_answer(text):
    """Extract final numeric answer from GSM8K chain-of-thought output."""
    # Look for #### pattern
    m = re.search(r'####\s*(-?[\d.,]+)', text)
    if m:
        return m.group(1).replace(",", "")
    # Last number in text
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums:
        return nums[-1]
    return None


def eval_gsm8k(model, tokenizer, n_shot=2, n_samples=None):
    """Evaluate GSM8K with 2-shot ChatML chain-of-thought."""
    qs = load_gsm8k(n_samples)
    if not qs:
        return {"error": "GSM8K dataset not available"}
    
    few_shot = qs[:n_shot]
    test_qs = qs[n_shot:]
    if n_samples:
        test_qs = test_qs[:n_samples]
    
    correct = 0
    for i, q in enumerate(test_qs):
        prompt = format_gsm8k_chat(few_shot, q, tokenizer)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        
        pred = extract_gsm8k_answer(response)
        gt = extract_gsm8k_answer(q["answer"])
        
        is_correct = (pred is not None and gt is not None and
                      abs(float(pred) - float(gt)) < 1e-6)
        if is_correct:
            correct += 1
        
        if (i + 1) % 10 == 0 or i == len(test_qs) - 1:
            print(f"    GSM8K: {i+1}/{len(test_qs)}, running acc={correct/max(i+1,1):.1%}")
    
    acc = correct / len(test_qs) if test_qs else 0
    return {"correct": correct, "total": len(test_qs), "accuracy": acc}


# 
# Main
# 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=str, default="256,512,full",
                       help="Comma-separated ranks (full=uncompressed)")
    parser.add_argument("--benchmarks", type=str, default="mmlu,gsm8k")
    parser.add_argument("--gsm8k-samples", type=int, default=100)
    parser.add_argument("--mmlu-subjects", type=int, default=10,
                       help="Number of MMLU subjects to evaluate")
    parser.add_argument("--mmlu-per-subject", type=int, default=30)
    args = parser.parse_args()
    
    ranks = []
    for r in args.ranks.split(","):
        r = r.strip()
        ranks.append(None if r.lower() == "full" else int(r))
    
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    
    print("=" * 60)
    print("TASK BENCHMARK HARNESS (Python/ChatML)")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Ranks: {args.ranks}")
    print(f"  Benchmarks: {args.benchmarks}")
    print("=" * 60)
    
    # Load model once
    print("[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, local_files_only=True)
    print("    Model loaded.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                              local_files_only=True)
    print("    Tokenizer loaded.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    
    all_results = {}
    
    for rank in ranks:
        rank_label = "full" if rank is None else str(rank)
        print(f"\n{'='*60}")
        print(f"  RANK: {rank_label}")
        print(f"{'='*60}")
        
        # Reload fresh model for each rank to avoid accumulated drift
        if rank is not None:
            print("  Reloading fresh model...")
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.float16, device_map="auto",
                trust_remote_code=True, local_files_only=True)
            
            print(f"  Applying GRC at k={rank}...")
            apply_grc(model, rank)
        
        rank_results = {}
        
        if "mmlu" in benchmarks:
            print(f"\n  [MMLU] Evaluating...")
            t0 = time.time()
            mmlu = eval_mmlu(model, tokenizer, subjects=MMLU_SUBJECTS[:args.mmlu_subjects],
                           n_shot=2, max_per_subject=args.mmlu_per_subject)
            elapsed = time.time() - t0
            if "_overall" in mmlu:
                print(f"  MMLU Overall: {mmlu['_overall']['accuracy']:.1%} "
                      f"({mmlu['_overall']['correct']}/{mmlu['_overall']['total']}) "
                      f"[{elapsed:.0f}s]")
            rank_results["mmlu"] = mmlu
        
        if "gsm8k" in benchmarks:
            print(f"\n  [GSM8K] Evaluating...")
            t0 = time.time()
            gsm8k = eval_gsm8k(model, tokenizer, n_shot=2, n_samples=args.gsm8k_samples)
            elapsed = time.time() - t0
            if "accuracy" in gsm8k:
                print(f"  GSM8K: {gsm8k['accuracy']:.1%} "
                      f"({gsm8k['correct']}/{gsm8k['total']}) "
                      f"[{elapsed:.0f}s]")
            rank_results["gsm8k"] = gsm8k
        
        all_results[rank_label] = rank_results
    
    # Summary
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    
    summary = {}
    for rank_label, rank_data in all_results.items():
        print(f"\n  k={rank_label}:")
        row = {}
        for bench in benchmarks:
            if bench in rank_data:
                d = rank_data[bench]
                if bench == "mmlu" and "_overall" in d:
                    acc = d["_overall"]["accuracy"]
                    print(f"    MMLU: {acc:.1%}")
                    row["mmlu"] = round(acc, 4)
                elif bench == "gsm8k" and "accuracy" in d:
                    acc = d["accuracy"]
                    print(f"    GSM8K: {acc:.1%}")
                    row["gsm8k"] = round(acc, 4)
        summary[rank_label] = row
    
    # Save
    out_file = OUTPUT / "results.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "full": all_results}, f, indent=2)
    print(f"\nSaved: {out_file}")
    
    # LaTeX-ready table
    print(f"\n  LaTeX table:")
    print(f"  {'Rank':<10} {'MMLU':<12} {'GSM8K':<12}")
    print(f"  {'-'*34}")
    for rank_label in all_results:
        row = summary.get(rank_label, {})
        mmlu_str = f"{row.get('mmlu', 0):.1%}" if 'mmlu' in row else "---"
        gsm8k_str = f"{row.get('gsm8k', 0):.1%}" if 'gsm8k' in row else "---"
        print(f"  {rank_label:<10} {mmlu_str:<12} {gsm8k_str:<12}")


if __name__ == "__main__":
    main()
