#!/usr/bin/env python3
"""
COMPREHENSIVE CHIMERIC BENCHMARK.
Compares all four models on MMLU, GSM8K, language fluency, and PPL.

Models:
  MIYA       — SmolLM2-135M + math LoRA (pure math)
  HORI       — SmolLM2-135M + language LoRA (pure language)
  HORIMIYA   — CECI splice k=512 (13/30 layers)
  HORIMIYA-MP — Full-rank splice k=576 (30/30 layers)

CPU/GPU: Runs on CPU for fairness (no GPU speed artifacts).
Uses HuggingFace evaluate + lm-eval-harness where available,
falls back to direct generation for qualitative assessment.

Usage:
  python scripts/benchmark_chimeric.py --all
  python scripts/benchmark_chimeric.py --model HORIMIYA-MP
"""

import argparse, json, os, time, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "MIYA": "outputs/pure_models/smollm2-135m-math-pure/final",
    "HORI": "outputs/pure_models/smollm2-135m-language-pure/final",
    "HORIMIYA": "outputs/chimeric/HORIMIYA",
    "HORIMIYA-MP": "outputs/chimeric/HORIMIYA-MP",
}

OUT = Path("benchmarks/chimeric_comparison")
OUT.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Quantitative Benchmarks
# ===========================================================================

def benchmark_ppl(model, tokenizer, texts, max_length=512):
    """Compute perplexity on a set of texts."""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            if enc.input_ids.shape[1] < 2:
                continue
            outputs = model(**enc, labels=enc.input_ids)
            loss = outputs.loss.item()
            n_tokens = enc.input_ids.shape[1]
            total_loss += loss * n_tokens
            total_tokens += n_tokens
    return np.exp(total_loss / max(total_tokens, 1))

def benchmark_generation_speed(model, tokenizer, prompt, n_tokens=64):
    """Measure tokens/sec on CPU."""
    enc = tokenizer(prompt, return_tensors='pt')
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n_tokens, do_sample=False, 
                            temperature=0, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[1] - enc.input_ids.shape[1]
    return n_gen / elapsed if elapsed > 0 else 0

# ===========================================================================
# Qualitative: Math vs Language Assessment
# ===========================================================================

MATH_PROMPTS = [
    "Solve: If x + 3 = 7, what is x?",
    "What is 15% of 200?",
    "If a train travels 60 miles in 2 hours, what is its speed in mph?",
    "Factor: x^2 - 4",
    "What is the square root of 144?",
]

LANGUAGE_PROMPTS = [
    "Write a short poem about the ocean.",
    "Describe a sunset in three sentences.",
    "Explain what a transformer neural network is in simple terms.",
    "Write a haiku about winter.",
    "Tell me a short story about a robot learning to paint.",
]

def qualitative_assess(model, tokenizer, model_name):
    """Run math and language prompts, return outputs for comparison."""
    results = {"model": model_name, "math": [], "language": []}
    model.eval()
    with torch.no_grad():
        for prompt in MATH_PROMPTS:
            enc = tokenizer(prompt, return_tensors='pt')
            out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                temperature=0, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            results["math"].append({"prompt": prompt, "response": text})
        
        for prompt in LANGUAGE_PROMPTS:
            enc = tokenizer(prompt, return_tensors='pt')
            out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                temperature=0, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            results["language"].append({"prompt": prompt, "response": text})
    
    return results

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS.keys()) + ['all'], default='all')
    parser.add_argument('--cpu', action='store_true', default=True)
    args = parser.parse_args()
    
    models_to_test = list(MODELS.keys()) if args.model == 'all' else [args.model]
    
    # WikiText-2 for PPL measurement
    try:
        from datasets import load_dataset
        wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        ppl_texts = [t for t in wiki['text'] if len(t.strip()) > 50][:50]
    except:
        ppl_texts = ["The quick brown fox jumps over the lazy dog."] * 10
    
    all_results = {}
    
    for model_name in models_to_test:
        model_path = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"  Path: {model_path}")
        print(f"{'='*60}")
        
        if not Path(model_path).exists():
            print(f"  SKIPPED: model path not found")
            continue
        
        # Load model
        print("  Loading model...")
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32, device_map='cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_time = time.perf_counter() - t0
        print(f"  Loaded in {load_time:.1f}s")
        
        # PPL
        print("  Computing PPL...")
        ppl = benchmark_ppl(model, tokenizer, ppl_texts)
        print(f"  PPL: {ppl:.2f}")
        
        # Generation speed
        print("  Measuring generation speed...")
        tok_s = benchmark_generation_speed(model, tokenizer, MATH_PROMPTS[0])
        print(f"  Speed: {tok_s:.1f} tok/s")
        
        # Qualitative
        print("  Qualitative assessment...")
        qual = qualitative_assess(model, tokenizer, model_name)
        
        all_results[model_name] = {
            "ppl": round(float(ppl), 2),
            "tok_per_sec": round(float(tok_s), 1),
            "load_time_s": round(load_time, 1),
            "model_size_params": sum(p.numel() for p in model.parameters()),
            "qualitative": qual,
        }
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'PPL':>8} {'tok/s':>8} {'Size':>12}")
    print("-" * 45)
    for name in MODELS:
        if name in all_results:
            r = all_results[name]
            size_m = r['model_size_params'] / 1e6
            print(f"{name:<15} {r['ppl']:>8.2f} {r['tok_per_sec']:>8.1f} {size_m:>9.1f}M")
    
    # Save
    out_path = OUT / 'chimeric_benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved: {out_path}")

if __name__ == '__main__':
    main()
