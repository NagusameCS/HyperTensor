#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
Paper VI full benchmark sweep — SmolLM2-135M at k∈{256,512,full}.
Covers MMLU (all subjects, streaming), GSM8K, HumanEval (pass@1).
Per peer review: completes the "only 16 MMLU questions" gap.
Runtime: ~3 GPU-hours on RTX 4070 Laptop (8GB).
"""

import json, os, sys, time, re, subprocess, tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT = Path("benchmarks/paper_vi_full_sweep")
OUT.mkdir(parents=True, exist_ok=True)
K_VALUES = [256, 512, 576]  # 576 = full dim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ===========================================================================
# GRC Projection (per-layer joint Gram eigendecomposition)
# ===========================================================================
def grc_project(model, k):
    if k >= 576: return 1.0
    signals = []
    for layer in model.model.layers:
        Wq = layer.self_attn.q_proj.weight.data.float()
        Wk = layer.self_attn.k_proj.weight.data.float()
        Wv = layer.self_attn.v_proj.weight.data.float()
        # All: shape (out_features, d_model). W^T @ W yields (d_model, d_model)
        G = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
        L, V = torch.linalg.eigh(G)  # ascending eigenvalues
        ke = min(k, len(L))
        P = V[:, -ke:]  # top-k eigenvectors, shape (d, k)
        signals.append(float(torch.sum(L[-ke:]) / torch.sum(L)))
        # Project: W_proj = W @ P @ P^T
        layer.self_attn.q_proj.weight.data.copy_(
            (Wq @ P @ P.T).to(layer.self_attn.q_proj.weight.dtype))
        layer.self_attn.k_proj.weight.data.copy_(
            (Wk @ P @ P.T).to(layer.self_attn.k_proj.weight.dtype))
        layer.self_attn.v_proj.weight.data.copy_(
            (Wv @ P @ P.T).to(layer.self_attn.v_proj.weight.dtype))
    return float(np.mean(signals))


# ===========================================================================
# MMLU — full 57-subject benchmark via HuggingFace
# ===========================================================================
def benchmark_mmlu_full(model, tokenizer, max_q=None):
    """MMLU via cais/mmlu. Streaming, all subjects. Returns accuracy."""
    try:
        ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    except Exception:
        print("    cais/mmlu not available, trying hendrycks_test...")
        ds = load_dataset("hendrycks_test", "all", split="test", streaming=True)
    correct, total = 0, 0
    t0 = time.time()
    for row in ds:
        if max_q and total >= max_q:
            break
        question = row.get("question", "")
        choices = row.get("choices", [])
        if isinstance(choices, list) and len(choices) >= 4:
            choices = choices[:4]
        else:
            choices = ["A", "B", "C", "D"]
        answer_idx = row.get("answer", 0)
        if isinstance(answer_idx, str):
            answer_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer_idx, 0)
        prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        option_ids = [tokenizer.encode(o, add_special_tokens=False)[0] for o in ["A", "B", "C", "D"]]
        option_logits = [logits[oid].item() for oid in option_ids]
        if np.argmax(option_logits) == answer_idx:
            correct += 1
        total += 1
        if total % 500 == 0:
            print(f"    MMLU: {total} done, acc={correct/total:.3f} ({time.time()-t0:.0f}s)")
    acc = correct / max(total, 1)
    print(f"    MMLU final: {correct}/{total} = {acc:.4f}")
    return acc, total


# ===========================================================================
# GSM8K
# ===========================================================================
def benchmark_gsm8k_full(model, tokenizer, max_q=None):
    ds = load_dataset("gsm8k", "main", split="test")
    correct, total = 0, 0
    t0 = time.time()
    for row in ds:
        if max_q and total >= max_q:
            break
        question = row["question"]
        gt_match = re.search(r'####\s*([\d,.-]+)', row["answer"])
        if not gt_match: continue
        gt_answer = gt_match.group(1).replace(",", "").strip()
        prompt = f"Question: {question}\nLet's solve this step by step.\n"
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=200, temperature=0, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        numbers = re.findall(r'[\d,.-]+', generated)
        if numbers:
            pred = numbers[-1].replace(",", "")
            try:
                if abs(float(pred) - float(gt_answer)) < 1e-6:
                    correct += 1
            except ValueError:
                pass
        total += 1
        if total % 20 == 0:
            print(f"    GSM8K: {total} done, acc={correct/total:.3f} ({time.time()-t0:.0f}s)")
    acc = correct / max(total, 1)
    print(f"    GSM8K final: {correct}/{total} = {acc:.4f}")
    return acc, total


# ===========================================================================
# HumanEval — pass@1 via subprocess sandbox
# ===========================================================================
def benchmark_humaneval(model, tokenizer, max_q=None):
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        print("    HumanEval dataset not available, skipping")
        return 0.0, 0
    n_pass, total = 0, 0
    for i, row in enumerate(ds):
        if max_q and total >= max_q:
            break
        prompt = row["prompt"]
        test = row["test"]
        entry = row["entry_point"]
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=256, temperature=0, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        program = prompt + completion + "\n" + test + f"\ncheck({entry})\n"
        try:
            r = subprocess.run([sys.executable, "-c", program], capture_output=True, timeout=10, text=True)
            ok = r.returncode == 0
        except Exception:
            ok = False
        n_pass += int(ok)
        total += 1
        if total % 10 == 0:
            print(f"    HumanEval: {total} done, pass@1={n_pass/total:.3f}")
    acc = n_pass / max(total, 1)
    print(f"    HumanEval final: {n_pass}/{total} pass@1={acc:.4f}")
    return acc, total


# ===========================================================================
# Main sweep
# ===========================================================================
def main():
    print("=" * 70)
    print("PAPER VI FULL BENCHMARK — SmolLM2-135M-Instruct")
    print(f"Ranks: {K_VALUES}")
    print("Benchmarks: MMLU (full), GSM8K, HumanEval")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"model": BASE_MODEL, "ranks": {}, "meta": {"device": DEVICE}}

    for k in K_VALUES:
        label = "full" if k >= 576 else str(k)
        print(f"\n{'='*50}\n  k={label} (k/d={k/576:.2f})\n{'='*50}")

        t_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
        model.eval()
        print(f"  Model loaded in {time.time()-t_load:.1f}s")

        if k < 576:
            sig = grc_project(model, k)
            print(f"  Signal preserved: {sig:.4f}")

        entry = {"k": k, "label": label, "signal_preserved": sig if k < 576 else 1.0}

        # MMLU
        print("  [MMLU full]")
        t0 = time.time()
        mmlu_acc, mmlu_n = benchmark_mmlu_full(model, tokenizer)
        entry["mmlu"] = {"accuracy": round(mmlu_acc, 6), "n": mmlu_n, "wall_s": round(time.time()-t0, 1)}

        # GSM8K
        print("  [GSM8K]")
        t0 = time.time()
        gsm8k_acc, gsm8k_n = benchmark_gsm8k_full(model, tokenizer)
        entry["gsm8k"] = {"accuracy": round(gsm8k_acc, 6), "n": gsm8k_n, "wall_s": round(time.time()-t0, 1)}

        # HumanEval
        print("  [HumanEval]")
        t0 = time.time()
        he_acc, he_n = benchmark_humaneval(model, tokenizer)
        entry["humaneval"] = {"accuracy": round(he_acc, 6), "n": he_n, "wall_s": round(time.time()-t0, 1)}

        results["ranks"][label] = entry
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — Asymmetric Degradation Check")
    print(f"{'Benchmark':<14} {'k=256':>8} {'k=512':>8} {'full':>8}")
    for bench in ["mmlu", "gsm8k", "humaneval"]:
        vals = []
        for k in ["256", "512", "full"]:
            if k in results["ranks"]:
                vals.append(f"{results['ranks'][k][bench]['accuracy']:.4f}")
            else:
                vals.append("   N/A  ")
        print(f"{bench:<14} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    out_path = OUT / "paper_vi_full_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
