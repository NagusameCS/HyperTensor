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
Paper VI Task-Level Impact — Self-Contained Benchmark
======================================================
Bypasses geodessical2 ChatML blocker. Uses transformers directly.
No external dataset downloads needed. Produces results for Paper VI.

Benchmarks:
  - MMLU-style: 20 hand-picked multiple-choice questions across 4 subjects
  - GSM8K-style: 10 math word problems with chain-of-thought
  - PPL: WikiText-2 sample (local if available, synthetic fallback)

Ranks: full, 1536, 1024, 512, 256
Model: SmolLM2-135M-Instruct (fits in 8GB VRAM)

Output: benchmarks/paper_vi_benchmark/results.json
"""
import json, os, re, sys, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "paper_vi_benchmark"
OUTPUT.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

#  MMLU-style questions (hand-picked, 5 subjects × 4 questions) 
MMLU_QUESTIONS = [
    # Math
    {"subject":"math","question":"What is the derivative of x^3?","choices":["x^2","3x^2","3x","x^3/3"],"answer":1},
    {"subject":"math","question":"If a triangle has angles 30° and 60°, what is the third angle?","choices":["30°","60°","90°","120°"],"answer":2},
    {"subject":"math","question":"What is 17 × 43?","choices":["731","714","688","756"],"answer":0},
    {"subject":"math","question":"Solve for x: 2x + 5 = 15","choices":["x=10","x=5","x=7.5","x=2.5"],"answer":1},
    # Science
    {"subject":"science","question":"What is the chemical symbol for water?","choices":["CO2","H2O","NaCl","O2"],"answer":1},
    {"subject":"science","question":"What planet is closest to the Sun?","choices":["Venus","Earth","Mercury","Mars"],"answer":2},
    {"subject":"science","question":"What gas do plants absorb during photosynthesis?","choices":["Oxygen","Nitrogen","Carbon dioxide","Hydrogen"],"answer":2},
    {"subject":"science","question":"What is the speed of light in vacuum?","choices":["3×10^6 m/s","3×10^8 m/s","3×10^10 m/s","300 km/s"],"answer":1},
    # History
    {"subject":"history","question":"In what year did World War II end?","choices":["1943","1944","1945","1946"],"answer":2},
    {"subject":"history","question":"Who was the first President of the United States?","choices":["Thomas Jefferson","George Washington","Abraham Lincoln","John Adams"],"answer":1},
    {"subject":"history","question":"The French Revolution began in which year?","choices":["1776","1789","1815","1848"],"answer":1},
    {"subject":"history","question":"Which ancient civilization built the pyramids at Giza?","choices":["Roman","Greek","Egyptian","Persian"],"answer":2},
    # Computer Science
    {"subject":"cs","question":"What does CPU stand for?","choices":["Central Processing Unit","Computer Personal Unit","Central Program Utility","Core Processing Unit"],"answer":0},
    {"subject":"cs","question":"Which data structure uses LIFO?","choices":["Queue","Stack","Array","Tree"],"answer":1},
    {"subject":"cs","question":"What is the time complexity of binary search?","choices":["O(n)","O(n^2)","O(log n)","O(1)"],"answer":2},
    {"subject":"cs","question":"What does 'HTTP' stand for?","choices":["HyperText Transfer Protocol","High Tech Transfer Protocol","HyperText Transmission Program","Home Tool Transfer Protocol"],"answer":0},
]

#  GSM8K-style math problems 
GSM8K_QUESTIONS = [
    {"question":"Janet has 24 eggs. She uses 4 eggs to make omelets and gives away 6 eggs to her neighbor. How many eggs does Janet have left?","answer":"14"},
    {"question":"A store sells apples for $0.50 each. If Tom buys 12 apples and pays with a $10 bill, how much change does he get?","answer":"4"},
    {"question":"Sarah runs 3 miles every day. How many miles does she run in 2 weeks?","answer":"42"},
    {"question":"A rectangle has a length of 8 cm and a width of 5 cm. What is its area in square centimeters?","answer":"40"},
    {"question":"If a train travels at 60 miles per hour, how far does it travel in 2.5 hours?","answer":"150"},
    {"question":"John has 3 boxes. Each box contains 24 pencils. He gives 18 pencils to his students. How many pencils does John have left?","answer":"54"},
    {"question":"A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices remain?","answer":"2"},
    {"question":"Mary saves $5 every week. After 8 weeks, she buys a book for $12. How much money does she have left?","answer":"28"},
]


def apply_grc(model, k):
    """GRC projection at rank k on Q,K,V attention weights. In-place."""
    n_layers = len(model.model.layers)
    nkv = model.config.num_key_value_heads
    nh = model.config.num_attention_heads
    has_gqa = nkv < nh
    
    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        dtype = attn.q_proj.weight.dtype
        
        Wq = attn.q_proj.weight.data.float()
        Wk = attn.k_proj.weight.data.float()
        Wv = attn.v_proj.weight.data.float()
        
        # Expand K,V for GQA if needed
        if has_gqa:
            nr = nh // nkv
            hd = Wq.shape[1] // nh  # head_dim = 576/9 = 64
            d_out = Wq.shape[0]  # 576
            Wk2 = torch.zeros(d_out, Wq.shape[1], device=Wk.device)
            Wv2 = torch.zeros(d_out, Wq.shape[1], device=Wv.device)
            for h in range(nh):
                kv = h // nr
                Wk2[:, h*hd:(h+1)*hd] = Wk[:, kv*hd:(kv+1)*hd]
                Wv2[:, h*hd:(h+1)*hd] = Wv[:, kv*hd:(kv+1)*hd]
            Wk, Wv = Wk2, Wv2
        
        # Joint SVD
        M = torch.cat([Wq, Wk, Wv], dim=0)
        try:
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        except Exception:
            M_cpu = M.cpu().numpy()
            U_cpu, S_cpu, Vt_cpu = np.linalg.svd(M_cpu, full_matrices=False)
            U, Vt = torch.from_numpy(U_cpu).to(Wq.device), torch.from_numpy(Vt_cpu).to(Wq.device)
        ke = min(k, len(S))
        P = Vt[:ke, :].T
        
        # Apply projection
        Wq_p = (Wq @ P @ P.T)
        Wk_p = (Wk @ P @ P.T)
        Wv_p = (Wv @ P @ P.T)
        
        # Undo GQA expansion
        if has_gqa:
            nr = nh // nkv
            hd = Wq.shape[1] // nh
            Wk_o = torch.zeros(Wk_p.shape[0], nkv*hd, device=Wk_p.device)
            Wv_o = torch.zeros(Wk_p.shape[0], nkv*hd, device=Wk_p.device)
            for kv_idx in range(nkv):
                Wk_o[:, kv_idx*hd:(kv_idx+1)*hd] = Wk_p[:, kv_idx*nr*hd:(kv_idx*nr+1)*hd]
                Wv_o[:, kv_idx*hd:(kv_idx+1)*hd] = Wv_p[:, kv_idx*nr*hd:(kv_idx*nr+1)*hd]
            attn.k_proj.weight.data = Wk_o.to(dtype)
            attn.v_proj.weight.data = Wv_o.to(dtype)
        else:
            attn.k_proj.weight.data = Wk_p.to(dtype)
            attn.v_proj.weight.data = Wv_p.to(dtype)
        
        attn.q_proj.weight.data = Wq_p.to(dtype)
    
    torch.cuda.empty_cache()


def eval_mmlu_style(model, tokenizer):
    """Evaluate on hand-picked MMLU-style questions."""
    results = {}
    by_subject = {}
    
    for q in MMLU_QUESTIONS:
        subj = q["subject"]
        if subj not in by_subject:
            by_subject[subj] = {"correct": 0, "total": 0}
        
        # Format as ChatML
        choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(q["choices"]))
        prompt = f"Question: {q['question']}\n{choices_text}\nAnswer with the letter of the correct choice."
        
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        response = response.strip().upper()
        
        m = re.search(r'\b([A-D])\b', response)
        pred = m.group(1) if m else "?"
        ans_letter = chr(65 + q["answer"])
        
        correct = pred == ans_letter
        by_subject[subj]["total"] += 1
        if correct:
            by_subject[subj]["correct"] += 1
    
    total_correct = sum(s["correct"] for s in by_subject.values())
    total_q = sum(s["total"] for s in by_subject.values())
    
    results["by_subject"] = {s: {"accuracy": v["correct"]/max(v["total"],1),
                                  "correct": v["correct"], "total": v["total"]}
                             for s, v in by_subject.items()}
    results["overall"] = {"accuracy": total_correct / max(total_q, 1),
                          "correct": total_correct, "total": total_q}
    return results


def eval_gsm8k_style(model, tokenizer):
    """Evaluate on hand-picked GSM8K-style math problems."""
    correct = 0
    results = []
    
    for q in GSM8K_QUESTIONS:
        prompt = f"Solve this math problem step by step. End your answer with #### followed by the final number.\n\n{q['question']}"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        m = re.search(r'####\s*(-?[\d.,]+)', response)
        if m:
            pred = m.group(1).replace(",", "")
        else:
            nums = re.findall(r'-?\d+(?:\.\d+)?', response)
            pred = nums[-1] if nums else None
        
        gt = q["answer"]
        is_correct = (pred is not None and abs(float(pred) - float(gt)) < 1e-6)
        if is_correct:
            correct += 1
        
        results.append({"question": q["question"][:60], "predicted": pred,
                        "expected": gt, "correct": is_correct})
    
    return {"accuracy": correct / len(GSM8K_QUESTIONS), "correct": correct,
            "total": len(GSM8K_QUESTIONS), "details": results}


def eval_ppl(model, tokenizer, text=None):
    """Measure perplexity on a short text sample."""
    if text is None:
        text = ("The capital of France is Paris. Paris is known for its art, "
                "culture, and cuisine. The Eiffel Tower is one of the most "
                "visited monuments in the world. French is the official language. "
                "The Seine River flows through the heart of the city. "
                "Many tourists visit the Louvre Museum every year.")
    
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss.item()
        ppl = np.exp(loss)
    
    return ppl


def main():
    ranks = [None, 1536, 1024, 512, 256]  # None = full
    rank_names = ["full", "1536", "1024", "512", "256"]
    
    print("=" * 60)
    print("PAPER VI — TASK-LEVEL IMPACT BENCHMARK")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Ranks: {', '.join(rank_names)}")
    print(f"  MMLU-style questions: {len(MMLU_QUESTIONS)}")
    print(f"  GSM8K-style questions: {len(GSM8K_QUESTIONS)}")
    print("=" * 60)
    
    all_results = {}
    
    for rank, rname in zip(ranks, rank_names):
        print(f"\n{'='*60}")
        print(f"  RANK: {rname}")
        print(f"{'='*60}")
        
        # Load fresh model for each rank
        print(f"  Loading model...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Loaded in {time.time()-t0:.1f}s")
        
        # Apply GRC if not full
        if rank is not None:
            print(f"  Applying GRC k={rank}...")
            t0 = time.time()
            apply_grc(model, rank)
            print(f"  GRC applied in {time.time()-t0:.1f}s")
        
        # MMLU
        print(f"  Running MMLU-style...")
        t0 = time.time()
        mmlu = eval_mmlu_style(model, tokenizer)
        print(f"  MMLU: {mmlu['overall']['accuracy']:.1%} ({mmlu['overall']['correct']}/{mmlu['overall']['total']}) "
              f"in {time.time()-t0:.1f}s")
        
        # GSM8K
        print(f"  Running GSM8K-style...")
        t0 = time.time()
        gsm8k = eval_gsm8k_style(model, tokenizer)
        print(f"  GSM8K: {gsm8k['accuracy']:.1%} ({gsm8k['correct']}/{gsm8k['total']}) "
              f"in {time.time()-t0:.1f}s")
        
        # PPL
        print(f"  Measuring PPL...")
        t0 = time.time()
        ppl = eval_ppl(model, tokenizer)
        print(f"  PPL: {ppl:.2f} in {time.time()-t0:.1f}s")
        
        all_results[rname] = {
            "mmlu": mmlu,
            "gsm8k": gsm8k,
            "ppl": round(ppl, 2),
        }
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    #  Save 
    output_file = OUTPUT / "results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    #  Summary Table 
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Rank':>8s}  {'MMLU':>8s}  {'GSM8K':>8s}  {'PPL':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for rname in rank_names:
        r = all_results[rname]
        print(f"  {rname:>8s}  {r['mmlu']['overall']['accuracy']:>7.1%}  "
              f"{r['gsm8k']['accuracy']:>7.1%}  {r['ppl']:>8.2f}")
    
    # Safe frontier: where MMLU drops <5% from full
    full_mmlu = all_results["full"]["mmlu"]["overall"]["accuracy"]
    safe_at = "none"
    for rname in reversed(rank_names):
        mmlu_acc = all_results[rname]["mmlu"]["overall"]["accuracy"]
        if mmlu_acc >= full_mmlu * 0.95:
            safe_at = rname
            break
    
    print(f"\n  Safe frontier (MMLU >= 95% of full): k={safe_at}")
    print(f"  Full MMLU: {full_mmlu:.1%}")
    print(f"\n  Saved: {output_file}")


if __name__ == "__main__":
    main()
