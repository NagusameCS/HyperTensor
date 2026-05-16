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
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
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
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
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

"""Baseline comparison: HyperRetro GRC vs bitsandbytes nf4 on WikiText-2 PPL.

Measures:
  - baseline (fp32, uncompressed)
  - bitsandbytes nf4 (4-bit, the industry standard)
  - HyperRetro GRC k=640 + distill r=8 (71% rank, our best result)
  - HyperRetro GRC k=448 + distill r=8 (50% rank)
"""
import sys, json, time, torch, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUT = Path("benchmarks/baseline_comparison")
OUT.mkdir(parents=True, exist_ok=True)


def ppl_wikitext(model, tokenizer, device="cuda", max_samples=200, seq_len=256):
    """Compute PPL on WikiText-2 test set."""
    model.eval()
    model = model.to(device)
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer.encode(text)
    
    stride = seq_len // 2
    nlls = []
    prev_end_loc = 0
    
    for i, begin_loc in enumerate(range(0, len(enc) - seq_len, stride)):
        if i >= max_samples:
            break
        end_loc = begin_loc + seq_len
        if prev_end_loc > begin_loc:
            begin_loc = prev_end_loc
        if begin_loc >= end_loc:
            continue
            
        input_ids = torch.tensor(enc[begin_loc:end_loc], dtype=torch.long).unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.item() * seq_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if i > 0 and i % 50 == 0:
            ppl_so_far = float(np.exp(sum(nlls) / (len(nlls) * seq_len)))
            print(f"  [{i}/{max_samples}] PPL={ppl_so_far:.4f}", flush=True)
    
    if not nlls:
        return float("inf")
    avg_nll = sum(nlls) / (len(nlls) * seq_len)
    return float(np.exp(avg_nll))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    
    # 1. Baseline fp32
    print("\n" + "="*60)
    print("1/4: BASELINE (fp32, uncompressed)")
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    t0 = time.time()
    results["baseline_fp32"] = ppl_wikitext(m, tokenizer, device)
    results["baseline_fp32_time"] = time.time() - t0
    print(f"  PPL={results['baseline_fp32']:.4f}  ({results['baseline_fp32_time']:.1f}s)")
    mem_fp32 = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024**3
    results["baseline_fp32_GB"] = mem_fp32
    del m; torch.cuda.empty_cache()
    
    # 2. bitsandbytes nf4
    print("\n" + "="*60)
    print("2/4: bitsandbytes nf4 (4-bit)")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config,
            device_map="auto",
        )
        t0 = time.time()
        results["bnb_nf4"] = ppl_wikitext(m, tokenizer, device)
        results["bnb_nf4_time"] = time.time() - t0
        print(f"  PPL={results['bnb_nf4']:.4f}  ({results['bnb_nf4_time']:.1f}s)")
        mem_bnb = torch.cuda.max_memory_allocated() / 1024**3
        results["bnb_nf4_GB"] = mem_bnb
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  bnb nf4 FAILED: {e}")
        results["bnb_nf4"] = None
        results["bnb_nf4_error"] = str(e)
    
    # 3. HyperRetro GRC k=640 + distill
    print("\n" + "="*60)
    print("3/4: HyperRetro GRC k=640 sink=4 + distill r=8")
    m = AutoModelForCausalLM.from_pretrained(
        "benchmarks/distill_e2e/qwen05b_k640_s4_r8",
        torch_dtype=torch.float32,
    )
    t0 = time.time()
    results["hyperretro_k640_distill"] = ppl_wikitext(m, tokenizer, device)
    results["hyperretro_k640_distill_time"] = time.time() - t0
    print(f"  PPL={results['hyperretro_k640_distill']:.4f}  ({results['hyperretro_k640_distill_time']:.1f}s)")
    del m; torch.cuda.empty_cache()
    
    # 4. HyperRetro GRC k=448 + distill
    print("\n" + "="*60)
    print("4/4: HyperRetro GRC k=448 sink=4 + distill r=8")
    m = AutoModelForCausalLM.from_pretrained(
        "benchmarks/distill_e2e/qwen05b_k448_s4_r8",
        torch_dtype=torch.float32,
    )
    t0 = time.time()
    results["hyperretro_k448_distill"] = ppl_wikitext(m, tokenizer, device)
    results["hyperretro_k448_distill_time"] = time.time() - t0
    print(f"  PPL={results['hyperretro_k448_distill']:.4f}  ({results['hyperretro_k448_distill_time']:.1f}s)")
    del m; torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    baseline = results["baseline_fp32"]
    rows = [
        ("baseline (fp32)", baseline, baseline, "—"),
    ]
    for label, key in [
        ("bnb nf4 4-bit", "bnb_nf4"),
        ("HyperRetro k=640 + distill", "hyperretro_k640_distill"),
        ("HyperRetro k=448 + distill", "hyperretro_k448_distill"),
    ]:
        ppl = results.get(key)
        if ppl is not None:
            delta = (ppl - baseline) / baseline * 100
            rows.append((label, ppl, baseline, f"{delta:+.1f}%"))
        else:
            rows.append((label, None, baseline, "FAILED"))
    
    for label, ppl, base, delta in rows:
        if ppl is not None:
            print(f"  {label:45s} PPL={ppl:.4f}  Δ={delta}")
        else:
            print(f"  {label:45s} {delta}")
    
    (OUT / "baseline_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT / 'baseline_comparison.json'}")


if __name__ == "__main__":
    main()
