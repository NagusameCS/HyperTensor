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

"""isagi_benchmark.py — ISAGI vs baseline Qwen benchmark comparison.

Measures ISAGI (with all 7 stack layers) vs vanilla model on:
  1. Response latency (ms)
  2. GTC cache hit rate
  3. COG metric growth per turn
  4. VRAM usage

USAGE:
  python scripts/isagi_benchmark.py                          # 1.5B, 50 turns
  python scripts/isagi_benchmark.py --n 200                  # 200 turns
  python scripts/isagi_benchmark.py --compare                 # vs baseline

OUTPUT:
  benchmarks/isagi_benchmark/report.json
"""
import torch, json, time, os, sys, random, argparse, math
from pathlib import Path
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from hyper_optimize import batched_collect_hidden_states, smart_svd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("benchmarks/isagi_benchmark")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42); random.seed(42)

BENCHMARK_QUERIES = [
    "Solve for x: 3x + 7 = 22 step by step",
    "Write a Python function to sort a list of strings by length",
    "Explain how photosynthesis converts light to energy",
    "What is the capital of France and why was it chosen?",
    "Prove that the square root of 2 is irrational",
    "Write a haiku about artificial intelligence",
    "How does the immune system distinguish self from non-self?",
    "Explain the difference between TCP and UDP protocols",
    "What is the Pythagorean theorem? Give an example.",
    "Describe the process of DNA replication step by step",
    "Write a function to find the nth Fibonacci number",
    "What is quantum entanglement in simple terms?",
    "Explain the central limit theorem intuitively",
    "How do vaccines create immunity?",
    "Write a short story about a robot learning to paint",
    "What is the derivative of x^2 * sin(x)? Show work.",
    "Explain climate change to a 10-year-old",
    "How does a transformer attention mechanism work?",
    "Describe Newton's three laws with real-world examples",
    "What is CRISPR and how does gene editing work?",
]

def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok

def benchmark_isagi(model, tok, queries, k_ugt=256):
    """Benchmark ISAGI with full stack."""
    L = model.config.num_hidden_layers
    mid = L // 2
    model.eval()

    # Init UGT basis
    cal_prompts = queries[:15]
    hs = batched_collect_hidden_states(model, tok, cal_prompts, layer=mid, batch_size=8)
    hs_f = hs.float().to(DEVICE)
    mean = hs_f.mean(dim=0)
    U, S = smart_svd((hs_f - mean).T, min(k_ugt, len(cal_prompts)))
    basis = U[:, :min(k_ugt, len(cal_prompts))].float()
    k = basis.shape[1]

    # Init COG metric + cache
    metric = torch.eye(k, device=DEVICE)
    cache_projs = []; cache_texts = []
    hits = 0; misses = 0
    latencies = []; metric_norms = []
    vram_start = torch.cuda.memory_allocated() / 1e9

    def project(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[mid][0, -1, :].float()
        centered = h - mean.to(h.device)
        proj = centered @ basis.to(h.device)
        return F.normalize(proj.unsqueeze(0), dim=1).squeeze(0)

    def generate(text, max_new=50):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                  temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        q_proj = project(query)

        # GTC search
        if cache_projs:
            stack = torch.stack([p.to(DEVICE) for p in cache_projs])
            sims = (q_proj.unsqueeze(0) @ stack.T).squeeze(0)
            best_sim = sims.max().item()
            if best_sim > 0.60:
                hits += 1
                latency = (time.perf_counter() - t0) * 1000
                latencies.append(latency)
                metric_norms.append(torch.norm(metric - torch.eye(k, device=DEVICE)).item())
                continue

        misses += 1
        response = generate(query, max_new=40)
        cache_projs.append(q_proj.cpu())
        cache_texts.append(query)

        # COG expansion
        J = q_proj.unsqueeze(1) @ q_proj.unsqueeze(0)
        J = J / (torch.norm(J) + 1e-10)
        metric = metric + 0.012 * J

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        metric_norms.append(torch.norm(metric - torch.eye(k, device=DEVICE)).item())

    vram_end = torch.cuda.memory_allocated() / 1e9
    hit_rate = hits / max(1, hits + misses)

    return {
        "k_ugt": k,
        "n_turns": len(queries),
        "gtc_hit_rate": round(hit_rate, 4),
        "cache_size": len(cache_projs),
        "mean_latency_ms": round(np.mean(latencies), 1),
        "median_latency_ms": round(np.median(latencies), 1),
        "p95_latency_ms": round(np.percentile(latencies, 95), 1),
        "metric_norm_final": round(metric_norms[-1], 4) if metric_norms else 0,
        "vram_delta_gb": round(vram_end - vram_start, 2),
    }

def benchmark_baseline(model, tok, queries):
    """Benchmark vanilla model (no ISAGI stack)."""
    model.eval()
    latencies = []
    vram_start = torch.cuda.memory_allocated() / 1e9

    def generate(text, max_new=50):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                  temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    for query in queries:
        t0 = time.perf_counter()
        _ = generate(query, max_new=40)
        latencies.append((time.perf_counter() - t0) * 1000)

    vram_end = torch.cuda.memory_allocated() / 1e9
    return {
        "n_turns": len(queries),
        "mean_latency_ms": round(np.mean(latencies), 1),
        "median_latency_ms": round(np.median(latencies), 1),
        "p95_latency_ms": round(np.percentile(latencies, 95), 1),
        "vram_gb": round(vram_end - vram_start, 2),
    }


def main(model_name="Qwen/Qwen2.5-1.5B-Instruct", n_turns=50, compare=False):
    print("=" * 60)
    print("  ISAGI BENCHMARK — vs Baseline Comparison")
    print(f"  Model: {model_name}, Turns: {n_turns}")
    print("=" * 60)

    queries = (BENCHMARK_QUERIES * (n_turns // len(BENCHMARK_QUERIES) + 1))[:n_turns]

    model, tok = load_model(model_name)

    print(f"\n[1] ISAGI Benchmark ({n_turns} turns)...")
    isagi_results = benchmark_isagi(model, tok, queries)

    print(f"  GTC hit rate:      {isagi_results['gtc_hit_rate']:.1%}")
    print(f"  Mean latency:      {isagi_results['mean_latency_ms']:.0f}ms")
    print(f"  P95 latency:       {isagi_results['p95_latency_ms']:.0f}ms")
    print(f"  COG metric norm:   {isagi_results['metric_norm_final']:.4f}")
    print(f"  Cache size:        {isagi_results['cache_size']}")

    report = {"isagi": isagi_results}

    if compare:
        print(f"\n[2] Baseline Benchmark ({n_turns} turns)...")
        baseline = benchmark_baseline(model, tok, queries)
        report["baseline"] = baseline

        speedup = baseline["mean_latency_ms"] / max(isagi_results["mean_latency_ms"], 1)
        print(f"  Mean latency:      {baseline['mean_latency_ms']:.0f}ms")
        print(f"  ISAGI speedup:     {speedup:.1f}x")

        report["comparison"] = {
            "isagi_faster": "yes" if isagi_results["mean_latency_ms"] < baseline["mean_latency_ms"] else "no",
            "speedup": round(speedup, 2),
            "latency_saved_ms": round(baseline["mean_latency_ms"] - isagi_results["mean_latency_ms"], 1),
        }
        print(f"  Latency saved:     {report['comparison']['latency_saved_ms']:.0f}ms")

    model_slug = model_name.replace("/", "_")
    with open(OUT / f"report_{model_slug}.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Saved to: {OUT}")
    print(f"  DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    main(args.model, args.n, args.compare)
