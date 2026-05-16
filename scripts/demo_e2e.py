#!/usr/bin/env python3
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

"""HyperRetro v0.2.0 — Comprehensive End-to-End Demo.

Demonstrates the complete pipeline:
  1. Compress a HuggingFace model via GRC
  2. Distill it (MSE + KL behavioral-residue)
  3. Run GPU speculative decoding
  4. Compare against bitsandbytes nf4
  5. Show per-layer compression statistics

Usage::

    python scripts/demo_e2e.py [--model Qwen/Qwen2.5-0.5B] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure hyperretro is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_DIR = Path("benchmarks/demo_e2e")
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Step 1: Compress
# ---------------------------------------------------------------------------

def demo_compress(model_id: str, k: int, device: str):
    header(f"Step 1: GRC Compression (k={k})")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hyperretro.hf.compress import compress_state_dict, CompressConfig

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
    )
    sd = model.state_dict()
    
    # Show original size
    total_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    attn_params = 0
    for name in sd:
        if any(p in name for p in ("q_proj", "k_proj", "v_proj", "c_attn", "qkv_proj")):
            attn_params += sd[name].numel()
    print(f"  Total params: {total_params:,}")
    print(f"  Attention params: {attn_params:,} ({attn_params/total_params*100:.1f}%)")

    t0 = time.time()
    cfg = CompressConfig(rank_k=k, sink_T=4, dtype="float32")
    stats = compress_state_dict(sd, cfg)
    elapsed = time.time() - t0
    print(f"  Compression time: {elapsed:.1f}s")
    print(f"  Layers compressed: {len(stats)}")

    # Show per-layer stats
    frob_q = [v["frob_relerr_q"] for v in stats.values()]
    print(f"  Mean Frobenius relative error (Q): {np.mean(frob_q):.4f}")
    print(f"  Max  Frobenius relative error (Q): {np.max(frob_q):.4f}")

    # Save
    out = DEMO_DIR / f"compressed_k{k}"
    model.load_state_dict(sd)
    model.save_pretrained(out, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(out)
    except Exception:
        pass
    print(f"  Saved to {out}")

    del model; torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Step 2: Distill (MSE + KL)
# ---------------------------------------------------------------------------

def demo_distill(model_id: str, k: int, corpus: str | None):
    header(f"Step 2: GRC Light Distillation (k={k})")

    from hyperretro.hf.distill import distill_hf_model

    for loss_type, label in [("mse", "MSE"), ("kl", "KL (behavioral-residue)")]:
        out = DEMO_DIR / f"distilled_k{k}_{loss_type}"
        if out.exists():
            print(f"  {label}: already exists at {out}, skipping")
            continue

        print(f"  Running {label} distillation...")
        t0 = time.time()
        report = distill_hf_model(
            model_id, str(out),
            rank_k=k, sink_T=4,
            lora_rank=8, lora_alpha=16,
            steps=200, batch_size=2, seq_len=128,
            corpus_path=corpus,
            device="cuda", dtype="float32",
            loss_type=loss_type, kl_temperature=4.0,
        )
        elapsed = time.time() - t0
        print(f"  {label}: {elapsed:.1f}s, "
              f"{report['n_layers_distilled']} layers distilled")
        import torch; torch.cuda.empty_cache()

    return {
        "mse": DEMO_DIR / f"distilled_k{k}_mse",
        "kl": DEMO_DIR / f"distilled_k{k}_kl",
    }


# ---------------------------------------------------------------------------
# Step 3: Speculative Decoding Bench
# ---------------------------------------------------------------------------

def demo_speculative(model_id: str, compressed_path: Path, k: int):
    header(f"Step 3: Speculative Decoding (k={k})")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hyperretro.vllm.draft import CompressedDrafter, DraftConfig

    prompt = (
        "The history of artificial intelligence began in antiquity "
        "with myths and stories of artificial beings"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Verifier: original model
    verifier = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
    ).to(device)
    verifier.eval()

    # Drafter: compressed model
    drafter_model = AutoModelForCausalLM.from_pretrained(
        compressed_path, torch_dtype=torch.float32,
    ).to(device)
    drafter_model.eval()
    drafter = CompressedDrafter(drafter_model, cfg=DraftConfig(n_drafts=4))

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    _ = drafter.propose(ids[:, :10], n_drafts=1)
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    n_drafts = 4
    n_trials = 10
    times = []
    for i in range(n_trials):
        prefix = ids[:, :min(10 + i, ids.size(1))]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        draft_ids, confs = drafter.propose(prefix, n_drafts=n_drafts)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    median_ms = float(np.median(times))
    tokens_per_sec = n_drafts / (median_ms / 1000)
    print(f"  Draft tokens per proposal: {n_drafts}")
    print(f"  Median ms/propose: {median_ms:.1f}")
    print(f"  Draft tokens/sec: {tokens_per_sec:.0f}")
    print(f"  Sample draft: {tok.decode(draft_ids.tolist())!r}")

    del verifier, drafter_model
    torch.cuda.empty_cache()

    return {"median_ms": median_ms, "tokens_per_sec": tokens_per_sec}


# ---------------------------------------------------------------------------
# Step 4: PPL Comparison
# ---------------------------------------------------------------------------

def demo_ppl(model_id: str, paths: dict[str, Path]):
    header("Step 4: Perplexity Comparison (WikiText-2)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok.encode(text)
    seq_len = 256
    stride = 128
    max_samples = 100

    def ppl_quick(model):
        model.eval()
        model = model.to(device)
        nlls = []
        prev = 0
        for i, begin in enumerate(range(0, len(enc) - seq_len, stride)):
            if i >= max_samples:
                break
            end = begin + seq_len
            if prev > begin:
                begin = prev
            if begin >= end:
                continue
            ids = torch.tensor(enc[begin:end], dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(ids, labels=ids)
                nlls.append(out.loss.item() * seq_len)
            prev = end
        if not nlls:
            return float("inf")
        return float(np.exp(sum(nlls) / (len(nlls) * seq_len)))

    results = {}
    
    # Baseline
    print("  Measuring baseline...")
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    results["baseline"] = ppl_quick(m)
    print(f"    baseline: {results['baseline']:.2f}")
    del m; torch.cuda.empty_cache()

    # Distilled models
    for label, p in paths.items():
        if p.exists():
            print(f"  Measuring {label}...")
            m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
            results[label] = ppl_quick(m)
            print(f"    {label}: {results[label]:.2f}")
            del m; torch.cuda.empty_cache()

    # bitsandbytes nf4 (if available)
    try:
        from transformers import BitsAndBytesConfig
        print("  Measuring bitsandbytes nf4...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="auto",
        )
        results["bnb_nf4"] = ppl_quick(m)
        print(f"    bnb_nf4: {results['bnb_nf4']:.2f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"    bnb_nf4: SKIPPED ({e})")

    # Summary
    print(f"\n  {'Method':<30s} {'PPL':>8s}  {'Δ base':>8s}")
    print(f"  {'-'*48}")
    base = results.get("baseline", 1)
    for label in ["baseline", "distilled_mse", "distilled_kl", "bnb_nf4"]:
        if label in results:
            delta = (results[label] - base) / base * 100
            print(f"  {label:<30s} {results[label]:8.2f}  {delta:+7.1f}%")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="HyperRetro v0.2.0 E2E Demo")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--k", type=int, default=640)
    p.add_argument("--corpus", default="data/wikitext2_train_5k.txt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-compress", action="store_true")
    p.add_argument("--skip-distill", action="store_true")
    p.add_argument("--skip-spec", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print("  HyperRetro v0.2.0 — End-to-End Demo")
    print(f"  Model: {args.model}  |  k={args.k}  |  Device: {args.device}")
    print("=" * 70)

    results = {"model": args.model, "k": args.k}

    # Step 1: Compress
    compressed = DEMO_DIR / f"compressed_k{args.k}"
    if not args.skip_compress or not compressed.exists():
        compressed = demo_compress(args.model, args.k, args.device)
    else:
        print(f"\n  Using existing compressed model: {compressed}")

    # Step 2: Distill
    distilled_paths = {
        "mse": DEMO_DIR / f"distilled_k{args.k}_mse",
        "kl": DEMO_DIR / f"distilled_k{args.k}_kl",
    }
    if not args.skip_distill:
        distilled_paths = demo_distill(args.model, args.k, args.corpus)

    # Step 3: Speculative
    if not args.skip_spec:
        spec = demo_speculative(args.model, compressed, args.k)
        results["spec"] = spec

    # Step 4: PPL
    if not args.skip_ppl:
        ppl_paths = {
            "distilled_mse": distilled_paths["mse"],
            "distilled_kl": distilled_paths["kl"],
        }
        ppl = demo_ppl(args.model, ppl_paths)
        results["ppl"] = ppl

    # Final summary
    header("Demo Complete")
    print(json.dumps(results, indent=2, default=str))
    
    report_path = DEMO_DIR / "demo_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
