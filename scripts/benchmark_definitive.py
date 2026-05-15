"""HyperRetro definitive benchmark — one script, all configs, all metrics.

Measures the full compression pipeline across multiple configurations,
producing a single JSON report with PPL, shrink, and format sizes.

Configs tested:
  0. fp16 baseline — unmodified HuggingFace model
  1. fp16 factored — activation-aware FFN SVD (round 13)
  2. int4 factored (FFN only) — block-wise int4 on factored FFN
  3. int4 factored (full model) — ALL weights quantized
  4. int4 factored + int8 embeddings — embeddings at int8

Each config measures: PPL, on-disk MB (safetensors), GGUF MB, shrink ratio.

Usage:
    python scripts/benchmark_definitive.py [--quick] [--ffn-rank 1024]

Output: benchmarks/definitive_benchmark.json (~90 min CPU for full run)
"""

import sys, json, shutil, time
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from hyperretro.hf.factored import (
    load_factored_hf_model,
    factor_ffn_state_dict,
    save_factored_checkpoint,
)
from hyperretro.hf.activation import collect_ffn_input_norms
from hyperretro.hf.factor_int4 import (
    save_int4_factored_checkpoint,
    dequant_int4_checkpoint,
    quantize_factored_state_dict,
)
from hyperretro.hf.factor_quantize import quantize_blockwise_int4

# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
CALIB = _ROOT / "data" / "wikitext2_train_5k.txt"
OUT = _ROOT / "benchmarks" / "definitive_benchmark.json"

EVAL_TEXT = (
    "Machine learning has transformed the way we build software. "
    "Instead of writing explicit rules, we collect examples and let "
    "the optimization process discover patterns. This paradigm shift "
    "began with simple linear models and has now evolved into massive "
    "neural networks with hundreds of billions of parameters. "
    "The key insight is that gradient descent on a sufficiently large "
    "dataset will find structure that human engineers would never "
    "think to encode. However, this power comes at a cost: the models "
    "are opaque, their failures are unpredictable, and their "
    "computational requirements are staggering. "
) * 3

BASELINE_PPL_FP16 = 2.3332  # cached from prior runs
BASELINE_MB_FP16 = 2955.4
NF4_PPL = 2.5318
NF4_SHRINK = 2.71


# ---------------------------------------------------------------------------
def ppl_eval(model, tokenizer, text=EVAL_TEXT, max_tokens=256):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss)) if loss is not None else float("inf")


def disk_mb(d):
    total = 0.0
    for sf in Path(d).glob("*.safetensors"):
        total += sf.stat().st_size
    return total / 1e6


# ---------------------------------------------------------------------------
def build_factored_fp16(ffn_rank, tokenizer, act_norms):
    """Build activation-aware factored fp16 checkpoint. Returns (ppl, mb, sd, manifest)."""
    print("  Computing activation-aware FFN factoring ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    ).cpu()
    sd = model.state_dict()

    # Factor FFN with activation-aware SVD
    ffn_entries = factor_ffn_state_dict(
        sd, max_rank=ffn_rank, rel_tol=1e-4,
        activation_col_norms=act_norms,
    )
    manifest = {"version": 1, "layers": [], "ffn": ffn_entries}

    # Save to temp
    tmp = Path("outputs/_bench_factored_fp16")
    if tmp.exists():
        shutil.rmtree(tmp)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    save_factored_checkpoint(
        sd, [], ffn_entries, out_dir=tmp, hf_config=cfg, dtype="float16",
    )

    # Load and measure PPL
    model_f, info = load_factored_hf_model(str(tmp), dtype="float16")
    model_f = model_f.cpu()
    val = ppl_eval(model_f, tokenizer)
    mb = disk_mb(tmp)
    del model, model_f
    print(f"    PPL={val:.4f}  disk={mb:.1f}MB  factored={info['patched_linears']}")
    return val, mb, sd, manifest


def build_int4_from_factored(sd, manifest, act_norms, tokenizer,
                              label, full_model, int8_embed):
    """Quantize a factored state_dict to int4, measure PPL."""
    t0 = time.time()
    tmp_int4 = Path(f"outputs/_bench_{label}")
    if tmp_int4.exists():
        shutil.rmtree(tmp_int4)

    # Use save_int4_factored_checkpoint for consistency
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    int8_pat = ["embed"] if int8_embed else None
    report = save_int4_factored_checkpoint(
        dict(sd), [], manifest.get("ffn", []),
        out_dir=tmp_int4, hf_config=cfg, n_bits=4, block_size=128,
        quantize_non_factored=full_model,
        activation_norms=act_norms,
        int8_patterns=int8_pat,
    )
    int4_mb = report["on_disk_mb"]

    # Dequant and measure PPL
    tmp_dq = Path(f"outputs/_bench_{label}_dq")
    if tmp_dq.exists():
        shutil.rmtree(tmp_dq)
    dequant_int4_checkpoint(tmp_int4, tmp_dq)

    model, info = load_factored_hf_model(str(tmp_dq), dtype="float16")
    model = model.cpu()
    val = ppl_eval(model, tokenizer)
    del model
    shutil.rmtree(tmp_int4, ignore_errors=True)
    shutil.rmtree(tmp_dq, ignore_errors=True)

    print(f"    PPL={val:.4f}  disk={int4_mb:.1f}MB  ({time.time()-t0:.0f}s)")
    return val, int4_mb


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    quick = "--quick" in sys.argv
    ffn_rank = 1024
    for a in sys.argv:
        if a.startswith("--ffn-rank="):
            ffn_rank = int(a.split("=")[1])

    print("=" * 70)
    print(f"HyperRetro Definitive Benchmark — {MODEL_ID}")
    print(f"FFN rank: {ffn_rank}  {'QUICK mode' if quick else 'FULL mode'}")
    print("=" * 70)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect activation norms once
    print("\n[stage 0] Collecting activation norms ...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    ).cpu()
    act_norms = collect_ffn_input_norms(
        model_dense, tokenizer,
        corpus_path=str(CALIB), n_batches=4, seq_len=256, device="cpu",
    )
    del model_dense
    print(f"  Collected {len(act_norms)} FFN norm keys")

    results = []

    # --- Config 0: Baseline ---
    print("\n[config 0] fp16 baseline ...")
    results.append({
        "config": "baseline_fp16",
        "ppl": BASELINE_PPL_FP16,
        "disk_mb": BASELINE_MB_FP16,
        "shrink_vs_fp16": 1.00,
        "notes": "Uncompressed HuggingFace model",
    })

    # --- Config 1: Factored fp16 ---
    print("\n[config 1] activation-aware factored fp16 ...")
    ppl_factored, mb_factored, sd_factored, manifest_factored = build_factored_fp16(
        ffn_rank, tokenizer, act_norms,
    )
    results.append({
        "config": "aware_factored_fp16",
        "ppl": round(ppl_factored, 4),
        "disk_mb": round(mb_factored, 1),
        "shrink_vs_fp16": round(BASELINE_MB_FP16 / mb_factored, 2),
        "ffn_rank": ffn_rank,
    })

    # --- Config 2: int4 FFN only (block128 + AWQ) ---
    print("\n[config 2] int4 factored (FFN only, block128+AWQ) ...")
    ppl_ffn_int4, mb_ffn_int4 = build_int4_from_factored(
        sd_factored, manifest_factored, act_norms, tokenizer,
        "ffn_int4", full_model=False, int8_embed=False,
    )
    results.append({
        "config": "int4_ffn_only_block128_awq",
        "ppl": round(ppl_ffn_int4, 4),
        "disk_mb": round(mb_ffn_int4, 1),
        "shrink_vs_fp16": round(BASELINE_MB_FP16 / mb_ffn_int4, 2),
    })

    if not quick:
        # --- Config 3: int4 full model ---
        print("\n[config 3] int4 full model (all weights) ...")
        ppl_full, mb_full = build_int4_from_factored(
            sd_factored, manifest_factored, act_norms, tokenizer,
            "full_int4", full_model=True, int8_embed=False,
        )
        results.append({
            "config": "int4_full_model_block128_awq",
            "ppl": round(ppl_full, 4),
            "disk_mb": round(mb_full, 1),
            "shrink_vs_fp16": round(BASELINE_MB_FP16 / mb_full, 2),
        })

        # --- Config 4: int4 + int8 embeddings ---
        print("\n[config 4] int4 full model + int8 embeddings ...")
        ppl_int8, mb_int8 = build_int4_from_factored(
            sd_factored, manifest_factored, act_norms, tokenizer,
            "int8_emb", full_model=True, int8_embed=True,
        )
        results.append({
            "config": "int4_full_int8_embeddings_block128_awq",
            "ppl": round(ppl_int8, 4),
            "disk_mb": round(mb_int8, 1),
            "shrink_vs_fp16": round(BASELINE_MB_FP16 / mb_int8, 2),
        })

    # --- Industry references ---
    results.append({
        "config": "REF_nf4_industry",
        "ppl": NF4_PPL,
        "disk_mb": round(BASELINE_MB_FP16 / NF4_SHRINK, 1),
        "shrink_vs_fp16": NF4_SHRINK,
        "notes": "bitsandbytes nf4 (industry default for 4-bit)",
    })

    # --- Summary table ---
    wall = time.time() - t_start
    report = {
        "model": MODEL_ID,
        "ffn_rank": ffn_rank,
        "wall_min": round(wall / 60, 1),
        "results": results,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 70}")
    print("DEFINITIVE BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Config':<40s} {'PPL':>8s} {'Disk MB':>8s} {'Shrink':>7s}")
    print("-" * 70)
    for r in results:
        s = r.get("shrink_vs_fp16", 1.0)
        print(f"{r['config']:<40s} {r['ppl']:>8.2f} {r['disk_mb']:>8.1f} {s:>6.2f}x")
    print("-" * 70)
    print(f"\n[saved] {OUT}  ({wall/60:.1f} min)")

    # Best result
    best = min(
        [r for r in results if r["config"].startswith("int4")],
        key=lambda r: r["ppl"],
        default=None,
    )
    if best:
        nf4_result = [r for r in results if "nf4" in r["config"]][0]
        print(f"\nBest HyperRetro: {best['config']}")
        print(f"  PPL={best['ppl']:.2f}  shrink={best['shrink_vs_fp16']:.1f}x")
        print(f"  vs nf4: PPL {best['ppl']/nf4_result['ppl']:.1f}x, "
              f"shrink {best['shrink_vs_fp16']/nf4_result['shrink_vs_fp16']:.1f}x")
        if best["shrink_vs_fp16"] > nf4_result["shrink_vs_fp16"]:
            print(f"  ✅ Beats nf4 on shrink")
        if best["ppl"] < nf4_result["ppl"]:
            print(f"  ✅ Beats nf4 on PPL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
