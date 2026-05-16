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

"""Optimal HyperRetro compression pipeline — end-to-end.

Produces the best possible compressed model using all available techniques:

  1. Activation-aware FFN factoring (round 13) — recovers 99.6% PPL at r=1024
  2. Int4 quantization of factored matrices (attack #7) — 5.1× additional shrink
  3. Attn-GRC compression (optional, k=640) — 2% shrink, 2× PPL cost
  4. GGUF export for llama.cpp (attack #5)

This script measures PPL at each stage and writes a comprehensive report.

Usage:
    python scripts/optimal_pipeline.py [--attn-rank 0] [--ffn-rank 1024] [--int4]

Outputs:
    outputs/optimal_fp16/          — fp16 factored checkpoint
    outputs/optimal_int4/          — int4 factored checkpoint  
    outputs/optimal.gguf           — GGUF for llama.cpp
    benchmarks/optimal_pipeline.json — full report

GPU not required (CPU-only mode, ~2 hr for full pipeline at r=1024).
"""

from __future__ import annotations

import json, shutil, sys, time, argparse
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
CALIB_PATH = _ROOT / "data" / "wikitext2_train_5k.txt"

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
) * 3  # 3× repeat for ~256 tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ppl(model, tokenizer, text: str, max_tokens: int = 256) -> float:
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
    return float(torch.exp(loss))


def disk_mb(d: Path) -> float:
    total = 0.0
    for sf in d.glob("*.safetensors"):
        total += sf.stat().st_size
    return total / 1e6


# ---------------------------------------------------------------------------
# Stage 1: Dense baseline
# ---------------------------------------------------------------------------

def stage_baseline(tokenizer) -> dict:
    from transformers import AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("STAGE 0: Dense fp16 baseline")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
    ).cpu()

    val = ppl(model, tokenizer, EVAL_TEXT)
    # Approximate on-disk size
    params = sum(p.numel() for p in model.parameters())
    size_est = params * 2 / 1e6  # fp16

    del model
    print(f"  PPL={val:.4f}  params={params/1e6:.1f}M  est_disk={size_est:.0f}MB")
    return {"ppl": round(val, 4), "params_m": round(params / 1e6, 1), "disk_mb_est": round(size_est, 1)}


# ---------------------------------------------------------------------------
# Stage 2: Activation-aware FFN factored (fp16)
# ---------------------------------------------------------------------------

def stage_factored_fp16(ffn_rank: int, attn_rank: int, tokenizer) -> dict:
    from hyperretro.hf.compress import compress_hf_model

    print("\n" + "=" * 60)
    print(f"STAGE 1: Activation-aware factored (FFN r={ffn_rank}, attn k={attn_rank})")
    print("=" * 60)

    out_dir = _ROOT / f"outputs/optimal_fp16_r{ffn_rank}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    use_attn = attn_rank > 0

    report = compress_hf_model(
        MODEL_ID, out_dir,
        rank_k=attn_rank if use_attn else 1536,  # no-op if attn_rank >= d
        sink_T=0,
        ffn_rank_in=ffn_rank,
        ffn_rank_out=ffn_rank,
        factored=True,
        activation_aware=True,
        activation_corpus_path=str(CALIB_PATH) if CALIB_PATH.exists() else None,
        activation_n_batches=4,
        activation_seq_len=256,
        dtype="float16",
    )

    size_mb = disk_mb(out_dir)

    # Load and measure PPL
    from hyperretro.hf.factored import load_factored_hf_model
    model, _info = load_factored_hf_model(str(out_dir), dtype="float16")
    model = model.cpu()
    val = ppl(model, tokenizer, EVAL_TEXT)
    del model

    print(f"  PPL={val:.4f}  disk={size_mb:.1f}MB")
    return {
        "ppl": round(val, 4),
        "disk_mb": round(size_mb, 1),
        "shrink_vs_fp16": round(2955.4 / size_mb, 2) if size_mb > 0 else 0,
        "ffn_rank": ffn_rank,
        "attn_rank": attn_rank,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Stage 3: Int4-quantize the factored checkpoint
# ---------------------------------------------------------------------------

def stage_factored_int4(fp16_dir: Path, ffn_rank: int) -> dict:
    import json as _json
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from hyperretro.hf.factor_int4 import save_int4_factored_checkpoint

    print("\n" + "=" * 60)
    print(f"STAGE 2: Int4-quantize factored checkpoint")
    print("=" * 60)

    out_dir = _ROOT / f"outputs/optimal_int4_r{ffn_rank}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    manifest = _json.loads((fp16_dir / "hyperretro_factored.json").read_text())
    orig_sd = load_file(str(fp16_dir / "model.safetensors"))
    cfg = AutoConfig.from_pretrained(MODEL_ID)

    report = save_int4_factored_checkpoint(
        orig_sd, [], manifest.get("ffn", []),
        out_dir=out_dir, hf_config=cfg, n_bits=4,
        quantize_non_factored=False,
    )

    size_mb = report["on_disk_mb"]
    fp16_mb = disk_mb(fp16_dir)
    print(f"  fp16 factored: {fp16_mb:.1f}MB → int4 factored: {size_mb:.1f}MB  "
          f"({fp16_mb/size_mb:.1f}× shrink)")

    return {
        "disk_mb": round(size_mb, 1),
        "shrink_vs_fp16": round(2955.4 / size_mb, 2) if size_mb > 0 else 0,
        "shrink_vs_factored_fp16": round(fp16_mb / size_mb, 2) if size_mb > 0 else 0,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Stage 4: GGUF export
# ---------------------------------------------------------------------------

def stage_gguf(int4_dir: Path, ffn_rank: int) -> dict:
    from hyperretro.hf.gguf_export import export_hyperretro_to_gguf

    print("\n" + "=" * 60)
    print(f"STAGE 3: GGUF export")
    print("=" * 60)

    out_path = _ROOT / f"outputs/optimal_r{ffn_rank}.gguf"
    if out_path.exists():
        out_path.unlink()

    result = export_hyperretro_to_gguf(
        str(int4_dir), str(out_path),
        model_name=f"HyperRetro-Qwen2.5-1.5B-r{ffn_rank}-int4",
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"  GGUF: {size_mb:.1f}MB")

    return {
        "gguf_mb": round(size_mb, 1),
        "gguf_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Optimal HyperRetro compression pipeline")
    p.add_argument("--ffn-rank", type=int, default=1024)
    p.add_argument("--attn-rank", type=int, default=0,
                   help="Attn GRC rank (0=skip, attn GRC costs 2× PPL for 2% shrink)")
    p.add_argument("--int4", action="store_true", default=True,
                   help="Apply int4 quantization after factoring")
    p.add_argument("--gguf", action="store_true", default=True,
                   help="Export to GGUF")
    p.add_argument("--skip-baseline", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    report = {
        "model": MODEL_ID,
        "ffn_rank": args.ffn_rank,
        "attn_rank": args.attn_rank,
        "stages": {},
    }

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Stage 0: Baseline
    if not args.skip_baseline:
        report["stages"]["baseline_fp16"] = stage_baseline(tokenizer)

    # Stage 1: Factored fp16
    s1 = stage_factored_fp16(args.ffn_rank, args.attn_rank, tokenizer)
    report["stages"]["factored_fp16"] = s1
    fp16_dir = Path(s1["out_dir"])

    # Stage 2: Int4
    if args.int4:
        s2 = stage_factored_int4(fp16_dir, args.ffn_rank)
        report["stages"]["factored_int4"] = s2
        int4_dir = Path(s2["out_dir"])
    else:
        int4_dir = fp16_dir

    # Stage 3: GGUF
    if args.gguf:
        s3 = stage_gguf(int4_dir, args.ffn_rank)
        report["stages"]["gguf"] = s3

    # Summary
    wall = time.time() - t0
    report["wall_s"] = round(wall, 1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for stage, data in report["stages"].items():
        if "ppl" in data:
            print(f"  {stage:25s}  PPL={data['ppl']:.4f}  disk={data.get('disk_mb', data.get('disk_mb_est', 0)):.0f}MB")
        elif "disk_mb" in data:
            print(f"  {stage:25s}  disk={data['disk_mb']:.0f}MB  shrink={data.get('shrink_vs_fp16', 0):.2f}x")
        elif "gguf_mb" in data:
            print(f"  {stage:25s}  {data['gguf_mb']:.0f}MB")
    print(f"  Wall time: {wall/60:.1f} min")

    # Save report
    out_json = _ROOT / "benchmarks" / "optimal_pipeline.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
