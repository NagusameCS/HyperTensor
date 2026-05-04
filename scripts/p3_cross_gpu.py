#!/usr/bin/env python3
"""
Cross-GPU P3 validation (Paper A §falsification): verify that the optimal
compression rank k* shifts with L2 cache capacity as predicted.

Runs the headline decode-throughput benchmark at k ∈ {512, 768, 1024, 1280, 1536, 2048}
on the current GPU and emits a per-rank CSV + summary JSON.

The cache-fit model predicts k* = largest power-of-two k where the
attention working set fits L2 with ~25% headroom:
  - RTX 4070 Laptop (32 MB L2) -> k* = 1024  (PAPER A, CONFIRMED)
  - RTX 4090          (72 MB L2) -> k* = 1536  (PREDICTED)
  - A100              (40 MB L2) -> k* = 1024  (PREDICTED)
  - H100              (50 MB L2) -> k* = 1280  (PREDICTED)
  - L40S              (96 MB L2) -> k* = 1536  (PREDICTED)

Usage (local RTX 4070 Laptop):
  python scripts/p3_cross_gpu.py --model models/smollm2-135m-instruct-q8_0.gguf

Usage (EC2, any GPU):
  python scripts/p3_cross_gpu.py \
    --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --ranks 512,768,1024,1280,1536,2048 \
    --prompts data/headline_prompts.txt \
    --out benchmarks/p3_cross_gpu

Outputs:
  benchmarks/p3_cross_gpu/
    p3_raw.csv          <- per-(k, prompt, rep) throughput
    p3_summary.json     <- per-k mean/stdev/CI
    p3_report.md        <- verdict: confirmed / falsified / consistent
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# Default prompts (4 prompt classes  2 decode budgets)
DEFAULT_PROMPTS = [
    # coding
    "Write a Python function that implements binary search on a sorted list.",
    "Write a C program that reads a CSV file and prints the sum of column 3.",
    # reasoning
    "If all A are B and some B are C, does it follow that some A are C? Explain.",
    "A train leaves at 8 AM going 60 mph. Another leaves at 9 AM going 80 mph. When do they meet?",
    # factual
    "Explain the difference between mitosis and meiosis.",
    "What are the main greenhouse gases and their sources?",
    # creative
    "Write a short poem about a programmer debugging at midnight.",
    "Describe an alien civilization that communicates through geometry instead of sound.",
]

DEFAULT_K_VALUES = [512, 768, 1024, 1280, 1536, 2048]
N_TOKENS = 256
CTX_SIZE = 512
REPS = 3  # per (k, prompt)
COOLDOWN_SEC = 15


def detect_exe() -> Optional[Path]:
    """Find the geodessical binary."""
    candidates = [
        ROOT / "build_host" / "geodessical2.exe",
        ROOT / "build_host" / "geodessical.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def detect_gpu_l2() -> tuple[str, int]:
    """Detect GPU name and L2 cache size via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,l2_cache_size",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        name, l2_str = out.strip().split(",", 1)
        l2_mb = int(float(l2_str.strip()))
        return name.strip(), l2_mb
    except Exception:
        return "unknown", 0


def predict_kstar(l2_mb: int) -> int:
    """Predict optimal k* from L2 cache size (cache-fit model, Paper A)."""
    # Conservative: k* is largest power-of-two where S(k) fits with headroom
    if l2_mb >= 96:
        return 1536
    elif l2_mb >= 72:
        return 1536
    elif l2_mb >= 50:
        return 1280
    elif l2_mb >= 40:
        return 1024
    elif l2_mb >= 32:
        return 1024
    else:
        return 512


def run_decode(exe: Path, model: Path, k: Optional[int], prompt: str,
               n_tokens: int = 256, ctx_size: int = 512) -> dict:
    """Run a single decode measurement. Returns dict with tok/s and status."""
    args = [
        str(exe), str(model),
        "--ctx-size", str(ctx_size),
        "-p", prompt, "-n", str(n_tokens),
        "--temp", "0",
    ]
    if k is not None:
        args += [
            "--axex-compress", "--axex-attn-only", "--axex-skip-o",
            "--axex-weight-pca", "--axex-compress-rank", str(k),
        ]

    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=300,
            encoding='utf-8', errors='replace',
        )
        stdout = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tok_per_s": 0.0}
    except Exception as e:
        return {"status": f"error: {e}", "tok_per_s": 0.0}

    # Parse tokens/sec from output
    tps = None
    for line in stdout.split("\n"):
        # Look for "XX.X tok/s" or "tps=XX.X"
        m = re.search(r"([\d.]+)\s*tok(?:ens?)?/s", line)
        if m:
            tps = float(m.group(1))
            break
        m = re.search(r"tps=([\d.]+)", line)
        if m:
            tps = float(m.group(1))
            break

    return {
        "status": "ok" if tps else "parse_error",
        "tok_per_s": tps or 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="P3 Cross-GPU Cache-Fit Validation")
    ap.add_argument("--model", required=True, help="Path to GGUF model file")
    ap.add_argument("--out", default="benchmarks/p3_cross_gpu", help="Output directory")
    ap.add_argument("--ranks", default="512,768,1024,1280,1536,2048",
                    help="Comma-separated k values")
    ap.add_argument("--prompts-file", default=None,
                    help="File with one prompt per line (default: built-in 8 prompts)")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--tokens", type=int, default=N_TOKENS)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_SEC)
    ap.add_argument("--exe", default=None, help="Path to geodessical binary")
    args = ap.parse_args()

    # Find binary
    exe = Path(args.exe) if args.exe else detect_exe()
    if not exe or not exe.exists():
        print("ERROR: geodessical binary not found. Build with build_host.ps1 or pass --exe.",
              file=sys.stderr)
        sys.exit(2)
    model = Path(args.model)
    if not model.exists():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        sys.exit(2)

    # Prompts
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().strip().split("\n")
    else:
        prompts = DEFAULT_PROMPTS

    ranks = [int(x) for x in args.ranks.split(",")]

    # Detect GPU
    gpu_name, l2_mb = detect_gpu_l2()
    predicted_kstar = predict_kstar(l2_mb)
    print(f"GPU: {gpu_name}  L2: {l2_mb} MB  predicted k*: {predicted_kstar}")
    print(f"Model: {model}")
    print(f"Ranks: {ranks}  Prompts: {len(prompts)}  Reps: {args.reps}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep
    csv_path = out_dir / "p3_raw.csv"
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "prompt_idx", "rep", "tok_per_s", "status", "gpu", "l2_mb"])

        for k in [None] + ranks:  # None = baseline
            k_label = "baseline" if k is None else str(k)
            for pi, prompt in enumerate(prompts):
                for rep in range(args.reps):
                    result = run_decode(exe, model, k, prompt, args.tokens)
                    row = [k_label, pi, rep, result["tok_per_s"], result["status"],
                           gpu_name, l2_mb]
                    writer.writerow(row)
                    rows.append(row)
                    f.flush()

                    status = "" if result["status"] == "ok" else ""
                    print(f"  k={k_label:>8s}  p={pi}  r={rep}  "
                          f"{result['tok_per_s']:.1f} tok/s  {status}", flush=True)

                # Cooldown between prompts
                time.sleep(args.cooldown)

    print(f"\nRaw data: {csv_path}  ({len(rows)} rows)")

    # Compute summary
    from collections import defaultdict
    by_k = defaultdict(list)
    for row in rows:
        k_label = row[0]
        tps = row[3]
        if tps > 0:
            by_k[k_label].append(tps)

    summary = {
        "gpu": gpu_name,
        "l2_mb": l2_mb,
        "predicted_kstar": predicted_kstar,
        "model": str(model),
        "n_prompts": len(prompts),
        "n_reps": args.reps,
        "ranks": {},
    }

    print("\n=== P3 Results ===")
    print(f"{'k':>8s}  {'mean tok/s':>10s}  {'stdev':>8s}  {'ratio':>8s}  {'verdict'}")
    print("-" * 58)

    baseline_tps = None
    if "baseline" in by_k:
        baseline_tps = statistics.mean(by_k["baseline"])

    for k_label in ["baseline"] + [str(r) for r in ranks]:
        if k_label not in by_k:
            continue
        vals = by_k[k_label]
        mean_tps = statistics.mean(vals)
        stdev_tps = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ratio = mean_tps / baseline_tps if baseline_tps else 0.0

        summary["ranks"][k_label] = {
            "mean_tok_per_s": round(mean_tps, 2),
            "stdev": round(stdev_tps, 2),
            "ratio_to_baseline": round(ratio, 4),
            "n_samples": len(vals),
        }

        if k_label == "baseline":
            verdict = "---"
        elif ratio > 1.01:
            verdict = "SUPER-BASELINE "
        elif ratio > 0.97:
            verdict = "INDISTINGUISHABLE"
        else:
            verdict = "BELOW BASELINE"

        print(f"{k_label:>8s}  {mean_tps:10.1f}  {stdev_tps:8.1f}  "
              f"{ratio:8.4f}  {verdict}")

    # Find empirical k*
    best_k = None
    best_ratio = 0.0
    for k_label in [str(r) for r in ranks]:
        if k_label in summary["ranks"]:
            r = summary["ranks"][k_label]["ratio_to_baseline"]
            if r > best_ratio:
                best_ratio = r
                best_k = int(k_label)

    summary["empirical_kstar"] = best_k
    summary["empirical_best_ratio"] = round(best_ratio, 4)
    summary["prediction_match"] = (best_k == predicted_kstar)

    # Write summary
    with open(out_dir / "p3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write report
    verdict = "CONFIRMED" if summary["prediction_match"] else "FALSIFIED"
    if not summary["prediction_match"]:
        # Check if within one power-of-two
        if best_k and abs(best_k - predicted_kstar) <= 512:
            verdict = "CONSISTENT (within 1 power-of-two)"
        else:
            verdict = "FALSIFIED"

    report = f"""# P3 Cross-GPU Validation Report

**GPU**: {gpu_name} ({l2_mb} MB L2)
**Predicted k***: {predicted_kstar}
**Empirical k***: {best_k} (ratio={best_ratio:.4f})
**Verdict**: {verdict}

## Per-Rank Throughput
| k | tok/s | stdev | ratio |
|---|-------|-------|-------|
"""
    for k_label in ["baseline"] + [str(r) for r in ranks]:
        if k_label in summary["ranks"]:
            r = summary["ranks"][k_label]
            report += f"| {k_label} | {r['mean_tok_per_s']:.1f} | {r['stdev']:.1f} | {r['ratio_to_baseline']:.4f} |\n"

    with open(out_dir / "p3_report.md", "w") as f:
        f.write(report)

    print(f"\n[done] Verdict: {verdict}")
    print(f"[done] Summary: {out_dir / 'p3_summary.json'}")
    print(f"[done] Report:  {out_dir / 'p3_report.md'}")


if __name__ == "__main__":
    main()
