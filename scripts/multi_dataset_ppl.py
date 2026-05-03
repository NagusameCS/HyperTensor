#!/usr/bin/env python3
"""
Multi-Dataset PPL Evaluator for Tier 2 Task-Level Benchmark Suite.

Evaluates WikiText-2, C4, and PTB perplexity on GRC-compressed models
across multiple compression ranks.  Extends Paper A's single-dataset PPL
measurement to multiple datasets, establishing the generality (or lack
thereof) of the PPL-vs-rank trade-off.

This is the first component of Paper F ("GRC Task-Level Impact").
Task-level evals (MMLU, GSM8K, HumanEval) follow in task_bench.py.

Usage:
  python scripts/multi_dataset_ppl.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --ranks 64,128,256,512,1024 \
    --datasets wikitext2,c4,ptb \
    --out benchmarks/multi_ppl

Requires: datasets to be pre-downloaded as .txt files in data/
  data/wikitext2_test.txt
  data/c4_val_5k.txt
  data/ptb_test.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def detect_exe() -> Optional[Path]:
    for p in [
        ROOT / "build_host" / "geodessical2.exe",
        ROOT / "build_host" / "geodessical.exe",
    ]:
        if p.exists():
            return p
    return None


def run_ppl(exe: Path, model: Path, k: Optional[int],
            dataset_path: str, ctx_size: int = 2048) -> dict:
    """Run perplexity evaluation via the geodessical binary's built-in PPL mode."""
    args = [str(exe), str(model), "--perplexity", dataset_path,
            "--ctx-size", str(ctx_size)]
    if k is not None:
        args += [
            "--axex-compress", "--axex-attn-only", "--axex-skip-o",
            "--axex-weight-pca", "--axex-compress-rank", str(k),
        ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=600,
                              encoding='utf-8', errors='replace')
        stdout = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "ppl": float("inf")}
    except Exception as e:
        return {"status": f"error: {e}", "ppl": float("inf")}

    # Parse PPL from output
    ppl = None
    for line in stdout.split("\n"):
        # Various PPL output formats
        for pat in [
            r"perplexity\s*[:=]\s*([\d.]+)",
            r"ppl\s*[:=]\s*([\d.]+)",
            r"final\s+ppl\s*[:=]\s*([\d.]+)",
            r"\[PPL\]\s*([\d.]+)",
        ]:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                ppl = float(m.group(1))
                break
        if ppl is not None:
            break

    return {
        "status": "ok" if ppl else "parse_error",
        "ppl": ppl or float("inf"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Multi-Dataset PPL Evaluator (Tier 2 / Paper F)"
    )
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/multi_ppl")
    ap.add_argument("--ranks", default="64,128,256,512,1024")
    ap.add_argument("--datasets", default="wikitext2,c4,ptb")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--ctx-size", type=int, default=2048)
    args = ap.parse_args()

    exe = Path(args.exe) if args.exe else detect_exe()
    if not exe or not exe.exists():
        print("ERROR: geodessical binary not found.", file=sys.stderr)
        sys.exit(2)
    model = Path(args.model)
    ranks = [int(x) for x in args.ranks.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate dataset files
    dataset_paths = {}
    for ds in datasets:
        candidates = [
            ROOT / "data" / f"{ds}_test.txt",
            ROOT / "data" / f"{ds}_val.txt",
            ROOT / "data" / f"{ds}_val_5k.txt",
            ROOT / "data" / f"{ds}.txt",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = str(c.resolve())
                break
        if found:
            dataset_paths[ds] = found
        else:
            print(f"WARNING: dataset '{ds}' not found in data/. Skipping.")
            print(f"  Tried: {[str(c) for c in candidates]}")

    if not dataset_paths:
        print("ERROR: no dataset files found.", file=sys.stderr)
        print("  Download WikiText-2 test set to data/wikitext2_test.txt")
        sys.exit(2)

    print(f"Model: {model}")
    print(f"Datasets: {list(dataset_paths.keys())}")
    print(f"Ranks: {ranks}")
    print()

    # Run sweep
    csv_path = out_dir / "multi_ppl.csv"
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "dataset", "ppl", "status"])

        # Baseline
        for ds_name, ds_path in dataset_paths.items():
            print(f"  baseline  {ds_name}  ...", end=" ", flush=True)
            result = run_ppl(exe, model, None, ds_path, args.ctx_size)
            row = ["baseline", ds_name, result["ppl"], result["status"]]
            writer.writerow(row)
            rows.append(row)
            f.flush()
            status = "" if result["status"] == "ok" else ""
            print(f"ppl={result['ppl']:.2f}  {status}", flush=True)

        # Compressed ranks
        for k in ranks:
            for ds_name, ds_path in dataset_paths.items():
                print(f"  k={k:>5d}  {ds_name:>12s}  ...", end=" ", flush=True)
                t0 = time.time()
                result = run_ppl(exe, model, k, ds_path, args.ctx_size)
                elapsed = time.time() - t0
                row = [str(k), ds_name, result["ppl"], result["status"]]
                writer.writerow(row)
                rows.append(row)
                f.flush()
                status = "" if result["status"] == "ok" else ""
                print(f"ppl={result['ppl']:.2f}  {elapsed:.0f}s  {status}", flush=True)

    # Summarize
    from collections import defaultdict
    by_k_ds = defaultdict(dict)
    for row in rows:
        k_label, ds, ppl_str, _status = row
        ppl = float(ppl_str)
        if ppl < float("inf"):
            by_k_ds[k_label][ds] = ppl

    # Baseline PPL per dataset
    baseline_ppl = by_k_ds.get("baseline", {})

    print("\n=== Multi-Dataset PPL ===")
    header = f"{'k':>8s}"
    for ds in dataset_paths:
        header += f"  {ds:>12s}"
    header += f"  {'Δ mean %':>10s}"
    print(header)
    print("-" * len(header))

    for k_label in ["baseline"] + [str(r) for r in ranks]:
        line = f"{k_label:>8s}"
        deltas = []
        for ds in dataset_paths:
            ppl = by_k_ds.get(k_label, {}).get(ds, float("inf"))
            base = baseline_ppl.get(ds, 1.0)
            delta = (ppl - base) / base * 100 if base > 0 else 0.0
            deltas.append(delta)
            line += f"  {ppl:12.2f}"
        if k_label != "baseline":
            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
            line += f"  {mean_delta:+.1f}%"
        else:
            line += f"  {'---':>10s}"
        print(line)

    # Write summary
    summary = {
        "model": str(model),
        "datasets": list(dataset_paths.keys()),
        "ranks": ranks,
        "baseline_ppl": baseline_ppl,
        "results": {k: v for k, v in by_k_ds.items()},
    }
    with open(out_dir / "multi_ppl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {csv_path}")
    print(f"[done] {out_dir / 'multi_ppl_summary.json'}")


if __name__ == "__main__":
    main()
