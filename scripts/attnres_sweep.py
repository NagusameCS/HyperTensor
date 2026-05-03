#!/usr/bin/env python3
"""
AttnRes  GRC interaction sweep for Paper C §attnres.

Tests the structural prediction: at moderate compression (k/d ∈ [0.4, 0.6]),
AttnRes should be a wash; at aggressive compression (k/d < 0.3), the narrowed
block-summary subspace causes softmax noise and degrades acceptance.

Sweeps: k ∈ {0.25, 0.35, 0.45, 0.55, 0.65}  d  with AttnRes on/off.
For SmolLM2-135M (d=576): k ∈ {144, 202, 259, 317, 374}.
For Llama-3.1-8B (d=4096): k ∈ {1024, 1434, 1843, 2253, 2662}.

Measures: decode throughput (tok/s), speculative acceptance rate α.

Usage:
  python scripts/attnres_sweep.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --d-model 576 \
    --out benchmarks/attnres_sweep

Outputs:
  benchmarks/attnres_sweep/
    attnres_raw.csv
    attnres_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    "Explain how a transformer attention mechanism works.",
    "Write a Python function that sorts a list using merge sort.",
    "What is the difference between TCP and UDP?",
    "Describe the process of photosynthesis in plants.",
    "How does public-key cryptography work?",
]

THRESH = 0.45
BATCH = 4
N_TOKENS = 64
CTX_SIZE = 512
REPS = 3
COOLDOWN_SEC = 10


def detect_exe() -> Optional[Path]:
    for p in [
        ROOT / "build_host" / "geodessical2.exe",
        ROOT / "build_host" / "geodessical.exe",
    ]:
        if p.exists():
            return p
    return None


def run_decode(exe: Path, model: Path, k: Optional[int], attnres: bool,
               attnres_strength: float, prompt: str) -> dict:
    """Run decode and parse throughput. Uses --ott-full --no-verifier for
    compressed runs (needed for parseable output with --axex-compress).
    Baseline runs include speculative mode for acceptance rate measurement."""
    args = [
        str(exe), str(model),
        "--ctx-size", str(CTX_SIZE),
        "-p", prompt, "-n", str(N_TOKENS), "--temp", "0",
    ]
    # For baseline (k=None), run speculative decode to get alpha
    # For compressed (k not None), use --ott-full --no-verifier for parseable TPS
    if k is None:
        args += [
            "--ott-full", "--ott-speculative",
            "--ott-spec-batch", str(BATCH),
            "--ott-spec-thresh", str(THRESH),
        ]
    else:
        args += ["--ott-full", "--no-verifier"]
        args += ["--axex-compress", "--axex-compress-rank", str(k)]
    if attnres:
        args += ["--attnres", "--attnres-strength", str(attnres_strength)]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120,
                              encoding='utf-8', errors='replace')
        stdout = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tok_per_s": 0.0, "alpha": 0.0}
    except Exception as e:
        return {"status": f"error: {e}", "tok_per_s": 0.0, "alpha": 0.0}

    tps = None
    alpha = None

    for line in stdout.split("\n"):
        m = re.search(r"([\d.]+)\s*tok(?:ens?)?/s", line)
        if m and tps is None:
            tps = float(m.group(1))
        # Also try TPS= format from --ott-full output
        m = re.search(r"TPS=([\d.]+)", line)
        if m and tps is None:
            tps = float(m.group(1))
        m = re.search(r"acceptance_rate=([\d.]+)%", line)
        if m:
            alpha = float(m.group(1))

    return {
        "status": "ok" if tps else "parse_error",
        "tok_per_s": tps or 0.0,
        "alpha": alpha if alpha is not None else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="AttnRes  GRC Interaction Sweep")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--d-model", type=int, required=True,
                    help="Model dimension d (e.g. 576 for SmolLM2, 4096 for Llama-8B)")
    ap.add_argument("--out", default="benchmarks/attnres_sweep")
    ap.add_argument("--exe", default=None, help="Path to geodessical binary")
    ap.add_argument("--attnres-strength", type=float, default=0.35,
                    help="AttnRes injection strength")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_SEC)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    exe = Path(args.exe) if args.exe else detect_exe()
    if not exe or not exe.exists():
        print("ERROR: geodessical binary not found.", file=sys.stderr)
        sys.exit(2)
    model = Path(args.model)
    if not model.exists():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        sys.exit(2)

    # Compute k values as fractions of d
    fractions = [0.25, 0.35, 0.45, 0.55, 0.65]
    k_values = [int(f * args.d_model) for f in fractions]

    print(f"Model d={args.d_model}")
    print(f"k fractions: {fractions}")
    print(f"k values: {k_values}")
    print(f"AttnRes strength: {args.attnres_strength}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "attnres_raw.csv"
    conditions = []

    # Build condition list: (k_label, k_value, attnres_bool)
    conditions.append(("baseline", None, False))
    for k, f in zip(k_values, fractions):
        for attnres in (False, True):
            label = f"k={k}({f:.2f}d)_attnres={attnres}"
            conditions.append((label, k, attnres))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "k", "attnres", "prompt_idx", "rep",
                         "tok_per_s", "alpha", "status"])

        for cond_label, k, attnres in conditions:
            for pi, prompt in enumerate(PROMPTS):
                for rep in range(args.reps):
                    if args.dry_run:
                        print(f"[dry] {cond_label} p={pi} r={rep}")
                        continue

                    result = run_decode(exe, model, k, attnres,
                                      args.attnres_strength, prompt)
                    row = [cond_label, k if k else "baseline", attnres,
                           pi, rep, result["tok_per_s"], result["alpha"],
                           result["status"]]
                    writer.writerow(row)
                    f.flush()

                    alpha_str = f"α={result['alpha']:.1f}%" if result["alpha"] > 0 else "α=N/A"
                    print(f"  {cond_label:>30s}  p={pi} r={rep}  "
                          f"{result['tok_per_s']:.1f} tok/s  {alpha_str}  "
                          f"{'' if result['status']=='ok' else ''}", flush=True)

                time.sleep(args.cooldown)

    if args.dry_run:
        print("[dry-run] done.")
        return

    # Summarise
    from collections import defaultdict
    by_cond = defaultdict(lambda: {"tps": [], "alpha": []})

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tps = float(row["tok_per_s"])
            alpha = float(row["alpha"])
            if tps > 0:
                by_cond[row["condition"]]["tps"].append(tps)
            if alpha > 0:
                by_cond[row["condition"]]["alpha"].append(alpha)

    summary = {"d_model": args.d_model, "attnres_strength": args.attnres_strength,
               "conditions": {}}
    print("\n=== Results ===")
    print(f"{'condition':>35s}  {'tok/s':>8s}  {'σ':>6s}  {'α%':>6s}  {'tok/s vs base':>12s}")

    base_tps = statistics.mean(by_cond.get("baseline", {}).get("tps", [1.0]))

    for cond_label, _k, _attnres in conditions:
        if cond_label not in by_cond:
            continue
        tps_vals = by_cond[cond_label]["tps"]
        alpha_vals = by_cond[cond_label]["alpha"]
        mean_tps = statistics.mean(tps_vals)
        stdev_tps = statistics.stdev(tps_vals) if len(tps_vals) > 1 else 0.0
        mean_alpha = statistics.mean(alpha_vals) if alpha_vals else 0.0
        ratio = mean_tps / base_tps if base_tps > 0 else 0.0

        summary["conditions"][cond_label] = {
            "mean_tok_per_s": round(mean_tps, 1),
            "stdev_tok_per_s": round(stdev_tps, 1),
            "mean_alpha": round(mean_alpha, 1),
            "ratio_vs_baseline": round(ratio, 4),
        }

        print(f"{cond_label:>35s}  {mean_tps:8.1f}  {stdev_tps:6.1f}  "
              f"{mean_alpha:6.1f}  {ratio:12.4f}")

    with open(out_dir / "attnres_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {out_dir / 'attnres_summary.json'}")


if __name__ == "__main__":
    main()
