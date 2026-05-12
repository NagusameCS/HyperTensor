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
PAPER I --- Bootstrap confidence intervals on the n=8 throughput sweep.

Background
----------
Paper I reports an empirical optimum at k* on a per-token-throughput sweep
with n=8 prompts per k condition. Peer review correctly flagged that no
confidence intervals were attached to the headline ratio, and that with
n=8 a non-parametric bootstrap is the right tool.

This script:

1. Reads a long-format CSV with at least these columns:
       k, prompt_idx, rep, tok_s
   (Default: benchmarks/paper_a_multi_k/multi_k_results.csv.)
2. Bootstraps each per-k throughput sample (resampling prompts with
   replacement, B=10000 by default) to produce a 95% percentile CI on
   the mean tok/s and on the speedup ratio relative to the "baseline"
   condition.
3. Reports k*, the speedup at k*, the 95% CI on that speedup, and the
   distribution over which k value is selected as k* across bootstrap
   resamples (i.e. the uncertainty in the LOCATION of the optimum).
4. Writes a JSON summary to benchmarks/paper_i_bootstrap_results.json.

This is a small, deterministic, CPU-only script. It does NOT re-run the
hardware sweep --- it only attaches statistical hygiene to the existing
measurements. The goal is to support a sentence in the Paper I revision
of the form:
    "k*=1536 (95% CI [1280, 1792]) yielded a 1.10x speedup
     (95% CI [1.07, 1.13])."
which can replace the current point-estimate phrasing.

Usage
-----
    python scripts/paper_i_bootstrap_ci.py
    python scripts/paper_i_bootstrap_ci.py --csv benchmarks/some_other.csv --B 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "benchmarks" / "paper_a_multi_k" / "multi_k_results.csv"
OUT_FILE = ROOT / "benchmarks" / "paper_i_bootstrap_results.json"


def read_csv(path: Path) -> dict[str, list[float]]:
    """Return per-condition list of tok_s values."""
    by_k: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tok_s = float(row["tok_s"])
            except (KeyError, ValueError):
                continue
            k = row.get("k", "?").strip()
            by_k[k].append(tok_s)
    return dict(by_k)


def bootstrap_mean_ci(samples: list[float], B: int, alpha: float, rng: np.random.Generator
                      ) -> tuple[float, float, float]:
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    boots = arr[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def bootstrap_ratio_ci(num: list[float], den: list[float], B: int, alpha: float,
                       rng: np.random.Generator) -> tuple[float, float, float]:
    a = np.asarray(num, dtype=float)
    b = np.asarray(den, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx_a = rng.integers(0, a.size, size=(B, a.size))
    idx_b = rng.integers(0, b.size, size=(B, b.size))
    boots = a[idx_a].mean(axis=1) / np.clip(b[idx_b].mean(axis=1), 1e-9, None)
    point = float(a.mean() / max(b.mean(), 1e-9))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def bootstrap_kstar(by_k: dict[str, list[float]], baseline_key: str, B: int,
                    rng: np.random.Generator) -> dict:
    """Distribution of argmax-k across bootstrap resamples.

    For each bootstrap iteration we resample each condition's prompts
    independently, recompute the per-condition mean speedup vs baseline,
    and record which k wins. The distribution over winning k is a
    non-parametric CI on the LOCATION of the optimum.
    """
    if baseline_key not in by_k:
        return {"error": f"baseline_key {baseline_key!r} not found",
                "available_keys": sorted(by_k.keys())}
    base = np.asarray(by_k[baseline_key], dtype=float)
    other_keys = [k for k in by_k if k != baseline_key]
    other_arrs = {k: np.asarray(by_k[k], dtype=float) for k in other_keys}

    wins = defaultdict(int)
    speedups_at_winner = []
    for _ in range(B):
        b_idx = rng.integers(0, base.size, size=base.size)
        b_mean = base[b_idx].mean()
        best_k, best_sp = None, -np.inf
        for k, arr in other_arrs.items():
            idx = rng.integers(0, arr.size, size=arr.size)
            sp = arr[idx].mean() / max(b_mean, 1e-9)
            if sp > best_sp:
                best_sp = sp
                best_k = k
        wins[best_k] += 1
        speedups_at_winner.append(float(best_sp))

    total = sum(wins.values())
    return {
        "argmax_distribution": {k: wins[k] / total for k in sorted(wins, key=lambda x: _k_sort_key(x))},
        "winning_speedup_mean": float(np.mean(speedups_at_winner)),
        "winning_speedup_p2_5": float(np.quantile(speedups_at_winner, 0.025)),
        "winning_speedup_p97_5": float(np.quantile(speedups_at_winner, 0.975)),
    }


def _k_sort_key(k: str):
    try:
        return (0, int(k))
    except ValueError:
        return (1, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    by_k = read_csv(csv_path)
    rng = np.random.default_rng(args.seed)

    per_condition = {}
    for k in sorted(by_k, key=_k_sort_key):
        s = by_k[k]
        m, lo, hi = bootstrap_mean_ci(s, args.B, args.alpha, rng)
        per_condition[k] = {
            "n": len(s),
            "mean_tok_s": m,
            "ci_low": lo,
            "ci_high": hi,
            "ci_half_width": (hi - lo) / 2 if not np.isnan(lo) else float("nan"),
        }

    speedup_per_k = {}
    if args.baseline in by_k:
        for k in sorted(by_k, key=_k_sort_key):
            if k == args.baseline:
                continue
            sp, lo, hi = bootstrap_ratio_ci(by_k[k], by_k[args.baseline], args.B, args.alpha, rng)
            speedup_per_k[k] = {"speedup_mean": sp, "ci_low": lo, "ci_high": hi}

    kstar = bootstrap_kstar(by_k, args.baseline, args.B, rng)

    summary = {
        "meta": {
            "csv": str(csv_path),
            "baseline_key": args.baseline,
            "B_bootstrap": args.B,
            "alpha": args.alpha,
            "seed": args.seed,
            "conditions": sorted(by_k, key=_k_sort_key),
            "n_per_condition": {k: len(v) for k, v in by_k.items()},
        },
        "per_condition_mean_tok_s_with_ci": per_condition,
        "speedup_vs_baseline_with_ci": speedup_per_k,
        "kstar_bootstrap": kstar,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"# wrote {out_path}\n")
    print(f"# per-condition tok/s (95% CI from B={args.B} bootstrap):")
    for k, row in per_condition.items():
        print(f"   k={k:<10}  n={row['n']:>2}  mean={row['mean_tok_s']:8.2f}  "
              f"95%CI=[{row['ci_low']:8.2f},{row['ci_high']:8.2f}]")
    if speedup_per_k:
        print("\n# speedup vs baseline (95% CI):")
        for k, row in speedup_per_k.items():
            print(f"   k={k:<10}  speedup={row['speedup_mean']:5.3f}  "
                  f"95%CI=[{row['ci_low']:5.3f},{row['ci_high']:5.3f}]")
    print("\n# k* bootstrap (distribution over which k wins each resample):")
    if "argmax_distribution" in kstar:
        for k, p in sorted(kstar["argmax_distribution"].items(), key=lambda kv: -kv[1])[:10]:
            print(f"   k={k:<10}  P(k*=k) = {p:.3f}")
        print(f"\n   winning speedup at k*: mean={kstar['winning_speedup_mean']:.3f}  "
              f"95%CI=[{kstar['winning_speedup_p2_5']:.3f},{kstar['winning_speedup_p97_5']:.3f}]")
    else:
        print(f"   (skipped: {kstar})")


if __name__ == "__main__":
    sys.exit(main())
