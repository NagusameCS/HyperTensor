"""Aggregate canonical-pack rank_sweep into a per-rank Pareto snippet for Paper A.
Reads:  benchmarks/whitepaper_pack_20260427_121815/rank_sweep_raw.csv
Writes: docs/data/rank_pareto.csv
        docs/data/rank_pareto.tex
"""
from __future__ import annotations

import csv
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "benchmarks", "whitepaper_pack_20260427_121815", "rank_sweep_raw.csv")
OUT_CSV = os.path.join(ROOT, "docs", "data", "rank_pareto.csv")
OUT_TEX = os.path.join(ROOT, "docs", "data", "rank_pareto.tex")


def parse_label(label: str):
    # Examples:
    #   baseline_coding_128
    #   grc_k1024_coding_128
    parts = label.split("_")
    if parts[0] == "baseline":
        return ("baseline", 0)
    if parts[0] == "grc":
        return ("grc", int(parts[1][1:]))  # strip leading 'k'
    return None


buckets = defaultdict(list)
with open(SRC, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        parsed = parse_label(row["label"])
        if not parsed:
            continue
        kind, rank = parsed
        try:
            tps = float(row["decode_tps"])
        except Exception:
            continue
        buckets[(kind, rank)].append(tps)

ranks = sorted({r for k, r in buckets if k == "grc"})
baseline_samples = buckets[("baseline", 0)]
b_mean = st.mean(baseline_samples)
b_std = st.stdev(baseline_samples) if len(baseline_samples) > 1 else 0.0

rows = [{
    "rank": 0,
    "decode_mean": round(b_mean, 3),
    "decode_std": round(b_std, 3),
    "n": len(baseline_samples),
    "ratio_to_baseline": 1.0,
}]
for r in ranks:
    samples = buckets[("grc", r)]
    m = st.mean(samples)
    s = st.stdev(samples) if len(samples) > 1 else 0.0
    rows.append({
        "rank": r,
        "decode_mean": round(m, 3),
        "decode_std": round(s, 3),
        "n": len(samples),
        "ratio_to_baseline": round(m / b_mean, 4),
    })

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {OUT_CSV}")

# Emit \input-able LaTeX table.
def fmt(row):
    if row["rank"] == 0:
        label = "baseline"
    else:
        label = f"$k\\!=\\!{row['rank']}$"
    return f"{label} & ${row['decode_mean']:.2f} \\pm {row['decode_std']:.2f}$ & {row['n']} & {row['ratio_to_baseline']:.4f} \\\\"

body = "\n".join(fmt(r) for r in rows)
tex = f"""% auto-generated from benchmarks/whitepaper_pack_20260427_121815/rank_sweep_raw.csv
% by scripts/build_rank_pareto_snippet.py -- do not hand-edit.
\\begin{{tabular}}{{lrrr}}
\\toprule
Condition & Decode tok/s & $n$ & Ratio vs.\\ baseline \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
"""
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"Wrote {OUT_TEX}")
