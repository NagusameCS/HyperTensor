"""Aggregate canonical-pack rank_sweep into a decode-length sweep snippet.

Honest framing: the existing pack has 128 vs 256 generated tokens for 4 prompt
templates (coding/reasoning/factual/creative) at k = 0/1024/1536/2048. We
report mean decode tok/s by (condition, generated_tokens) so the paper can
discuss how throughput holds across the two decode lengths we measured.

Reads:  benchmarks/whitepaper_pack_20260427_121815/rank_sweep_raw.csv
Writes: docs/data/decode_length_sweep.csv
        docs/data/decode_length_sweep.tex
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "benchmarks", "whitepaper_pack_20260427_121815", "rank_sweep_raw.csv")
OUT_CSV = os.path.join(ROOT, "docs", "data", "decode_length_sweep.csv")
OUT_TEX = os.path.join(ROOT, "docs", "data", "decode_length_sweep.tex")


def parse_label(label: str):
    parts = label.split("_")
    if parts[0] == "baseline":
        return ("baseline", 0)
    if parts[0] == "grc":
        return ("grc", int(parts[1][1:]))
    return None


buckets: dict[tuple, list[float]] = defaultdict(list)
with open(SRC, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        parsed = parse_label(row["label"])
        if not parsed:
            continue
        kind, rank = parsed
        try:
            tps = float(row["decode_tps"])
            tokens = int(row["tokens"])
        except Exception:
            continue
        buckets[(kind, rank, tokens)].append(tps)

ranks = sorted({r for k, r, _ in buckets if k == "grc"})
token_lengths = sorted({t for _, _, t in buckets})

rows = []
for tlen in token_lengths:
    base = buckets[("baseline", 0, tlen)]
    b_mean = st.mean(base)
    b_std = st.stdev(base) if len(base) > 1 else 0.0
    rec = {
        "decode_tokens": tlen,
        "baseline_mean": round(b_mean, 3),
        "baseline_std": round(b_std, 3),
        "n_baseline": len(base),
    }
    for r in ranks:
        samples = buckets[("grc", r, tlen)]
        m = st.mean(samples)
        s = st.stdev(samples) if len(samples) > 1 else 0.0
        rec[f"grc_k{r}_mean"] = round(m, 3)
        rec[f"grc_k{r}_std"] = round(s, 3)
        rec[f"grc_k{r}_ratio"] = round(m / b_mean, 4)
    rows.append(rec)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {OUT_CSV}")

# Compact LaTeX table: decode tokens × baseline / k=1024 / k=1536 / k=2048
def fmt_pair(mean, std):
    return f"${mean:.2f} \\pm {std:.2f}$"

body_lines = []
for r in rows:
    parts = [
        str(r["decode_tokens"]),
        fmt_pair(r["baseline_mean"], r["baseline_std"]),
    ]
    for k in ranks:
        parts.append(fmt_pair(r[f"grc_k{k}_mean"], r[f"grc_k{k}_std"]))
    body_lines.append(" & ".join(parts) + " \\\\")
body = "\n".join(body_lines)

cols = "r" + "r" * (1 + len(ranks))
header_extra = " & ".join(f"$k\\!=\\!{k}$" for k in ranks)
tex = f"""% auto-generated from benchmarks/whitepaper_pack_20260427_121815/rank_sweep_raw.csv
% by scripts/build_decode_length_snippet.py -- do not hand-edit.
\\begin{{tabular}}{{{cols}}}
\\toprule
Decode tok & Baseline & {header_extra} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
"""
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"Wrote {OUT_TEX}")
