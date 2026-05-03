"""Build quality snippet (perplexity + per-prompt decode CI) from canonical pack.

Reads:
  benchmarks/whitepaper_pack_20260427_121815/ci_ppl_5run.csv
  benchmarks/whitepaper_pack_20260427_121815/ci_pack_summary.csv
Writes:
  docs/data/quality_anchor.csv
  docs/data/quality_anchor.tex
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPL_SRC = os.path.join(ROOT, "benchmarks", "whitepaper_pack_20260427_121815", "ci_ppl_5run.csv")
CI_SRC = os.path.join(ROOT, "benchmarks", "whitepaper_pack_20260427_121815", "ci_pack_summary.csv")
OUT_CSV = os.path.join(ROOT, "docs", "data", "quality_anchor.csv")
OUT_TEX = os.path.join(ROOT, "docs", "data", "quality_anchor.tex")

ppl: dict[str, list[float]] = defaultdict(list)
with open(PPL_SRC, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            ppl[row["mode"]].append(float(row["ppl"]))
        except Exception:
            pass

base = ppl["baseline"]
grc = ppl["grc_k2048"]
b_mean = st.mean(base)
b_std = st.stdev(base) if len(base) > 1 else 0.0
g_mean = st.mean(grc)
g_std = st.stdev(grc) if len(grc) > 1 else 0.0
delta_pct = 100.0 * (g_mean - b_mean) / b_mean

cases = []
with open(CI_SRC, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cases.append(row)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "condition", "mean", "std_or_ci95", "n", "pct_vs_baseline"])
    w.writerow(["wikitext_ppl", "baseline", round(b_mean, 4), round(b_std, 4), len(base), 100.0])
    w.writerow(["wikitext_ppl", "grc_k2048", round(g_mean, 4), round(g_std, 4), len(grc), round(100.0 + delta_pct, 3)])
    for c in cases:
        w.writerow([
            "decode_tps_" + c["case_name"],
            "baseline",
            c["baseline_decode_mean"],
            c["baseline_decode_ci95"],
            5,
            100.0,
        ])
        w.writerow([
            "decode_tps_" + c["case_name"],
            "grc_k2048",
            c["grc_decode_mean"],
            c["grc_decode_ci95"],
            5,
            c["decode_pct_of_baseline_mean"],
        ])
print(f"Wrote {OUT_CSV}")

case_lines = []
for c in cases:
    case_lines.append(
        " & ".join([
            c["case_name"].replace("_", "\\_"),
            f"${float(c['baseline_decode_mean']):.2f} \\pm {float(c['baseline_decode_ci95']):.2f}$",
            f"${float(c['grc_decode_mean']):.2f} \\pm {float(c['grc_decode_ci95']):.2f}$",
            f"{float(c['decode_pct_of_baseline_mean']):.1f}\\%",
        ]) + " \\\\"
    )
case_body = "\n".join(case_lines)

tex = f"""% auto-generated from benchmarks/whitepaper_pack_20260427_121815/{{ci_ppl_5run,ci_pack_summary}}.csv
% by scripts/build_quality_anchor_snippet.py -- do not hand-edit.
\\begin{{tabular}}{{lrrr}}
\\toprule
Metric & Baseline & GRC ($k\\!=\\!2048$) & Ratio \\\\
\\midrule
WikiText perplexity ($n\\!=\\!{len(base)}$) & ${b_mean:.4f} \\pm {b_std:.4f}$ & ${g_mean:.4f} \\pm {g_std:.4f}$ & ${1.0 + delta_pct/100.0:.4f}\\times$ \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textit{{Decode tok/s (mean $\\pm$ 95\\% CI, $n\\!=\\!5$)}}}} \\\\
{case_body}
\\bottomrule
\\end{{tabular}}
"""
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"Wrote {OUT_TEX}")
