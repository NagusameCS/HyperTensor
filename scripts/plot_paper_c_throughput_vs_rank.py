"""Generate compression-cost curve for Paper C from the canonical pack.

Reads:  benchmarks/whitepaper_pack_20260427_121815/rank_sweep_raw.csv
Writes: docs/figures/paper-c/throughput_vs_rank.png
        docs/figures/paper-c/throughput_vs_rank.pdf

The plot shows decode tok/s vs compression rank k for every prompt class,
with the baseline drawn as a horizontal band.  This is the empirical
"throughput under compression" curve that Paper C's closed-form throughput
model (Eq. 1-2) predicts: a peak at the cache-fit rank then a soft decline
as L2 pressure rises.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "benchmarks", "whitepaper_pack_20260427_121815", "rank_sweep_raw.csv")
OUT_DIR = os.path.join(ROOT, "docs", "figures", "paper-c")
os.makedirs(OUT_DIR, exist_ok=True)


def parse(label: str):
    parts = label.split("_")
    if parts[0] == "baseline":
        return ("baseline", 0)
    if parts[0] == "grc":
        return ("grc", int(parts[1][1:]))
    return None


# (kind, rank) -> {prompt -> [tps,...]}
data: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
with open(SRC, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        parsed = parse(row["label"])
        if not parsed:
            continue
        kind, rank = parsed
        # Recover prompt class from label, e.g. "grc_k1024_coding_128"
        toks = row["label"].split("_")
        if kind == "baseline":
            prompt_class = toks[1]
        else:
            prompt_class = toks[2]
        try:
            data[(kind, rank)][prompt_class].append(float(row["decode_tps"]))
        except Exception:
            continue

ranks = sorted({r for k, r in data if k == "grc"})
classes = sorted({p for d in data.values() for p in d.keys()})

# Baseline reference
base = data[("baseline", 0)]
base_per_class = {c: sum(base[c]) / len(base[c]) for c in base}
base_grand_mean = sum(base_per_class.values()) / len(base_per_class)

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)

# Baseline band: min-max across classes
b_vals = list(base_per_class.values())
ax.axhspan(min(b_vals), max(b_vals), color="0.85", alpha=0.6, zorder=0,
           label=f"Baseline range ({min(b_vals):.1f}-{max(b_vals):.1f} tok/s)")
ax.axhline(base_grand_mean, color="0.4", linestyle="--", linewidth=1, zorder=1,
           label=f"Baseline mean ({base_grand_mean:.2f} tok/s)")

colors = {"coding": "#c0392b", "reasoning": "#2d6be4", "factual": "#1a7a4a", "creative": "#8e44ad"}
markers = {"coding": "o", "reasoning": "s", "factual": "^", "creative": "D"}

for c in classes:
    xs, ys = [], []
    for r in ranks:
        samples = data[("grc", r)].get(c, [])
        if samples:
            xs.append(r)
            ys.append(sum(samples) / len(samples))
    ax.plot(xs, ys, color=colors.get(c, "k"), marker=markers.get(c, "x"),
            markersize=6, linewidth=1.5, label=c)

ax.set_xlabel("Compression rank $k$")
ax.set_ylabel("Decode throughput (tok/s)")
ax.set_title("Throughput under compression (Llama-3.1-8B Q4\\_K\\_M, RTX 4070 Laptop)")
ax.set_xticks(ranks)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)

# Annotate the cache-fit peak
peak_rank = ranks[0]
peak_y = max(sum(data[("grc", peak_rank)][c]) / len(data[("grc", peak_rank)][c])
             for c in classes if data[("grc", peak_rank)].get(c))
ax.annotate(f"cache-fit peak\n($k\\!=\\!{peak_rank}$)",
            xy=(peak_rank, peak_y), xytext=(peak_rank + 100, peak_y + 1.5),
            arrowprops=dict(arrowstyle="->", color="#c0392b"),
            fontsize=8, color="#c0392b")

plt.tight_layout()
png_path = os.path.join(OUT_DIR, "throughput_vs_rank.png")
pdf_path = os.path.join(OUT_DIR, "throughput_vs_rank.pdf")
plt.savefig(png_path, dpi=140)
plt.savefig(pdf_path)
print(f"Wrote {png_path}")
print(f"Wrote {pdf_path}")
