"""Extract per-layer Eckart-Young vs GRC bound data for Paper A appendix.

Reads docs/figures/eckart_young_bound.json (per-layer / per-rank / per-matrix
relative Frobenius errors) and emits a compact LaTeX-ready table plus a
matplotlib bar chart of the EY excess-factor by layer at k=512 (the only
rank where GRC is non-degenerate enough to compute a finite ratio).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "figures" / "eckart_young_bound.json"
OUT_DIR = ROOT / "docs" / "figures" / "paper-a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    data = json.loads(SRC.read_text())
    layers = data["layers"]
    summary = data["summary"]

    # Build flat table.
    rows = []
    for L, ranks in layers.items():
        for k, slots in ranks.items():
            row = {"layer": int(L), "k": int(k)}
            for slot, vals in slots.items():
                row[f"{slot}_grc"] = vals["grc_relF2"]
                row[f"{slot}_ey"] = vals["eckart_young_relF2"]
            rows.append(row)
    rows.sort(key=lambda r: (r["layer"], r["k"]))

    # CSV.
    csv_path = OUT_DIR / "ey_per_layer.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("layer,k,Q_grc,Q_ey,K_grc,K_ey,V_grc,V_ey\n")
        for r in rows:
            f.write(
                f"{r['layer']},{r['k']},"
                f"{r['Q_grc']:.6f},{r['Q_ey']:.6f},"
                f"{r['K_grc']:.6f},{r['K_ey']:.6f},"
                f"{r['V_grc']:.6f},{r['V_ey']:.6f}\n"
            )
    print(f"Wrote {csv_path}")

    # Plot: per-layer per-slot relF2 at k=1024 (the cache-fit optimum).
    layer_ids = sorted(int(L) for L in layers)
    slots = ("Q", "K", "V")
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    width = 0.12
    xs = np.arange(len(layer_ids))
    colors_grc = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c"}
    colors_ey = {"Q": "#aec7e8", "K": "#ffbb78", "V": "#98df8a"}
    for i, slot in enumerate(slots):
        grc = [layers[str(L)]["1024"][slot]["grc_relF2"] for L in layer_ids]
        ey = [layers[str(L)]["1024"][slot]["eckart_young_relF2"] for L in layer_ids]
        ax.bar(xs + (2 * i - 2) * width, grc, width, label=f"{slot} (GRC)",
               color=colors_grc[slot], edgecolor="black", linewidth=0.4)
        ax.bar(xs + (2 * i - 1) * width, ey, width, label=f"{slot} (EY)",
               color=colors_ey[slot], edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layer_ids])
    ax.set_ylabel("relative $\\|\\cdot\\|_F^2$ error")
    ax.set_title("Per-layer GRC vs Eckart--Young oracle at $k=1024$ "
                 "(Llama-3.1-8B Q4\\_K\\_M)")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"ey_per_layer_k1024.{ext}"
        fig.savefig(out, dpi=150)
        print(f"Wrote {out}")

    # Also print a compact LaTeX-ready table for k=1024.
    print("\nAt k=1024:")
    print(f"  EY mean rel-F2:  {summary['1024']['ey_mean_relF2']:.4f}")
    print(f"  GRC mean rel-F2: {summary['1024']['grc_mean_relF2']:.4f}")
    print(f"  Ratio:           {summary['1024']['grc_mean_relF2']/summary['1024']['ey_mean_relF2']:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
