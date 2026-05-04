"""Parse NCU CSV traces and compute L2 cache hit rate for Paper A cache-fit test.

Inputs: docs/figures/paper-a/ncu/baseline_l2.csv and grc_k1024_l2.csv
Outputs: comparison plot + summary JSON for paper embedding.
"""
import csv
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NCU_DIR = ROOT / "docs" / "figures" / "paper-a" / "ncu"


def parse(path: Path):
    """Return list of dicts keyed by launch ID with metric values."""
    launches: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # Skip leading non-CSV lines (==PROF== etc.)
        reader = None
        header_found = False
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"ID"'):
                header_found = True
                rows.append(line)
                continue
            if header_found:
                rows.append(line)
        reader = csv.DictReader(rows)
        for row in reader:
            lid = row.get("ID")
            if lid is None:
                continue
            d = launches.setdefault(lid, {})
            metric = row.get("Metric Name", "")
            val_str = (row.get("Metric Value", "") or "").replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue
            d[metric] = val
    return launches


def hit_rates(launches: dict[str, dict]) -> list[float]:
    out = []
    for lid, d in launches.items():
        hit = d.get("lts__t_sectors_op_read_lookup_hit.sum")
        total = d.get("lts__t_sectors_op_read.sum")
        if hit is None or total is None or total <= 0:
            continue
        out.append(100.0 * hit / total)
    return out


def main() -> None:
    baseline = parse(NCU_DIR / "baseline_l2.csv")
    grc = parse(NCU_DIR / "grc_k1024_l2.csv")
    b_rates = hit_rates(baseline)
    g_rates = hit_rates(grc)

    summary = {
        "baseline": {
            "n_launches": len(b_rates),
            "mean_hit_rate_pct": statistics.mean(b_rates) if b_rates else None,
            "median_hit_rate_pct": statistics.median(b_rates) if b_rates else None,
            "stdev_hit_rate_pct": statistics.stdev(b_rates) if len(b_rates) > 1 else None,
        },
        "grc_k1024": {
            "n_launches": len(g_rates),
            "mean_hit_rate_pct": statistics.mean(g_rates) if g_rates else None,
            "median_hit_rate_pct": statistics.median(g_rates) if g_rates else None,
            "stdev_hit_rate_pct": statistics.stdev(g_rates) if len(g_rates) > 1 else None,
        },
    }
    if b_rates and g_rates:
        summary["delta_pp"] = summary["grc_k1024"]["mean_hit_rate_pct"] - summary["baseline"]["mean_hit_rate_pct"]

    out_json = NCU_DIR / "l2_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if b_rates:
        ax.hist(b_rates, bins=20, alpha=0.55, label=f"Baseline (mean={statistics.mean(b_rates):.1f}%, n={len(b_rates)})", color="#444")
    if g_rates:
        ax.hist(g_rates, bins=20, alpha=0.55, label=f"GRC k=1024 (mean={statistics.mean(g_rates):.1f}%, n={len(g_rates)})", color="#1f77b4")
    ax.set_xlabel("L2 read hit rate (%) --- kernel_gemv_q4_k")
    ax.set_ylabel("Launch count")
    ax.set_title("Llama-3.1-8B Q4_K_M decode: L2 cache-fit (Paper A §6)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = NCU_DIR / "l2_hit_rate_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
