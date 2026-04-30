#!/usr/bin/env python3
"""
scripts/paperA_proof/parse_multi_k.py

Aggregate the multi-k NCU sweep produced by multi_k_ncu_sweep.ps1 into a
single summary JSON + CSV table suitable for inclusion in Paper A.

Outputs:
  docs/figures/paper-a/ncu_sweep/sweep_summary.json
  docs/figures/paper-a/ncu_sweep/sweep_summary.csv
  docs/figures/paper-a/ncu_sweep/l2_hit_rate_vs_k.png   (matplotlib)
"""
from __future__ import annotations
import csv, json, math, re, statistics, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "docs" / "figures" / "paper-a" / "ncu_sweep"

# Attention-projection kernel signature (the GRC-affected one)
ATTN_KERNEL_NAME_RE = re.compile(r"gemv_q4_k", re.IGNORECASE)
ATTN_GRID_TARGET = "(512, 1, 1)"

CSV_HEADER_TOKENS = ["ID", "Process Name", "Kernel Name", "Grid Size",
                     "Block Size", "Metric Name", "Metric Value"]


def parse_ncu_csv(path: Path):
    """Yield (kernel, grid, metric, value) tuples; tolerant to NCU header rows."""
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # ncu's --csv emits a banner; find the first row that looks like the header
    header_idx = None
    for i, row in enumerate(rows):
        if "Kernel Name" in row and "Metric Name" in row:
            header_idx = i
            break
    if header_idx is None:
        return
    cols = rows[header_idx]
    name_col = cols.index("Kernel Name")
    grid_col = cols.index("Grid Size")
    metric_col = cols.index("Metric Name")
    value_col = cols.index("Metric Value")
    for r in rows[header_idx + 1:]:
        if len(r) <= value_col:
            continue
        kernel = r[name_col]
        grid = r[grid_col]
        metric = r[metric_col]
        raw = r[value_col].replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        yield kernel, grid, metric, value


def aggregate_attention_kernel(rows):
    """Mean across the 200 sampled launches of the attention projection kernel."""
    by_metric = defaultdict(list)
    for k, g, m, v in rows:
        if not ATTN_KERNEL_NAME_RE.search(k):
            continue
        if g.replace(" ", "") != ATTN_GRID_TARGET.replace(" ", ""):
            continue
        by_metric[m].append(v)
    return {m: statistics.mean(vs) for m, vs in by_metric.items() if vs}


def aggregate_all_gemv(rows):
    """GEMV-weighted mean across all gemv kernels (whole-decode metric)."""
    weighted_hit = 0.0
    weighted_w = 0.0
    dram_total = 0.0
    for k, g, m, v in rows:
        if "gemv" not in k.lower():
            continue
        if m == "lts__t_sectors_op_read.sum":
            weighted_w += v
        if m == "dram__bytes_read.sum":
            dram_total += v
    # Need to re-walk to do weighted hit-rate
    sectors_by_kg = defaultdict(float)
    hit_by_kg = defaultdict(float)
    for k, g, m, v in rows:
        if "gemv" not in k.lower():
            continue
        key = (k, g)
        if m == "lts__t_sectors_op_read.sum":
            sectors_by_kg[key] = v
        elif m == "lts__t_sector_hit_rate.pct":
            hit_by_kg[key] = v
    total_sectors = sum(sectors_by_kg.values())
    if total_sectors > 0:
        weighted_hit = sum(hit_by_kg[k] * sectors_by_kg[k] for k in sectors_by_kg) / total_sectors
    else:
        weighted_hit = float("nan")
    return {
        "gemv_weighted_l2_hit_rate_pct": weighted_hit,
        "gemv_total_dram_read_mb": dram_total / (1024 * 1024),
    }


def main():
    csvs = sorted(SWEEP_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs in {SWEEP_DIR}", file=sys.stderr)
        sys.exit(1)

    # group by k
    by_k = defaultdict(list)  # k_label -> list of (rep, agg_attn, agg_all)
    for c in csvs:
        m = re.match(r"(baseline|k(\d+))_rep(\d+)\.csv", c.name)
        if not m:
            continue
        if m.group(1) == "baseline":
            k_label = "baseline"
        else:
            k_label = int(m.group(2))
        rep = int(m.group(3))
        rows = list(parse_ncu_csv(c))
        agg_attn = aggregate_attention_kernel(rows)
        agg_all = aggregate_all_gemv(rows)
        by_k[k_label].append((rep, agg_attn, agg_all))

    # build summary
    summary = {
        "kernel_filter": "gemv_q4_k grid=(512,1,1) (attention projection)",
        "n_reps_per_k": None,
        "by_k": {},
    }
    for k_label, runs in by_k.items():
        attn_hits = [r[1].get("lts__t_sector_hit_rate.pct", float("nan"))
                     for r in runs]
        attn_dram = [r[1].get("dram__bytes_read.sum", float("nan")) / (1024 * 1024)
                     for r in runs]
        all_hits = [r[2]["gemv_weighted_l2_hit_rate_pct"] for r in runs]
        all_dram = [r[2]["gemv_total_dram_read_mb"] for r in runs]
        attn_hits = [x for x in attn_hits if not math.isnan(x)]
        attn_dram = [x for x in attn_dram if not math.isnan(x)]
        all_hits = [x for x in all_hits if not math.isnan(x)]
        all_dram = [x for x in all_dram if not math.isnan(x)]

        def m_sd(xs):
            if not xs: return (None, None)
            if len(xs) == 1: return (xs[0], 0.0)
            return (statistics.mean(xs), statistics.stdev(xs))

        am, asd = m_sd(attn_hits)
        dm, dsd = m_sd(attn_dram)
        wm, wsd = m_sd(all_hits)
        wdm, wdsd = m_sd(all_dram)
        summary["by_k"][str(k_label)] = {
            "n": len(runs),
            "attention_kernel_l2_hit_pct_mean": am,
            "attention_kernel_l2_hit_pct_std": asd,
            "attention_kernel_dram_mb_mean": dm,
            "attention_kernel_dram_mb_std": dsd,
            "gemv_weighted_l2_hit_pct_mean": wm,
            "gemv_weighted_l2_hit_pct_std": wsd,
            "gemv_total_dram_mb_mean": wdm,
            "gemv_total_dram_mb_std": wdsd,
        }
    summary["n_reps_per_k"] = {k: v["n"] for k, v in summary["by_k"].items()}

    out_json = SWEEP_DIR / "sweep_summary.json"
    out_csv = SWEEP_DIR / "sweep_summary.csv"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "n", "attn_l2_hit_pct_mean", "attn_l2_hit_pct_std",
                    "attn_dram_mb_mean", "attn_dram_mb_std",
                    "gemv_weighted_l2_hit_pct_mean", "gemv_total_dram_mb_mean"])
        order = ["baseline"] + sorted([k for k in summary["by_k"] if k != "baseline"], key=int)
        for k in order:
            d = summary["by_k"][k]
            w.writerow([
                k, d["n"],
                f"{d['attention_kernel_l2_hit_pct_mean']:.3f}" if d["attention_kernel_l2_hit_pct_mean"] is not None else "",
                f"{d['attention_kernel_l2_hit_pct_std']:.3f}"  if d["attention_kernel_l2_hit_pct_std"]  is not None else "",
                f"{d['attention_kernel_dram_mb_mean']:.3f}"    if d["attention_kernel_dram_mb_mean"]    is not None else "",
                f"{d['attention_kernel_dram_mb_std']:.3f}"     if d["attention_kernel_dram_mb_std"]     is not None else "",
                f"{d['gemv_weighted_l2_hit_pct_mean']:.3f}"    if d["gemv_weighted_l2_hit_pct_mean"]    is not None else "",
                f"{d['gemv_total_dram_mb_mean']:.3f}"          if d["gemv_total_dram_mb_mean"]          is not None else "",
            ])
    print(f"Wrote {out_json} and {out_csv}")

    # Try to plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks_sorted = sorted([int(k) for k in summary["by_k"] if k != "baseline"])
        attn_means = [summary["by_k"][str(k)]["attention_kernel_l2_hit_pct_mean"] for k in ks_sorted]
        attn_stds  = [summary["by_k"][str(k)]["attention_kernel_l2_hit_pct_std"]  for k in ks_sorted]
        baseline_attn = summary["by_k"]["baseline"]["attention_kernel_l2_hit_pct_mean"]
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(ks_sorted, attn_means, yerr=attn_stds, fmt="o-",
                     capsize=3, label="Attention proj. kernel (gemv_q4_k grid=512)")
        if baseline_attn is not None:
            plt.axhline(baseline_attn, color="grey", linestyle=":",
                        label=f"Baseline = {baseline_attn:.2f}%")
        plt.xlabel("Compression rank k")
        plt.ylabel("L2 sector hit-rate (%)")
        plt.title("Multi-k NCU sweep: attention-projection L2 hit-rate vs. k\n"
                  "RTX 4070 Laptop, Llama-3.1-8B Q4_K_M")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = SWEEP_DIR / "l2_hit_rate_vs_k.png"
        plt.savefig(out_png, dpi=140)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
