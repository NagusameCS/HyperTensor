#!/usr/bin/env python3
"""
scripts/paperA_proof/parse_expB.py

Aggregate the Exp B (P2) thrash-sweep CSVs produced by expB_thrash_sweep.ps1
into a single summary JSON + CSV.

For each (delta MB, condition in {baseline, grc_k1024}, rep) it pulls:
  - lts__t_sector_hit_rate.pct  (weighted across gemv kernels by sectors)
  - dram__bytes_read.sum         (total over the 200 sampled launches)

The headline derived metric is:

    speedup_proxy(delta) = baseline_dram(delta) / grc_dram(delta)

If the cache-fit hypothesis is correct, this proxy should approach 1.0
(GRC speed-up vanishes) as delta crowds the persistent allocation out of
L2.  If the fusion hypothesis is correct, the proxy should be flat.

Outputs:
  docs/figures/paper-a/expB_thrash/expB_summary.json
  docs/figures/paper-a/expB_thrash/expB_summary.csv
  docs/figures/paper-a/expB_thrash/expB_dram_vs_delta.png
"""
from __future__ import annotations
import csv, json, math, re, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPB_DIR = ROOT / "docs" / "figures" / "paper-a" / "expB_thrash"

LABEL_RE = re.compile(r"delta(\d+)_(baseline|grc_k1024)_rep(\d+)\.ncu\.csv")


def parse_ncu(path: Path):
    rows = list(csv.reader(path.open(encoding="utf-8", errors="replace")))
    header_idx = next((i for i, r in enumerate(rows)
                       if "Kernel Name" in r and "Metric Name" in r), None)
    if header_idx is None:
        return []
    cols = rows[header_idx]
    name_col   = cols.index("Kernel Name")
    metric_col = cols.index("Metric Name")
    value_col  = cols.index("Metric Value")
    out = []
    for r in rows[header_idx + 1:]:
        if len(r) <= value_col:
            continue
        try:
            v = float(r[value_col].replace(",", ""))
        except ValueError:
            continue
        out.append((r[name_col], r[metric_col], v))
    return out


def gemv_aggregate(rows):
    """Weighted L2 hit-rate + total DRAM read across all gemv kernels."""
    sectors_by_kernel = defaultdict(float)
    hit_by_kernel = defaultdict(float)
    dram_total = 0.0
    for k, m, v in rows:
        if "gemv" not in k.lower():
            continue
        if m == "lts__t_sectors_op_read.sum":
            sectors_by_kernel[k] = v
        elif m == "lts__t_sector_hit_rate.pct":
            hit_by_kernel[k] = v
        elif m == "dram__bytes_read.sum":
            dram_total += v
    total_sec = sum(sectors_by_kernel.values())
    if total_sec > 0:
        weighted_hit = sum(hit_by_kernel[k] * sectors_by_kernel[k]
                           for k in sectors_by_kernel) / total_sec
    else:
        weighted_hit = float("nan")
    return weighted_hit, dram_total / (1024 * 1024)


def main():
    csvs = sorted(EXPB_DIR.glob("*.ncu.csv"))
    if not csvs:
        print(f"No CSVs in {EXPB_DIR}", file=sys.stderr)
        sys.exit(1)

    bucket = defaultdict(list)  # (delta, cond) -> [(rep, hit_pct, dram_mb)]
    for c in csvs:
        m = LABEL_RE.match(c.name)
        if not m:
            continue
        delta = int(m.group(1))
        cond  = m.group(2)
        rep   = int(m.group(3))
        if c.stat().st_size < 50_000:
            print(f"[skip] {c.name} too small")
            continue
        rows = parse_ncu(c)
        hit, dram = gemv_aggregate(rows)
        if math.isnan(hit):
            print(f"[skip] {c.name}: no gemv data")
            continue
        bucket[(delta, cond)].append((rep, hit, dram))

    summary = {
        "metric_definitions": {
            "l2_hit_pct": "sector-weighted gemv L2 hit-rate",
            "dram_mb":    "sum of dram__bytes_read across the 200 sampled gemv launches",
            "speedup_proxy": "baseline_dram / grc_dram (>1 means GRC moves less DRAM)",
        },
        "by_delta": {},
    }

    deltas = sorted({d for (d, _) in bucket})
    rows_csv = []
    for d in deltas:
        entry = {"delta_mb": d}
        for cond in ("baseline", "grc_k1024"):
            runs = bucket.get((d, cond), [])
            if not runs:
                entry[cond] = None
                continue
            hits = [r[1] for r in runs]
            drams = [r[2] for r in runs]
            entry[cond] = {
                "n": len(runs),
                "hit_pct_mean": statistics.mean(hits),
                "hit_pct_std":  statistics.stdev(hits) if len(hits) > 1 else 0.0,
                "dram_mb_mean": statistics.mean(drams),
                "dram_mb_std":  statistics.stdev(drams) if len(drams) > 1 else 0.0,
            }
        if entry.get("baseline") and entry.get("grc_k1024"):
            entry["speedup_proxy"] = (
                entry["baseline"]["dram_mb_mean"] / entry["grc_k1024"]["dram_mb_mean"]
                if entry["grc_k1024"]["dram_mb_mean"] > 0 else None
            )
            entry["hit_delta_pp"] = (
                entry["grc_k1024"]["hit_pct_mean"] - entry["baseline"]["hit_pct_mean"]
            )
        else:
            entry["speedup_proxy"] = None
            entry["hit_delta_pp"] = None
        summary["by_delta"][str(d)] = entry
        rows_csv.append([
            d,
            entry["baseline"]["hit_pct_mean"]   if entry.get("baseline")   else "",
            entry["baseline"]["dram_mb_mean"]   if entry.get("baseline")   else "",
            entry["grc_k1024"]["hit_pct_mean"]  if entry.get("grc_k1024")  else "",
            entry["grc_k1024"]["dram_mb_mean"]  if entry.get("grc_k1024")  else "",
            entry["speedup_proxy"] if entry["speedup_proxy"] is not None else "",
            entry["hit_delta_pp"]  if entry["hit_delta_pp"]  is not None else "",
        ])

    out_json = EXPB_DIR / "expB_summary.json"
    out_csv  = EXPB_DIR / "expB_summary.csv"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_mb", "baseline_hit_pct", "baseline_dram_mb",
                    "grc_hit_pct", "grc_dram_mb", "speedup_proxy", "hit_delta_pp"])
        for r in rows_csv:
            w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in r])
    print(f"Wrote {out_json} and {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = [int(k) for k in summary["by_delta"]
              if summary["by_delta"][k].get("speedup_proxy") is not None]
        ds.sort()
        if ds:
            sp = [summary["by_delta"][str(d)]["speedup_proxy"] for d in ds]
            hd = [summary["by_delta"][str(d)]["hit_delta_pp"]  for d in ds]
            fig, ax1 = plt.subplots(figsize=(7, 4.2))
            ax1.plot(ds, sp, "o-", color="tab:blue", label="DRAM speedup proxy")
            ax1.axhline(1.0, color="grey", linestyle=":", label="No GRC benefit")
            ax1.set_xlabel("L2 thrasher delta (MB)")
            ax1.set_ylabel("speedup proxy = baseline_dram / grc_dram", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2 = ax1.twinx()
            ax2.plot(ds, hd, "s--", color="tab:red", label="L2 hit-rate delta")
            ax2.set_ylabel("L2 hit-rate delta (pp)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            plt.title("Exp B (P2): GRC benefit vs. L2 contention\n"
                      "RTX 4070 Laptop, Llama-3.1-8B Q4_K_M, k=1024")
            fig.tight_layout()
            out_png = EXPB_DIR / "expB_dram_vs_delta.png"
            plt.savefig(out_png, dpi=140)
            print(f"Wrote {out_png}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
