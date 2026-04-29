"""Parse NCU CSV output for Llama-8B GEMV kernels and emit a summary CSV.

Reads docs/figures/paper-a/ncu/{baseline,k1024}_raw.txt and writes
docs/figures/paper-a/ncu/summary.csv + summary.json.

Focus: kernel_gemv_q4_k and kernel_gemv_q6_k (the attention/FFN GEMVs).
The grid (1792,1,1) is the FFN big matmul; (512,1,1) is attn-projection;
(128,1,1) is small (Q/K/V on small heads or output proj after compression).
"""
from __future__ import annotations
import csv, json, re, statistics, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
NCU_DIR = REPO / "docs" / "figures" / "paper-a" / "ncu"
METRICS = [
    "dram__bytes_read.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

def parse_ncu_csv(path: Path):
    """Yield dicts: {kernel, grid, block, metric, avg_value (float)}."""
    if not path.exists():
        return
    # auto-detect UTF-16 BOM (Tee-Object writes UTF-16LE on PS5)
    with path.open("rb") as fb:
        head = fb.read(4)
    enc = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    with path.open(encoding=enc, errors="replace") as f:
        # ncu --csv emits prelude then a CSV block. Find header row.
        rows = list(csv.reader(f))
    # Find the header (it contains "ID","Process Name",...,"Metric Name","Metric Unit","Minimum","Maximum","Average")
    hdr_idx = None
    for i, r in enumerate(rows):
        if r and "Metric Name" in r and "Average" in r:
            hdr_idx = i
            break
    if hdr_idx is None:
        return
    hdr = rows[hdr_idx]
    col = {name: i for i, name in enumerate(hdr)}
    for r in rows[hdr_idx + 1:]:
        if len(r) < len(hdr):
            continue
        try:
            kernel = r[col["Kernel Name"]] if "Kernel Name" in col else r[3]
            grid = r[col["Grid Size"]] if "Grid Size" in col else r[4]
            block = r[col["Block Size"]] if "Block Size" in col else r[5]
            metric = r[col["Metric Name"]]
            unit = r[col["Metric Unit"]]
            avg = r[col["Average"]].replace(",", "")
            avg_f = float(avg)
        except (KeyError, ValueError, IndexError):
            continue
        # short kernel name
        kshort = re.sub(r"\(.*?\).*", "", kernel).strip()
        yield {
            "kernel": kshort,
            "grid": grid,
            "block": block,
            "metric": metric,
            "unit": unit,
            "avg": avg_f,
        }

def aggregate(rows):
    """Group by (kernel, grid). Return dict[(kernel,grid)] -> dict[metric] -> mean."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["kernel"], r["grid"])
        groups[key][r["metric"]].append(r["avg"])
    out = {}
    for key, mvals in groups.items():
        out[key] = {m: statistics.mean(v) for m, v in mvals.items()}
    return out

def main():
    base = list(parse_ncu_csv(NCU_DIR / "baseline_raw.txt"))
    comp = list(parse_ncu_csv(NCU_DIR / "k1024_raw.txt"))
    if not base:
        print("ERROR: no baseline rows", file=sys.stderr); sys.exit(1)
    base_agg = aggregate(base)
    comp_agg = aggregate(comp)

    # Build a unified table over all (kernel, grid) keys appearing in either run
    all_keys = sorted(set(base_agg.keys()) | set(comp_agg.keys()))

    out_csv = NCU_DIR / "summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "grid",
                    "base_l2_hit_pct", "k1024_l2_hit_pct", "delta_l2_hit_pct",
                    "base_dram_MB", "k1024_dram_MB", "dram_reduction_pct",
                    "base_warps_pct", "k1024_warps_pct"])
        for key in all_keys:
            b = base_agg.get(key, {})
            c = comp_agg.get(key, {})
            b_l2 = b.get("lts__t_sector_hit_rate.pct")
            c_l2 = c.get("lts__t_sector_hit_rate.pct")
            b_dr = b.get("dram__bytes_read.sum")
            c_dr = c.get("dram__bytes_read.sum")
            b_w = b.get("sm__warps_active.avg.pct_of_peak_sustained_active")
            c_w = c.get("sm__warps_active.avg.pct_of_peak_sustained_active")
            d_l2 = (c_l2 - b_l2) if (b_l2 is not None and c_l2 is not None) else None
            d_dr = (1.0 - c_dr / b_dr) * 100.0 if (b_dr and c_dr) else None
            w.writerow([
                key[0], key[1],
                f"{b_l2:.2f}" if b_l2 is not None else "",
                f"{c_l2:.2f}" if c_l2 is not None else "",
                f"{d_l2:+.2f}" if d_l2 is not None else "",
                f"{b_dr/1e6:.2f}" if b_dr else "",
                f"{c_dr/1e6:.2f}" if c_dr else "",
                f"{d_dr:+.1f}" if d_dr is not None else "",
                f"{b_w:.1f}" if b_w is not None else "",
                f"{c_w:.1f}" if c_w is not None else "",
            ])
    print(f"wrote {out_csv}")

    # Headline: weighted mean L2 hit-rate over all GEMV kernels
    def weighted_l2(agg):
        total_sectors = 0.0
        weighted = 0.0
        for (k, g), m in agg.items():
            if "gemv" not in k.lower(): continue
            sec = m.get("lts__t_sectors_op_read.sum", 0.0)
            hit = m.get("lts__t_sector_hit_rate.pct", 0.0)
            total_sectors += sec
            weighted += sec * hit
        return (weighted / total_sectors) if total_sectors else None

    def total_dram(agg):
        return sum(m.get("dram__bytes_read.sum", 0.0) for m in agg.values())

    summary = {
        "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
        "gpu": "RTX 4070 Laptop (sm_89, 32 MB L2)",
        "ncu_version": "2026.1.0.0",
        "decode_tokens_profiled": 16,
        "kernels_profiled": "all (--launch-count 200 --launch-skip 50)",
        "baseline": {
            "weighted_l2_hit_rate_gemv_pct": weighted_l2(base_agg),
            "total_dram_read_MB_per_kernel_avg": total_dram(base_agg) / 1e6 / max(1, sum(1 for k in base_agg if "gemv" in k[0].lower())),
            "n_kernel_groups": len(base_agg),
        },
        "k1024_skip_o": {
            "weighted_l2_hit_rate_gemv_pct": weighted_l2(comp_agg),
            "total_dram_read_MB_per_kernel_avg": total_dram(comp_agg) / 1e6 / max(1, sum(1 for k in comp_agg if "gemv" in k[0].lower())),
            "n_kernel_groups": len(comp_agg),
        },
    }
    bl = summary["baseline"]["weighted_l2_hit_rate_gemv_pct"]
    cl = summary["k1024_skip_o"]["weighted_l2_hit_rate_gemv_pct"]
    if bl is not None and cl is not None:
        summary["headline"] = {
            "l2_hit_rate_baseline_pct": round(bl, 2),
            "l2_hit_rate_k1024_pct": round(cl, 2),
            "absolute_improvement_pct_points": round(cl - bl, 2),
        }
    out_json = NCU_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
