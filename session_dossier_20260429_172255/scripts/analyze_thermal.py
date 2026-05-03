"""
analyze_thermal.py — turn nvidia-smi CSV trace into a paper-ready summary.

Input: docs/figures/paper-b/v02_empirical/thermal_sustained_8b.csv
       header: ts,gpu_C,sm_MHz,mem_MHz,power_W,util_pct
Plus a parallel decode log to compute mean tok/s and energy.

Outputs:
  docs/figures/paper-b/v02_empirical/thermal_summary.json
  docs/figures/paper-b/v02_empirical/thermal_summary.md
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "docs" / "figures" / "paper-b" / "v02_empirical"

def parse_csv(p: Path):
    rows = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "gpu_C":    float(row["gpu_C"]),
                    "sm_MHz":   float(row["sm_MHz"]),
                    "mem_MHz":  float(row["mem_MHz"]),
                    "power_W":  float(row["power_W"]),
                    "util_pct": float(row["util_pct"]),
                })
            except (ValueError, KeyError):
                continue
    return rows

def parse_decode_log(p: Path):
    raw = p.read_bytes()
    # Tee-Object on Windows writes UTF-16 LE; detect via NUL bytes.
    if b"\x00" in raw[:200]:
        try:
            txt = raw.decode("utf-16")
        except UnicodeDecodeError:
            txt = raw.decode("utf-16-le", errors="ignore")
    else:
        txt = raw.decode("utf-8", errors="ignore")
    runs = []
    # Match per-run lines: "[GD] N tokens in M ms (X tok/s)"
    for m in re.finditer(r"\[GD\]\s+(\d+)\s+tokens in\s+(\d+)\s+ms\s+\(([\d.]+)\s+tok/s\)", txt):
        runs.append({
            "tokens": int(m.group(1)),
            "ms":     int(m.group(2)),
            "tok_s":  float(m.group(3)),
        })
    decode_only = [float(m.group(1))
                   for m in re.finditer(r"Decode-only:[^,]*,\s*([\d.]+)\s+tok/s", txt)]
    return runs, decode_only

def stats(xs):
    if not xs:
        return None
    n = len(xs)
    s = sorted(xs)
    return {
        "n": n,
        "min":  s[0],
        "p10":  s[max(0, n//10)],
        "p50":  s[n//2],
        "p90":  s[min(n-1, (9*n)//10)],
        "max":  s[-1],
        "mean": sum(xs)/n,
    }

def main():
    csv_p = OUT / "thermal_sustained_8b.csv"
    log_p = OUT / "thermal_sustained_8b.log"
    rows = parse_csv(csv_p)
    runs, decode_only = parse_decode_log(log_p)

    if not rows:
        print("[thermal] no telemetry rows.", file=sys.stderr); sys.exit(1)

    # Detect "active" vs "idle" by util_pct ≥ 20%
    active = [r for r in rows if r["util_pct"] >= 20.0]
    idle   = [r for r in rows if r["util_pct"] <  20.0]

    summary = {
        "samples_total": len(rows),
        "samples_active": len(active),
        "samples_idle":   len(idle),
        "gpu_C": {
            "all":    stats([r["gpu_C"]   for r in rows]),
            "active": stats([r["gpu_C"]   for r in active]),
        },
        "sm_MHz": {
            "active": stats([r["sm_MHz"]  for r in active]),
        },
        "mem_MHz": {
            "active": stats([r["mem_MHz"] for r in active]),
        },
        "power_W": {
            "active": stats([r["power_W"] for r in active]),
        },
        "util_pct": {
            "all":    stats([r["util_pct"] for r in rows]),
        },
        "decode_runs": runs,
        "decode_only_tps": decode_only,
    }

    # Throttling probe: split active into first-third vs last-third, compare
    # mean tok/s and mean SM clock; if last-third drops > 5 % we report
    # thermal throttling, otherwise we report headroom.
    if len(active) >= 9:
        third = len(active) // 3
        early = active[:third]
        late  = active[-third:]
        early_clk = sum(r["sm_MHz"]  for r in early) / len(early)
        late_clk  = sum(r["sm_MHz"]  for r in late ) / len(late)
        early_pwr = sum(r["power_W"] for r in early) / len(early)
        late_pwr  = sum(r["power_W"] for r in late ) / len(late)
        early_T   = sum(r["gpu_C"]   for r in early) / len(early)
        late_T    = sum(r["gpu_C"]   for r in late ) / len(late)
        summary["sustained"] = {
            "early_third": {"sm_MHz": early_clk, "power_W": early_pwr, "gpu_C": early_T},
            "late_third":  {"sm_MHz": late_clk,  "power_W": late_pwr,  "gpu_C": late_T},
            "delta_sm_MHz": late_clk - early_clk,
            "delta_sm_pct": 100.0 * (late_clk - early_clk) / max(early_clk, 1.0),
            "delta_T_C":    late_T   - early_T,
        }

    if runs:
        first_tps = runs[0]["tok_s"]
        last_tps  = runs[-1]["tok_s"]
        summary["throughput_drift"] = {
            "first_run_tok_s": first_tps,
            "last_run_tok_s":  last_tps,
            "delta_tok_s_pct": 100.0 * (last_tps - first_tps) / max(first_tps, 1e-6),
        }
        # Tokens-per-joule estimate (uses active power mean)
        if active:
            mean_pwr = sum(r["power_W"] for r in active) / len(active)
            mean_tps = sum(r["tok_s"]   for r in runs)   / len(runs)
            summary["tokens_per_joule_estimate"] = mean_tps / max(mean_pwr, 1e-6)
            summary["mean_active_power_W"]  = mean_pwr
            summary["mean_decode_tok_s"]    = mean_tps

    (OUT / "thermal_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Thermal Rank — Sustained-Decode Empirical Trace", "",
          "Hardware: RTX 4070 Laptop (8 GB VRAM, ~40 TFLOPS FP16 peak); "
          "model: Meta-Llama-3.1-8B-Instruct Q4_K_M (8.31 B params, "
          "4693 MB on-disk).",
          "Telemetry: `nvidia-smi` polled at 1 Hz "
          f"(N={summary['samples_total']} samples, "
          f"{summary['samples_active']} active, "
          f"{summary['samples_idle']} idle).",
          "",
          "## Active-window summary (util ≥ 20 %)", "",
          "| metric | min | p50 | mean | p90 | max |",
          "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("GPU temp (°C)", "gpu_C"),
                       ("SM clock (MHz)", "sm_MHz"),
                       ("MEM clock (MHz)","mem_MHz"),
                       ("Power (W)",      "power_W")):
        s = summary[key].get("active")
        if not s:
            continue
        md.append(f"| {label} | {s['min']:.0f} | {s['p50']:.0f} | "
                  f"{s['mean']:.1f} | {s['p90']:.0f} | {s['max']:.0f} |")

    if "sustained" in summary:
        s = summary["sustained"]
        md += ["", "## Sustained-decode drift (early third vs late third of active window)",
               "",
               "| | SM MHz | Power W | GPU °C |",
               "|---|---:|---:|---:|",
               f"| early | {s['early_third']['sm_MHz']:.0f} | "
               f"{s['early_third']['power_W']:.1f} | "
               f"{s['early_third']['gpu_C']:.1f} |",
               f"| late  | {s['late_third']['sm_MHz']:.0f} | "
               f"{s['late_third']['power_W']:.1f} | "
               f"{s['late_third']['gpu_C']:.1f} |",
               f"| Δ     | {s['delta_sm_MHz']:+.0f} "
               f"({s['delta_sm_pct']:+.1f} %) | — | "
               f"{s['delta_T_C']:+.1f} |"]

    if "throughput_drift" in summary:
        s = summary["throughput_drift"]
        md += ["", "## Decode throughput drift",
               "",
               f"- First decode: **{s['first_run_tok_s']:.1f} tok/s**",
               f"- Last decode:  **{s['last_run_tok_s']:.1f} tok/s**",
               f"- Δ: {s['delta_tok_s_pct']:+.1f} %"]

    if "tokens_per_joule_estimate" in summary:
        md += ["", "## Energy efficiency",
               "",
               f"- Mean decode rate: {summary['mean_decode_tok_s']:.1f} tok/s",
               f"- Mean active power: {summary['mean_active_power_W']:.1f} W",
               f"- **Tokens per joule (TpJ) ≈ "
               f"{summary['tokens_per_joule_estimate']:.3f}**"]

    md += ["", "## Interpretation for paper-B §Thermal Rank", "",
           "The Thermal Rank module (`runtime/nn/thermal_rank.c`) consumes "
           "exactly the telemetry channels measured here (NVML "
           "`nvmlDeviceGetTemperature`, `nvmlDeviceGetPowerUsage`). The "
           "thresholds in code default to T_low=65 °C → full rank, "
           "T_high=85 °C → min rank, with a linear interpolation between. "
           "The empirically observed active-window distribution (above) "
           "shows whether the workload reaches the actuation band. If the "
           "p90 active temp sits below T_low, the rank is held at full and "
           "Thermal Rank reduces to a no-op for this workload (which is "
           "the *correct* behaviour — no throttling pressure means no rank "
           "reduction is needed); the feature only differentiates from "
           "fixed-rank operation under thermal load.",
           "",
           f"_Generated from `thermal_sustained_8b.csv` "
           f"({summary['samples_total']} samples, "
           f"{summary['samples_active']} active)._"]
    (OUT / "thermal_summary.md").write_text("\n".join(md))
    print(f"Wrote {OUT/'thermal_summary.json'} and thermal_summary.md")

if __name__ == "__main__":
    main()
