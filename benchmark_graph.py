#!/usr/bin/env python3
"""
benchmark_graph.py — Resource consumption and throughput graphs for HyperTensor benchmarks.

Reads:
  benchmark_extended.csv     — per-trial results (GPU%, VRAM, Power per run)
  benchmark_timeseries.csv   — (optional) continuous sampler log with stage labels

Outputs (saved to benchmark_graphs/):
  throughput.png    — decode t/s comparison by runtime/backend/model
  resources.png     — GPU%, peak VRAM, avg power by runtime/backend/model
  efficiency.png    — decode t/s vs power W scatter (efficiency frontier)
  timeseries.png    — GPU%, VRAM, Power over real time with stage color bands
                      (only generated when benchmark_timeseries.csv exists)
"""

import sys, os, csv, math
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Install with: pip install matplotlib numpy")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CSV_FILE   = SCRIPT_DIR / "benchmark_extended.csv"
TS_FILE    = SCRIPT_DIR / "benchmark_timeseries.csv"
OUT_DIR    = SCRIPT_DIR / "benchmark_graphs"

PALETTE = {
    "Geodessical/GPU": "#1565C0",
    "Geodessical/CPU": "#90CAF9",
    "Ollama/GPU":      "#BF360C",
    "Ollama/CPU":      "#FFCCBC",
}
STAGE_COLORS = {
    "GeoCPU": "#E3F2FD",
    "GeoGPU": "#BBDEFB",
    "OllGPU": "#FBE9E7",
    "OllCPU": "#FFE0B2",
}
MODELS_ORDER = ["smollm2-135m", "phi35-mini", "gemma4-2b"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def fnum(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except:
        return default

def avg(vals):
    v = [x for x in vals if x > 0]
    return round(sum(v) / len(v), 1) if v else 0.0

def model_idx(m):
    return MODELS_ORDER.index(m) if m in MODELS_ORDER else 99

def combo(r):
    return f"{r['Runtime']}/{r['Backend']}"


# ── Load summary CSV ───────────────────────────────────────────────────────────
if not CSV_FILE.exists():
    print(f"ERROR: {CSV_FILE} not found. Run benchmark_extended.ps1 first.")
    sys.exit(1)

all_rows = load_csv(CSV_FILE)
rows = [r for r in all_rows if not r.get('Err', '').strip()]
print(f"Loaded {len(rows)} valid rows (of {len(all_rows)} total) from {CSV_FILE.name}")

OUT_DIR.mkdir(exist_ok=True)

models   = sorted(set(r['Model']   for r in rows), key=model_idx)
combos   = ["Geodessical/GPU", "Geodessical/CPU", "Ollama/GPU", "Ollama/CPU"]
combos   = [c for c in combos if any(combo(r) == c for r in rows)]

x = np.arange(len(models))
bar_w = min(0.18, 0.8 / max(len(combos), 1))


# ── Figure 1: Decode throughput comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

for i, c in enumerate(combos):
    rt, be = c.split('/')
    vals = []
    for m in models:
        mr = [r for r in rows if r['Model'] == m and r['Runtime'] == rt and r['Backend'] == be]
        vals.append(avg([fnum(r['DecodeTS']) for r in mr]))
    offset = (i - len(combos) / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, label=c, color=PALETTE.get(c, '#999'),
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Decode Throughput (tokens/s)', fontsize=11)
ax.set_title('HyperTensor — Decode Throughput by Runtime / Backend / Model', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(OUT_DIR / 'throughput.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'throughput.png'}")


# ── Figure 2: Resource consumption (GPU%, VRAM, Power) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics = [
    ('GpuPctAvg', 'Avg GPU Utilization (%)',     'GPU Util %',  'max'),
    ('VramMB',    'Peak VRAM Used (MB)',           'VRAM MB',     'max'),
    ('PowerW',    'Avg GPU Power Draw (W)',        'Power W',     'avg'),
]

for ax, (col, title, ylabel, agg) in zip(axes, metrics):
    for i, c in enumerate(combos):
        rt, be = c.split('/')
        vals = []
        for m in models:
            mr = [r for r in rows if r['Model'] == m and r['Runtime'] == rt and r['Backend'] == be]
            nums = [fnum(r[col]) for r in mr]
            if agg == 'max':
                vals.append(max(nums) if nums else 0)
            else:
                vals.append(avg(nums))
        offset = (i - len(combos) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=c, color=PALETTE.get(c, '#999'),
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha='right', fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle('HyperTensor — GPU Resource Consumption by Runtime / Backend / Model',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'resources.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'resources.png'}")


# ── Figure 3: Efficiency scatter (t/s vs Power W) ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
markers_map = {'smollm2-135m': 'o', 'phi35-mini': 's', 'gemma4-2b': '^'}
plotted = set()

for c in combos:
    rt, be = c.split('/')
    color = PALETTE.get(c, '#999')
    for m in models:
        mr = [r for r in rows if r['Model'] == m and r['Runtime'] == rt and r['Backend'] == be]
        ts = avg([fnum(r['DecodeTS']) for r in mr])
        pw = avg([fnum(r['PowerW'])   for r in mr])
        if ts > 0 and pw > 0:
            mk = markers_map.get(m, 'D')
            label = f"{c} — {m}" if (c, m) not in plotted else None
            ax.scatter(pw, ts, color=color, marker=mk, s=100, zorder=5,
                       edgecolors='white', linewidth=0.8, label=label)
            ax.annotate(f"{m}\n{c}", (pw, ts),
                        textcoords='offset points', xytext=(6, 4), fontsize=6, color='#444')
            plotted.add((c, m))

# Efficiency guide lines (iso-lines: t/s per watt)
pw_range = np.linspace(0.1, max(1, ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 300), 200)
for tpw in [1, 2, 5, 10]:
    ax.plot(pw_range, [tpw * p for p in pw_range], '--', color='#ccc', linewidth=0.7, zorder=0)
    ax.text(pw_range[-1], tpw * pw_range[-1], f'{tpw} t/s/W',
            fontsize=6, color='#aaa', va='center')

ax.set_xlabel('Avg GPU Power Draw (W)', fontsize=11)
ax.set_ylabel('Decode Throughput (tokens/s)', fontsize=11)
ax.set_title('HyperTensor — Throughput vs Power Efficiency\n(upper-left = more efficient)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.25)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(OUT_DIR / 'efficiency.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'efficiency.png'}")


# ── Figure 4: Time-series with stage bands ────────────────────────────────────
if TS_FILE.exists() and TS_FILE.stat().st_size > 500:
    print(f"Loading time-series data from {TS_FILE.name}...")
    ts_rows = load_csv(TS_FILE)
    ts_rows = [r for r in ts_rows if r.get('Timestamp', '').strip().lstrip('-').isdigit()]

    if len(ts_rows) < 5:
        print("Too few time-series samples, skipping timeseries chart.")
    else:
        t0 = fnum(ts_rows[0]['Timestamp'])
        # Convert ms timestamps to seconds from start
        times  = [(fnum(r['Timestamp']) - t0) / 1000.0 for r in ts_rows]
        gpupct = [fnum(r['GpuPct'])   for r in ts_rows]
        vramMB = [fnum(r['VramMB'])   for r in ts_rows]
        powerW = [fnum(r['PowerW'])   for r in ts_rows]
        stages = [r.get('Stage', 'idle').strip() for r in ts_rows]

        def stage_key(s):
            """Return top-level phase key like 'GeoCPU', 'GeoGPU', etc."""
            parts = s.split('|')
            return parts[0] if parts else 'idle'

        def stage_label(s):
            """Human-readable label: GeoCPU|smollm2-135m → 'GeoCPU\nsmollm2'"""
            parts = s.split('|')
            if len(parts) >= 2:
                model_short = parts[1].replace('smollm2-135m', 'smollm2').replace('phi35-mini', 'phi3.5').replace('gemma4-2b', 'gemma4')
                return f"{parts[0]}\n{model_short}"
            return parts[0] if parts else s

        # Find contiguous stage-key spans for background bands
        spans = []  # [(start_idx, end_idx, key, label)]
        cur_key = stage_key(stages[0])
        cur_lbl = stage_label(stages[0])
        span_start = 0
        for idx in range(1, len(stages)):
            sk = stage_key(stages[idx])
            if sk != cur_key:
                spans.append((span_start, idx - 1, cur_key, cur_lbl))
                span_start = idx
                cur_key = sk
                cur_lbl = stage_label(stages[idx])
        spans.append((span_start, len(stages) - 1, cur_key, cur_lbl))

        fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
        series_cfg = [
            (gpupct, 'GPU Utilization (%)',   '#1565C0', 0,   100),
            (vramMB, 'VRAM Used (MB)',         '#2E7D32', 0,   None),
            (powerW, 'Power Draw (W)',          '#B71C1C', 0,   None),
        ]

        # Legend patches for stage types
        legend_patches = [
            mpatches.Patch(facecolor=STAGE_COLORS.get(k, '#eee'), alpha=0.6, label=k)
            for k in ['GeoCPU', 'GeoGPU', 'OllGPU', 'OllCPU']
        ]

        for ax, (yvals, ylabel, ycolor, ymin, ymax) in zip(axes, series_cfg):
            # Draw stage background bands
            prev_key = None
            for (si, ei, key, lbl) in spans:
                t_start = times[si]
                t_end   = times[min(ei, len(times) - 1)]
                bg = STAGE_COLORS.get(key, '#F5F5F5')
                ax.axvspan(t_start, t_end, alpha=0.35, color=bg, zorder=0)

            # Plot data line
            ax.plot(times, yvals, color=ycolor, linewidth=0.9, zorder=2)

            # Add stage labels along x-axis (top of each span) only on first subplot
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(bottom=ymin, top=ymax)
            ax.grid(alpha=0.2, zorder=1)

        # Add stage label annotations only on the top subplot
        top_ax = axes[0]
        for (si, ei, key, lbl) in spans:
            t_mid = (times[si] + times[min(ei, len(times) - 1)]) / 2
            ypos  = top_ax.get_ylim()[1] * 0.97
            top_ax.text(t_mid, ypos, lbl, ha='center', va='top',
                        fontsize=5.5, color='#333', rotation=0, zorder=3,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, edgecolor='none'))

        axes[-1].set_xlabel('Time (seconds from benchmark start)', fontsize=10)
        top_ax.legend(handles=legend_patches, loc='upper right', fontsize=7,
                      title='Stage', title_fontsize=7)

        total_min = round(times[-1] / 60, 1) if times else 0
        fig.suptitle(
            f'HyperTensor — GPU Resource Usage Across All Benchmark Stages  ({total_min} min total)',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'timeseries.png', dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR / 'timeseries.png'}")
else:
    print(f"No time-series data found at {TS_FILE.name}.")
    print("Re-run benchmark_extended.ps1 with the updated version to capture it.")

print(f"\nAll graphs saved to: {OUT_DIR}/")
