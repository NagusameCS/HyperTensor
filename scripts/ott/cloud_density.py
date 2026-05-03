#!/usr/bin/env python3
"""
C2 — Cloud Density Bridge Analysis
====================================
Measures the offline-to-online density gap and estimates how cloud density
affects acceptance rate.

Background (from paper):
  The OD (OneDecode) table is baked offline from a fixed seed corpus.  At
  runtime the geodesic operates on live hidden states that may lie far from
  any offline anchor.  When the nearest anchor distance exceeds the manifold
  curvature radius the draft diverges, producing Type II rejections.

  This script:
    1. Reads the live telemetry TSV (hidden-state activations for every
       emitted token — the "online" cloud)
    2. Reads the axiom beta report JSON (offline manifold stats: n_anchors,
       coverage radius, mean nearest-neighbour distance)
    3. Computes pairwise distances among online states and compares them
       against the offline coverage radius
    4. Produces a density chart (matplotlib, saved as PNG) and a JSON
       summary with the offline-to-online bridge mapping

Usage:
    python scripts/ott/cloud_density.py \\
        --telemetry test_live.tsv \\
        --axiom-report axiom_beta_report.json \\
        [--out-dir .]

Outputs:
    cloud_density_summary.json
    cloud_density_chart.png  (requires matplotlib)
"""

import argparse
import json
import math
import sys
from pathlib import Path

#  Helpers 

def cosine_distance(a: list, b: list) -> float:
    """1 - cosine_similarity (range [0, 2])."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - dot / (na * nb)


def l2_distance(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


#  TSV parser 

def load_telemetry(path: Path):
    """Return list of {step, source, token_id, hidden} dicts."""
    records = []
    dim = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                # header: # geodessical_live_telemetry_v1\tlayer=-1\tdim=576\tevery=2
                if "dim=" in line:
                    for part in line.split("\t"):
                        if part.startswith("dim="):
                            dim = int(part.split("=")[1])
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                step   = int(parts[0])
                source = parts[1]
                tok_id = int(parts[2])
                hidden = [float(x) for x in parts[3:]]
            except ValueError:
                continue
            records.append({"step": step, "source": source,
                             "token_id": tok_id, "hidden": hidden})
    return records, dim


#  Density analysis 

def analyse_online_density(records: list, metric: str = "cosine"):
    """Compute consecutive and all-pairs stats on online hidden states."""
    if len(records) < 2:
        return {}

    dist_fn = cosine_distance if metric == "cosine" else l2_distance
    hiddens = [r["hidden"] for r in records]

    # Consecutive distances (trajectory smoothness)
    consec = []
    for i in range(1, len(hiddens)):
        consec.append(dist_fn(hiddens[i - 1], hiddens[i]))

    # Nearest-neighbour distances (cloud density)
    nn_dists = []
    for i, h in enumerate(hiddens):
        best = float("inf")
        for j, h2 in enumerate(hiddens):
            if i == j:
                continue
            d = dist_fn(h, h2)
            if d < best:
                best = d
        nn_dists.append(best)

    return {
        "n_online_states": len(records),
        "consecutive": {
            "mean":   sum(consec) / len(consec),
            "p50":    percentile(consec, 50),
            "p90":    percentile(consec, 90),
            "max":    max(consec),
        },
        "nn_distance": {
            "mean":   sum(nn_dists) / len(nn_dists),
            "p50":    percentile(nn_dists, 50),
            "p90":    percentile(nn_dists, 90),
            "max":    max(nn_dists),
        },
    }


def load_axiom_report(path: Path) -> dict:
    """Load axiom beta JSON and extract relevant manifold parameters."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Navigate the report structure flexibly
    result = {}

    # Try top-level keys
    for key in ("n_anchors", "coverage_radius", "mean_nn_dist",
                 "geodesic_score", "phase5_score"):
        if key in data:
            result[key] = data[key]

    # Try nested in phases/sections
    for section in data.values():
        if isinstance(section, dict):
            for key in ("n_anchors", "coverage_radius", "mean_nn_dist",
                        "geodesic_score", "manifold_radius"):
                if key in section and key not in result:
                    result[key] = section[key]

    return result


def compute_bridge_gap(online_stats: dict, offline_stats: dict) -> dict:
    """Quantify offline-to-online gap and estimate density enrichment needed."""
    gap = {}

    offline_radius = offline_stats.get("coverage_radius",
                     offline_stats.get("manifold_radius", None))
    online_nn_mean = online_stats.get("nn_distance", {}).get("mean", None)
    online_nn_p90  = online_stats.get("nn_distance", {}).get("p90", None)

    gap["offline_coverage_radius"] = offline_radius
    gap["online_nn_mean"] = online_nn_mean
    gap["online_nn_p90"]  = online_nn_p90

    if offline_radius and online_nn_p90:
        # Density ratio: how many more online states fall outside the offline
        # coverage radius than inside it → proxy for acceptance gap
        gap["density_ratio"] = online_nn_p90 / offline_radius
        if gap["density_ratio"] > 1.5:
            enrichment_factor = math.ceil(gap["density_ratio"] ** 2)
            gap["recommended_enrichment_factor"] = enrichment_factor
            gap["recommendation"] = (
                f"Online p90 NN distance ({online_nn_p90:.4f}) is "
                f"{gap['density_ratio']:.1f}x the offline coverage radius "
                f"({offline_radius:.4f}).  Re-bake the OD table with "
                f"~{enrichment_factor}x more seed samples (--axiom-samples) "
                f"or widen coverage with larger --axiom-probe to close the gap."
            )
        else:
            gap["recommendation"] = (
                "Online cloud is within offline coverage radius.  "
                "Acceptance rate is limited by manifold divergence (Type II), "
                "not by density mismatch — focus on GRC calibration (C3) instead."
            )

    return gap


def try_plot(records: list, online_stats: dict, out_dir: Path):
    """Attempt to produce a density chart; skip gracefully if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[C2] matplotlib not available — skipping chart. "
              "Install with: pip install matplotlib", file=sys.stderr)
        return None

    steps   = [r["step"] for r in records]
    hiddens = np.array([r["hidden"] for r in records], dtype=np.float32)

    # Compute consecutive cosine distances
    dists = []
    for i in range(1, len(hiddens)):
        dot = np.dot(hiddens[i - 1], hiddens[i])
        na  = np.linalg.norm(hiddens[i - 1])
        nb  = np.linalg.norm(hiddens[i])
        if na > 1e-12 and nb > 1e-12:
            dists.append(1.0 - float(dot / (na * nb)))
        else:
            dists.append(1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Top: consecutive trajectory distances
    ax = axes[0]
    ax.plot(steps[1:], dists, color="steelblue", linewidth=1.2, label="Consec. cosine dist")
    if online_stats.get("consecutive", {}).get("p90"):
        ax.axhline(online_stats["consecutive"]["p90"], color="orange",
                   linestyle="--", label=f"p90={online_stats['consecutive']['p90']:.4f}")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Cosine distance")
    ax.set_title("Online Cloud: Consecutive Hidden-State Distances (trajectory smoothness)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: source breakdown bar chart
    ax2 = axes[1]
    sources = {}
    for r in records:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    colors = {"geo": "steelblue", "xfmr": "salmon", "bonus": "limegreen", "direct": "goldenrod"}
    bars = ax2.bar(list(sources.keys()),
                   list(sources.values()),
                   color=[colors.get(k, "gray") for k in sources.keys()])
    ax2.set_title("Token Source Distribution (emitted tokens)")
    ax2.set_ylabel("Count")
    for bar, v in zip(bars, sources.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(v), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    chart_path = out_dir / "cloud_density_chart.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"[C2] Chart saved to {chart_path}")
    return chart_path


#  Main 

def main():
    parser = argparse.ArgumentParser(description="C2 Cloud Density Bridge Analysis")
    parser.add_argument("--telemetry", required=True,
                        help="Path to test_live.tsv (--ott-live-telemetry output)")
    parser.add_argument("--axiom-report", default="axiom_beta_report.json",
                        help="Path to axiom_beta_report.json")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine",
                        help="Distance metric for density analysis")
    parser.add_argument("--out-dir", default=".", help="Directory for output files")
    args = parser.parse_args()

    tel_path   = Path(args.telemetry)
    axiom_path = Path(args.axiom_report)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tel_path.exists():
        print(f"[C2] Error: telemetry file not found: {tel_path}", file=sys.stderr)
        sys.exit(1)

    records, dim = load_telemetry(tel_path)
    if not records:
        print(f"[C2] No telemetry records found in {tel_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[C2] Loaded {len(records)} online states  (dim={dim}  metric={args.metric})")

    online_stats = analyse_online_density(records, metric=args.metric)

    offline_stats = {}
    if axiom_path.exists():
        offline_stats = load_axiom_report(axiom_path)
        print(f"[C2] Offline axiom stats: {offline_stats}")
    else:
        print(f"[C2] axiom report not found at {axiom_path} — skipping offline comparison",
              file=sys.stderr)

    bridge_gap = compute_bridge_gap(online_stats, offline_stats)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  C2 Cloud Density Report")
    print(f"{'='*60}")
    print(f"  Online states: {online_stats.get('n_online_states', 0)}")
    cs = online_stats.get("consecutive", {})
    print(f"  Consecutive distance  mean={cs.get('mean', 0):.6f}  "
          f"p50={cs.get('p50', 0):.6f}  p90={cs.get('p90', 0):.6f}  max={cs.get('max', 0):.6f}")
    nn = online_stats.get("nn_distance", {})
    print(f"  Nearest-neighbour dist mean={nn.get('mean', 0):.6f}  "
          f"p50={nn.get('p50', 0):.6f}  p90={nn.get('p90', 0):.6f}  max={nn.get('max', 0):.6f}")
    if bridge_gap.get("density_ratio"):
        print(f"\n  Offline-to-online density ratio: {bridge_gap['density_ratio']:.2f}x")
    if bridge_gap.get("recommendation"):
        print(f"\n  Recommendation:\n  {bridge_gap['recommendation']}")
    print(f"{'='*60}\n")

    # Save JSON
    report = {
        "telemetry_path": str(tel_path),
        "axiom_report_path": str(axiom_path) if axiom_path.exists() else None,
        "metric": args.metric,
        "online_stats": online_stats,
        "offline_stats": offline_stats,
        "bridge_gap": bridge_gap,
    }
    json_path = out_dir / "cloud_density_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[C2] Summary written to {json_path}")

    # Chart
    try_plot(records, online_stats, out_dir)


if __name__ == "__main__":
    main()
