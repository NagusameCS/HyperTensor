"""
Statistical hypothesis tests for the GRC super-baseline result at k=1024.

Reads paired baseline / GRC throughput measurements from the validated
whitepaper pack, runs:

  H1: paired t-test, one-sided   (GRC decode > baseline decode at k=1024)
  H2: Wilcoxon signed-rank test  (non-parametric robustness check)
  H3: bootstrap 95% CI on the mean ratio (decode_grc / decode_baseline)
  H4: same tests for k=1536 (expecting GRC ≤ baseline)
  H5: paired t-test on CI 12-rep coding/256 and reasoning/256 data

Outputs:
  docs/figures/statistical_tests.json
  docs/figures/throughput_paired.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACK = Path("benchmarks/whitepaper_pack_20260427_121815")
OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rank_sweep():
    """Return dict[(prompt, tokens, rank)] -> {'baseline': float, 'grc': float}."""
    rows = list(csv.DictReader(open(PACK / "rank_sweep_relative_to_baseline.csv")))
    out = {}
    for r in rows:
        key = (r["prompt"], int(r["tokens"]), int(r["rank"]))
        out[key] = {
            "baseline_decode": float(r["baseline_decode_tps"]),
            "grc_decode": float(r["grc_decode_tps"]),
            "baseline_overall": float(r["baseline_overall_tps"]),
            "grc_overall": float(r["grc_overall_tps"]),
        }
    return out


def load_ci_raw():
    """12 paired reps per (case, mode). Returns dict[case] -> dict['baseline','grc'] -> array."""
    rows = list(csv.DictReader(open(PACK / "ci_pack_raw.csv")))
    cases: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        label = r["label"]
        # labels like "outlier_baseline_coding_256" or "outlier_grc_coding_256_k2048"
        if "_baseline_" in label:
            mode = "baseline"
            case = label.split("_baseline_", 1)[1]
        elif "_grc_" in label:
            mode = "grc"
            case = label.split("_grc_", 1)[1].rsplit("_k", 1)[0]
        else:
            continue
        cases.setdefault(case, {"baseline": [], "grc": []})[mode].append(
            float(r["decode_tps"]))
    return cases


def paired_test(baseline, grc, hypothesis="greater"):
    """Run paired t-test and Wilcoxon. hypothesis = 'greater' tests grc > baseline."""
    baseline = np.asarray(baseline, dtype=float)
    grc = np.asarray(grc, dtype=float)
    diff = grc - baseline

    # Paired t-test (one-sided)
    t_stat, t_p = stats.ttest_rel(grc, baseline, alternative=hypothesis)

    # Wilcoxon signed-rank (one-sided)
    try:
        w_stat, w_p = stats.wilcoxon(grc, baseline, alternative=hypothesis,
                                     zero_method="pratt")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")

    # Bootstrap 95% CI on mean ratio
    rng = np.random.default_rng(42)
    n = len(diff)
    ratios = grc / baseline
    boot_means = np.array([
        rng.choice(ratios, size=n, replace=True).mean() for _ in range(10000)
    ])

    return {
        "n": n,
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std(ddof=1)),
        "grc_mean": float(grc.mean()),
        "grc_std": float(grc.std(ddof=1)),
        "mean_diff": float(diff.mean()),
        "mean_ratio": float(ratios.mean()),
        "ratio_ci95_lo": float(np.percentile(boot_means, 2.5)),
        "ratio_ci95_hi": float(np.percentile(boot_means, 97.5)),
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "alternative": hypothesis,
        "significant_at_0.05": bool(t_p < 0.05),
    }


def main():
    results = {}

    #  Rank sweep paired data 
    sweep = load_rank_sweep()
    for k in [1024, 1536]:
        baseline = []
        grc = []
        for (prompt, toks, rank), v in sweep.items():
            if rank != k:
                continue
            baseline.append(v["baseline_decode"])
            grc.append(v["grc_decode"])
        hyp = "greater" if k == 1024 else "two-sided"
        results[f"rank_sweep_k{k}_decode"] = paired_test(baseline, grc, hyp)

    #  CI 12-rep data 
    ci = load_ci_raw()
    for case, modes in ci.items():
        if len(modes["baseline"]) != len(modes["grc"]):
            n = min(len(modes["baseline"]), len(modes["grc"]))
            modes["baseline"] = modes["baseline"][:n]
            modes["grc"] = modes["grc"][:n]
        results[f"ci_{case}_decode"] = paired_test(
            modes["baseline"], modes["grc"], "two-sided")

    #  Output JSON 
    out_path = OUT_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    #  Print summary 
    print(f"\n{'='*70}\nSTATISTICAL TEST RESULTS\n{'='*70}")
    for key, r in results.items():
        sig = " SIGNIFICANT" if r["significant_at_0.05"] else " NOT SIGNIFICANT"
        print(f"\n[{key}]  n={r['n']}  alt={r['alternative']}")
        print(f"  baseline: {r['baseline_mean']:.3f} ± {r['baseline_std']:.3f}")
        print(f"  GRC:      {r['grc_mean']:.3f} ± {r['grc_std']:.3f}")
        print(f"  ratio:    {r['mean_ratio']:.4f} (95% CI [{r['ratio_ci95_lo']:.4f}, {r['ratio_ci95_hi']:.4f}])")
        print(f"  t-test:   t={r['t_stat']:.3f}, p={r['t_pvalue']:.4g}  {sig}")
        print(f"  Wilcoxon: W={r['wilcoxon_stat']:.3f}, p={r['wilcoxon_pvalue']:.4g}")

    #  Plot paired throughput 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, k in zip(axes, [1024, 1536]):
        baseline, grc, labels = [], [], []
        for (prompt, toks, rank), v in sorted(sweep.items()):
            if rank != k:
                continue
            baseline.append(v["baseline_decode"])
            grc.append(v["grc_decode"])
            labels.append(f"{prompt[:4]}/{toks}")
        x = np.arange(len(labels))
        for i in range(len(labels)):
            color = "#1a7a4a" if grc[i] > baseline[i] else "#c0392b"
            ax.plot([0, 1], [baseline[i], grc[i]], color=color, alpha=0.55, lw=1.2)
            ax.scatter([0], [baseline[i]], color="#5c5c5c", s=22, zorder=3)
            ax.scatter([1], [grc[i]], color=color, s=22, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", f"GRC k={k}"])
        ax.set_ylabel("Decode throughput (tok/s)")
        r = results[f"rank_sweep_k{k}_decode"]
        ax.set_title(f"k={k}: ratio = {r['mean_ratio']*100:.2f}%, "
                     f"t-test p = {r['t_pvalue']:.3g}")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Paired throughput: 8 prompt configurations per rank",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "throughput_paired.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n Wrote {OUT_DIR / 'statistical_tests.json'}")
    print(f" Wrote {OUT_DIR / 'throughput_paired.png'}")


if __name__ == "__main__":
    main()
