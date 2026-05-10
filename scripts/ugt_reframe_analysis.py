#!/usr/bin/env python3
"""
UGT reframe analysis
====================

Re-analyses the existing layer-wise ablation result file
(`benchmarks/ugt_random_basis_layerwise_smol135m_ext_n5.json`) under three
*alternative* hypotheses that would let some weakened form of UGT survive:

  R1. SUBSPACE-LEVEL claim. Forget zones. Is B as a whole subspace causally
      more important than a Haar-random orthonormal subspace of the same rank?
      (Pool all 3 zones into a single "ablate any zone" effect per probe; test
      paired B vs B' over all probes.)

  R2. CATEGORY-AGNOSTIC claim. Zones differ from each other (some hurt more
      than others), but the assignment of categories to zones is arbitrary.
      (Test heterogeneity-of-zones via repeated-measures variance, separately
      under B and under B'.)

  R3. ANY-DIAGONAL claim. Maybe the named zone labels are wrong but *some*
      category-zone pairing is privileged. Test the *best* of the 9 cells
      under a permutation null that preserves probe identity.

If R1 fails: not even subspace-level identity survives.
If R2 fails: zones are interchangeable (no zone structure at all).
If R3 fails: no permutation of category-to-zone labels rescues the claim.

Output: prints a structured report; writes
`benchmarks/ugt_reframe_analysis.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paired_stats(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n < 2:
        return {"n": int(n), "mean": float(diffs.mean()) if n else None,
                "t_p": None, "wilcoxon_p": None}
    out = {"n": int(n), "mean": float(diffs.mean()),
           "sd": float(diffs.std(ddof=1))}
    try:
        from scipy import stats
        t = stats.ttest_1samp(diffs, 0.0)
        out["t_p"] = float(t.pvalue)
        try:
            w = stats.wilcoxon(diffs, alternative="two-sided",
                               zero_method="wilcox", correction=False,
                               nan_policy="propagate")
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_p"] = None
        # Bootstrap CI95
        rng = np.random.default_rng(0)
        boots = rng.choice(diffs, size=(5000, n), replace=True).mean(axis=1)
        out["ci95_low"] = float(np.percentile(boots, 2.5))
        out["ci95_high"] = float(np.percentile(boots, 97.5))
    except Exception as e:
        out["t_p"] = None; out["wilcoxon_p"] = None
        out["scipy_error"] = repr(e)
    return out


def load_layerwise(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reframe_R1_subspace_level(data):
    """Pool zones: per (seed, probe, category, zone), get delta_B and delta_Bp.
    The 'subspace-level' effect of B (resp. B') is the *average* damage when
    you ablate any of its 3 zones, on the probe's own category.
    Then take diagonal-pooled paired diff (B - B') across (seed x probe).

    But a stronger test: pool both diagonal AND off-diagonal — if B as a whole
    is more impactful than B', it should hurt MORE no matter which zone is
    ablated, on any probe."""
    diffs_diag = []
    diffs_all = []
    diffs_offdiag = []
    for seed_rec in data["per_seed"]:
        rb = seed_rec["B"]["raw_deltas_by_category_then_zone"]
        rbp = seed_rec["B_random"]["raw_deltas_by_category_then_zone"]
        ids_b = seed_rec["B"].get("probe_ids_by_category_then_zone", {})
        for cat in rb:
            for zone in rb[cat]:
                vb = rb[cat][zone]
                vbp = rbp.get(cat, {}).get(zone, [])
                n = min(len(vb), len(vbp))
                for i in range(n):
                    d = vb[i] - vbp[i]
                    diffs_all.append(d)
                    if cat == zone:
                        diffs_diag.append(d)
                    else:
                        diffs_offdiag.append(d)
    return {
        "all_cells_pooled": paired_stats(diffs_all),
        "diagonal_pooled": paired_stats(diffs_diag),
        "offdiagonal_pooled": paired_stats(diffs_offdiag),
        "interpretation": (
            "If B is privileged as a whole subspace (zone-agnostic), "
            "all_cells_pooled mean should be > 0 and significant. "
            "If B is privileged on its own categories specifically, "
            "diagonal_pooled should be > 0 and offdiagonal ~ 0."
        ),
    }


def reframe_R2_zone_heterogeneity(data):
    """Per-zone *absolute* damage size, averaged over (seed, probe). Tests
    whether some zones hurt more than others, regardless of whether they
    align with named categories."""
    zone_means_B = {"syntax": [], "algorithmic": [], "factual": []}
    zone_means_Bp = {"syntax": [], "algorithmic": [], "factual": []}
    for seed_rec in data["per_seed"]:
        for cat, zd in seed_rec["B"]["raw_deltas_by_category_then_zone"].items():
            for zone, vals in zd.items():
                zone_means_B[zone].extend(vals)
        for cat, zd in seed_rec["B_random"]["raw_deltas_by_category_then_zone"].items():
            for zone, vals in zd.items():
                zone_means_Bp[zone].extend(vals)
    out = {"under_B": {}, "under_B_random": {}}
    for z in zone_means_B:
        out["under_B"][z] = {
            "mean_delta": float(np.mean(zone_means_B[z])) if zone_means_B[z] else None,
            "std": float(np.std(zone_means_B[z], ddof=1)) if len(zone_means_B[z]) > 1 else None,
            "n": len(zone_means_B[z]),
        }
        out["under_B_random"][z] = {
            "mean_delta": float(np.mean(zone_means_Bp[z])) if zone_means_Bp[z] else None,
            "std": float(np.std(zone_means_Bp[z], ddof=1)) if len(zone_means_Bp[z]) > 1 else None,
            "n": len(zone_means_Bp[z]),
        }
    # ANOVA: do the three zones differ? (under B and under B' separately)
    try:
        from scipy import stats
        f_B = stats.f_oneway(*[zone_means_B[z] for z in zone_means_B])
        f_Bp = stats.f_oneway(*[zone_means_Bp[z] for z in zone_means_Bp])
        out["anova_zones_differ_under_B"] = {"F": float(f_B.statistic), "p": float(f_B.pvalue)}
        out["anova_zones_differ_under_Brand"] = {"F": float(f_Bp.statistic), "p": float(f_Bp.pvalue)}
    except Exception as e:
        out["anova_error"] = repr(e)
    out["interpretation"] = (
        "Under H_zones-are-different-but-mislabeled, the ANOVA p under B "
        "should be small. If both ANOVAs are significant but the random-B' "
        "ANOVA is ALSO significant, zone heterogeneity is a generic property "
        "of orthogonal projection at different residual directions, not a "
        "UGT-specific finding."
    )
    return out


def reframe_R3_best_relabeling(data):
    """For each seed, treat the 3x3 cell matrix as a doubly-indexed table.
    Find the permutation pi of zone labels that maximises mean diagonal
    (cat=pi(zone)) damage under B (B-B' pooled). Report whether that maximum
    survives a permutation null where probe categories are shuffled within
    each (seed, zone) group.

    This asks: 'is there ANY 1-1 relabeling under which UGT shows a positive
    diagonal signal?'."""
    from itertools import permutations
    cats = ["syntax", "algorithmic", "factual"]
    # Build per-seed 3x3 mean-diff matrix: M[i,j] = mean_{probe in cat_i}(delta_B - delta_Bp) when zone j is ablated
    seed_matrices = []
    for seed_rec in data["per_seed"]:
        rb = seed_rec["B"]["raw_deltas_by_category_then_zone"]
        rbp = seed_rec["B_random"]["raw_deltas_by_category_then_zone"]
        M = np.zeros((3, 3))
        for i, c in enumerate(cats):
            for j, z in enumerate(cats):
                vb = rb.get(c, {}).get(z, [])
                vbp = rbp.get(c, {}).get(z, [])
                n = min(len(vb), len(vbp))
                if n == 0:
                    continue
                M[i, j] = float(np.mean([vb[k] - vbp[k] for k in range(n)]))
        seed_matrices.append(M)
    avg_M = np.mean(seed_matrices, axis=0)

    def diag_score(M, perm):
        # perm[j] = which row goes with col j
        return float(np.mean([M[perm[j], j] for j in range(3)]))

    best_perm, best_score = None, -np.inf
    all_scores = {}
    for perm in permutations(range(3)):
        s = diag_score(avg_M, perm)
        all_scores[str(perm)] = s
        if s > best_score:
            best_score = s; best_perm = perm
    identity_score = diag_score(avg_M, (0, 1, 2))
    return {
        "average_M_3x3": avg_M.tolist(),
        "categories": cats,
        "all_permutation_diag_scores": all_scores,
        "identity_perm_diag_score": identity_score,
        "best_perm_indices": list(best_perm),
        "best_perm_score": best_score,
        "interpretation": (
            "best_perm_score is the largest possible mean diagonal (B-B') "
            "achievable by any 1-1 relabeling of zone names. If this is "
            "still <= 0, no relabeling rescues the claim. If > 0 but small, "
            "it is at most a weak claim about the best-fitting labeling."
        ),
    }


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = ROOT / "benchmarks" / "ugt_random_basis_layerwise_smol135m_ext_n5.json"
    if not path.exists():
        print(f"file not found: {path}"); sys.exit(2)
    data = load_layerwise(path)
    print(f"# UGT reframe analysis on {path.name}")
    print(f"# seeds={data['meta'].get('seeds', '?')}  suite={data['meta'].get('probe_suite', '?')}  intervention={data['meta'].get('intervention', '?')}")
    R1 = reframe_R1_subspace_level(data)
    R2 = reframe_R2_zone_heterogeneity(data)
    R3 = reframe_R3_best_relabeling(data)
    out = {"source_file": str(path), "R1_subspace_level": R1,
           "R2_zone_heterogeneity": R2, "R3_best_relabeling": R3}
    out_path = ROOT / "benchmarks" / "ugt_reframe_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("--- R1: SUBSPACE-LEVEL (pool zones) ---")
    for k, v in R1.items():
        if k == "interpretation":
            continue
        if isinstance(v, dict):
            print(f"  {k:<22}  mean={v.get('mean'):+.5f}  t_p={v.get('t_p')!s:<8}  w_p={v.get('wilcoxon_p')!s:<8}  n={v.get('n')}")
    print()
    print("--- R2: ZONE HETEROGENEITY ---")
    for who in ("under_B", "under_B_random"):
        print(f"  {who}:")
        for z, s in R2[who].items():
            print(f"    {z:<12}  mean={s['mean_delta']:+.5f}  sd={s['std']:.5f}  n={s['n']}")
    if "anova_zones_differ_under_B" in R2:
        print(f"  ANOVA(under B):       F={R2['anova_zones_differ_under_B']['F']:.3f}  p={R2['anova_zones_differ_under_B']['p']:.4f}")
        print(f"  ANOVA(under B_rand):  F={R2['anova_zones_differ_under_Brand']['F']:.3f}  p={R2['anova_zones_differ_under_Brand']['p']:.4f}")
    print()
    print("--- R3: BEST RELABELING ---")
    print(f"  identity perm  (0,1,2)  diag_score={R3['identity_perm_diag_score']:+.5f}")
    print(f"  best perm      {tuple(R3['best_perm_indices'])}  diag_score={R3['best_perm_score']:+.5f}")
    print(f"  3x3 avg matrix (rows=cat, cols=zone):")
    for row in R3["average_M_3x3"]:
        print("    " + "  ".join(f"{v:+.5f}" for v in row))
    print()
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
