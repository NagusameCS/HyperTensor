#!/usr/bin/env python3
"""
scripts/paperA_proof/decide_l2_mystery.py

Given the multi-k NCU sweep summary (and optionally the Exp B summary),
issue an honest verdict on the L2-cache-fit hypothesis from Paper A.

Hypothesis (cache-fit):
  the GRC speed-up over baseline is mediated by the GRC-fused projection
  fitting the persistent working-set in L2 (32 MB on RTX 4070 Laptop),
  while the legacy three-launch path streams ~28-30 MB of weights per
  attention block and gets cold-misses.

Quantitative budget (per attention block, fused path):
    S(k) = 2 * d * k_bytes_compressed + 2 * k * k_bytes_compressed + headers
         ~= 4096 * k * 1.0 + k * k * 1.0 + small   (Q8_0, ~1 byte/weight)
         ~= 4096*k + k^2     bytes

Predictions (Paper A, sec:falsification):
    P1A: GRC L2 hit-rate > baseline L2 hit-rate by at least 3 pp.
    P1B: GRC L2 hit-rate is non-increasing in k for k >= 384.
    P1C: A monotone trend across the multi-k sweep is at least suggestive
         of cache-fit; flat curves are consistent (all sizes still fit).
    P2:  (Exp B) GRC speed-up shrinks as the L2 thrasher delta grows.

Output: a markdown fragment + JSON verdict at
    docs/figures/paper-a/ncu_sweep/l2_verdict.{md,json}

Status terminology (per user instruction; honest hedging):
    REFUTED      --- observation contradicts the prediction
    CONFIRMED    --- observation positively matches the prediction
    CONSISTENT   --- observation does not contradict, but does not require
                   the cache-fit explanation either
    INCONCLUSIVE --- not enough data to decide
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "docs" / "figures" / "paper-a" / "ncu_sweep" / "sweep_summary.json"
EXPB  = ROOT / "docs" / "figures" / "paper-a" / "expB_thrash" / "expB_summary.json"
OUT_MD   = ROOT / "docs" / "figures" / "paper-a" / "ncu_sweep" / "l2_verdict.md"
OUT_JSON = ROOT / "docs" / "figures" / "paper-a" / "ncu_sweep" / "l2_verdict.json"

L2_BYTES = 32 * 1024 * 1024  # RTX 4070 Laptop


def budget_mb(k: int, d: int = 4096) -> float:
    """Approx working-set per attention block, fused GRC path (bytes)."""
    return (4096.0 * k + k * k) / (1024 * 1024)


def fmt(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def verdict_p1a(baseline_pct, grc_pcts):
    if baseline_pct is None or not grc_pcts:
        return "INCONCLUSIVE", "no baseline or no GRC k points"
    deltas = [g - baseline_pct for g in grc_pcts]
    min_d, max_d = min(deltas), max(deltas)
    if min_d >= 3.0:
        return "CONFIRMED", f"every GRC k beats baseline by >=3pp (min delta {min_d:.2f}pp)"
    if max_d >= 3.0:
        return "CONSISTENT", f"some GRC k beat baseline by >=3pp (max {max_d:.2f}pp, min {min_d:.2f}pp)"
    if max_d > 0:
        return "CONSISTENT", f"GRC > baseline but by <3pp (max {max_d:.2f}pp)"
    return "REFUTED", f"GRC L2 hit-rate not above baseline (max delta {max_d:.2f}pp)"


def verdict_p1b(ks_pcts):
    """Monotone non-increasing in k from k>=384?"""
    pts = sorted([(k, v) for k, v in ks_pcts.items() if v is not None])
    if len(pts) < 3:
        return "INCONCLUSIVE", f"only {len(pts)} points"
    diffs = [pts[i+1][1] - pts[i][1] for i in range(len(pts)-1)]
    n_up = sum(1 for d in diffs if d > 0.5)   # tolerate 0.5pp noise
    n_down = sum(1 for d in diffs if d < -0.5)
    if n_up == 0 and n_down >= 1:
        return "CONFIRMED", f"non-increasing with {n_down}/{len(diffs)} clear drops"
    if n_up >= 1 and n_down >= 1:
        return "CONSISTENT", f"non-monotone ({n_up} up, {n_down} down >0.5pp); plateau under L2 budget"
    if n_up >= 2:
        return "REFUTED", f"GRC hit-rate increases with k ({n_up}/{len(diffs)} up-steps)"
    return "CONSISTENT", f"flat (no >0.5pp transitions in {len(diffs)} steps); all k still under L2 budget"


def verdict_p2(expB):
    if not expB or "by_delta" not in expB:
        return "INCONCLUSIVE", "no Exp B data"
    deltas = sorted(int(d) for d in expB["by_delta"])
    sps = [expB["by_delta"][str(d)].get("speedup_proxy") for d in deltas]
    sps = [(d, s) for d, s in zip(deltas, sps) if s is not None]
    if len(sps) < 2:
        return "INCONCLUSIVE", f"only {len(sps)} delta points"
    first, last = sps[0][1], sps[-1][1]
    if first > 1.05 and last <= 1.05 and last < first - 0.05:
        return "CONFIRMED", f"speedup proxy collapses {first:.3f} -> {last:.3f} as delta grows"
    if first > 1.05 and last < first - 0.05:
        return "CONSISTENT", f"speedup proxy decays {first:.3f} -> {last:.3f}"
    if abs(first - last) <= 0.05:
        return "CONSISTENT-OR-FUSION", f"flat speedup proxy ({first:.3f} -> {last:.3f}); fusion-only also explains this"
    return "INCONCLUSIVE", f"first={first:.3f}, last={last:.3f}, mixed trend"


def main():
    if not SWEEP.exists():
        print(f"missing {SWEEP}", file=sys.stderr); sys.exit(1)
    sweep = json.loads(SWEEP.read_text())
    expB = json.loads(EXPB.read_text()) if EXPB.exists() else None

    by_k = sweep.get("by_k", {})
    baseline = by_k.get("baseline", {}).get("attention_kernel_l2_hit_pct_mean")
    ks = {int(k): v["attention_kernel_l2_hit_pct_mean"]
          for k, v in by_k.items() if k != "baseline" and v.get("attention_kernel_l2_hit_pct_mean") is not None}

    p1a_v, p1a_r = verdict_p1a(baseline, list(ks.values()))
    p1b_v, p1b_r = verdict_p1b(ks)
    p2_v,  p2_r  = verdict_p2(expB)

    overall = {
        ("CONFIRMED", "CONFIRMED"): "CONFIRMED",
        ("CONFIRMED", "CONSISTENT"): "CONSISTENT-WITH-CONFIRMATION",
    }.get((p1a_v, p1b_v), p1a_v if p1a_v == p1b_v else "MIXED")

    verdict = {
        "L2_bytes_4070_laptop": L2_BYTES,
        "baseline_attn_l2_hit_pct": baseline,
        "grc_attn_l2_hit_pct_by_k": ks,
        "approx_budget_mb_by_k": {str(k): budget_mb(k) for k in ks},
        "P1A_above_baseline": {"verdict": p1a_v, "reason": p1a_r},
        "P1B_monotone_in_k":  {"verdict": p1b_v, "reason": p1b_r},
        "P2_thrasher_kills_speedup": {"verdict": p2_v, "reason": p2_r},
        "overall_intra_run":  overall,
    }
    OUT_JSON.write_text(json.dumps(verdict, indent=2))

    md = []
    md.append("# Paper A: L2 cache-fit hypothesis verdict\n")
    md.append("Auto-generated by `scripts/paperA_proof/decide_l2_mystery.py`. ")
    md.append("Status terms: CONFIRMED, CONSISTENT, REFUTED, INCONCLUSIVE.\n\n")
    md.append("## Inputs\n")
    md.append(f"- Multi-k NCU sweep: `{SWEEP.relative_to(ROOT)}`\n")
    md.append(f"- Exp B thrash sweep: " + (f"`{EXPB.relative_to(ROOT)}`" if expB else "(absent)") + "\n\n")
    md.append("## Per-k attention-kernel L2 hit-rate\n\n")
    md.append("| k | budget S(k) (MB) | fits in 32 MB L2? | attn L2 hit-rate (%) |\n")
    md.append("|---|---:|---|---:|\n")
    md.append(f"| baseline | (legacy 3-launch) | streams ~28-30 MB/block | {fmt(baseline,3)} |\n")
    for k in sorted(ks):
        b = budget_mb(k)
        fits = "yes" if b * 1024 * 1024 < L2_BYTES else "no"
        md.append(f"| {k} | {b:.2f} | {fits} | {fmt(ks[k],3)} |\n")
    md.append("\n## Verdicts\n\n")
    md.append(f"- **P1A (GRC > baseline by >=3pp):** {p1a_v} - {p1a_r}\n")
    md.append(f"- **P1B (non-increasing in k):** {p1b_v} - {p1b_r}\n")
    md.append(f"- **P2 (thrasher kills GRC speed-up):** {p2_v} - {p2_r}\n")
    md.append(f"- **Overall (Exp A only):** {overall}\n")
    OUT_MD.write_text("".join(md))

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print("".join(md))


if __name__ == "__main__":
    main()
