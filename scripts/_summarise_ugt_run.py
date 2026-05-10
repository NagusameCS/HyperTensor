"""Quick summary of a UGT random-basis ablation results JSON."""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"missing: {p}")
        return 1
    d = json.loads(p.read_text(encoding="utf-8"))
    meta = d.get("meta", {})
    print(f"# {p.name}")
    print(f"# model={meta.get('model')}  k={meta.get('k')}  zones={meta.get('zones')}")
    print(f"# probe_suite={meta.get('probe_suite','?')}  n_probes={meta.get('n_probes','?')}  seeds={meta.get('seeds') or meta.get('seeds_completed')}")
    agg = d["aggregated"]["by_category_then_ablated_zone"]
    cats = ["syntax", "algorithmic", "factual"]
    print("\n# Diagonal cells (cat == zone) -- predicted positive under H_meaningful")
    for cat in cats:
        if cat not in agg or cat not in agg.get(cat, {}):
            continue
        cell = agg[cat][cat]
        ps = cell.get("paired_stats", {})
        diag = cell.get("B_minus_Brand")
        ci_lo = ps.get("ci95_low")
        ci_hi = ps.get("ci95_high")
        tp = ps.get("t_p_two_sided")
        wp = ps.get("wilcoxon_p_two_sided")
        n = ps.get("n")
        print(f"  {cat:<12}/{cat:<12}  B-B'={diag:+.4f}  "
              f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  "
              f"t_p={tp:.3f}  w_p={wp if wp is None else f'{wp:.3f}'}  "
              f"n_pairs={n}")
    print("\n# Off-diagonal cells (cat != zone) -- predicted near zero under H_meaningful")
    for cat in cats:
        for z in cats:
            if cat == z or cat not in agg or z not in agg.get(cat, {}):
                continue
            cell = agg[cat][z]
            ps = cell.get("paired_stats", {})
            tp = ps.get("t_p_two_sided")
            d_val = cell.get("B_minus_Brand")
            sig = tp is not None and tp < 0.05
            mark = " ***" if sig else ""
            print(f"  {cat:<12}/{z:<12}  B-B'={d_val:+.4f}  t_p={tp:.3f}{mark}")
    # Bonferroni reminder
    print("\n# Multiple-comparisons note: 9 cells, Bonferroni-adjusted alpha for FWER=0.05 is 0.0056")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/ugt_random_basis_ablation_smol135m_n8.json"))
