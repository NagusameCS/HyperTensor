import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/ugt_random_basis_layerwise_qwen7b_k200_ext_n3.partial.json"
d = json.load(open(p, "r"))
print("seeds completed:", d["meta"].get("seeds_completed"), "/", d["meta"].get("seeds_target"))
for sr in d["per_seed"]:
    print(f"seed={sr['seed']} purity_B={sr['purity_B']:.4f}")
print("--- aggregate so far ---")
agg = d["aggregated"]["by_category_then_ablated_zone"]
for cat, zb in agg.items():
    for z, row in zb.items():
        delta = row["B_minus_Brand"]
        ps = row.get("paired_stats", {}) or {}
        n = ps.get("n", 0)
        tp = ps.get("t_p_two_sided") or ps.get("t_p")
        wp = ps.get("wilcoxon_p_two_sided") or ps.get("wilcoxon_p")
        ci_lo = ps.get("ci95_low"); ci_hi = ps.get("ci95_high")
        ci = f"[{ci_lo:+.4f},{ci_hi:+.4f}]" if ci_lo is not None else "[--,--]"
        tps = f"{tp:.3f}" if isinstance(tp, (int, float)) else str(tp)
        wps = f"{wp:.3f}" if isinstance(wp, (int, float)) else str(wp)
        print(f"  cat={cat:<12} z={z:<12} delta={delta:+.5f} CI95={ci} t_p={tps} w_p={wps} n={n}")
