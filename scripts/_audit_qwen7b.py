import json, sys, numpy as np
p = 'benchmarks/ugt_random_basis_layerwise_qwen7b_k200_ext_n3.json'
d = json.load(open(p, encoding='utf-8'))
print('top-level keys:', list(d.keys()))
print('config:', d.get('config'))
ps = d.get('per_seed') or d.get('seeds') or []
print('n_seeds entries:', len(ps))
agg = d['aggregated']['by_category_then_ablated_zone']
print()
for cat in ['syntax','algorithmic','factual']:
    for z in ['syntax','algorithmic','factual']:
        cell = agg[cat][z]
        keys = list(cell.keys())
        bv = cell.get('B_values') or cell.get('B_paired') or cell.get('B') or []
        rv = cell.get('Brand_values') or cell.get('Brand_paired') or cell.get('Brand') or []
        diff = cell.get('paired_diffs') or []
        print(f'{cat}/{z}: keys={keys[:8]} n_diff={len(diff)} delta={cell.get("B_minus_Brand"):+.4f}')
        if diff:
            arr = np.asarray(diff, dtype=float)
            print(f'   diffs: min={arr.min():+.4f} max={arr.max():+.4f} mean={arr.mean():+.4f} n_zero={(np.abs(arr)<1e-10).sum()}/{len(arr)}')
