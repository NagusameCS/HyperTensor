import json, numpy as np
p = 'benchmarks/ugt_random_basis_layerwise_qwen7b_k200_ext_n3.json'
d = json.load(open(p, encoding='utf-8'))
ps = d['per_seed']
print(f'seeds = {[s.get("seed") for s in ps]}')
print(f'meta = {d.get("meta")}')
print()
# For each seed, print per-cell mean delta under B and B'
for s in ps:
    seed = s.get('seed')
    print(f'--- seed {seed} ---')
    cells = s.get('cells') or s.get('by_category_then_ablated_zone') or {}
    if not cells:
        print('  keys:', list(s.keys()))
        continue
    for cat in ['syntax','algorithmic','factual']:
        for z in ['syntax','algorithmic','factual']:
            c = cells.get(cat, {}).get(z, {})
            print(f'  {cat}/{z}: keys={list(c.keys())[:10]}')
            break
        break
    break
print()
# Dump full structure of first seed
print('full first-seed structure:')
import pprint
pprint.pprint(ps[0], depth=3, width=120)
