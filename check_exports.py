import json, os, glob
for f in glob.glob('axiom_vis/*/manifold_export.json'):
    d = json.load(open(f))
    m = d['model']
    c = d['manifold']
    cv = d['curvature']
    slug = os.path.dirname(f).split('/')[1]
    cloud_n = len(c['cloud'])
    curv_n = len(cv['points'])
    r_min = min(p['R'] for p in cv['points'])
    r_max = max(p['R'] for p in cv['points'])
    print(f"{slug:20s}  {m['name'][:35]:35s}  arch={m['arch']:8s}  dim={m['dim']:5d}  cloud={cloud_n:3d}  curv={curv_n:3d}  R=[{r_min:.4f}, {r_max:.4f}]")
