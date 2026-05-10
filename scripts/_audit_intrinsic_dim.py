"""
Tier-list verification (#8): intrinsic-dim grid (PCA-95 vs TwoNN vs MLE Levina-Bickel)
on the legacy axiom_vis cloud points -- the same data the volume's
"Cross-Architecture Manifold Evidence" table is built from.

Volume claim: "Intrinsic-dimensionality estimates (PCA, 95% variance retained)"
              SmolLM2-135M -> 17, Gemma-4-E2B -> 25, Phi-3.5-mini -> 11.

Audit plan:
  1. Verify those numbers are actually PCA-95 (vs TwoNN).
  2. Add MLE/Levina-Bickel column the volume implies but doesn't report.
  3. Sanity-check sample sufficiency (n vs d).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from ablation_utils import intrinsic_dim_compare

models = ['smollm2-135m', 'gemma-4-e2b', 'phi-3.5-mini']
volume_claim = {'smollm2-135m': 17, 'gemma-4-e2b': 25, 'phi-3.5-mini': 11}

print(f"{'model':<14} {'d':>6} {'n':>4} {'json.intrinsic_dim':>18} {'json.pca_comp':>14} {'json.twonn_raw':>16} {'recompute_PCA95':>16} {'recompute_TwoNN':>16} {'recompute_MLE_LB':>17}")
out = {}
for m in models:
    p1 = json.load(open(f'legacy/axiom_vis/{m}/phase1_manifold.json'))
    cloud = np.array(p1['cloud'], dtype=float)
    n, k_amb = cloud.shape  # cloud is in projected 3D! Not the ambient d.
    d_model = p1['model_dim']
    json_id = p1['intrinsic_dim']
    json_pca = p1['pca_components']
    json_twonn = p1['twonn_raw']

    # Recompute on the cloud as stored (already projected to 3D, so this is misleading)
    res_proj = intrinsic_dim_compare(cloud, max_dim=k_amb)

    out[m] = dict(
        d_model=d_model, n_samples=n, cloud_dim=k_amb,
        json_intrinsic_dim=json_id, json_pca_components=json_pca, json_twonn_raw=json_twonn,
        recompute_on_3d_cloud=res_proj,
    )
    print(f"{m:<14} {d_model:>6} {n:>4} {json_id:>18} {json_pca:>14} {json_twonn:>16.3f} {res_proj['pca_95']:>16} {res_proj['twonn']:>16} {res_proj['mle_levina_bickel']:>17}")

print()
print("KEY OBSERVATIONS:")
print("- 'cloud' in phase1_manifold.json is already a 3-D projection (PCA top-3 likely),")
print("  so re-running intrinsic-dim on the cloud = upper bound 3.")
print("- The 17/25/11 in the volume's PCA-95 column actually equal `intrinsic_dim`,")
print("  which inside the JSON is set to `twonn_raw` (rounded). The PCA-95 component")
print("  count (`pca_components`) is 46/54/31 -- 2.7-3x larger than the table claims.")
print("- n_samples=64 for all three; for Phi-3.5 (d=3072), 64 samples is below standard")
print("  rule-of-thumb (n > 10*k_intrinsic) and any high-d intrinsic-dim estimate is noisy.")

os.makedirs('benchmarks', exist_ok=True)
json.dump(out, open('benchmarks/intrinsic_dim_grid_audit.json','w'), indent=2)
print()
print("wrote benchmarks/intrinsic_dim_grid_audit.json")
