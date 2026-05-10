"""
Tier-list verification (#8 + #14): proper intrinsic-dim grid on real hidden states
+ per-layer manifold-curvature-ratio (MCR) computation.

Pulls hidden states from SmolLM2-135M-Instruct on wikitext, runs:
  - PCA-95 (cumulative variance)
  - TwoNN (Facco et al.)
  - MLE Levina-Bickel
on n>>d samples.

Then computes per-layer MCR = sigma_{k+1}^2 / sum_i sigma_i^2 at k=k_PCA95
and reports the full layer profile.
"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, 'scripts')
from ablation_utils import intrinsic_dim_compare
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
N_TEXTS = 64    # number of independent text segments
SEQ = 64        # tokens per segment -> 64*64 = 4096 hidden-state vectors >> d=576

print(f"loading {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
d = model.config.hidden_size
L = model.config.num_hidden_layers
print(f"d={d} L={L}")

# Read wikitext
texts = []
with open('data/wikitext2_train_5k.txt', encoding='utf-8') as f:
    buf = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if len(buf) >= 4:
            texts.append(' '.join(buf))
            buf = []
        if len(texts) >= N_TEXTS:
            break

print(f"texts collected: {len(texts)}")

# Forward, capture hidden states from every layer
all_states = [[] for _ in range(L+1)]   # layer 0 = embeddings, 1..L = after each block
with torch.no_grad():
    for t in texts:
        enc = tok(t, return_tensors='pt', truncation=True, max_length=SEQ).to(device)
        out = model(**enc, output_hidden_states=True)
        for li, h in enumerate(out.hidden_states):
            # h: (1, seq, d)
            all_states[li].append(h.squeeze(0).cpu().float().numpy())
states = [np.concatenate(xs, axis=0) for xs in all_states]   # (n_total, d) per layer
n_total = states[0].shape[0]
print(f"total hidden vectors per layer: {n_total} (vs d={d})")

# Run intrinsic-dim grid on a representative subset of layers
LAYERS = [0, L//4, L//2, 3*L//4, L]
grid_rows = []
for li in LAYERS:
    X = states[li]
    res = intrinsic_dim_compare(X, max_dim=d)
    grid_rows.append((li, res['pca_95'], res['twonn'], res['mle_levina_bickel']))
    print(f"layer {li:>3}: PCA-95={res['pca_95']:>4} TwoNN={res['twonn']:>4} MLE-LB={res['mle_levina_bickel']:>4}")

# Per-layer MCR (manifold-curvature-ratio)
# MCR_l = sigma_{l, k_l+1}^2 / sum_i sigma_{l,i}^2 evaluated at PCA-95 cut
print("\nPer-layer MCR (k = PCA-95 truncation):")
mcr_rows = []
for li in range(L+1):
    X = states[li]
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s**2
    cum = np.cumsum(s2) / s2.sum()
    k95 = int(np.searchsorted(cum, 0.95)) + 1
    mcr = float(s2[min(k95, len(s2)-1)] / s2.sum())
    mcr_rows.append({'layer': li, 'k_PCA95': k95, 'mcr_at_k95': mcr,
                     'top_singular_share': float(s2[0]/s2.sum())})
    if li % 4 == 0 or li == L:
        print(f"  layer {li:>3}: k_PCA95={k95:>4}  MCR={mcr:.5f}  top-1 share={s2[0]/s2.sum():.3f}")

# Spread of MCR across layers tells us if rank allocation should vary.
mcr_arr = np.array([r['mcr_at_k95'] for r in mcr_rows])
k95_arr = np.array([r['k_PCA95'] for r in mcr_rows])
print(f"\nk_PCA95 across layers: min={k95_arr.min()} max={k95_arr.max()} mean={k95_arr.mean():.1f} std={k95_arr.std():.1f}")
print(f"MCR across layers:      min={mcr_arr.min():.5f} max={mcr_arr.max():.5f} mean={mcr_arr.mean():.5f} cv={mcr_arr.std()/mcr_arr.mean():.3f}")
print(f"  -> low CV ({mcr_arr.std()/mcr_arr.mean():.2f}) means flat profile; MCR-driven per-layer allocation will degenerate to ~uniform.")

out = {
    'model': MODEL,
    'd': d, 'L': L, 'n_total': int(n_total),
    'intrinsic_dim_grid': [
        {'layer': li, 'pca_95': p, 'twonn': t, 'mle_lb': m}
        for li,p,t,m in grid_rows
    ],
    'per_layer_mcr': mcr_rows,
    'k95_summary': dict(min=int(k95_arr.min()), max=int(k95_arr.max()),
                        mean=float(k95_arr.mean()), std=float(k95_arr.std())),
    'mcr_summary': dict(min=float(mcr_arr.min()), max=float(mcr_arr.max()),
                        mean=float(mcr_arr.mean()),
                        cv=float(mcr_arr.std()/mcr_arr.mean())),
}
os.makedirs('benchmarks', exist_ok=True)
json.dump(out, open('benchmarks/intrinsic_dim_real_grid.json','w'), indent=2)
print('\nwrote benchmarks/intrinsic_dim_real_grid.json')
