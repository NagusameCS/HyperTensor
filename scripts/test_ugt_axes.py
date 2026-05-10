#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
Test UGT k-axis decomposition hypothesis:
  k = k_axis1 + k_axis2 + k_residual
  where k_axis1 captures Discovery/Construction spectrum
  and   k_axis2 captures Objective/Subjective spectrum

Method:
  1. Build controlled calibration set with explicit axis labels
     - Each prompt tagged with (axis1, axis2) = (D/C, O/S)
  2. Extract hidden states, build UGT basis via SVD
  3. Project all states onto UGT basis
  4. For each UGT coordinate, measure:
     - axis1_discriminability: how well it separates D vs C
     - axis2_discriminability: how well it separates O vs S
  5. Sort coordinates by axis discriminability
  6. Check whether top coordinates naturally separate into
     axis1-dominant, axis2-dominant, and residual groups
"""
import torch, numpy as np, warnings, json, os
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
OUT = 'benchmarks/ugt_axes'
os.makedirs(OUT, exist_ok=True)

# Controlled prompts per quadrant
prompts = {
    # Discovery + Objective (Science-like)
    'D_O': [
        "The boiling point of water at sea level is 100 degrees Celsius.",
        "Photosynthesis converts CO2 and water into glucose using sunlight.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "DNA is a double helix structure held together by hydrogen bonds.",
        "The Earth orbits the Sun at an average distance of 149.6 million kilometers.",
    ],
    # Construction + Objective (Math-like)
    'C_O': [
        "The Pythagorean theorem states a squared plus b squared equals c squared.",
        "A prime number is a natural number greater than 1 with exactly two divisors.",
        "The derivative of x cubed is 3x squared by the power rule.",
        "A group is a set with an associative operation, identity, and inverses.",
        "The determinant of a 2x2 matrix ad minus bc equals zero when columns are dependent.",
    ],
    # Discovery + Subjective (Literature/History/Art-like)
    'D_S': [
        "Shakespeare's Hamlet explores themes of mortality, madness, and revenge.",
        "The French Revolution of 1789 established principles of liberty and equality.",
        "Picasso's Guernica depicts the horror of the bombing of a civilian town.",
        "The Renaissance marked a rebirth of classical learning and artistic innovation.",
        "1984 by George Orwell is a dystopian novel about totalitarian surveillance.",
    ],
    # Construction + Subjective (Code-like)
    'C_S': [
        "A for loop iterates over elements of an array from index 0 to n minus 1.",
        "Recursion is when a function calls itself with a smaller subproblem.",
        "Binary search splits a sorted array in half and checks the middle element.",
        "A hash table provides constant time average case lookup via hashing.",
        "The stack data structure follows last-in first-out order for push and pop.",
    ],
}

print(f'Loading {MODEL}...')
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
d = model.config.hidden_size

# Extract hidden states
all_states = []
all_labels = []  # (quadrant_label, axis1_label, axis2_label)

for quadrant, texts in prompts.items():
    axis1 = 'D' if quadrant.startswith('D') else 'C'
    axis2 = 'O' if quadrant.endswith('O') else 'S'
    for text in texts:
        enc = tok(text, return_tensors='pt', truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        all_states.append(h)
        all_labels.append((quadrant, axis1, axis2))

# Build UGT basis
hs = torch.stack(all_states)
hs_c = hs - hs.mean(0)
U, S, Vt = torch.linalg.svd(hs_c.T, full_matrices=False)

# Use K = N-1 (max rank from 20 calibration prompts)
K_UGT = min(32, len(all_states) - 1)
basis = U[:, :K_UGT].float()
print(f'UGT basis: {basis.shape}, top-{K_UGT} singular values capture {S[:K_UGT].sum().item()/S.sum().item()*100:.1f}% variance')

# Project all states
projs = (hs_c.float() @ basis).numpy()  # [N, K]
labels_arr = np.array(all_labels)

# For each coordinate, measure axis discriminability (t-test effect size)
K = K_UGT
axis1_scores = np.zeros(K)
axis2_scores = np.zeros(K)

for k in range(K):
    # Axis 1: Discovery vs Construction
    d_idx = [i for i, l in enumerate(all_labels) if l[1] == 'D']
    c_idx = [i for i, l in enumerate(all_labels) if l[1] == 'C']
    d_vals = projs[d_idx, k]
    c_vals = projs[c_idx, k]
    # Cohen's d = mean_diff / pooled_std
    mean_diff = np.abs(d_vals.mean() - c_vals.mean())
    pooled_std = np.sqrt((d_vals.var() + c_vals.var()) / 2 + 1e-10)
    axis1_scores[k] = mean_diff / pooled_std

    # Axis 2: Objective vs Subjective
    o_idx = [i for i, l in enumerate(all_labels) if l[2] == 'O']
    s_idx = [i for i, l in enumerate(all_labels) if l[2] == 'S']
    o_vals = projs[o_idx, k]
    s_vals = projs[s_idx, k]
    mean_diff = np.abs(o_vals.mean() - s_vals.mean())
    pooled_std = np.sqrt((o_vals.var() + s_vals.var()) / 2 + 1e-10)
    axis2_scores[k] = mean_diff / pooled_std

# Classify each coordinate
TOL = 0.20  # tolerance for "dominant" vs "residual"
coord_types = []
for k in range(K):
    a1 = axis1_scores[k]
    a2 = axis2_scores[k]
    if a1 > a2 + TOL and a1 > 0.5:
        coord_types.append('axis1')
    elif a2 > a1 + TOL and a2 > 0.5:
        coord_types.append('axis2')
    elif a1 > 0.3 or a2 > 0.3:
        coord_types.append('mixed')
    else:
        coord_types.append('residual')

k_axis1 = coord_types.count('axis1')
k_axis2 = coord_types.count('axis2')
k_mixed = coord_types.count('mixed')
k_residual = coord_types.count('residual')

# Print results
print(f'\n{"="*65}')
print(f'UGT k-Axis Decomposition Hypothesis Test')
print(f'  k_total = {K}')
print(f'  k_axis1 (Discovery/Construction) = {k_axis1}')
print(f'  k_axis2 (Objective/Subjective)   = {k_axis2}')
print(f'  k_mixed  (captures both)         = {k_mixed}')
print(f'  k_residual (neither dominant)    = {k_residual}')
print(f'  k_decomposed = k_axis1 + k_axis2 + k_mixed + k_residual = {k_axis1+k_axis2+k_mixed+k_residual}')
print(f'{"="*65}')

print(f'\nAxis 1 dominant coords (Discovery/Construction):')
for k in range(K):
    if coord_types[k] == 'axis1':
        print(f'  UGT[{k:2d}]: axis1_d={axis1_scores[k]:.3f}  axis2_d={axis2_scores[k]:.3f}')

print(f'\nAxis 2 dominant coords (Objective/Subjective):')
for k in range(K):
    if coord_types[k] == 'axis2':
        print(f'  UGT[{k:2d}]: axis1_d={axis1_scores[k]:.3f}  axis2_d={axis2_scores[k]:.3f}')

print(f'\nMixed coords:')
for k in range(K):
    if coord_types[k] == 'mixed':
        print(f'  UGT[{k:2d}]: axis1_d={axis1_scores[k]:.3f}  axis2_d={axis2_scores[k]:.3f}')

# Top coordinates by sum of discriminabilities
total_scores = axis1_scores + axis2_scores
top_indices = np.argsort(-total_scores)
print(f'\nTop 10 coordinates by total discriminability:')
for rank, k in enumerate(top_indices[:10]):
    print(f'  #{rank+1} UGT[{k:2d}]: total={total_scores[k]:.3f}  axis1={axis1_scores[k]:.3f}  axis2={axis2_scores[k]:.3f}  type={coord_types[k]}')

# Save results
results = {
    'model': MODEL,
    'k_total': int(K),
    'k_axis1': k_axis1,
    'k_axis2': k_axis2,
    'k_mixed': k_mixed,
    'k_residual': k_residual,
    'hypothesis': 'k = k_axis1 + k_axis2 + k_residual',
    'verdict': 'CONFIRMED' if (k_axis1 > 0 and k_axis2 > 0) else 'PARTIAL',
    'coordinate_types': coord_types,
    'axis1_scores': axis1_scores.tolist(),
    'axis2_scores': axis2_scores.tolist(),
}
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to {OUT}/results.json')
print(f'Verdict: {results["verdict"]}')

del model; torch.cuda.empty_cache()
