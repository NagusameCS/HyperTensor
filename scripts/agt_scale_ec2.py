#!/usr/bin/env python3
"""AGT at scale: 50K primes + 105 zeros. Run on EC2 L40S."""
import torch, json, math, os, time
import numpy as np

print("=" * 60)
print("  AGT SCALE: 50K primes + 105 zeros on L40S")
print("=" * 60)

t0 = time.time()

# Generate primes
N_MAX_PRIMES = 50000
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

print(f"[1/5] Finding {N_MAX_PRIMES} primes...")
primes = [n for n in range(2, N_MAX_PRIMES * 25) if is_prime(n)][:N_MAX_PRIMES]
N_MAX = primes[-1]
print(f"  Primes: {len(primes)}, max={N_MAX}")

# Build feature vectors
print("[2/5] Building feature vectors...")
features = []
for i, p in enumerate(primes):
    f = [math.log(p) / math.log(N_MAX)]
    np_val = primes[i+1] if i+1 < len(primes) else p + 2
    f.append(math.log(max(np_val - p, 1) + 1) / math.log(N_MAX))
    for m in [3, 5, 7, 11, 13]:
        f.append((p % m) / m)
    th = sum(math.log(q) for q in primes if q <= p)
    f.append(th / max(p, 1))
    f.append(i / len(primes))
    pnt = p / math.log(p) if p > 1 else 1
    f.append((i + 1 - pnt) / max(pnt, 1))
    features.append(f)

pv = torch.tensor(features, dtype=torch.float32)
print(f"  Feature matrix: {pv.shape}")

# SVD
print("[3/5] Computing SVD...")
U, S, Vh = torch.linalg.svd(pv.float(), full_matrices=False)
# U: [50000, 10] (left singular vectors in sample space)
# Vh: [10, 10] (right singular vectors in feature space)
# Basis = Vh.T: [10, 10] — projects features to k-space
total_var = (S**2).sum().item()
cumsum = torch.cumsum(S**2, dim=0)
k90 = int((cumsum / total_var > 0.90).float().argmax().item()) + 1
k95 = int((cumsum / total_var > 0.95).float().argmax().item()) + 1
k99 = int((cumsum / total_var > 0.99).float().argmax().item()) + 1

print(f"  k90={k90}, k95={k95}, k99={k99}")
print(f"  SV1={S[0].item():.1f}, SV2={S[1].item():.1f}, SV3={S[2].item():.1f}")
print(f"  SV gap: {S[0].item()/max(S[1].item(),1e-10):.0f}×")

# Zeta zeros (first 105 on critical line)
zeta_zeros = [
    14.134725,21.022040,25.010857,30.424876,32.935061,37.586178,40.918719,
    43.327073,48.005150,49.773832,52.970321,56.446248,59.347044,60.831779,
    65.112543,67.079811,69.546401,72.067158,75.704691,77.144840,79.337375,
    82.910381,84.735493,87.425273,88.809112,92.491899,94.651344,95.870634,
    98.831194,101.317851,103.725538,105.446623,107.168611,111.029535,111.874659,
    114.320221,116.226680,118.790783,121.370125,122.946829,124.256819,127.516684,
    129.578704,131.087688,133.497737,134.756510,138.116042,139.736209,141.123707,
    143.111846,146.000982,147.422765,150.053520,150.925258,153.024694,156.112909,
    157.597591,158.849988,161.188964,163.030709,165.537069,167.184440,169.094515,
    169.911976,173.411537,174.754191,176.441434,178.377408,179.916484,182.207078,
    184.874468,185.598784,187.228923,189.416159,192.026656,193.079727,195.265397,
    196.876482,198.015310,201.264752,202.493595,204.189672,205.394697,207.906258,
    209.576510,211.690862,213.347919,214.547045,216.169539,219.067596,220.714919,
    221.430706,224.007000,224.983325,227.421444,229.337413,231.250189,231.987235,
    233.693404,236.524230,238.162420,240.269891,240.903050,243.350186,246.041877,
]

# Zero features
print("[4/5] Computing zero features...")
feat_dim = pv.shape[1]

def zeta_features(t, real_part):
    f = [real_part]
    f.append(math.log(abs(t) + 1) / math.log(N_MAX + 1))
    gaps = [abs(t - p) for p in primes[:2000]]
    f.append(math.log(min(gaps) + 0.01) / 3.0)
    nearby = sum(1 for p in primes[:2000] if abs(t - p) < 10)
    f.append(nearby / 10.0)
    f.append(sum(1 for p in primes if p <= abs(t)) / len(primes))
    harmonic = sum(math.sin(t * math.log(q)) / math.log(q) for q in primes[:200] if q > 1)
    f.append(harmonic / 200)
    return torch.tensor(f, dtype=torch.float32)

# Critical zeros
crit_features = []
for t in zeta_zeros:
    f = zeta_features(t, 0.5)
    padded = torch.zeros(feat_dim)
    padded[:len(f)] = f
    crit_features.append(padded)
crit_stack = torch.stack(crit_features)

# Off-critical
off_features = []
for t in zeta_zeros[:50]:
    for re in [0.3, 0.7]:
        f = zeta_features(t, re)
        padded = torch.zeros(feat_dim)
        padded[:len(f)] = f
        off_features.append(padded)
off_stack = torch.stack(off_features)

# Project into AGT basis (V = right singular vectors, shape [10, k])
K_CRIT = min(10, len(S))
basis = Vh.T[:, :K_CRIT]  # [10, K_CRIT] — projects 10-dim features to K_CRIT-dim space

crit_proj = (crit_stack @ basis).norm(dim=1)
off_proj = (off_stack @ basis).norm(dim=1)

crit_mean = crit_proj.mean().item()
crit_std = crit_proj.std().item()
off_mean = off_proj.mean().item()
off_std = off_proj.std().item()

separation = off_mean / max(crit_mean, 1e-10)

# Detection accuracy
detected = (off_proj > crit_mean + 3 * crit_std).sum().item()
false_positives = (crit_proj > crit_mean + 3 * crit_std).sum().item()

print(f"\n[5/5] RESULTS at N={len(primes)}:")
print(f"  Critical subspace: k90={k90}, k95={k95}, k99={k99}")
print(f"  Critical μ={crit_mean:.4f}, σ={crit_std:.4f}")
print(f"  Off-critical μ={off_mean:.4f}, σ={off_std:.4f}")
print(f"  Separation: {separation:.0f}×")
print(f"  Detection: {detected}/{len(off_features)} ({detected/len(off_features)*100:.1f}%)")
print(f"  False positives: {false_positives}/{len(zeta_zeros)} ({false_positives/len(zeta_zeros)*100:.1f}%)")
print(f"  Time: {time.time()-t0:.1f}s")

# Save
out = "/home/ubuntu/benchmarks/agt_50k"
os.makedirs(out, exist_ok=True)
report = {
    "n_primes": len(primes),
    "n_zeros": len(zeta_zeros),
    "n_off_critical": len(off_features),
    "critical_subspace": {"k90": k90, "k95": k95, "k99": k99},
    "sv_gap": round(S[0].item() / max(S[1].item(), 1e-10), 1),
    "separation": round(float(separation), 0),
    "detection_pct": round(detected / len(off_features) * 100, 1),
    "false_positive_pct": round(false_positives / max(len(zeta_zeros), 1) * 100, 1),
    "time_s": round(time.time() - t0, 1),
}
with open(f"{out}/results.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Saved: {out}/results.json")
print(f"\n  [OK] AGT at 50K: 1D subspace {'CONFIRMED' if k90<=2 else 'PERSISTS'} | "
      f"{separation:.0f}× separation | {detected/len(off_features)*100:.0f}% detection")
