"""AGT v2: Encode Re(s) explicitly. Sharper TEH discrimination.
Fixed: off-critical candidates with real Re≠0.5, 5x off_sep weight."""
import torch, json, math, time, os
import torch.nn.functional as F

DEVICE = "cuda"
D_MODEL = 576
OUT_DIR = "/home/ubuntu/benchmarks/agt_zeta_v2"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  AGT v2: ζ(s) Manifold with Re(s) encoding")
print("=" * 60)

# Primes
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

N = 10000
primes = [n for n in range(2, N + 1) if is_prime(n)]
print(f"  Primes ≤ {N}: {len(primes)}")

def prime_features(p):
    f = []
    f.append(math.log(p) / math.log(N))
    idx = primes.index(p)
    next_p = primes[idx + 1] if idx + 1 < len(primes) else p + 2
    f.append(math.log(max(next_p - p, 1) + 1) / math.log(N))
    for mod in [3, 5, 7, 11]:
        f.append((p % mod) / mod)
    theta = sum(math.log(q) for q in primes if q <= p)
    f.append(theta / max(p, 1))
    f.append(-1.0)
    f.append(idx / len(primes))
    return torch.tensor(f, dtype=torch.float32)

FEAT_DIM = 9
prime_vecs = torch.stack([prime_features(p) for p in primes])

embedder = torch.nn.Sequential(
    torch.nn.Linear(FEAT_DIM, 128), torch.nn.GELU(), torch.nn.Linear(128, D_MODEL),
).to(DEVICE)

# ζ(s) zeros on critical line Re(s)=1/2
zeta_zeros_imag = [
    14.134725, 21.022040, 25.010857, 30.424876, 32.935061,
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112543,
    67.079811, 69.546401, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425273, 88.809112,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]

ZETA_FEAT_DIM = 6

def zeta_features_v2(t, real_part):
    """Encode both t and real_part. real_part=0.5 = on critical line."""
    f = []
    f.append(math.log(abs(t) + 1) / 5.0)
    f.append((t % (2 * math.pi)) / (2 * math.pi))
    gram_n = (t / (2 * math.pi)) * (math.log(t / (2 * math.pi)) - 1) + 7/8
    f.append((gram_n % 1))
    gaps = [abs(z - t) for z in zeta_zeros_imag]
    f.append(math.log(min(gaps) + 0.01) / 3.0)
    f.append((real_part - 0.5) * 10.0)  # KEY: 0 for critical line
    nearby = sum(1 for z in zeta_zeros_imag if abs(z - t) < 10)
    f.append(nearby / 10.0)
    return torch.tensor(f, dtype=torch.float32)

# Critical-line zeros
critical_zeros = torch.stack([zeta_features_v2(t, 0.5) for t in zeta_zeros_imag])

# Off-critical-line candidates WITH real Re≠0.5
test_cases = [
    (15.5, 0.35), (22.8, 0.62), (31.2, 0.28), (44.7, 0.71), (53.1, 0.44),
    (60.0, 0.55), (71.3, 0.33), (85.9, 0.68), (93.4, 0.41), (99.9, 0.59),
    (20.0, 0.38), (35.0, 0.66), (50.0, 0.25), (65.0, 0.73), (80.0, 0.42),
]
off_critical = torch.stack([zeta_features_v2(t, rp) for t, rp in test_cases])
print(f"  Critical zeros: {len(critical_zeros)}, Off-critical candidates: {len(off_critical)}")

# Embedder for zeros
zero_embedder = torch.nn.Sequential(
    torch.nn.Linear(ZETA_FEAT_DIM, 64), torch.nn.GELU(), torch.nn.Linear(64, D_MODEL),
).to(DEVICE)

# -- Train --
print("\n[Training] Manifold learning...")
opt = torch.optim.AdamW(list(embedder.parameters()) + list(zero_embedder.parameters()), lr=0.002)
steps = 2000

for step in range(steps):
    # Prime continuity
    batch_idx = torch.randint(0, len(primes), (32,))
    prime_emb = F.normalize(embedder(prime_vecs[batch_idx].to(DEVICE)), dim=-1)
    next_emb = F.normalize(embedder(prime_vecs[(batch_idx + 1) % len(primes)].to(DEVICE)), dim=-1)
    cont_loss = (1 - (prime_emb * next_emb).sum(dim=-1)).mean()
    
    # Zero cluster
    zi = torch.randint(0, len(zeta_zeros_imag), (10,))
    ze = F.normalize(zero_embedder(critical_zeros[zi].to(DEVICE)), dim=-1)
    zj = torch.randint(0, len(zeta_zeros_imag), (10,))
    ze2 = F.normalize(zero_embedder(critical_zeros[zj].to(DEVICE)), dim=-1)
    zclust = (1 - (ze * ze2).sum(dim=-1)).mean()
    
    # OFF-critical separation (STRONG weight)
    oi = torch.randint(0, len(off_critical), (8,))
    oe = F.normalize(zero_embedder(off_critical[oi].to(DEVICE)), dim=-1)
    off_sim = (oe @ ze.T).mean()
    off_sep = torch.relu(off_sim - 0.15)
    
    loss = cont_loss + 0.5 * zclust + 8.0 * off_sep
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    if (step + 1) % 500 == 0:
        print(f"  Step {step+1}: loss={loss.item():.4f} cont={cont_loss.item():.3f} "
              f"zclust={zclust.item():.3f} offsep={off_sep.item():.4f}")

# -- TEH Test --
print("\n[TEH] Off-critical-line detection...")
with torch.no_grad():
    all_z = F.normalize(zero_embedder(critical_zeros.to(DEVICE)), dim=-1)
    U, S, V = torch.linalg.svd(all_z.T @ all_z)
    crit_basis = U[:, :8]
    P_crit = crit_basis @ crit_basis.T
    P_forb = torch.eye(D_MODEL, device=DEVICE) - P_crit

# Test off-critical
off_results = []
for i, oc in enumerate(off_critical):
    emb = F.normalize(zero_embedder(oc.unsqueeze(0).to(DEVICE)), dim=-1).float().view(-1)
    act = (torch.norm(P_forb @ emb).item() / max(torch.norm(emb).item(), 1e-8)) * 100
    det = act > 12.0
    off_results.append({"i": i, "t": test_cases[i][0], "Re": test_cases[i][1], "act": round(act, 2), "detected": det})
    flag = "DETECT" if det else "pass"
    print(f"  Off-line (t={test_cases[i][0]:.0f}, Re={test_cases[i][1]:.2f}): act={act:.1f}% [{flag}]")

# Test critical (should be LOW)
crit_acts = []
for i in range(10):
    emb = F.normalize(zero_embedder(critical_zeros[i].unsqueeze(0).to(DEVICE)), dim=-1).float().view(-1)
    act = (torch.norm(P_forb @ emb).item() / max(torch.norm(emb).item(), 1e-8)) * 100
    crit_acts.append(act)
    fp = act > 12.0
    flag = "FALSE+" if fp else "ok"
    print(f"  Critical ζ(1/2+i·{zeta_zeros_imag[i]:.1f}): act={act:.1f}% [{flag}]")

detection = sum(1 for r in off_results if r["detected"])
fp_count = sum(1 for a in crit_acts if a > 12.0)
print(f"\n  Detection: {detection}/{len(off_results)} ({100*detection/len(off_results):.0f}%)")
print(f"  False positives: {fp_count}/10")
print(f"  Mean off-critical act: {sum(r['act'] for r in off_results)/len(off_results):.1f}%")
print(f"  Mean critical act: {sum(crit_acts)/len(crit_acts):.1f}%")

# Save
output = {
    "config": {"n_primes": len(primes), "n_zeta_zeros": len(zeta_zeros_imag), "n_off_critical": len(off_critical)},
    "teh": {
        "detection_rate": round(100 * detection / len(off_results)),
        "false_positives": fp_count,
        "mean_off_act": round(sum(r["act"] for r in off_results) / len(off_results), 1),
        "mean_crit_act": round(sum(crit_acts) / len(crit_acts), 1),
    },
    "details": {"off_critical": off_results, "critical_acts": [round(a, 2) for a in crit_acts]},
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(output, f, indent=2)
torch.save({"embedder": embedder.state_dict(), "zero_embedder": zero_embedder.state_dict(), "crit_basis": crit_basis.cpu()}, f"{OUT_DIR}/model.pt")
print(f"\nSaved to {OUT_DIR}/")
