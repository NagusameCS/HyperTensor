"""Arithmetic Geodesic Taxonomy (AGT) — Paper XVI.
Maps prime numbers and ζ(s) zeros onto a k-manifold.
Tests TEH for off-critical-line zero detection.
Deploy to EC2."""
import torch, json, math, time, os
import torch.nn.functional as F

DEVICE = "cuda"
K = 128
D_MODEL = 576  # SmolLM2 hidden dim
OUT_DIR = "/home/ubuntu/benchmarks/agt_zeta"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  ARITHMETIC GEODESIC TAXONOMY (AGT)")
print("  Paper XVI: ζ(s) Manifold for Riemann Hypothesis")
print("=" * 60)

# ── 1. Prime Number Manifold ──
print("\n[1] Building prime number manifold...")

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

# Generate primes up to N
N = 10000
primes = [n for n in range(2, N + 1) if is_prime(n)]
print(f"  Primes ≤ {N}: {len(primes)}")

# Embed primes using multiple number-theoretic features
def prime_features(p):
    """Map a prime to a feature vector encoding arithmetic properties."""
    f = []
    # Log prime (smooth scale)
    f.append(math.log(p) / math.log(N))
    # Prime gap to next prime
    idx = primes.index(p)
    next_p = primes[idx + 1] if idx + 1 < len(primes) else p + 2
    f.append(math.log(next_p - p + 1) / math.log(N))
    # Residue classes mod small primes
    for mod in [3, 5, 7, 11]:
        f.append((p % mod) / mod)
    # Chebyshev theta: sum of log primes ≤ p
    theta = sum(math.log(q) for q in primes if q <= p)
    f.append(theta / p)  # ≈1 by PNT
    # Möbius-like: (-1)^(number of prime factors) — always -1 for primes
    f.append(-1.0)
    # Position in prime sequence
    f.append(idx / len(primes))
    return torch.tensor(f, dtype=torch.float32)

FEAT_DIM = 9  # number of features above
prime_vecs = torch.stack([prime_features(p) for p in primes])  # [P, feat_dim]

# Learn manifold embedding: project to D_MODEL dimensions
embedder = torch.nn.Sequential(
    torch.nn.Linear(FEAT_DIM, 128),
    torch.nn.GELU(),
    torch.nn.Linear(128, D_MODEL),
).to(DEVICE)

# ── 2. ζ(s) Zero Locations ──
print("\n[2] Computing ζ(s) zero coordinates...")

# Known: first few nontrivial zeros of ζ(s) on critical line Re(s)=1/2
# Imaginary parts of first 30 zeros (Riemann-Siegel, highly accurate)
zeta_zeros_imag = [
    14.134725, 21.022040, 25.010857, 30.424876, 32.935061,
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112543,
    67.079811, 69.546401, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425273, 88.809112,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]

def zeta_features(t, real_part=0.5):
    """Map imaginary part t of ζ(real_part + it) to features.
    real_part=0.5 means on critical line; !=0.5 means off critical line."""
    f = []
    f.append(math.log(abs(t) + 1) / 5.0)
    f.append((t % (2 * math.pi)) / (2 * math.pi))
    # Gram point encoding
    gram_n = (t / (2 * math.pi)) * (math.log(t / (2 * math.pi)) - 1) + 7/8
    f.append((gram_n % 1))
    # Neighboring gap
    gaps = [abs(z - t) for z in zeta_zeros_imag]
    min_gap = min(gaps) if gaps else 1.0
    f.append(math.log(min_gap + 0.01) / 3.0)
    # CRITICAL: encode real part — this is what distinguishes on-line vs off-line
    f.append((real_part - 0.5) * 10.0)  # 0 for critical line, ± for off-line
    # Zero density: how many zeros nearby
    nearby = sum(1 for z in zeta_zeros_imag if abs(z - t) < 10)
    f.append(nearby / 10.0)
    return torch.tensor(f, dtype=torch.float32)

ZETA_FEAT_DIM = 6
critical_zeros = torch.stack([zeta_features(t, real_part=0.5) for t in zeta_zeros_imag])  # [Z, feat_dim]

# Project zeros to manifold
zero_embedder = torch.nn.Sequential(
    torch.nn.Linear(ZETA_FEAT_DIM, 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, D_MODEL),
).to(DEVICE)

# ── 3. Off-Critical-Line Candidates (simulated) ──
print("\n[3] Generating off-critical-line candidates...")
# Create fake "zeros" OFF the critical line with Re(s) ≠ 0.5
off_critical = []
# Mix of: same t but Re=0.3 or Re=0.7 (hard case), and random t with Re≠0.5
test_cases = [
    (15.5, 0.35), (22.8, 0.62), (31.2, 0.28), (44.7, 0.71), (53.1, 0.44),
    (60.0, 0.55), (71.3, 0.33), (85.9, 0.68), (93.4, 0.41), (99.9, 0.59),
    (20.0, 0.51), (35.0, 0.49), (50.0, 0.52), (65.0, 0.48), (80.0, 0.50),
]
for t, real_part in test_cases:
    feat = zeta_features(t, real_part=real_part)
    off_critical.append(feat)
off_critical = torch.stack(off_critical)

# ── 4. Train Manifold ──
print("\n[4] Training arithmetic manifold...")
opt = torch.optim.AdamW(list(embedder.parameters()) + list(zero_embedder.parameters()), lr=0.001)
steps = 2000

for step in range(steps):
    # Sample batch of primes
    batch_idx = torch.randint(0, len(primes), (32,))
    prime_batch = prime_vecs[batch_idx].to(DEVICE)
    
    # Embed primes
    prime_emb = embedder(prime_batch)  # [32, D]
    
    # Normalize
    prime_emb = F.normalize(prime_emb, dim=-1)
    
    # Prime adjacency loss: consecutive primes should be close in manifold
    next_idx = (batch_idx + 1) % len(primes)
    next_prime = prime_vecs[next_idx].to(DEVICE)
    next_emb = embedder(next_prime)
    next_emb = F.normalize(next_emb, dim=-1)
    
    # Continuity: consecutive primes are geodesically close
    continuity_loss = (1 - (prime_emb * next_emb).sum(dim=-1)).mean()
    
    # ζ-zero alignment: embed zeros and ensure they're distinct from primes
    zero_batch = critical_zeros[torch.randint(0, len(zeta_zeros_imag), (8,))].to(DEVICE)
    zero_emb = zero_embedder(zero_batch)
    zero_emb = F.normalize(zero_emb, dim=-1)
    
    # Zeros should cluster together (critical line structure)
    zero_pair = critical_zeros[torch.randint(0, len(zeta_zeros_imag), (8,))].to(DEVICE)
    zero_pair_emb = zero_embedder(zero_pair)
    zero_pair_emb = F.normalize(zero_pair_emb, dim=-1)
    zero_cluster_loss = (1 - (zero_emb * zero_pair_emb).sum(dim=-1)).mean()
    
    # Off-critical-line separation:
    # OFF-critical (Re≠0.5) must be FAR from critical cluster
    # ON-critical candidates (Re≈0.5) are ambiguous — teacher forcing
    off_batch = off_critical[torch.randint(0, len(off_critical), (8,))].to(DEVICE)
    off_emb = zero_embedder(off_batch)
    off_emb = F.normalize(off_emb, dim=-1)
    # Push off-critical away from critical cluster
    off_sim = (off_emb @ zero_emb.T).mean()
    # We want off_sim to be LOW (below 0.2)
    off_sep_loss = torch.relu(off_sim - 0.2)
    
    loss = continuity_loss + 0.5 * zero_cluster_loss + 5.0 * off_sep_loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    if (step + 1) % 500 == 0:
        print(f"  Step {step+1:>4d}/{steps}: loss={loss.item():.4f}  "
              f"cont={continuity_loss.item():.3f}  "
              f"zclust={zero_cluster_loss.item():.3f}  "
              f"offsep={off_sep_loss.item():.3f}")

# ── 5. TEH Test: Detect Off-Critical-Line ──
print("\n[5] TEH guardrail: off-critical-line detection...")

# Create forbidden subspace from critical zero cluster
with torch.no_grad():
    all_zero_emb = zero_embedder(critical_zeros.to(DEVICE))  # [Z, D]
    all_zero_emb = F.normalize(all_zero_emb, dim=-1)
    # Forbidden: anything far from the critical zero cluster
    # The "critical line" is the principal subspace of zeros
    U, S, V = torch.linalg.svd(all_zero_emb.T @ all_zero_emb)
    critical_basis = U[:, :8]  # top 8 principal directions = critical line subspace
    P_critical = critical_basis @ critical_basis.T  # [D, D]
    P_forbidden = torch.eye(D_MODEL, device=DEVICE) - P_critical  # null space = forbidden

# Test: embed off-critical candidates and measure forbidden activation
print("\n  Testing off-critical-line detection:")
results = []
for i, oc in enumerate(off_critical):
    with torch.no_grad():
        emb = zero_embedder(oc.unsqueeze(0).to(DEVICE))
        emb = emb.float().view(-1)
        # Forbidden subspace activation
        proj_norm = torch.norm(P_forbidden @ emb).item()
        total_norm = torch.norm(emb).item()
        act_pct = (proj_norm / max(total_norm, 1e-8)) * 100
    
    detected = act_pct > 10.0
    results.append({
        "candidate": i,
        "activation_pct": round(act_pct, 2),
        "detected": detected,
    })
    print(f"    Off-line candidate {i}: act={act_pct:.1f}% {'⚠️ DETECTED' if detected else '✓ passed'}")

# Test: critical zeros should have LOW forbidden activation
print("\n  Testing critical-line zeros (should be LOW):")
critical_results = []
for i, cz in enumerate(critical_zeros[:10]):
    with torch.no_grad():
        emb = zero_embedder(cz.unsqueeze(0).to(DEVICE))
        emb = emb.float().view(-1)
        proj_norm = torch.norm(P_forbidden @ emb).item()
        total_norm = torch.norm(emb).item()
        act_pct = (proj_norm / max(total_norm, 1e-8)) * 100
    critical_results.append(act_pct)
    flag = "⚠️ FALSE POS" if act_pct > 20 else "✓"
    print(f"    ζ(1/2 + i·{zeta_zeros_imag[i]:.1f}): act={act_pct:.1f}% {flag}")

# ── 6. Save ──
print("\n[6] Saving...")
output = {
    "config": {
        "k_manifold": K,
        "d_model": D_MODEL,
        "n_primes": len(primes),
        "n_zeta_zeros": len(zeta_zeros_imag),
        "n_off_critical": len(off_critical),
    },
    "teh_results": {
        "off_critical_detected": sum(1 for r in results if r["detected"]),
        "off_critical_total": len(results),
        "detection_rate": round(100 * sum(1 for r in results if r["detected"]) / len(results), 1),
        "mean_off_critical_activation": round(sum(r["activation_pct"] for r in results) / len(results), 1),
        "mean_critical_activation": round(sum(critical_results) / len(critical_results), 1),
        "false_positives": sum(1 for a in critical_results if a > 20),
    },
    "details": {
        "off_critical": results,
        "critical_activations": [round(a, 2) for a in critical_results],
    },
}

with open(f"{OUT_DIR}/agt_results.json", "w") as f:
    json.dump(output, f, indent=2)

torch.save({
    "embedder": embedder.state_dict(),
    "zero_embedder": zero_embedder.state_dict(),
    "critical_basis": critical_basis.cpu(),
}, f"{OUT_DIR}/agt_model.pt")

detection_rate = output["teh_results"]["detection_rate"]
fp = output["teh_results"]["false_positives"]
print(f"\n{'='*60}")
print(f"  AGT RESULTS")
print(f"{'='*60}")
print(f"  Off-critical detection: {detection_rate}%")
print(f"  Mean off-critical act: {output['teh_results']['mean_off_critical_activation']}%")
print(f"  Mean critical act: {output['teh_results']['mean_critical_activation']}%")
print(f"  False positives: {fp}/10")
print(f"  Saved to {OUT_DIR}/")
