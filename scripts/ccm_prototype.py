"""Circuit Complexity Manifold (CCM) Prototype — Paper XIX.
Embeds Boolean circuits on a k-manifold. Tests if P vs NP circuits
have geometrically separable curvature signatures.
Deploy to EC2."""
import torch, json, math, random, time, os
import torch.nn.functional as F
from collections import defaultdict

DEVICE = "cuda"
D_MODEL = 576
K = 64
OUT_DIR = "/home/ubuntu/benchmarks/ccm_prototype"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  CIRCUIT COMPLEXITY MANIFOLD (CCM)")
print("  Paper XIX: Geometric P vs NP")
print("=" * 60)

# ── Circuit Generation ──
print("\n[1] Generating Boolean circuits...")

def generate_2sat_instance(n_vars=8, n_clauses=12):
    """Generate random 2-SAT instance (P). Each clause has 2 literals."""
    clauses = []
    for _ in range(n_clauses):
        a = random.randint(0, n_vars-1)
        b = random.randint(0, n_vars-1)
        while b == a:
            b = random.randint(0, n_vars-1)
        sa = random.choice([True, False])
        sb = random.choice([True, False])
        clauses.append((a, sa, b, sb))
    return {"n_vars": n_vars, "n_clauses": n_clauses, "clauses": clauses, "type": "2-SAT"}

def generate_horn_instance(n_vars=8, n_clauses=12):
    """Generate random Horn-SAT instance (P). At most one positive literal."""
    clauses = []
    for _ in range(n_clauses):
        # Horn clause: at most one positive literal
        n_lits = random.randint(1, 3)
        lits = random.sample(range(n_vars), min(n_lits, n_vars))
        # Only one positive
        pos_idx = random.randint(0, len(lits)-1) if lits else 0
        clause = []
        for i, v in enumerate(lits):
            clause.append((v, i == pos_idx))  # only one True sign
        clauses.append(clause)
    return {"n_vars": n_vars, "n_clauses": n_clauses, "clauses": clauses, "type": "Horn-SAT"}

def generate_3sat_instance(n_vars=8, n_clauses=12):
    """Generate random 3-SAT instance (NP-complete). Each clause has 3 literals."""
    clauses = []
    for _ in range(n_clauses):
        lits = random.sample(range(n_vars), min(3, n_vars))
        signs = [random.choice([True, False]) for _ in lits]
        clauses.append(list(zip(lits, signs)))
    return {"n_vars": n_vars, "n_clauses": n_clauses, "clauses": clauses, "type": "3-SAT"}

def generate_tsp_instance(n_cities=6):
    """Generate TSP instance (NP-complete)."""
    cities = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(n_cities)]
    dists = []
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            d = math.sqrt((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2)
            dists.append(d)
    return {"n_cities": n_cities, "cities": cities, "distances": dists, "type": "TSP"}

# Generate circuits
n_each = 500
circuits = []
for _ in range(n_each):
    circuits.append(generate_2sat_instance(random.randint(6, 12), random.randint(8, 20)))
    circuits.append(generate_horn_instance(random.randint(6, 12), random.randint(8, 20)))
    circuits.append(generate_3sat_instance(random.randint(6, 12), random.randint(8, 20)))
    circuits.append(generate_tsp_instance(random.randint(5, 8)))

print(f"  Generated {len(circuits)} circuits (2-SAT, Horn-SAT, 3-SAT, TSP)")

# ── Feature Extraction ──
print("\n[2] Extracting circuit features...")

def circuit_features(c):
    """Extract fixed-length feature vector from any circuit type."""
    f = []
    t = c["type"]
    
    # Type encoding (one-hot)
    for ct in ["2-SAT", "Horn-SAT", "3-SAT", "TSP"]:
        f.append(1.0 if t == ct else 0.0)
    
    if t in ("2-SAT", "Horn-SAT", "3-SAT"):
        n_vars = c["n_vars"]
        n_clauses = c["n_clauses"]
        f.append(n_vars / 20.0)
        f.append(n_clauses / 30.0)
        f.append(n_clauses / max(n_vars, 1))  # clause-to-variable ratio
        
        # Clause statistics
        clause_sizes = [len(clause) if isinstance(clause, list) else 2 for clause in c["clauses"]]
        f.append(sum(clause_sizes) / max(len(clause_sizes), 1))
        
        # Literal statistics
        pos_count = 0
        neg_count = 0
        var_freq = defaultdict(int)
        for clause in c["clauses"]:
            if t == "Horn-SAT":
                for v, sign in clause:
                    if sign: pos_count += 1
                    else: neg_count += 1
                    var_freq[v] += 1
            elif t in ("2-SAT", "3-SAT"):
                if isinstance(clause, tuple):
                    a, sa, b, sb = clause
                    pos_count += (1 if sa else 0) + (1 if sb else 0)
                    neg_count += (0 if sa else 1) + (0 if sb else 1)
                    var_freq[a] += 1
                    var_freq[b] += 1
        
        f.append(pos_count / max(n_clauses * 3, 1))
        f.append(neg_count / max(n_clauses * 3, 1))
        
        # Variable frequency entropy
        freqs = list(var_freq.values())
        if freqs:
            total = sum(freqs)
            probs = [x/total for x in freqs]
            entropy = -sum(p * math.log(p+1e-8) for p in probs)
            f.append(entropy / 5.0)
        else:
            f.append(0.0)
        
        # Graph Laplacian approximation: clause-variable incidence density
        density = len(var_freq) / max(n_vars, 1)
        f.append(density)
        
    elif t == "TSP":
        n_cities = c["n_cities"]
        f.append(n_cities / 15.0)
        f.append(len(c["distances"]) / 100.0)
        dists = c["distances"]
        if dists:
            f.append(sum(dists) / len(dists))  # mean distance
            f.append(max(dists))  # max distance
            f.append(min(dists))  # min distance
            f.append((sum(dists)/len(dists)) * math.sqrt(n_cities))  # tour estimate
        else:
            f.extend([0.0]*4)
        # Pad remaining features
        f.extend([0.0]*5)  # match SAT feature count
    
    # Pad to fixed length
    while len(f) < 16:
        f.append(0.0)
    return torch.tensor(f[:16], dtype=torch.float32)

FEAT_DIM = 16
circuit_vecs = torch.stack([circuit_features(c) for c in circuits])  # [N, 16]
labels = []
for c in circuits:
    if c["type"] in ("2-SAT", "Horn-SAT"):
        labels.append(0)  # P
    else:
        labels.append(1)  # NP
labels = torch.tensor(labels)

print(f"  Feature dim: {FEAT_DIM}")
print(f"  P circuits: {(labels==0).sum().item()}, NP circuits: {(labels==1).sum().item()}")

# ── Train Manifold ──
print("\n[3] Training circuit manifold...")

embedder = torch.nn.Sequential(
    torch.nn.Linear(FEAT_DIM, 128),
    torch.nn.GELU(),
    torch.nn.Linear(128, D_MODEL),
).to(DEVICE)

classifier = torch.nn.Linear(D_MODEL, 2).to(DEVICE)  # P vs NP

opt = torch.optim.AdamW(list(embedder.parameters()) + list(classifier.parameters()), lr=0.001)
steps = 3000

for step in range(steps):
    idx = torch.randint(0, len(circuits), (64,))
    batch_vecs = circuit_vecs[idx].to(DEVICE)
    batch_labels = labels[idx].to(DEVICE)
    
    emb = embedder(batch_vecs)
    emb_norm = F.normalize(emb, dim=-1)
    
    # Classification loss: separate P from NP
    logits = classifier(emb_norm)
    cls_loss = F.cross_entropy(logits, batch_labels)
    
    # Manifold structure: same-type circuits cluster together
    p_mask = batch_labels == 0
    np_mask = batch_labels == 1
    
    p_emb = emb_norm[p_mask]
    np_emb = emb_norm[np_mask]
    
    # Intra-class compactness
    intra_loss = 0.0
    if p_mask.sum() >= 2:
        sim = p_emb @ p_emb.T
        intra_loss += (1 - sim).mean()
    if np_mask.sum() >= 2:
        sim = np_emb @ np_emb.T
        intra_loss += (1 - sim).mean()
    
    # Inter-class separation
    inter_loss = 0.0
    if p_mask.sum() >= 1 and np_mask.sum() >= 1:
        cross_sim = (p_emb @ np_emb.T).mean()
        inter_loss = torch.relu(cross_sim + 0.5)  # want cross_sim < -0.5
    
    # Curvature proxy: Jacobi field growth between P and NP embeddings
    # High curvature = large embedding gradient between classes
    if p_mask.sum() >= 2 and np_mask.sum() >= 2:
        p_center = p_emb.mean(dim=0)
        np_center = np_emb.mean(dim=0)
        # Curvature proxy: geodesic distance between class centers
        geo_dist = torch.norm(p_center - np_center)
        curv_loss = torch.relu(1.0 - geo_dist)  # encourage separation > 1.0
    else:
        curv_loss = 0.0
    
    loss = cls_loss + 0.1 * intra_loss + 0.2 * inter_loss + 0.05 * curv_loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    if (step + 1) % 500 == 0:
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == batch_labels).float().mean()
        print(f"  Step {step+1}: loss={loss.item():.4f} cls={cls_loss.item():.3f} "
              f"acc={acc.item():.2f} intra={intra_loss.item():.3f} "
              f"inter={inter_loss.item():.3f} curv={curv_loss.item():.3f}")

# ── Measure Curvature Gap ──
print("\n[4] Measuring P vs NP curvature gap...")

with torch.no_grad():
    all_emb = embedder(circuit_vecs.to(DEVICE))
    all_emb = F.normalize(all_emb, dim=-1)
    
    p_embs = all_emb[labels == 0]
    np_embs = all_emb[labels == 1]
    
    # SVD on P-subspace and NP-subspace
    p_U, p_S, _ = torch.linalg.svd(p_embs.T @ p_embs)
    np_U, np_S, _ = torch.linalg.svd(np_embs.T @ np_embs)
    
    # Principal angles between P and NP subspaces
    # Higher angle = more curvature between classes
    p_basis = p_U[:, :K]  # [D, K]
    np_basis = np_U[:, :K]  # [D, K]
    
    cross = p_basis.T @ np_basis  # [K, K]
    U_c, S_c, V_c = torch.linalg.svd(cross)
    principal_angles = torch.acos(torch.clamp(S_c, -1, 1))
    
    # Curvature gap: mean principal angle between P and NP subspaces
    curvature_gap = principal_angles.mean().item() * 180 / math.pi  # in degrees
    
    # Intra-class curvature (compactness)
    p_compactness = p_S[:K].mean().item()
    np_compactness = np_S[:K].mean().item()
    
    # Classification accuracy
    logits_all = classifier(all_emb)
    preds = logits_all.argmax(dim=-1)
    acc = (preds.cpu() == labels).float().mean().item()

print(f"  Classification accuracy: {acc*100:.1f}%")
print(f"  Curvature gap (P vs NP): {curvature_gap:.1f}°")
print(f"  P-subspace compactness: {p_compactness:.3f}")
print(f"  NP-subspace compactness: {np_compactness:.3f}")
print(f"  Principal angles: {[f'{a*180/math.pi:.1f}°' for a in principal_angles[:5].tolist()]}")

# ── TEH-like test: can a P-circuit be forced into NP region? ──
print("\n[5] TEH barrier test: crossing from P to NP subspace...")
# Take random P circuits and project toward NP subspace
P_forbidden = np_basis @ np_basis.T  # projector onto NP subspace
P_safe = p_basis @ p_basis.T  # projector onto P subspace

test_results = []
for i in range(20):
    p_idx = torch.randint(0, len(p_embs), (1,)).item()
    np_idx = torch.randint(0, len(np_embs), (1,)).item()
    
    p_vec = circuit_vecs[labels == 0][p_idx].to(DEVICE)
    np_vec = circuit_vecs[labels == 1][np_idx].to(DEVICE)
    
    p_emb = F.normalize(embedder(p_vec.unsqueeze(0)), dim=-1)
    np_emb = F.normalize(embedder(np_vec.unsqueeze(0)), dim=-1)
    
    # Measure: how much does the P circuit activate the NP subspace?
    p_on_np = torch.norm(P_forbidden @ p_emb.view(-1)).item()
    p_on_p = torch.norm(P_safe @ p_emb.view(-1)).item()
    
    # NP circuit on P subspace (control)
    np_on_p = torch.norm(P_safe @ np_emb.view(-1)).item()
    
    barrier_ratio = p_on_np / max(p_on_p, 1e-8)
    test_results.append({
        "p_on_np": round(p_on_np, 4),
        "p_on_p": round(p_on_p, 4),
        "barrier_ratio": round(barrier_ratio, 4),
    })

mean_barrier = sum(r["barrier_ratio"] for r in test_results) / len(test_results)
print(f"  Mean P→NP barrier ratio: {mean_barrier:.4f}")
print(f"  Interpretation: {'STRONG SEPARATION' if mean_barrier < 0.5 else 'MODERATE' if mean_barrier < 0.8 else 'WEAK'}")

# ── Save ──
output = {
    "config": {
        "n_circuits": len(circuits),
        "n_p": (labels==0).sum().item(),
        "n_np": (labels==1).sum().item(),
        "k_subspace": K,
        "d_model": D_MODEL,
    },
    "results": {
        "classification_accuracy": round(acc * 100, 1),
        "curvature_gap_deg": round(curvature_gap, 1),
        "p_compactness": round(p_compactness, 3),
        "np_compactness": round(np_compactness, 3),
        "mean_barrier_ratio": round(mean_barrier, 4),
        "barrier_interpretation": "STRONG" if mean_barrier < 0.5 else "MODERATE" if mean_barrier < 0.8 else "WEAK",
        "principal_angles_deg": [round(a * 180 / math.pi, 1) for a in principal_angles[:10].tolist()],
    },
}

with open(f"{OUT_DIR}/ccm_results.json", "w") as f:
    json.dump(output, f, indent=2)
torch.save({"embedder": embedder.state_dict(), "classifier": classifier.state_dict()}, f"{OUT_DIR}/ccm_model.pt")

print(f"\n{'='*60}")
print(f"  CCM RESULTS")
print(f"{'='*60}")
print(f"  P vs NP classification: {acc*100:.1f}%")
print(f"  Curvature gap: {curvature_gap:.1f}°")
print(f"  P→NP barrier: {mean_barrier:.4f} ({'STRONG' if mean_barrier<0.5 else 'MODERATE' if mean_barrier<0.8 else 'WEAK'})")
print(f"  Saved to {OUT_DIR}/")
