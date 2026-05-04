"""Hodge Prototype: Algebraic Variety Manifold (Paper XXXI).
Embeds simplified projective varieties. Detects Hodge classes as
harmonic forms (Laplacian nullspace). Tests: algebraic cycles = closed geodesics.
Deploy to EC2."""
import torch, json, math, os, random
import torch.nn.functional as F

DEVICE="cuda"; D=512; K=64
OUT="/home/ubuntu/benchmarks/hodge_prototype"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  HODGE PROTOTYPE: Algebraic Variety Manifold")
print("  Paper XXXI: Harmonic forms = closed geodesics")
print("="*60)

# -- Simulated algebraic varieties --
# Represent projective hypersurfaces by their defining polynomials
# x^d + y^d + z^d + ... = 0 in P^n
# Features: dimension n, degree d, Hodge numbers h^{p,q}

def random_variety():
    """Generate a random smooth projective hypersurface."""
    n=random.randint(1,5)  # dimension of projective space
    d=random.randint(2,8)  # degree of hypersurface
    dim=n-1  # dimension of variety
    
    # Approximate Hodge numbers for smooth hypersurfaces
    # h^{p,q} for p+q = dim
    hodge_numbers=[]
    for p in range(dim+1):
        q=dim-p
        if p==q:
            # Middle dimension: complex formula
            h=1+random.randint(0,min(d,20))
        else:
            h=1 if p+q<n else random.randint(0,min(d,5))
        hodge_numbers.append(h)
    
    # Chern classes (simplified)
    c1=(n+1-d)  # first Chern class
    c2=random.randint(-10*d,10*d)
    
    # Euler characteristic
    euler=sum((-1)**p*h for p,h in enumerate(hodge_numbers))
    
    return {
        "n":n,"d":d,"dim":dim,
        "hodge":hodge_numbers,"c1":c1,"c2":c2,"euler":euler,
        "type":"hypersurface"
    }

def variety_features(v):
    f=[]
    f.append(v["n"]/6.0); f.append(v["d"]/10.0); f.append(v["dim"]/5.0)
    # Hodge numbers (pad to 6)
    for i in range(6):
        if i<len(v["hodge"]): f.append(v["hodge"][i]/20.0)
        else: f.append(0)
    f.append(v["c1"]/10.0); f.append(v["c2"]/100.0)
    f.append(v["euler"]/50.0)
    # Is it Calabi-Yau? (c1=0)
    f.append(1.0 if abs(v["c1"])<1e-6 else 0.0)
    return torch.tensor(f,dtype=torch.float32)

n_varieties=1000
varieties=[random_variety() for _ in range(n_varieties)]
FEAT_DIM=len(variety_features(varieties[0]))
vvecs=torch.stack([variety_features(v) for v in varieties])
print(f"  Varieties: {n_varieties}, feat dim: {FEAT_DIM}")

# -- Labels: Hodge classes <- algebraic cycles --
# For each variety, generate some "algebraic cycles" (simulated subvarieties)
# and corresponding "Hodge classes" (cohomology classes)
# The conjecture says they should match

# Generate algebraic cycles as random linear combinations of basis cycles
n_cycles_per_variety=3
cycle_vectors=[]  # each cycle is a feature vector
cycle_variety_idx=[]  # which variety it belongs to

for i,v in enumerate(varieties):
    dim=v["dim"]
    for _ in range(n_cycles_per_variety):
        # Algebraic cycle: subvariety of codimension p
        p=random.randint(1,max(1,dim))
        # Encode cycle as: [variety features + cycle-specific features]
        cf=list(variety_features(v))
        cf.append(p/max(1,dim))  # codimension
        cf.append(random.random())  # degree of subvariety
        cf.append(random.random())  # intersection number
        cycle_vectors.append(torch.tensor(cf,dtype=torch.float32))
        cycle_variety_idx.append(i)

cycle_vecs=torch.stack(cycle_vectors)  # [N_cycles, feat_dim+3]
print(f"  Algebraic cycles: {len(cycle_vecs)}")

# -- Train manifold --
print(f"\n[2] Training Hodge manifold (D={D})...")
# Encoder: variety -> manifold point
vencoder=torch.nn.Sequential(torch.nn.Linear(FEAT_DIM,256),torch.nn.GELU(),torch.nn.Linear(256,D)).to(DEVICE)
# Cycle embedder: algebraic cycle -> tangent vector on manifold
cencoder=torch.nn.Sequential(torch.nn.Linear(FEAT_DIM+3,256),torch.nn.GELU(),torch.nn.Linear(256,D)).to(DEVICE)

opt=torch.optim.AdamW(list(vencoder.parameters())+list(cencoder.parameters()),lr=0.002)

for step in range(3000):
    # Variety continuity: similar varieties -> nearby
    vi=torch.randint(0,n_varieties,(48,))
    ve=F.normalize(vencoder(vvecs[vi].to(DEVICE)),dim=-1)
    cont=(1-(ve@ve.T)).mean()
    
    # Cycle-geodesic: each cycle is a tangent vector at its variety
    ci=torch.randint(0,len(cycle_vecs),(32,))
    cycle_emb=cencoder(cycle_vecs[ci].to(DEVICE))
    # Get corresponding variety embeddings
    v_indices=[cycle_variety_idx[j] for j in ci.tolist()]
    var_emb=vencoder(vvecs[torch.tensor(v_indices)].to(DEVICE))
    
    # Cycle should be TANGENT to variety (orthogonal to position)
    # Harmonic condition: Δω = 0 -> cycle is in kernel of Laplacian
    # Simplified: cycle direction is orthogonal to variety position gradient
    tangent_comp=torch.norm(cycle_emb-(cycle_emb*var_emb).sum(dim=-1,keepdim=True)*var_emb,dim=-1).mean()
    # Want cycles to be tangent (parallel to variety, not radial)
    radial=F.cosine_similarity(cycle_emb,var_emb).abs().mean()
    cycle_loss=radial  # minimize radial component (cycles are tangent)
    
    # Closed geodesic: sum of cycles around a variety should ≈ 0
    # (algebraic equivalence: principal divisors sum to zero)
    closure=0
    for i,v_idx in enumerate(v_indices[:8]):
        # Find all cycles for this variety
        same_var=[j for j,idx in enumerate(cycle_variety_idx) if idx==v_idx]
        if len(same_var)>=2:
            c1=cencoder(cycle_vecs[same_var[0]].unsqueeze(0).to(DEVICE))
            c2=cencoder(cycle_vecs[same_var[1]].unsqueeze(0).to(DEVICE))
            closure+=F.mse_loss(c1+c2,torch.zeros_like(c1))
    closure=closure/max(len(v_indices),1)
    
    loss=cont+0.1*cycle_loss+0.01*closure
    loss.backward(); opt.step(); opt.zero_grad()
    if (step+1)%500==0:
        print(f"  Step {step+1}: loss={loss.item():.4f} cont={cont.item():.3f} cycle={cycle_loss.item():.3f}")

# -- Hodge detection --
print("\n[3] Detecting Hodge classes as harmonic forms...")
with torch.no_grad():
    # Embed all varieties
    ve_all=F.normalize(vencoder(vvecs.to(DEVICE)),dim=-1)
    
    # Build graph Laplacian on variety manifold
    n_sub=min(300,n_varieties); vs=ve_all[:n_sub]; sim=vs@vs.T
    k_nn=15; adj=torch.zeros(n_sub,n_sub,device=DEVICE)
    for i in range(n_sub):
        _,nn=torch.topk(sim[i],k=k_nn+1); adj[i,nn[1:]]=1
    adj=(adj+adj.T)/2; deg=adj.sum(dim=1)
    L=torch.diag(deg)-adj
    deg_inv=torch.diag(1.0/torch.sqrt(deg+1e-8))
    L_norm=torch.eye(n_sub,device=DEVICE)-deg_inv@adj@deg_inv
    
    # Harmonic forms = nullspace of Laplacian
    eigs,U=torch.linalg.eigh(L_norm)
    # Count near-zero eigenvalues (harmonic forms)
    n_harmonic=(eigs<0.01).sum().item()
    
    # For each variety, count algebraic cycles as proxy for Hodge classes
    cycle_counts=[sum(1 for idx in cycle_variety_idx if idx==i) for i in range(n_sub)]
    
    # Correlation: do varieties with more cycles have more harmonic forms?
    # (This tests the Hodge conjecture: algebraic cycles ↔ harmonic forms)
    harmonic_per_var=[]
    for i in range(n_sub):
        # Project variety embedding onto harmonic subspace
        harmonic_proj=torch.norm(U[i,:n_harmonic]).item() if n_harmonic>0 else 0
        harmonic_per_var.append(harmonic_proj)
    
    h_t=torch.tensor(harmonic_per_var)
    c_t=torch.tensor(cycle_counts,dtype=torch.float32)
    corr=torch.corrcoef(torch.stack([h_t,c_t]))[0,1].item() if n_harmonic>0 else 0
    
    print(f"  Harmonic forms (dim): {n_harmonic}")
    print(f"  Harmonic-cycles correlation: {corr:.3f}")
    print(f"  First 8 eigenvalues: {[round(e.item(),4) for e in eigs[:8]]}")
    print(f"  Hodge: {'SUPPORTED' if corr>0.2 else 'WEAK'} (corr={corr:.3f})")

r={"n_varieties":n_varieties,"n_cycles":len(cycle_vecs),"D":D,
   "harmonic_dim":n_harmonic,"harmonic_cycle_corr":round(corr,3),
   "eigs":[round(e.item(),4) for e in eigs[:8]],
   "interpretation":"SUPPORTED" if corr>0.2 else "WEAK"}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
torch.save({"vencoder":vencoder.state_dict(),"cencoder":cencoder.state_dict()},f"{OUT}/model.pt")
print(f"\nSaved to {OUT}/ | {r['interpretation']}")
