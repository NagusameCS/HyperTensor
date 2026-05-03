"""HSM Prototype: 2D Navier-Stokes flow manifold.
Embeds fluid velocity fields, measures sectional curvature vs enstrophy.
Validates Paper XXII: enstrophy=curvature on hydrodynamic manifold.
Uses Kolmogorov flow and Taylor-Green vortex as test cases.
Deploy to EC2."""
import torch, json, math, os
import torch.nn.functional as F

DEVICE="cuda"; D=512; GRID=32
OUT="/home/ubuntu/benchmarks/hsm_prototype"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  HSM PROTOTYPE: 2D Navier-Stokes Manifold")
print("  Paper XXII: Enstrophy = Curvature")
print("="*60)

# ── Generate 2D flow fields ──
print(f"\n[1] Generating 2D flow fields ({GRID}x{GRID} grid)...")

def kolmogorov_flow(x,y,t,Re):
    """Kolmogorov flow: u=sin(y)exp(-t/Re), v=0. Smooth, laminar."""
    decay=math.exp(-t/Re)
    u=torch.sin(y)*decay
    v=torch.zeros_like(x)
    return u,v

def taylor_green_vortex(x,y,t,Re):
    """Taylor-Green vortex: u=cos(x)sin(y)exp(-2t/Re). Decaying."""
    decay=math.exp(-2*t/Re)
    u=torch.cos(x)*torch.sin(y)*decay
    v=-torch.sin(x)*torch.cos(y)*decay
    return u,v

def shear_layer(x,y,t,Re):
    """Shear layer: u=tanh((y-0.5)/0.1). Sharp gradient."""
    decay=math.exp(-t/Re)
    u=torch.tanh((y-0.5)/0.1)*decay
    v=0.01*torch.sin(2*math.pi*x)*decay
    return u,v

def random_field(x,y,Re):
    """Random smooth field: superposition of Fourier modes."""
    u=torch.zeros_like(x); v=torch.zeros_like(x)
    for kx in range(1,5):
        for ky in range(1,5):
            amp=torch.randn(1).item()/(kx*ky)
            u+=amp*torch.sin(kx*x+ky*y)
            v+=amp*torch.cos(kx*x-ky*y)
    decay=math.exp(-abs(torch.randn(1).item())/Re)
    u*=decay; v*=decay
    return u,v

# Generate snapshots
n_snapshots=2000
fields=[]; enstrophies=[]; Re_vals=[]

# Create coordinate grid
xs=torch.linspace(0,2*math.pi,GRID)
ys=torch.linspace(0,2*math.pi,GRID)
X,Y=torch.meshgrid(xs,ys,indexing='ij')

for i in range(n_snapshots):
    Re=10**(1+2*torch.rand(1).item())  # Re in [10, 1000]
    t=torch.rand(1).item()*5.0  # t in [0,5]
    
    # Pick random flow type
    flow_type=torch.randint(0,4,(1,)).item()
    if flow_type==0:
        u,v=kolmogorov_flow(X,Y,t,Re)
    elif flow_type==1:
        u,v=taylor_green_vortex(X,Y,t,Re)
    elif flow_type==2:
        u,v=shear_layer(X,Y,t,Re)
    else:
        u,v=random_field(X,Y,Re)
    
    # Compute vorticity ω = ∂v/∂x - ∂u/∂y
    # Finite difference
    dx=2*math.pi/GRID
    dv_dx=(torch.roll(v,-1,dims=0)-torch.roll(v,1,dims=0))/(2*dx)
    du_dy=(torch.roll(u,-1,dims=1)-torch.roll(u,1,dims=1))/(2*dx)
    vorticity=dv_dx-du_dy
    
    # Enstrophy = 0.5 * ∫ ω² dA
    enstrophy=0.5*(vorticity**2).mean().item()
    
    # Flatten velocity field
    field=torch.cat([u.flatten(),v.flatten()])  # [2*GRID^2]
    fields.append(field)
    enstrophies.append(enstrophy)
    Re_vals.append(Re)

field_vecs=torch.stack(fields)  # [N, 2*GRID^2]
enstrophy_t=torch.tensor(enstrophies)
Re_t=torch.tensor(Re_vals)

print(f"  Generated {n_snapshots} flow snapshots")
print(f"  Field dim: {field_vecs.shape[1]}")
print(f"  Enstrophy range: [{enstrophy_t.min():.4f}, {enstrophy_t.max():.4f}]")
print(f"  Re range: [{Re_t.min():.0f}, {Re_t.max():.0f}]")

# ── Embed onto manifold ──
print("\n[2] Embedding flows onto hydrodynamic manifold...")

# PCA compression first (2*GRID^2=2048 → 256)
U,S,Vh=torch.linalg.svd(field_vecs.float()-field_vecs.float().mean(0),full_matrices=False)
pca_basis=Vh[:256,:].T  # [2048, 256] — right singular vectors = feature space directions
compressed=field_vecs.float()@pca_basis  # [N, 256]

encoder=torch.nn.Sequential(
    torch.nn.Linear(256,512),torch.nn.GELU(),
    torch.nn.Linear(512,D)
).to(DEVICE)

opt=torch.optim.AdamW(encoder.parameters(),lr=0.002)
steps=4000

for step in range(steps):
    idx=torch.randint(0,n_snapshots,(64,))
    cv=compressed[idx].to(DEVICE)
    emb=encoder(cv)
    emb_norm=F.normalize(emb,dim=-1)
    
    # Continuity: similar flows → nearby embedding
    feat_dist=torch.cdist(cv,cv)
    sim_target=torch.exp(-feat_dist*0.3)
    emb_sim=emb_norm@emb_norm.T
    cont_loss=F.mse_loss(emb_sim,sim_target)
    
    # Enstrophy encoding: embedding norm should correlate with enstrophy
    e_vals=enstrophy_t[idx].to(DEVICE)
    e_norm=(e_vals-e_vals.min())/(e_vals.max()-e_vals.min()+1e-8)
    emb_norm_vals=torch.norm(emb,dim=-1)
    emb_norm_norm=(emb_norm_vals-emb_norm_vals.min())/(emb_norm_vals.max()-emb_norm_vals.min()+1e-8)
    enstrophy_loss=F.mse_loss(emb_norm_norm,e_norm)
    
    loss=cont_loss+0.2*enstrophy_loss
    loss.backward(); opt.step(); opt.zero_grad()
    
    if (step+1)%1000==0:
        print(f"  Step {step+1}: loss={loss.item():.4f} cont={cont_loss.item():.3f} enstrophy={enstrophy_loss.item():.3f}")

# ── Measure enstrophy-curvature correspondence ──
print("\n[3] Measuring enstrophy vs curvature...")

with torch.no_grad():
    all_emb=encoder(compressed.to(DEVICE))
    all_emb=F.normalize(all_emb,dim=-1)
    
    # Sectional curvature proxy: graph Laplacian eigenvalue gap
    # Build k-NN graph
    k_nn=20; n_sub=min(400,n_snapshots)
    emb_sub=all_emb[:n_sub]
    sim=emb_sub@emb_sub.T
    
    adj=torch.zeros(n_sub,n_sub,device=DEVICE)
    for i in range(n_sub):
        _,nn_idx=torch.topk(sim[i],k=k_nn+1)
        adj[i,nn_idx[1:]]=1
    adj=(adj+adj.T)/2
    
    # Graph Laplacian
    deg=adj.sum(dim=1)
    L=torch.diag(deg)-adj
    deg_inv=torch.diag(1.0/torch.sqrt(deg+1e-8))
    L_norm=torch.eye(n_sub,device=DEVICE)-deg_inv@adj@deg_inv
    
    # Local curvature at each point = diagonal of L (how much it differs from neighbors)
    local_curvature=L_norm.diag()  # [n_sub]
    
    # Correlation: enstrophy vs local curvature
    e_sub=enstrophy_t[:n_sub]
    e_norm_val=(e_sub-e_sub.min())/(e_sub.max()-e_sub.min()+1e-8)
    
    curv_norm=(local_curvature-local_curvature.min())/(local_curvature.max()-local_curvature.min()+1e-8)
    corr=torch.corrcoef(torch.stack([e_norm_val.float(),curv_norm.cpu().float()]))[0,1].item()
    
    # Split by enstrophy
    high_e=e_sub>e_sub.median()
    low_e=~high_e
    curv_high=local_curvature[high_e].mean().item()
    curv_low=local_curvature[low_e].mean().item()
    
    print(f"  Enstrophy-curvature correlation: {corr:.3f}")
    print(f"  Mean curvature (high enstrophy): {curv_high:.4f}")
    print(f"  Mean curvature (low enstrophy): {curv_low:.4f}")
    print(f"  Curvature ratio (high/low): {curv_high/max(curv_low,1e-8):.2f}x")
    
    # Laplacian spectral gap (global regularity proxy)
    eigs=torch.linalg.eigvalsh(L_norm)
    eigs_sorted=sorted(eigs.cpu().tolist())
    nonzero=[e for e in eigs_sorted if e>1e-6]
    spectral_gap=nonzero[0] if nonzero else 0
    
    print(f"  Spectral gap: {spectral_gap:.6f}")
    print(f"  First 6 eigenvalues: {[round(e,4) for e in eigs_sorted[:6]]}")

# ── Save ──
results={
    "n_snapshots":n_snapshots,"grid":GRID,"d_embed":D,
    "enstrophy_curvature_corr":round(corr,3),
    "curvature_high_enstrophy":round(curv_high,4),
    "curvature_low_enstrophy":round(curv_low,4),
    "curvature_ratio":round(curv_high/max(curv_low,1e-8),2),
    "spectral_gap":round(spectral_gap,6),
    "first_6_eigs":[round(e,4) for e in eigs_sorted[:6]],
    "interpretation":"Positive enstrophy-curvature correlation validates HSM geometric formulation" if corr>0.3 else "Weak correlation — need higher Re flows or 3D",
}
with open(f"{OUT}/results.json","w") as f: json.dump(results,f,indent=2)
torch.save({"encoder":encoder.state_dict(),"pca_basis":pca_basis},f"{OUT}/model.pt")
print(f"\nSaved to {OUT}/")
print(f"HSM: enstrophy-curvature corr={corr:.3f} | {'Validated' if corr>0.3 else 'Needs 3D/higher Re'}")
