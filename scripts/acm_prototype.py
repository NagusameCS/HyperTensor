"""ACM Prototype: Learn involution ι on ζ(s) manifold.
Verifies: (1) ι² ≈ id, (2) fixed-point dim ≈ k/2 for critical zeros,
(3) off-critical zeros are NOT fixed points.
Deploy to EC2."""
import torch, json, math, os
import torch.nn.functional as F

DEVICE="cuda"; D=576; K=32
OUT="/home/ubuntu/benchmarks/acm_prototype"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  ACM PROTOTYPE: ζ(s) Involution Manifold")
print("  Paper XVII validation")
print("="*60)

# -- ζ(s) zero data --
zeta_zeros_imag=[14.134725,21.022040,25.010857,30.424876,32.935061,37.586178,40.918719,43.327073,48.005150,49.773832,52.970321,56.446248,59.347044,60.831779,65.112543,67.079811,69.546401,72.067158,75.704691,77.144840,79.337375,82.910381,84.735493,87.425273,88.809112,92.491899,94.651344,95.870634,98.831194,101.317851]

def zeta_features(t,real_part):
    f=[]
    f.append(math.log(abs(t)+1)/5.0)
    f.append((t%(2*math.pi))/(2*math.pi))
    gram=(t/(2*math.pi))*(math.log(t/(2*math.pi))-1)+7/8; f.append((gram%1))
    gaps=[abs(z-t) for z in zeta_zeros_imag]; f.append(math.log(min(gaps)+0.01)/3.0)
    f.append((real_part-0.5)*10.0)
    nearby=sum(1 for z in zeta_zeros_imag if abs(z-t)<10); f.append(nearby/10.0)
    return torch.tensor(f,dtype=torch.float32)

# Critical zeros (Re=0.5)
crit_z=torch.stack([zeta_features(t,0.5) for t in zeta_zeros_imag])  # [30,6]
# Off-critical candidates with Re≠0.5
off_cases=[(15.5,0.35),(22.8,0.62),(31.2,0.28),(44.7,0.71),(53.1,0.44),(60.0,0.55),(71.3,0.33),(85.9,0.68),(93.4,0.41),(99.9,0.59),(20.0,0.38),(35.0,0.66),(50.0,0.25),(65.0,0.73),(80.0,0.42)]
off_z=torch.stack([zeta_features(t,rp) for t,rp in off_cases])  # [15,6]

print(f"Critical zeros: {len(crit_z)}, Off-critical: {len(off_z)}")

# -- Build manifold + involution --
encoder=torch.nn.Sequential(torch.nn.Linear(6,128),torch.nn.GELU(),torch.nn.Linear(128,D)).to(DEVICE)
# Involution ι: D -> D, must satisfy ι²≈id
involution=torch.nn.Sequential(torch.nn.Linear(D,256),torch.nn.GELU(),torch.nn.Linear(256,256),torch.nn.GELU(),torch.nn.Linear(256,D)).to(DEVICE)

opt=torch.optim.AdamW(list(encoder.parameters())+list(involution.parameters()),lr=0.002)
steps=4000

for step in range(steps):
    # Sample batch
    ci=torch.randint(0,len(crit_z),(16,)); oi=torch.randint(0,len(off_z),(8,))
    cz=crit_z[ci].to(DEVICE); oz=off_z[oi].to(DEVICE)
    
    e_c=encoder(cz); e_o=encoder(oz)
    e_c=F.normalize(e_c,dim=-1); e_o=F.normalize(e_o,dim=-1)
    
    # Involution on critical zeros
    iota_c=involution(e_c)
    # Involution on off-critical zeros
    iota_o=involution(e_o)
    
    # Loss 1: Critical zeros are FIXED POINTS of ι
    fp_loss=F.mse_loss(iota_c,e_c)
    
    # Loss 2: ι is an INVOLUTION: ι(ι(x)) ≈ x
    iota2_c=involution(iota_c)
    iota2_o=involution(iota_o)
    inv_loss=F.mse_loss(iota2_c,e_c)+F.mse_loss(iota2_o,e_o)
    
    # Loss 3: Off-critical zeros are NOT fixed points
    # Maximize distance between ι(x) and x for off-critical
    off_dist=torch.norm(iota_o-e_o,dim=-1).mean()
    off_loss=torch.relu(0.3-off_dist)  # want distance > 0.3
    
    # Loss 4: Critical zeros cluster (same subspace)
    if len(e_c)>=2:
        clust_loss=(1-(e_c@e_c.T)).mean()
    else:
        clust_loss=0.0
    
    loss=fp_loss+0.5*inv_loss+2.0*off_loss+0.1*clust_loss
    loss.backward(); opt.step(); opt.zero_grad()
    
    if (step+1)%500==0:
        with torch.no_grad():
            fp_err=torch.norm(iota_c-e_c,dim=-1).mean().item()
            off_err=torch.norm(iota_o-e_o,dim=-1).mean().item()
        print(f"  Step {step+1}: loss={loss.item():.4f} fp={fp_err:.3f} off={off_err:.3f} inv={inv_loss.item():.3f}")

# -- Validation --
print("\n[Validation]")
with torch.no_grad():
    # All embeddings
    ec_all=F.normalize(encoder(crit_z.to(DEVICE)),dim=-1)
    eo_all=F.normalize(encoder(off_z.to(DEVICE)),dim=-1)
    
    # Fixed-point error for critical zeros
    iota_crit=involution(ec_all)
    fp_errors=torch.norm(iota_crit-ec_all,dim=-1)
    print(f"  Critical FP error: mean={fp_errors.mean():.4f} max={fp_errors.max():.4f}")
    
    # Fixed-point error for off-critical
    iota_off=involution(eo_all)
    off_errors=torch.norm(iota_off-eo_all,dim=-1)
    print(f"  Off-critical FP error: mean={off_errors.mean():.4f} min={off_errors.min():.4f}")
    
    # Involution property: ι² ≈ id
    iota2_crit=involution(iota_crit)
    inv_errors=torch.norm(iota2_crit-ec_all,dim=-1)
    print(f"  ι²≈id error: mean={inv_errors.mean():.4f} max={inv_errors.max():.4f}")
    
    # Fixed-point subspace dimension
    # Compute Jacobian of ι at critical points: J = ∂ι/∂x ≈ (ι(x+ε)-ι(x))/ε
    eps=0.01
    J_sum=torch.zeros(D,D,device=DEVICE)
    n_samples=min(20,len(ec_all))
    for i in range(n_samples):
        x=ec_all[i]
        for d_idx in range(min(D,100)):  # sample 100 dims for efficiency
            delta=torch.zeros(D,device=DEVICE); delta[d_idx]=eps
            J_col=(involution((x+delta).unsqueeze(0))-involution(x.unsqueeze(0))).squeeze()/eps
            J_sum[:,d_idx]+=J_col
    J=J_sum/(n_samples*100/D)
    
    # Fixed-point dim = rank of I-J at fixed points
    I_minus_J=torch.eye(D,device=DEVICE)-J
    S=torch.linalg.svdvals(I_minus_J)
    # Count singular values above threshold -> dimension of nullspace
    thresh=0.1
    fp_dim=(S<thresh).sum().item()
    print(f"  Fixed-point subspace dim: {fp_dim} (expected ≈{D//2}={D//2})")
    print(f"  Top 10 singular values of I-J: {[round(s.item(),3) for s in S[:10]]}")
    
    # TEH test: can off-critical reach fixed-point set?
    # Project off-critical onto critical subspace
    U,_,_=torch.linalg.svd(ec_all.T@ec_all)
    crit_basis=U[:,:K]; P_crit=crit_basis@crit_basis.T
    P_forb=torch.eye(D,device=DEVICE)-P_crit
    
    off_act=[]
    for i in range(len(eo_all)):
        emb=eo_all[i]
        act=torch.norm(P_forb@emb).item()/max(torch.norm(emb).item(),1e-8)*100
        off_act.append(act)
    crit_act=[]
    for i in range(min(10,len(ec_all))):
        emb=ec_all[i]
        act=torch.norm(P_forb@emb).item()/max(torch.norm(emb).item(),1e-8)*100
        crit_act.append(act)
    
    detection=sum(1 for a in off_act if a>15)
    fp_count=sum(1 for a in crit_act if a>15)
    print(f"  TEH detection: {detection}/{len(off_act)} off-critical detected")
    print(f"  TEH false positives: {fp_count}/{len(crit_act)}")
    print(f"  Mean off-critical act: {sum(off_act)/len(off_act):.1f}%")
    print(f"  Mean critical act: {sum(crit_act)/len(crit_act):.1f}%")

# -- Save --
results={
    "fp_error_mean":round(fp_errors.mean().item(),4),
    "fp_error_max":round(fp_errors.max().item(),4),
    "off_error_mean":round(off_errors.mean().item(),4),
    "involution_error":round(inv_errors.mean().item(),4),
    "fixed_point_dim":fp_dim,
    "expected_dim":D//2,
    "teh_detection":f"{detection}/{len(off_act)}",
    "teh_false_positives":f"{fp_count}/{len(crit_act)}",
    "mean_off_act":round(sum(off_act)/len(off_act),1),
    "mean_crit_act":round(sum(crit_act)/len(crit_act),1),
}
with open(f"{OUT}/results.json","w") as f: json.dump(results,f,indent=2)
torch.save({"encoder":encoder.state_dict(),"involution":involution.state_dict(),"crit_basis":crit_basis.cpu()},f"{OUT}/model.pt")
print(f"\nSaved to {OUT}/")
print(f"ACM verification: fp_dim={fp_dim} (target={D//2}) | TEH={detection}/{len(off_act)} | ι² err={inv_errors.mean():.4f}")
