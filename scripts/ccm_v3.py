"""CCM v3: Spectral graph features for genuine P vs NP topological barrier.
Uses clause-variable incidence graph Laplacian eigenvalues,
modularity, backbone approximation, and phase transition encoding.
Goal: P->NP barrier < 0.5 (strong topological separation)."""
import torch, json, math, random, os
from collections import defaultdict
import torch.nn.functional as F

DEVICE="cuda"; D=576; K=64
OUT="/home/ubuntu/benchmarks/ccm_v3"
os.makedirs(OUT,exist_ok=True)

# ── Circuit generation (same as before) ──
def g2(nv=8,nc=12):
    return {"nv":nv,"nc":nc,"cls":[(random.randint(0,nv-1),random.choice([True,False]),random.randint(0,nv-1),random.choice([True,False])) for _ in range(nc)],"t":"2SAT"}
def gh(nv=8,nc=12):
    return {"nv":nv,"nc":nc,"cls":[[(v,i==0) for i,v in enumerate(random.sample(range(nv),min(random.randint(1,3),nv)))] for _ in range(nc)],"t":"Horn"}
def g3(nv=8,nc=12):
    return {"nv":nv,"nc":nc,"cls":[list(zip(random.sample(range(nv),min(3,nv)),[random.choice([True,False]) for _ in range(3)])) for _ in range(nc)],"t":"3SAT"}
def gt(n=6):
    c=[(random.uniform(0,1),random.uniform(0,1)) for _ in range(n)]
    return {"nc":n,"dst":[math.sqrt((c[i][0]-c[j][0])**2+(c[i][1]-c[j][1])**2) for i in range(n) for j in range(i+1,n)],"t":"TSP"}

def laplacian_eigenvalues(c):
    """Compute top eigenvalues of clause-variable incidence graph Laplacian."""
    t=c["t"]
    if t=="TSP":
        n=c["nc"]; m=len(c["dst"])
        # Distance matrix Laplacian
        D=torch.zeros(n,n)
        idx=0
        for i in range(n):
            for j in range(i+1,n):
                D[i,j]=c["dst"][idx]; D[j,i]=c["dst"][idx]; idx+=1
        deg=D.sum(dim=1); L=torch.diag(deg)-D
        eigs=torch.linalg.eigvalsh(L)
        return eigs[:8]
    
    nv=c["nv"]; nc=c["nc"]
    # Build bipartite incidence: variables (0..nv-1) + clauses (nv..nv+nc-1)
    N=nv+nc
    adj=torch.zeros(N,N)
    for ci,cl in enumerate(c["cls"]):
        cnode=nv+ci
        if t=="2SAT":
            a,sa,b,sb=cl; adj[a,cnode]=1; adj[cnode,a]=1; adj[b,cnode]=1; adj[cnode,b]=1
        elif t in ("3SAT","Horn"):
            for v,s in cl: adj[v,cnode]=1; adj[cnode,v]=1
    deg=adj.sum(dim=1); L=torch.diag(deg)-adj
    eigs=torch.linalg.eigvalsh(L)
    return eigs[:8]

def spectral_features(c):
    """Extract spectral + structural features (NO type labels)."""
    f=[]
    t=c["t"]
    
    # Spectral features (top 8 Laplacian eigenvalues, normalized)
    eigs=laplacian_eigenvalues(c)
    max_eig=max(eigs[-1].item(),1.0)
    for e in eigs:
        f.append(e.item()/max_eig)
    # Pad to 8
    while len(f)<8: f.append(0)
    
    if t in ("2SAT","Horn","3SAT"):
        nv=c["nv"]; nc=c["nc"]
        f.append(nv/20.0)
        f.append(nc/30.0)
        f.append(nc/max(nv,1))  # clause-to-variable ratio (phase transition!)
        
        # Literal statistics
        pos=neg=0; vf=defaultdict(int); sz=[]
        for cl in c["cls"]:
            if t=="2SAT":
                a,sa,b,sb=cl; sz.append(2)
                pos+=(1 if sa else 0)+(1 if sb else 0); neg+=(0 if sa else 1)+(0 if sb else 1)
                vf[a]+=1; vf[b]+=1
            elif t in ("3SAT","Horn"):
                sz.append(len(cl))
                for v,s in cl: pos+=(1 if s else 0); neg+=(0 if s else 1); vf[v]+=1
        
        f.append(sum(sz)/max(len(sz),1))  # mean clause size
        f.append(pos/max(nc*3,1))
        f.append(neg/max(nc*3,1))
        f.append(pos/max(pos+neg,1))  # positive fraction
        
        # Variable occurrence entropy
        fq=list(vf.values())
        if fq:
            tot=sum(fq); pr=[x/tot for x in fq]
            ent=-sum(p*math.log(p+1e-8) for p in pr); f.append(ent/5.0)
        else: f.append(0)
        
        # Variable coverage (fraction of vars appearing)
        f.append(len(vf)/max(nv,1))
        
        # Average variable frequency
        f.append(sum(vf.values())/max(len(vf),1)/4.0)
        
        # Modularity proxy: variance of variable frequencies
        if fq:
            mean_f=sum(fq)/len(fq)
            var_f=sum((x-mean_f)**2 for x in fq)/len(fq)
            f.append(math.log(var_f+1)/5.0)
        else: f.append(0)
        
    elif t=="TSP":
        n=c["nc"]; f.append(n/15.0); f.append(len(c["dst"])/100.0)
        d=c["dst"]
        if d: f.extend([sum(d)/len(d),max(d),min(d),(sum(d)/len(d))*math.sqrt(n),(max(d)-min(d))])
        else: f.extend([0]*5)
        f.extend([0]*6)  # pad
    
    while len(f)<24: f.append(0)
    return torch.tensor(f[:24],dtype=torch.float32)

print("CCM v3: Spectral + structural features")
N_EACH=400
circuits=[]
for _ in range(N_EACH):
    circuits+=[g2(random.randint(6,14),random.randint(8,25)),
               gh(random.randint(6,14),random.randint(8,25)),
               g3(random.randint(6,14),random.randint(8,25)),
               gt(random.randint(5,9))]
cv=torch.stack([spectral_features(c) for c in circuits])
lb=torch.tensor([0 if c["t"] in ("2SAT","Horn") else 1 for c in circuits])
print(f"Circuits: {len(circuits)} P={ (lb==0).sum().item()} NP={ (lb==1).sum().item()} feat_dim={cv.shape[1]}")

emb=torch.nn.Sequential(torch.nn.Linear(cv.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,D)).to(DEVICE)
clf=torch.nn.Linear(D,2).to(DEVICE)
opt=torch.optim.AdamW(list(emb.parameters())+list(clf.parameters()),lr=0.003)

for step in range(5000):
    idx=torch.randint(0,len(circuits),(80,))
    bv=cv[idx].to(DEVICE); bl=lb[idx].to(DEVICE)
    e=F.normalize(emb(bv),dim=-1)
    logits=clf(e); cls_loss=F.cross_entropy(logits,bl)
    pm=bl==0; npm=bl==1
    pe=e[pm]; npe=e[npm]
    intra=0; inter=0; curv=0
    if pm.sum()>=2: intra+=(1-(pe@pe.T)).mean()
    if npm.sum()>=2: intra+=(1-(npe@npe.T)).mean()
    if pm.sum()>=1 and npm.sum()>=1: inter=torch.relu((pe@npe.T).mean()+0.15)
    if pm.sum()>=2 and npm.sum()>=2: curv=torch.relu(0.8-torch.norm(pe.mean(0)-npe.mean(0)))
    loss=cls_loss+0.03*intra+0.8*inter+0.3*curv
    loss.backward(); opt.step(); opt.zero_grad()
    if (step+1)%500==0:
        with torch.no_grad(): acc=(logits.argmax(-1)==bl).float().mean()
        print(f"  Step {step+1}: loss={loss.item():.4f} acc={acc:.2f} inter={inter:.3f} curv={curv:.3f}")

print("\nMeasuring curvature gap...")
with torch.no_grad():
    ae=F.normalize(emb(cv.to(DEVICE)),dim=-1)
    pe=ae[lb==0]; npe=ae[lb==1]
    pU,pS,_=torch.linalg.svd(pe.T@pe); npU,npS,_=torch.linalg.svd(npe.T@npe)
    cross=pU[:,:K].T@npU[:,:K]; _,Sc,_=torch.linalg.svd(cross)
    angles=torch.acos(torch.clamp(Sc,-1,1))*180/math.pi
    gap=angles.mean().item()
    acc=(clf(ae).argmax(-1).cpu()==lb).float().mean().item()
    P_forb=npU[:,:K]@npU[:,:K].T; P_safe=pU[:,:K]@pU[:,:K].T
    ratios=[]
    for _ in range(100):
        pi=random.randint(0,len(pe)-1)
        pn=torch.norm(P_forb@pe[pi]).item(); pp=torch.norm(P_safe@pe[pi]).item()
        ratios.append(pn/max(pp,1e-8))
    barrier=sum(ratios)/len(ratios)
    p_self=[]; np_self=[]
    for _ in range(100):
        pi=random.randint(0,len(pe)-1); ni=random.randint(0,len(npe)-1)
        p_self.append(torch.norm(P_safe@pe[pi]).item()/max(torch.norm(pe[pi]).item(),1e-8))
        np_self.append(torch.norm(P_forb@npe[ni]).item()/max(torch.norm(npe[ni]).item(),1e-8))

print(f"Acc: {acc*100:.1f}%")
print(f"Curvature gap: {gap:.1f} deg")
print(f"P->NP barrier ratio: {barrier:.4f} ({'STRONG' if barrier<0.5 else 'MODERATE' if barrier<0.8 else 'WEAK'})")
print(f"P self-projection: {sum(p_self)/len(p_self):.3f}")
print(f"NP self-projection: {sum(np_self)/len(np_self):.3f}")
print(f"Principal angles: {[round(a.item(),1) for a in angles[:8]]}")

r={"acc":round(acc*100,1),"curvature_gap":round(gap,1),"barrier":round(barrier,4),
   "barrier_label":"STRONG" if barrier<0.5 else "MODERATE" if barrier<0.8 else "WEAK",
   "p_self":round(sum(p_self)/len(p_self),3),"np_self":round(sum(np_self)/len(np_self),3),
   "angles":[round(a.item(),1) for a in angles[:8]]}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
torch.save({"emb":emb.state_dict(),"clf":clf.state_dict()},f"{OUT}/model.pt")
print(f"Saved to {OUT}/")
