"""CCM v2: Structural features only (NO type one-hot). Real P vs NP test."""
import torch, json, math, random, os
from collections import defaultdict
import torch.nn.functional as F

DEVICE="cuda"; D=576; K=64
OUT="/home/ubuntu/benchmarks/ccm_v2"
os.makedirs(OUT,exist_ok=True)

def g2(nv=8,nc=12):
    return {"nv":nv,"nc":nc,"cls":[(random.randint(0,nv-1),random.choice([True,False]),random.randint(0,nv-1),random.choice([True,False])) for _ in range(nc)],"t":"2SAT"}
def gh(nv=8,nc=12):
    return {"nv":nv,"nc":nc,"cls":[[(v,i==0) for i,v in enumerate(random.sample(range(nv),min(random.randint(1,3),nv)))] for _ in range(nc)],"t":"Horn"}
def g3(nv=8,nc=12):
    return {"nv":nv,"nc":nc,"cls":[list(zip(random.sample(range(nv),min(3,nv)),[random.choice([True,False]) for _ in range(3)])) for _ in range(nc)],"t":"3SAT"}
def gt(n=6):
    c=[(random.uniform(0,1),random.uniform(0,1)) for _ in range(n)]
    return {"nc":n,"dst":[math.sqrt((c[i][0]-c[j][0])**2+(c[i][1]-c[j][1])**2) for i in range(n) for j in range(i+1,n)],"t":"TSP"}

def feat(c):
    f=[]
    t=c["t"]
    if t in ("2SAT","Horn","3SAT"):
        nv=c["nv"]; nc=c["nc"]
        f+=[nv/20.0, nc/30.0, nc/max(nv,1)]
        pos=neg=0; vf=defaultdict(int); sz=[]
        for cl in c["cls"]:
            if t=="2SAT":
                a,sa,b,sb=cl; sz.append(2); pos+=(1 if sa else 0)+(1 if sb else 0); neg+=(0 if sa else 1)+(0 if sb else 1); vf[a]+=1; vf[b]+=1
            elif t in ("3SAT","Horn"):
                sz.append(len(cl))
                for v,s in cl: pos+=(1 if s else 0); neg+=(0 if s else 1); vf[v]+=1
        f+=[sum(sz)/max(len(sz),1), pos/max(nc*3,1), neg/max(nc*3,1)]
        fq=list(vf.values())
        if fq:
            tot=sum(fq); pr=[x/tot for x in fq]; ent=-sum(p*math.log(p+1e-8) for p in pr); f.append(ent/5.0)
        else: f.append(0)
        f+=[len(vf)/max(nv,1), sum(vf.values())/max(len(vf),1)/3.0, pos/max(pos+neg,1)]
    elif t=="TSP":
        n=c["nc"]; f+=[n/15.0, len(c["dst"])/100.0]
        d=c["dst"]
        if d: f+=[sum(d)/len(d), max(d), min(d), (sum(d)/len(d))*math.sqrt(n)]
        else: f+=[0]*4
        f+=[0]*5
    while len(f)<16: f.append(0)
    return torch.tensor(f[:16],dtype=torch.float32)

print("CCM v2: Structural features only")
N=500
circuits=[]
for _ in range(N):
    circuits+=[g2(random.randint(6,12),random.randint(8,20)), gh(random.randint(6,12),random.randint(8,20)), g3(random.randint(6,12),random.randint(8,20)), gt(random.randint(5,8))]
cv=torch.stack([feat(c) for c in circuits])
lb=torch.tensor([0 if c["t"] in ("2SAT","Horn") else 1 for c in circuits])
print(f"Circuits: {len(circuits)} P={ (lb==0).sum().item()} NP={ (lb==1).sum().item()}")

emb=torch.nn.Sequential(torch.nn.Linear(16,128),torch.nn.GELU(),torch.nn.Linear(128,D)).to(DEVICE)
clf=torch.nn.Linear(D,2).to(DEVICE)
opt=torch.optim.AdamW(list(emb.parameters())+list(clf.parameters()),lr=0.003)

for step in range(4000):
    idx=torch.randint(0,len(circuits),(64,))
    bv=cv[idx].to(DEVICE); bl=lb[idx].to(DEVICE)
    e=F.normalize(emb(bv),dim=-1)
    logits=clf(e); cls_loss=F.cross_entropy(logits,bl)
    pm=bl==0; npm=bl==1
    pe=e[pm]; npe=e[npm]
    intra=0; inter=0; curv=0
    if pm.sum()>=2: intra+=(1-(pe@pe.T)).mean()
    if npm.sum()>=2: intra+=(1-(npe@npe.T)).mean()
    if pm.sum()>=1 and npm.sum()>=1: inter=torch.relu((pe@npe.T).mean()+0.2)
    if pm.sum()>=2 and npm.sum()>=2: curv=torch.relu(0.5-torch.norm(pe.mean(0)-npe.mean(0)))
    loss=cls_loss+0.05*intra+0.5*inter+0.2*curv
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
    for _ in range(50):
        pi=random.randint(0,len(pe)-1)
        pn=torch.norm(P_forb@pe[pi]).item(); pp=torch.norm(P_safe@pe[pi]).item()
        ratios.append(pn/max(pp,1e-8))
    barrier=sum(ratios)/len(ratios)

print(f"Acc: {acc*100:.1f}%")
print(f"Curvature gap: {gap:.1f} deg")
print(f"P->NP barrier: {barrier:.4f}")
print(f"Principal angles: {[round(a.item(),1) for a in angles[:8]]}")

r={"acc":round(acc*100,1),"curvature_gap":round(gap,1),"barrier":round(barrier,4),"barrier_label":"STRONG" if barrier<0.5 else "MODERATE" if barrier<0.8 else "WEAK","angles":[round(a.item(),1) for a in angles[:8]]}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
torch.save({"emb":emb.state_dict(),"clf":clf.state_dict()},f"{OUT}/model.pt")
print(f"Saved to {OUT}/")
print(f"INTERPRETATION: P vs NP geometrically {'SEPARABLE' if gap>30 else 'OVERLAPPING' if gap<15 else 'PARTIALLY SEPARABLE'}")
