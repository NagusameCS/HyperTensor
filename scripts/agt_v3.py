"""AGT v3: Scaled. 10K primes, 100+ zeros, D=768, K=32."""
import torch, json, math, os, random
import torch.nn.functional as F

DEVICE="cuda"; D=768; K_CRIT=32
OUT="/home/ubuntu/benchmarks/agt_v3"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  AGT v3: 10K primes + 100+ zeros")
print("="*60)

N_MAX=100000
def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

primes=[n for n in range(2,N_MAX+1) if is_prime(n)]
print(f"  Primes: {len(primes)}")

def pfeat(p):
    idx=primes.index(p)
    f=[math.log(p)/math.log(N_MAX)]
    np=primes[idx+1] if idx+1<len(primes) else p+2
    f.append(math.log(max(np-p,1)+1)/math.log(N_MAX))
    for m in [3,5,7,11,13]: f.append((p%m)/m)
    th=sum(math.log(q) for q in primes if q<=p)
    f.append(th/max(p,1)); f.append(-1.0); f.append(idx/len(primes))
    pnt=p/math.log(p) if p>1 else 1
    f.append((idx+1-pnt)/max(pnt,1))
    return torch.tensor(f,dtype=torch.float32)

FD=len(pfeat(primes[0])); pv=torch.stack([pfeat(p) for p in primes])

zeta_zeros_imag=[
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
print(f"  Zeros: {len(zeta_zeros_imag)}")

ZFD=7
def zfeat(t,rp):
    f=[math.log(abs(t)+1)/5.0, (t%(2*math.pi))/(2*math.pi)]
    gram=(t/(2*math.pi))*(math.log(t/(2*math.pi))-1)+7/8; f.append(gram%1)
    gaps=[abs(z-t) for z in zeta_zeros_imag]; f.append(math.log(min(gaps)+0.01)/3.0)
    f.append((rp-0.5)*10.0)
    nearby=sum(1 for z in zeta_zeros_imag if abs(z-t)<10); f.append(nearby/10.0)
    f.append(sum(1 for z in zeta_zeros_imag if z<=t)/len(zeta_zeros_imag))
    return torch.tensor(f,dtype=torch.float32)

crit_z=torch.stack([zfeat(t,0.5) for t in zeta_zeros_imag])

# Off-critical
off_cases=[]
for t in zeta_zeros_imag[:30]:
    off_cases.append((t+random.uniform(-1.5,1.5),random.choice([0.3,0.35,0.4,0.55,0.6,0.65,0.7])))
for _ in range(30):
    t=20+80*random.random()
    off_cases.append((t,0.5+0.3*(random.random()-0.5)))
off_z=torch.stack([zfeat(t,rp) for t,rp in off_cases])
print(f"  Crit: {len(crit_z)}, Off: {len(off_z)}")

# Train
print(f"\n[2] Training D={D} manifold...")
emb=torch.nn.Sequential(torch.nn.Linear(FD,256),torch.nn.GELU(),torch.nn.Linear(256,D)).to(DEVICE)
zemb=torch.nn.Sequential(torch.nn.Linear(ZFD,128),torch.nn.GELU(),torch.nn.Linear(128,D)).to(DEVICE)
opt=torch.optim.AdamW(list(emb.parameters())+list(zemb.parameters()),lr=0.002)

for step in range(3000):
    bi=torch.randint(0,len(primes),(48,))
    pe=F.normalize(emb(pv[bi].to(DEVICE)),dim=-1)
    ne=F.normalize(emb(pv[(bi+1)%len(primes)].to(DEVICE)),dim=-1)
    cont=(1-(pe*ne).sum(dim=-1)).mean()
    zi=torch.randint(0,len(crit_z),(16,))
    ze=F.normalize(zemb(crit_z[zi].to(DEVICE)),dim=-1)
    zj=torch.randint(0,len(crit_z),(16,))
    ze2=F.normalize(zemb(crit_z[zj].to(DEVICE)),dim=-1)
    zcl=(1-(ze*ze2).sum(dim=-1)).mean()
    oi=torch.randint(0,len(off_z),(12,))
    oe=F.normalize(zemb(off_z[oi].to(DEVICE)),dim=-1)
    off_sim=(oe@ze.T).mean()
    off_sep=torch.relu(off_sim-0.1)
    loss=cont+0.5*zcl+10.0*off_sep
    loss.backward(); opt.step(); opt.zero_grad()
    if (step+1)%500==0:
        print(f"  Step {step+1}: loss={loss.item():.4f} cont={cont.item():.4f} zcl={zcl.item():.4f} offsep={off_sep.item():.4f}")

# TEH
print("\n[3] TEH detection...")
with torch.no_grad():
    az=F.normalize(zemb(crit_z.to(DEVICE)),dim=-1)
    U,S,_=torch.linalg.svd(az.T@az)
    cb=U[:,:K_CRIT]; Pf=torch.eye(D,device=DEVICE)-cb@cb.T
    total=S.sum(); cs=torch.cumsum(S,0)
    k90=(cs<0.9*total).sum().item()+1; k95=(cs<0.95*total).sum().item()+1
    
    off_r=[]; crit_a=[]
    for oc in off_z:
        e=F.normalize(zemb(oc.unsqueeze(0).to(DEVICE)),dim=-1).view(-1)
        a=torch.norm(Pf@e).item()/max(torch.norm(e).item(),1e-8)*100
        off_r.append(a)
    for i in range(min(20,len(crit_z))):
        e=F.normalize(zemb(crit_z[i].unsqueeze(0).to(DEVICE)),dim=-1).view(-1)
        a=torch.norm(Pf@e).item()/max(torch.norm(e).item(),1e-8)*100
        crit_a.append(a)
    
    det=sum(1 for a in off_r if a>12.0); fp=sum(1 for a in crit_a if a>12.0)
    mo=sum(off_r)/len(off_r); mc=sum(crit_a)/len(crit_a)
    print(f"  Detection: {det}/{len(off_r)} ({100*det/len(off_r):.0f}%)")
    print(f"  False+: {fp}/{len(crit_a)}")
    print(f"  Mean off: {mo:.1f}% | Mean crit: {mc:.1f}% | Ratio: {mo/max(mc,1e-8):.0f}x")
    print(f"  Subspace: k90={k90}, k95={k95}, total={len(S)}")
    print(f"  Top SVs: {[round(s.item(),1) for s in S[:8]]}")

r={"config":{"n_primes":len(primes),"n_zeros":len(zeta_zeros_imag),"n_off":len(off_z),"D":D,"K":K_CRIT},
   "subspace":{"k90":k90,"k95":k95},"teh":{"det":round(100*det/len(off_r)),"fp":fp,"mo":round(mo,1),"mc":round(mc,1),"ratio":round(mo/max(mc,1e-8))}}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
torch.save({"emb":emb.state_dict(),"zemb":zemb.state_dict(),"cb":cb.cpu()},f"{OUT}/model.pt")
print(f"\nSaved to {OUT}/")
