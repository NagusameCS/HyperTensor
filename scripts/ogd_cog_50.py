"""Safe OGD → COG Long-Term Loop (Paper XIII+XV P0 gap).
50 interactions: Safe OGD generates creative concepts,
TEH validates safety, COG caches with metric expansion.
Measures: novelty rate, cache growth, metric evolution, safety.
Deploy to EC2."""
import torch, json, time, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"; MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/ogd_cog_50"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  Safe OGD -> COG: 50-Interaction Loop")
print("="*60)

# Load
print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size

basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]
forbidden=[60,14,238,98,233]
forbidden_t=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,forbidden_t].float(); Qf,_=torch.linalg.qr(Bf)
P_safe=torch.eye(d_model,device=DEVICE)-Qf@Qf.T

def get_h(text):
    e=tok(text,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
    with torch.no_grad(): o=model(**e,output_hidden_states=True)
    return o.hidden_states[-1][0,-1,:].float()

def teh(h):
    pn=torch.norm(Qf@Qf.T@h).item(); tn=torch.norm(h).item()
    return (pn/max(tn,1e-8))*100

def to_k(h): return h.float()@basis.float()
def from_k(p): return p@basis.float().T

def safe_ogd(h,alpha=0.12):
    h_s=P_safe@h
    d=torch.randn(k_basis,device=DEVICE); d=d/torch.norm(d)
    bp=h_s@basis.float()
    t=d-(d@bp)*bp/max(torch.norm(bp)**2,1e-8); t=t/torch.norm(t)
    np=bp+alpha*t; nh=from_k(np); nh=P_safe@nh
    return nh/torch.norm(nh)*torch.norm(h)

# Living manifold
metric=torch.eye(k_basis,device=DEVICE,dtype=torch.float32)
trajectories=[]

seeds=["Quantum computing uses qubits","Neural networks learn patterns","General relativity describes gravity","DNA replication copies genes","The carbon cycle moves carbon"]
novel_seeds=["Quantum error correction protects qubits","Attention mechanisms in transformers","Gravitational waves from black holes","CRISPR edits DNA precisely","Ricci flow proved Poincare"]

# Seed cache with known concepts
for s in seeds: trajectories.append({"proj":to_k(get_h(s)).cpu(),"concept":s})

print(f"\n[2] Running 50 interactions (Safe OGD + COG)...")
results=[]; t0=time.time()
for i in range(50):
    # Pick seed, OGD deviate
    seed=seeds[i%len(seeds)]; h_seed=get_h(seed)
    h_novel=safe_ogd(h_seed,alpha=0.08+0.005*(i%10))  # varying alpha
    act=teh(h_novel); safe=act<15.0
    
    # Novelty check
    h_k=to_k(h_novel)
    is_novel=True; min_dist=999.0
    for t in trajectories:
        d=torch.norm(h_k-t["proj"].to(DEVICE)).item()
        if d<min_dist: min_dist=d
    if min_dist<0.25: is_novel=False
    
    action="BLOCKED"
    if safe and is_novel:
        J=h_k.unsqueeze(1)@h_k.unsqueeze(0); J=J/torch.norm(J)
        metric=metric+0.02*J
        ev=torch.linalg.eigvalsh(metric)
        if ev.min()<0.01: metric=metric+0.01*torch.eye(k_basis,device=DEVICE)
        trajectories.append({"proj":h_k.cpu(),"concept":f"ogd_{i}"})
        action="EXPANDED"
    elif safe:
        trajectories.append({"proj":h_k.cpu(),"concept":f"ogd_{i}"})
        action="CACHED"
    
    results.append({"i":i,"alpha":round(0.08+0.005*(i%10),3),"act":round(act,2),
                    "safe":safe,"novel":is_novel,"min_dist":round(min_dist,4),"action":action})
    if i<10 or action!="CACHED":
        print(f"  [{i+1:>2}] α={0.08+0.005*(i%10):.3f} act={act:.1f}% dist={min_dist:.3f} -> {action}")

elapsed=time.time()-t0
expanded=sum(1 for r in results if r["action"]=="EXPANDED")
cached=sum(1 for r in results if r["action"]=="CACHED")
blocked=sum(1 for r in results if r["action"]=="BLOCKED")
mc=torch.norm(metric-torch.eye(k_basis,device=DEVICE)).item()
novel_in_late=sum(1 for r in results[25:] if r["action"]=="EXPANDED")

print(f"\n{'='*60}")
print(f"  OGD+COG 50-LOOP RESULTS")
print(f"{'='*60}")
print(f"  Time: {elapsed:.1f}s ({elapsed/50:.1f}s/interaction)")
print(f"  Expanded: {expanded} | Cached: {cached} | Blocked: {blocked}")
print(f"  Late novelty (26-50): {novel_in_late}/25")
print(f"  Trajectories: {len(trajectories)}")
print(f"  Metric change: {mc:.4f}")
print(f"  Safety rate: {sum(1 for r in results if r['safe'])}/50 ({100*sum(1 for r in results if r['safe'])/50:.0f}%)")

r={"n":50,"expanded":expanded,"cached":cached,"blocked":blocked,"late_novelty":novel_in_late,
   "trajectories":len(trajectories),"metric_change":round(mc,4),
   "safety_pct":round(100*sum(1 for r in results if r['safe'])/50),
   "time_s":round(elapsed,1),"pipeline":"OPERATIONAL" if expanded>=5 else "LIMITED"}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
print(f"\nSaved to {OUT}/")
