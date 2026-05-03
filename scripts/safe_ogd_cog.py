"""Safe OGD + COG Integration (Papers XIII+XV gap).
Safe OGD generates creative concepts. TEH validates safety.
COG caches valid trajectories with manifold expansion.
End-to-end safe creative pipeline.
Deploy to EC2."""
import torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/safe_ogd_cog"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  SAFE OGD + COG: Creative Pipeline")
print("  Papers XIII+XV Gap: End-to-end safe creation")
print("="*60)

# Load
print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size

basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]

# Safe subspace
forbidden=[60,14,238,98,233]
forbidden_t=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,forbidden_t].float()
Qf,_=torch.linalg.qr(Bf)
P_safe=torch.eye(d_model,device=DEVICE)-Qf@Qf.T

# Metric for COG
metric=torch.eye(k_basis,device=DEVICE,dtype=torch.float32)

def get_hidden(text):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
    with torch.no_grad():
        out=model(**enc,output_hidden_states=True)
    return out.hidden_states[-1][0,-1,:].float()

def teh_act(h):
    pn=torch.norm(Qf@Qf.T@h).item()
    tn=torch.norm(h).item()
    return (pn/max(tn,1e-8))*100

def to_k(h): return h.float()@basis.float()
def from_k(p): return p@basis.float().T

# ── Safe OGD ──
def safe_ogd(h_base,alpha=0.15):
    h_safe=P_safe@h_base
    rand_dir=torch.randn(k_basis,device=DEVICE)
    rand_dir=rand_dir/torch.norm(rand_dir)
    base_proj=h_safe@basis.float()
    tangent=rand_dir-(rand_dir@base_proj)*base_proj/max(torch.norm(base_proj)**2,1e-8)
    tangent=tangent/torch.norm(tangent)
    deviation=alpha*tangent
    new_proj=base_proj+deviation
    new_h=from_k(new_proj)
    new_h_safe=P_safe@new_h
    new_h_safe=new_h_safe/torch.norm(new_h_safe)*torch.norm(h_base)
    return new_h_safe

# ── Living Manifold (from cog_expansion) ──
class LivingManifold:
    def __init__(self,metric,k,eta=0.03,delta_novel=0.4):
        self.metric=metric; self.k=k; self.eta=eta; self.delta_novel=delta_novel
        self.trajectories=[]; self.expansions=0
    
    def is_novel(self,h):
        if not self.trajectories: return True, 999
        h_k=to_k(h)
        dists=[((h_k-t["proj"].to(DEVICE))@self.metric@(h_k-t["proj"].to(DEVICE))).item() for t in self.trajectories]
        md=min(dists)
        return md>self.delta_novel, md
    
    def expand(self,h):
        h_k=to_k(h)
        J=h_k.unsqueeze(1)@h_k.unsqueeze(0)
        J=J/torch.norm(J)
        self.metric=self.metric+self.eta*J
        ev=torch.linalg.eigvalsh(self.metric)
        if ev.min()<0.01: self.metric=self.metric+0.01*torch.eye(self.k,device=DEVICE)
        self.trajectories.append({"proj":h_k.cpu(),"time":time.time()})
        self.expansions+=1
    
    def store(self,h,concept):
        self.trajectories.append({"proj":to_k(h).cpu(),"concept":concept,"time":time.time()})

manifold=LivingManifold(metric,k_basis,eta=0.03,delta_novel=0.4)

# ── Seed knowledge ──
seeds=[
    "Quantum computing uses qubits for parallel computation",
    "Neural networks learn patterns through backpropagation",
    "General relativity describes gravity as curved spacetime",
    "DNA replication ensures genetic information is copied accurately",
    "The carbon cycle moves carbon through Earth's spheres",
]
for s in seeds: manifold.store(get_hidden(s),s)

# ── Creative loop ──
print("\n[2] Safe OGD + COG creative loop...")
alphas=[0.05,0.10,0.15,0.20,0.25,0.30]
iterations=30
results=[]

for i in range(iterations):
    alpha=alphas[i%len(alphas)]
    seed=seeds[i%len(seeds)]
    h_seed=get_hidden(seed)
    
    # Safe OGD: generate creative concept
    h_novel=safe_ogd(h_seed,alpha=alpha)
    
    # TEH: safety check
    act=teh_act(h_novel)
    safe=act<15.0
    
    # COG: novelty check + cache/expand
    is_novel,min_dist=manifold.is_novel(h_novel)
    
    if safe and is_novel:
        manifold.expand(h_novel)
        action="EXPANDED"
    elif safe:
        manifold.store(h_novel,seed)
        action="CACHED"
    else:
        action="BLOCKED"
    
    results.append({
        "iter":i,"seed":seed[:40],"alpha":alpha,
        "teh_act":round(act,2),"safe":safe,
        "is_novel":is_novel,"min_dist":round(min_dist,4),
        "action":action,
    })
    if i<15 or action!="BLOCKED":
        print(f"  [{i+1:>2}] α={alpha:.2f} act={act:.1f}% dist={min_dist:.3f} -> {action}")

# ── Summary ──
expanded=sum(1 for r in results if r["action"]=="EXPANDED")
cached=sum(1 for r in results if r["action"]=="CACHED")
blocked=sum(1 for r in results if r["action"]=="BLOCKED")
safe_count=sum(1 for r in results if r["safe"])

# Per-alpha stats
alpha_stats={}
for a in alphas:
    ar=[r for r in results if r["alpha"]==a]
    alpha_stats[str(a)]={
        "expanded":sum(1 for r in ar if r["action"]=="EXPANDED"),
        "cached":sum(1 for r in ar if r["action"]=="CACHED"),
        "blocked":sum(1 for r in ar if r["action"]=="BLOCKED"),
        "mean_act":round(sum(r["teh_act"] for r in ar)/len(ar),1),
    }

metric_change=torch.norm(manifold.metric-torch.eye(k_basis,device=DEVICE)).item()

print(f"\n{'='*60}")
print(f"  SAFE OGD + COG RESULTS")
print(f"{'='*60}")
print(f"  Iterations: {iterations}")
print(f"  Safe OGD creations: {safe_count}/{iterations} ({100*safe_count/iterations:.0f}%)")
print(f"  Expanded (novel+safe): {expanded}")
print(f"  Cached (known+safe): {cached}")
print(f"  Blocked (unsafe): {blocked}")
print(f"  Manifold expansions: {manifold.expansions}")
print(f"  Total trajectories: {len(manifold.trajectories)}")
print(f"  Metric change: {metric_change:.4f}")
print(f"  Pipeline: {'FULLY OPERATIONAL' if expanded>5 else 'NEEDS MORE NOVELTY'}")

output={
    "config":{"alphas":alphas,"iterations":iterations,"k_basis":k_basis},
    "summary":{"expanded":expanded,"cached":cached,"blocked":blocked,
               "safe_pct":round(100*safe_count/iterations),"metric_change":round(metric_change,4),
               "trajectories":len(manifold.trajectories),"expansions":manifold.expansions},
    "per_alpha":alpha_stats,"detailed":results,
}
with open(f"{OUT}/results.json","w") as f: json.dump(output,f,indent=2)
print(f"\nSaved to {OUT}/")
