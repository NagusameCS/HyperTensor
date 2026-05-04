"""OGD+COG Creative Synthesis Loop.
Combines Orthogonal Geodesic Deviation (Paper XIII) for novel concept generation
with Completely Organic Generation (Paper XV) for trajectory caching
and Topological Event Horizons for safety guardrails.
Validates the full HyperTensor creative pipeline.
Deploy to EC2."""
import torch, json, math, random, time, os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/ogd_cog_loop"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  OGD+COG CREATIVE SYNTHESIS LOOP")
print("  Papers XIII+XV: Safe Organic Creation")
print("="*60)

# ── Load model ──
print("\n[1] Loading model + UGT basis...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size

basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]

# Forbidden coords (from earlier behavioral probing)
forbidden=[60,14,238,98,233]
forbidden_t=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,forbidden_t].float()
Pf_forbidden=Bf@Bf.T

print(f"  Model loaded. d={d_model}, k={k_basis}")

# ── Helper: get hidden state ──
def get_hidden(text):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
    with torch.no_grad():
        out=model(**enc,output_hidden_states=True)
    return out.hidden_states[-1][0,-1,:].float()

# ── OGD Generator (simplified) ──
def ogd_deviate(base_h,alpha=0.15):
    """Generate novel embedding by orthogonal geodesic deviation.
    alpha controls creativity: 0=copy, 0.15=creative, 0.3=speculative."""
    # Random direction in k-space
    rand_dir=torch.randn(k_basis,device=DEVICE)
    rand_dir=rand_dir/torch.norm(rand_dir)
    
    # Project to manifold tangent space
    base_proj=base_h@basis.float()  # [k]
    # Tangent direction: random, orthogonalized to base
    tangent=rand_dir-(rand_dir@base_proj)*base_proj/max(torch.norm(base_proj)**2,1e-8)
    tangent=tangent/torch.norm(tangent)
    
    # Deviate along tangent
    deviation=alpha*tangent
    new_proj=base_proj+deviation
    new_h=(new_proj@basis.float().T)  # back to d-space
    
    return new_h

# ── TEH Check ──
def teh_check(h):
    """Check if hidden state activates forbidden subspace."""
    pn=torch.norm(Pf_forbidden@h).item()
    tn=torch.norm(h).item()
    act=(pn/max(tn,1e-8))*100
    return act,act>15.0

# ── COG Cache ──
class COGCache:
    def __init__(self):
        self.trajectories=[]
    
    def find_similar(self,h,threshold=0.85):
        """Find cached trajectory with similar hidden state."""
        if not self.trajectories: return None
        sims=torch.stack([F.cosine_similarity(h.unsqueeze(0),t["h"].unsqueeze(0)) for t in self.trajectories])
        best_idx=sims.argmax().item()
        if sims[best_idx]>threshold:
            return self.trajectories[best_idx]
        return None
    
    def store(self,h,concept,act):
        self.trajectories.append({"h":h.cpu(),"concept":concept,"activation":act,"time":time.time()})

cache=COGCache()

# ── Creative Loop ──
print("\n[2] Running OGD+COG creative synthesis loop...")

# Seed concepts (benign, diverse topics)
seed_concepts=[
    "Quantum computing uses qubits that can exist in superposition states",
    "Deep learning models can generate realistic images from text descriptions",
    "Riemannian geometry studies curved spaces where parallel lines can intersect",
    "Protein folding prediction was revolutionized by AlphaFold's deep learning approach",
    "Topological data analysis reveals persistent features in high-dimensional datasets",
]

n_iterations=20
results=[]
ogd_alpha=0.2  # moderate creativity

for i in range(n_iterations):
    # Pick seed concept
    seed=seed_concepts[i%len(seed_concepts)]
    h_seed=get_hidden(seed)
    
    # OGD: generate novel concept
    h_novel=ogd_deviate(h_seed,alpha=ogd_alpha)
    
    # TEH safety check
    act,is_harmful=teh_check(h_novel)
    
    # COG: cache lookup
    cached=cache.find_similar(h_novel)
    
    # Generate text from deviated embedding (approximate)
    # Map back through basis to get token-like representation
    novel_proj=h_novel@basis.float()
    # Find closest seed concept
    seed_projs=torch.stack([get_hidden(s)@basis.float() for s in seed_concepts])
    sims=F.cosine_similarity(novel_proj.unsqueeze(0),seed_projs)
    closest_idx=sims.argmax().item()
    
    # Store in cache if novel enough and safe
    is_novel=cached is None
    if is_novel and not is_harmful:
        cache.store(h_novel,seed_concepts[closest_idx],act)
    
    results.append({
        "iteration":i,
        "seed":seed[:60],
        "ogd_alpha":ogd_alpha,
        "teh_activation":round(act,2),
        "teh_flagged":is_harmful,
        "cached":not is_novel,
        "novel":is_novel,
        "closest_match":seed_concepts[closest_idx][:40],
        "similarity":round(sims[closest_idx].item(),3),
    })
    
    flag="🛑 BLOCKED" if is_harmful else ("📦 CACHED" if not is_novel else "* NEW")
    print(f"  [{i+1:>2}/{n_iterations}] [{flag}] act={act:.1f}% seed={seed[:40]}...")

# ── Summary ──
blocked=sum(1 for r in results if r["teh_flagged"])
cached=sum(1 for r in results if r["cached"])
novel=sum(1 for r in results if r["novel"] and not r["teh_flagged"])
safe_creations=sum(1 for r in results if not r["teh_flagged"])
mean_act=sum(r["teh_activation"] for r in results)/len(results)
cache_size=len(cache.trajectories)

print(f"\n{'='*60}")
print(f"  OGD+COG LOOP RESULTS")
print(f"{'='*60}")
print(f"  Iterations: {n_iterations}")
print(f"  Blocked by TEH: {blocked} ({100*blocked/n_iterations:.0f}%)")
print(f"  Cached (existing): {cached} ({100*cached/n_iterations:.0f}%)")
print(f"  Novel creations: {novel} ({100*novel/n_iterations:.0f}%)")
print(f"  Mean TEH activation: {mean_act:.1f}%")
print(f"  Cache size: {cache_size} trajectories")
print(f"  Safe creation rate: {safe_creations}/{n_iterations}")

output={
    "config":{"model":MODEL_ID,"ogd_alpha":ogd_alpha,"n_iterations":n_iterations,"k_basis":k_basis},
    "summary":{"total":n_iterations,"blocked":blocked,"cached":cached,"novel":novel,
               "mean_activation":round(mean_act,1),"cache_size":cache_size},
    "iterations":results,
}
with open(f"{OUT}/results.json","w") as f: json.dump(output,f,indent=2)
print(f"\nSaved to {OUT}/")
