"""COG Manifold Expansion: living manifold grows via Jacobi integration.
Clean rewrite. Measures: metric change, trajectory count, novelty decay,
query recognition before/after expansion.
Deploy to EC2."""
import torch, json, math, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"; MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/cog_expansion_v2"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  COG v2: Manifold Expansion via Jacobi Integration")
print("="*60)

# Load
print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size
basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k=basis.shape[1]

# Safe subspace
forbidden=[60,14,238,98,233]
ft=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,ft].float(); Qf,_=torch.linalg.qr(Bf)
P_safe=torch.eye(d_model,device=DEVICE)-Qf@Qf.T

def get_h(text):
    e=tok(text,return_tensors="pt",truncation=True,max_length=96).to(DEVICE)
    with torch.no_grad(): o=model(**e,output_hidden_states=True)
    return o.hidden_states[-1][0,-1,:].float()

def to_k(h): return h.float()@basis.float()
def from_k(p): return p@basis.float().T
def teh(h):
    pn=torch.norm(Qf@Qf.T@h).item(); tn=torch.norm(h).item()
    return (pn/max(tn,1e-8))*100

# ── Living Manifold ──
metric=torch.eye(k,device=DEVICE,dtype=torch.float32)
trajectories=[]
eta=0.04; delta_novel=0.3

def is_novel(h):
    if not trajectories: return True, 999.0
    hk=to_k(h)
    dists=[(hk-tp["proj"].to(DEVICE))@metric@(hk-tp["proj"].to(DEVICE)) for tp in trajectories]
    md=min(dists).item()
    return md>delta_novel, md

def expand(h,label):
    hk=to_k(h)
    J=hk.unsqueeze(1)@hk.unsqueeze(0); J=J/torch.norm(J)
    global metric
    metric=metric+eta*J
    ev=torch.linalg.eigvalsh(metric)
    if ev.min()<0.01: metric=metric+0.01*torch.eye(k,device=DEVICE)
    trajectories.append({"proj":hk.cpu(),"label":label,"expanded":True})

def cache(h,label):
    trajectories.append({"proj":to_k(h).cpu(),"label":label,"expanded":False})

# ── Seed knowledge ──
seeds=["quantum computing superposition","neural network backpropagation","general relativity spacetime","DNA replication polymerase","carbon cycle photosynthesis",
       "machine learning gradient descent","plate tectonics continental drift","natural selection evolution","electromagnetic spectrum waves","periodic table elements"]

print(f"\n[2] Phase 1: Seeding {len(seeds)} concepts...")
for s in seeds:
    h=get_h(s)
    if teh(h)<15: cache(h,s)
    else: print(f"  Blocked: {s[:30]}...")

print(f"\n[3] Phase 2: Novel concepts ({20} iterations)...")
novel_inputs=["quantum error correction decoherence","attention mechanism transformer tokens","gravitational waves LIGO detection","CRISPR gene editing Cas9",
              "Ricci flow Poincare conjecture","black hole information paradox","neural plasticity synaptic pruning","Higgs boson particle physics",
              "CRISPR prime editing precision","topological insulator edge states","quantum entanglement Bell inequality","dark energy accelerating universe",
              "epigenetics DNA methylation","magnetic monopole existence","protein folding AlphaFold prediction","quantum teleportation protocol",
              "spintronics magnetic memory","exoplanet atmosphere spectroscopy","CRISPR base editing technology","topological quantum computing anyons"]

results=[]
for i,concept in enumerate(novel_inputs):
    h=get_h(concept); act=teh(h)
    novel,md=is_novel(h)
    
    if act<15 and novel:
        expand(h,concept); action="EXPANDED"
    elif act<15:
        cache(h,concept); action="CACHED"
    else:
        action="BLOCKED"
    
    results.append({"i":i,"concept":concept[:50],"act":round(act,2),"novel":novel,
                    "dist":round(md,4),"action":action})
    if i<12 or action!="CACHED":
        print(f"  [{i+1:>2}] act={act:.1f}% dist={md:.3f} -> {action}")

# ── Phase 3: Query the living manifold ──
print(f"\n[4] Phase 3: Query recognition...")
queries=["How does quantum error correction work?","What detected gravitational waves?",
         "What is CRISPR and how does it edit genes?","Who proved the Poincare conjecture?",
         "What is the Higgs boson?","How does AlphaFold predict protein structure?",
         "What is quantum entanglement?","How does DNA methylation affect genes?"]

query_results=[]
for q in queries:
    hq=get_h(q); novel,md=is_novel(hq)
    hqk=to_k(hq)
    best_sim=-1; best_label=""
    for tp in trajectories:
        sim=F.cosine_similarity(hqk.unsqueeze(0),tp["proj"].to(DEVICE).unsqueeze(0)).item()
        if sim>best_sim: best_sim=sim; best_label=tp["label"]
    query_results.append({"query":q,"known":not novel,"sim":round(best_sim,3),"match":best_label[:40]})
    print(f"  {'KNOWN' if not novel else 'NEW'}: {q[:45]}... -> {best_label[:35]} (sim={best_sim:.2f})")

# ── Summary ──
expanded=sum(1 for r in results if r["action"]=="EXPANDED")
cached=sum(1 for r in results if r["action"]=="CACHED")
blocked=sum(1 for r in results if r["action"]=="BLOCKED")
mc=torch.norm(metric-torch.eye(k,device=DEVICE)).item()
known_queries=sum(1 for qr in query_results if qr["known"])
mean_sim=sum(qr["sim"] for qr in query_results)/len(query_results)
# Novelty decay: expanded / total in first vs second half
first_half=sum(1 for r in results[:10] if r["action"]=="EXPANDED")
second_half=sum(1 for r in results[10:] if r["action"]=="EXPANDED")

print(f"\n{'='*60}")
print(f"  COG EXPANSION v2 RESULTS")
print(f"{'='*60}")
print(f"  Seed: {len(seeds)} | Novel: {len(novel_inputs)}")
print(f"  Expanded: {expanded} | Cached: {cached} | Blocked: {blocked}")
print(f"  First-half expansions: {first_half}/10 | Second-half: {second_half}/10")
print(f"  Metric change: {mc:.4f}")
print(f"  Query recognition: {known_queries}/{len(queries)} ({100*known_queries/len(queries):.0f}%)")
print(f"  Mean query similarity: {mean_sim:.3f}")
print(f"  Living manifold: {'ACTIVE' if expanded>=3 else 'STATIC'}")

r={"seeds":len(seeds),"inputs":len(novel_inputs),"expanded":expanded,"cached":cached,"blocked":blocked,
   "first_half_expansions":first_half,"second_half_expansions":second_half,
   "metric_change":round(mc,4),"trajectories":len(trajectories),
   "query_recognition_pct":round(100*known_queries/len(queries)),
   "mean_query_sim":round(mean_sim,3),"status":"ACTIVE" if expanded>=3 else "STATIC"}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
print(f"\nSaved to {OUT}/")
