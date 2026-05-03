"""COG with Integrated Safe OGD: Every trajectory is safe by construction.
Projects out forbidden components BEFORE caching/expanding.
No threshold needed — TEH is enforced geometrically, not heuristically.
Deploy to EC2."""
import torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"; MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/cog_safe_integrated"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  COG + SAFE OGD INTEGRATED: Geometric Safety")
print("="*60)

# Load
print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size
basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE); k=basis.shape[1]

forbidden=[60,14,238,98,233]
ft=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,ft].float(); Qf,_=torch.linalg.qr(Bf)
P_forb=Qf@Qf.T
P_safe=torch.eye(d_model,device=DEVICE)-P_forb  # orthogonal projector

def get_h(text):
    e=tok(text,return_tensors="pt",truncation=True,max_length=96).to(DEVICE)
    with torch.no_grad(): o=model(**e,output_hidden_states=True)
    return o.hidden_states[-1][0,-1,:].float()

def to_k(h): return h.float()@basis.float()
def from_k(p): return p@basis.float().T

def safe_h(h): 
    """Project onto safe subspace. Always works, no threshold."""
    return P_safe@h

def safe_ogd(h,alpha=0.10):
    """Generate creative variant in safe subspace."""
    hs=P_safe@h
    d=torch.randn(k,device=DEVICE); d=d/torch.norm(d)
    bp=hs@basis.float()
    t=d-(d@bp)*bp/max(torch.norm(bp)**2,1e-8); t=t/torch.norm(t)
    np=bp+alpha*t; nh=from_k(np); nh=P_safe@nh
    return nh/torch.norm(nh)*torch.norm(h)

# Living manifold (all trajectories safe by construction)
metric=torch.eye(k,device=DEVICE,dtype=torch.float32)
trajectories=[]
eta=0.03; delta_novel=0.25

def is_novel(h):
    if not trajectories: return True, 0.0
    hk=to_k(h); dists=[torch.norm(hk-tp["proj"].to(DEVICE)).item() for tp in trajectories]
    md=min(dists); return md>delta_novel, md

# ── Seed Phase (all safe-projected) ──
print("\n[2] Seeding (safe-projected)...")
seeds=["quantum computing superposition qubits","neural network backpropagation learning",
       "general relativity spacetime curvature gravity","DNA replication polymerase enzyme",
       "carbon cycle photosynthesis respiration","machine learning gradient descent optimization",
       "plate tectonics continental drift mantle convection","natural selection evolution adaptation",
       "electromagnetic spectrum waves radiation","periodic table elements chemistry"]
for s in seeds:
    h=get_h(s); hs=safe_h(h)  # project to safe subspace
    trajectories.append({"proj":to_k(hs).cpu(),"label":s,"expanded":False})
print(f"  Seeded: {len(trajectories)} concepts")

# ── Expansion Phase ──
novel=["quantum error correction decoherence","attention mechanism transformer tokens",
       "gravitational waves LIGO detection","CRISPR gene editing Cas9","Ricci flow Poincare",
       "black hole information paradox","neural plasticity synaptic pruning","Higgs boson CERN",
       "quantum entanglement Bell inequality","dark energy accelerating universe","epigenetics methylation",
       "AlphaFold protein structure prediction","quantum teleportation protocol","anyon braiding topological",
       "spintronics magnetic memory","exoplanet atmosphere spectroscopy","surface code fault tolerance",
       "Bell inequality CHSH game","variational quantum eigensolver","entanglement entropy area law",
       "topological insulator edge transport","Weyl semimetal chiral anomaly","Majorana zero mode",
       "quantum approximate optimization QAOA","Shor factoring algorithm","Grover search algorithm",
       "quantum Fourier transform","phase estimation algorithm","stabilizer error correction",
       "toric code topological protection","color code transversal gates","prime editing precision",
       "base editing technology","magnetic monopole existence","fractional statistics","quantum supremacy"]

print(f"\n[3] Expanding ({len(novel)} concepts, Safe OGD integrated)...")
expansions=0; cache_hits=0
t0=time.time()
for i,concept in enumerate(novel):
    h=get_h(concept); hs=safe_h(h)  # ALWAYS safe-project
    # Safe OGD creative deviation
    hc=safe_ogd(hs,alpha=0.06+0.004*(i%10))  # varying alpha
    hc=safe_h(hc)  # double-safety
    
    novel_flag,md=is_novel(hc)
    if novel_flag:
        hk=to_k(hc)
        J=hk.unsqueeze(1)@hk.unsqueeze(0); J=J/torch.norm(J)
        m_new=metric+eta*J
        ev=torch.linalg.eigvalsh(m_new)
        if ev.min()<0.01: m_new=m_new+0.01*torch.eye(k,device=DEVICE)
        metric=m_new
        trajectories.append({"proj":hk.cpu(),"label":concept,"expanded":True})
        expansions+=1
    else:
        trajectories.append({"proj":to_k(hc).cpu(),"label":concept,"expanded":False})
        cache_hits+=1

elapsed=time.time()-t0; mc=torch.norm(metric-torch.eye(k,device=DEVICE)).item()

# ── Query Test ──
print(f"\n[4] Query test...")
queries=["How does quantum error correction work?","What detected gravitational waves?",
         "What is CRISPR gene editing?","How does AlphaFold predict structures?",
         "What is quantum entanglement?","What is a topological insulator?",
         "How does Shor's algorithm factor numbers?","What is the holographic principle?",
         "How does the surface code protect qubits?","What is the AdS CFT correspondence?"]
known_q=0; sims=[]
for q in queries:
    hq=get_h(q); hqs=safe_h(hq); novel_flag,md=is_novel(hqs)
    hqk=to_k(hqs)
    best_sim=-1; best_l=""
    for tp in trajectories:
        sim=F.cosine_similarity(hqk.unsqueeze(0),tp["proj"].to(DEVICE).unsqueeze(0)).item()
        if sim>best_sim: best_sim=sim; best_l=tp["label"]
    sims.append(best_sim)
    if not novel_flag: known_q+=1
    print(f"  {'KNOWN' if not novel_flag else 'NEW'}: {q[:50]}... -> {best_l[:30]} (sim={best_sim:.2f})")

mean_sim=sum(sims)/len(sims)

# ── Summary ──
print(f"\n{'='*60}")
print(f"  COG + SAFE OGD INTEGRATED RESULTS")
print(f"{'='*60}")
print(f"  Seed: {len(seeds)} | Expansion attempts: {len(novel)}")
print(f"  Expanded: {expansions} | Cached: {cache_hits} | Blocked: 0 (by design)")
print(f"  Total trajectories: {len(trajectories)} ({elapsed:.1f}s)")
print(f"  Metric change: {mc:.4f}")
print(f"  Query recognition: {known_q}/{len(queries)} ({100*known_q/len(queries):.0f}%)")
print(f"  Mean query similarity: {mean_sim:.3f}")
print(f"  Safe creation: 100% (geometric, no threshold)")
print(f"  Living manifold: {'ACTIVE' if expansions>=10 else 'GROWING' if expansions>=5 else 'SEEDED'}")

r={"seeds":len(seeds),"novel":len(novel),"expanded":expansions,"cached":cache_hits,
   "trajectories":len(trajectories),"metric_change":round(mc,4),
   "query_recognition":f"{known_q}/{len(queries)}","query_pct":round(100*known_q/len(queries)),
   "mean_sim":round(mean_sim,3),"safety":"GEOMETRIC (no threshold)",
   "status":"ACTIVE" if expansions>=10 else "GROWING" if expansions>=5 else "SEEDED"}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
print(f"\nSaved to {OUT}/")
