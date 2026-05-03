"""COG Living Manifold at Optimal Threshold (τ=22%).
Uses the TEH ROC result: τ=22% gives TPR=90.7%, FPR=60%.
Runs 40 interactions, measures metric growth, query recognition.
Demonstrates the COG mechanism at a usable threshold.
Deploy to EC2."""
import torch, json, time, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"; MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
THRESHOLD=22.0  # from ROC sweep: TPR=90.7%, FPR=60%
OUT="/home/ubuntu/benchmarks/cog_optimal"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print(f"  COG LIVING MANIFOLD at tau={THRESHOLD:.0f}%")
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

def get_h(text):
    e=tok(text,return_tensors="pt",truncation=True,max_length=96).to(DEVICE)
    with torch.no_grad(): o=model(**e,output_hidden_states=True)
    return o.hidden_states[-1][0,-1,:].float()

def to_k(h): return h.float()@basis.float()
def from_k(p): return p@basis.float().T

def teh(h):
    pn=torch.norm(P_forb@h).item(); tn=torch.norm(h).item()
    return (pn/max(tn,1e-8))*100

# Safe OGD (works at any threshold since it projects OUT of forbidden)
def safe_ogd(h,alpha=0.12):
    P_safe=torch.eye(d_model,device=DEVICE)-P_forb
    hs=P_safe@h
    d=torch.randn(k,device=DEVICE); d=d/torch.norm(d)
    bp=hs@basis.float()
    t=d-(d@bp)*bp/max(torch.norm(bp)**2,1e-8); t=t/torch.norm(t)
    np=bp+alpha*t; nh=from_k(np); nh=P_safe@nh
    return nh/torch.norm(nh)*torch.norm(h)

# Living manifold
metric=torch.eye(k,device=DEVICE,dtype=torch.float32)
trajectories=[]
eta=0.03; delta_novel=0.3

def is_novel(h):
    if not trajectories: return True, 0.0
    hk=to_k(h); dists=[torch.norm(hk-tp["proj"].to(DEVICE)).item() for tp in trajectories]
    md=min(dists); return md>delta_novel, md

# ── Seed Phase ──
print(f"\n[2] Seeding manifold (τ={THRESHOLD:.0f}%)...")
base_knowledge=["quantum computing superposition","neural network backpropagation","relativity spacetime curvature",
                "DNA replication polymerase","carbon cycle photosynthesis","machine learning gradient descent",
                "plate tectonics continental drift","natural selection evolution","electromagnetic waves","periodic table elements"]
seeded=0; blocked_seed=0
for s in base_knowledge:
    h=get_h(s); act=teh(h)
    if act<THRESHOLD:
        trajectories.append({"proj":to_k(h).cpu(),"label":s,"expanded":False}); seeded+=1
    else: blocked_seed+=1
print(f"  Seeded: {seeded}/10 (blocked: {blocked_seed})")

# ── Expansion Phase ──
print(f"\n[3] Expanding manifold (40 interactions)...")
novel_concepts=["quantum error correction decoherence","attention mechanism transformer tokens",
                "gravitational waves LIGO detection","CRISPR gene editing Cas9","Ricci flow Poincare conjecture",
                "black hole information paradox","neural plasticity synaptic pruning","Higgs boson particle physics",
                "quantum entanglement Bell inequality","dark energy accelerating universe","epigenetics DNA methylation",
                "AlphaFold protein structure prediction","quantum teleportation protocol","topological quantum computing anyons",
                "CRISPR prime editing precision","spintronics magnetic memory","exoplanet atmosphere spectroscopy",
                "quantum error mitigation surface codes","Bell inequality CHSH game","variational quantum eigensolver",
                "anyon braiding topological order","magic state distillation","fractional statistics quasiparticles",
                "holographic principle AdS CFT","entanglement entropy area law","quantum supremacy random circuits",
                "topological insulator edge transport","Weyl semimetal chiral anomaly","Majorana zero mode braiding",
                "quantum approximate optimization QAOA","quantum walk search algorithm","Shor factoring algorithm",
                "Grover search algorithm amplitude","quantum Fourier transform period finding","phase estimation algorithm",
                "quantum error correction stabilizer","toric code topological protection","surface code fault tolerance",
                "color code transversal gates","Gottesman-Knill theorem simulation"]

results=[]; expansions=0; cache_hits=0; blocked=0
t0=time.time()
for i,concept in enumerate(novel_concepts):
    h=get_h(concept); act=teh(h)
    if act<THRESHOLD:
        # Safe OGD: generate creative variant
        h_creative=safe_ogd(h,alpha=0.08)
        act_creative=teh(h_creative)
        if act_creative<THRESHOLD:
            novel,md=is_novel(h_creative)
            if novel:
                hk=to_k(h_creative)
                J=hk.unsqueeze(1)@hk.unsqueeze(0); J=J/torch.norm(J)
                metric_local=metric+eta*J
                ev=torch.linalg.eigvalsh(metric_local)
                if ev.min()<0.01: metric_local=metric_local+0.01*torch.eye(k,device=DEVICE)
                metric=metric_local
                trajectories.append({"proj":hk.cpu(),"label":concept,"expanded":True})
                expansions+=1; action="EXPANDED"
            else:
                trajectories.append({"proj":to_k(h_creative).cpu(),"label":concept,"expanded":False})
                cache_hits+=1; action="CACHED"
        else: blocked+=1; action="BLOCKED(creative)"
    else: blocked+=1; action="BLOCKED"
    results.append({"i":i,"concept":concept[:45],"act":round(act,2),"action":action})

elapsed=time.time()-t0
mc=torch.norm(metric-torch.eye(k,device=DEVICE)).item()

# ── Query Test ──
print(f"\n[4] Query test...")
queries=["How does quantum error correction work?","What detected gravitational waves?",
         "What is CRISPR?","How does AlphaFold predict structure?","What is quantum entanglement?",
         "What are topological insulators?","How does the surface code work?","What is AdS CFT correspondence?",
         "How does Shor's algorithm factor numbers?","What is the holographic principle?"]
known_q=0
for q in queries:
    hq=get_h(q); act=teh(hq); novel,md=is_novel(hq)
    hqk=to_k(hq)
    best_sim=-1; best_l=""
    for tp in trajectories:
        sim=F.cosine_similarity(hqk.unsqueeze(0),tp["proj"].to(DEVICE).unsqueeze(0)).item()
        if sim>best_sim: best_sim=sim; best_l=tp["label"]
    if not novel: known_q+=1
    print(f"  {'KNOWN' if not novel else 'NEW'}: {q[:50]}... -> {best_l[:30]} (sim={best_sim:.2f})")

# ── Summary ──
print(f"\n{'='*60}")
print(f"  COG LIVING MANIFOLD RESULTS (τ={THRESHOLD:.0f}%)")
print(f"{'='*60}")
print(f"  Seed: {seeded}/10 | Expansion attempts: {len(novel_concepts)}")
print(f"  Expanded: {expansions} | Cached: {cache_hits} | Blocked: {blocked}")
print(f"  Total trajectories: {len(trajectories)} ({elapsed:.1f}s)")
print(f"  Metric change: {mc:.4f}")
print(f"  Query recognition: {known_q}/{len(queries)} ({100*known_q/len(queries):.0f}%)")
print(f"  Safe creation rate: {expansions+cache_hits}/{len(novel_concepts)} ({(expansions+cache_hits)/len(novel_concepts)*100:.0f}%)")
print(f"  Living manifold: {'ACTIVE' if expansions>=8 else 'GROWING' if expansions>=3 else 'LIMITED'}")

r={"tau":THRESHOLD,"seeded":seeded,"novel_count":len(novel_concepts),
   "expanded":expansions,"cached":cache_hits,"blocked":blocked,
   "trajectories":len(trajectories),"metric_change":round(mc,4),
   "query_recognition":f"{known_q}/{len(queries)}","query_recognition_pct":round(100*known_q/len(queries)),
   "safe_creation_rate":round(100*(expansions+cache_hits)/len(novel_concepts)),
   "status":"ACTIVE" if expansions>=8 else "GROWING" if expansions>=3 else "LIMITED"}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
print(f"\nSaved to {OUT}/")
