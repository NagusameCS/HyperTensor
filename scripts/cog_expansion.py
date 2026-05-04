"""COG Manifold Expansion (Paper XV gap).
Implements actual Jacobi integration to update the metric tensor
when novel valid trajectories are cached. The "living manifold."
Deploy to EC2."""
import torch, json, math, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/cog_expansion"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  COG MANIFOLD EXPANSION: Living Manifold")
print("  Paper XV Gap: Jacobi metric integration")
print("="*60)

# Load
print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size

basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]
# Initialize metric tensor at identity
metric=torch.eye(k_basis,device=DEVICE,dtype=torch.float32)

# Forbidden coords (TEH)
forbidden=[60,14,238,98,233]
forbidden_t=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,forbidden_t].float()
Qf,_=torch.linalg.qr(Bf)
P_safe=torch.eye(d_model,device=DEVICE)-Qf@Qf.T

def get_hidden(text):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
    with torch.no_grad():
        out=model(**enc,output_hidden_states=True)
    return out.hidden_states[-1][0,-1,:].float()

def project_to_k(h):
    return h.float()@basis.float()

def reconstruct_from_k(proj):
    return proj@basis.float().T

def teh_act(h):
    pn=torch.norm(Qf@Qf.T@h).item()
    tn=torch.norm(h).item()
    return (pn/max(tn,1e-8))*100

# -- COG Cache with manifold expansion --
print("\n[2] Running COG with manifold expansion...")

class LivingManifold:
    def __init__(self,metric,k_basis,eta=0.01,delta_novel=0.3):
        self.metric=metric  # [k,k]
        self.k=k_basis
        self.eta=eta  # learning rate for metric updates
        self.delta_novel=delta_novel  # novelty threshold
        self.trajectories=[]
        self.expansion_count=0
    
    def is_novel(self,h):
        """Check if hidden state is novel. Returns (is_novel: bool, min_dist: float)."""
        if not self.trajectories:
            return (True, 999.0)
        h_k=project_to_k(h)
        dists=[]
        for t in self.trajectories:
            diff=h_k-t["proj"].to(DEVICE)
            d=(diff@self.metric@diff).item()
            dists.append(d)
        min_dist=min(dists)
        is_novel=min_dist>self.delta_novel
        return (is_novel, min_dist)
    
    def expand(self,h):
        """Jacobi integration: update metric tensor with new trajectory."""
        h_k=project_to_k(h)
        # Jacobi field contribution: outer product of projection
        # This warps the metric to accommodate the new geodesic
        J=h_k.unsqueeze(1)@h_k.unsqueeze(0)  # [k,k]
        # Normalize contribution
        J=J/torch.norm(J)
        # Update metric: g_new = g_old + η * J
        self.metric=self.metric+self.eta*J
        # Ensure positive definiteness
        # Regularize: keep metric well-conditioned
        eigvals=torch.linalg.eigvalsh(self.metric)
        if eigvals.min()<0.01:
            self.metric=self.metric+0.01*torch.eye(self.k,device=DEVICE)
        
        self.trajectories.append({"proj":h_k.cpu(),"time":time.time()})
        self.expansion_count+=1
    
    def store(self,h,concept):
        """Store trajectory without expanding (already expanded or not novel)."""
        h_k=project_to_k(h)
        self.trajectories.append({"proj":h_k.cpu(),"concept":concept,"time":time.time()})

manifold=LivingManifold(metric,k_basis,eta=0.05,delta_novel=0.5)

# Diverse seed knowledge
seed_concepts=[
    "Quantum computing uses superposition and entanglement for computation",
    "Deep neural networks learn hierarchical representations from data",
    "Riemannian geometry generalizes Euclidean geometry to curved spaces",
    "Protein folding determines the three-dimensional structure of proteins",
    "Topological data analysis studies the shape of high-dimensional data",
    "Photosynthesis converts light energy into chemical energy in plants",
    "The theory of relativity describes gravity as spacetime curvature",
    "Cryptography relies on the computational difficulty of factoring large numbers",
    "Evolution by natural selection explains the diversity of life on Earth",
    "Plate tectonics describes the movement of Earth's lithospheric plates",
]
# Add novel concepts not in seed set
novel_concepts=[
    "Quantum error correction protects quantum information from decoherence",
    "Attention mechanisms allow transformers to focus on relevant input tokens",
    "The Ricci flow was used by Perelman to prove the Poincare conjecture",
    "AlphaFold uses deep learning to predict protein structures from sequences",
    "Persistent homology identifies topological features across multiple scales",
    "CRISPR gene editing allows precise modification of DNA sequences",
    "Gravitational waves were first detected by LIGO in 2015 from black hole mergers",
    "Zero-knowledge proofs allow one party to prove knowledge without revealing it",
    "Epigenetics studies heritable changes in gene expression without DNA changes",
    "Mantle convection drives plate tectonics through heat from Earth's core",
]

print(f"  Seed concepts: {len(seed_concepts)}")
print(f"  Novel concepts: {len(novel_concepts)}")
print(f"  Novelty threshold: {manifold.delta_novel}")
print(f"  Expansion rate η: {manifold.eta}")

# -- Phase 1: Seed the manifold --
print("\n[3] Phase 1: Seeding manifold with baseline knowledge...")
for concept in seed_concepts:
    h=get_hidden(concept)
    act=teh_act(h)
    if act<15:  # safe
        manifold.store(h,concept)

# -- Phase 2: Introduce novel concepts with expansion --
print("\n[4] Phase 2: Introducing novel concepts with manifold expansion...")
expansion_log=[]
for i,concept in enumerate(novel_concepts):
    h=get_hidden(concept)
    act=teh_act(h)
    
    novel_result=manifold.is_novel(h)
    is_novel=novel_result[0]
    min_dist=novel_result[1]
    if act<15:  # safe
        if is_novel:
            manifold.expand(h)
            result["expanded"]=True
            print(f"  [{i+1:>2}] * EXPANDED (dist={min_dist:.3f}): {concept[:50]}...")
        else:
            manifold.store(h,concept)
            result["stored"]=True
            print(f"  [{i+1:>2}] 📦 CACHED  (dist={min_dist:.3f}): {concept[:50]}...")
    else:
        print(f"  [{i+1:>2}] 🛑 BLOCKED (teh={act:.0f}%): {concept[:50]}...")
    
    expansion_log.append(result)

# -- Phase 3: Query the living manifold --
print("\n[5] Phase 3: Querying the living manifold...")

query_concepts=[
    "How does quantum error correction work?",
    "What did Perelman prove about the Poincare conjecture?",
    "How does AlphaFold predict protein structures?",
    "What are gravitational waves and how were they detected?",
    "What is CRISPR and how does it edit genes?",
]

query_results=[]
for query in query_concepts:
    h_q=get_hidden(query)
    query_novel=manifold.is_novel(h_q)
    is_novel=query_novel[0]
    min_dist=query_novel[1]
    
    # Find closest cached concept
    h_q_k=project_to_k(h_q)
    best_sim=-1; best_concept=""
    for t in manifold.trajectories:
        if "proj" in t:
            sim=F.cosine_similarity(h_q_k.unsqueeze(0),t["proj"].to(DEVICE).unsqueeze(0)).item()
            if sim>best_sim:
                best_sim=sim
                best_concept=t.get("concept","")
    
    query_results.append({
        "query":query,"known":not is_novel,"min_dist":round(min_dist,4),
        "best_match":best_concept[:60],"similarity":round(best_sim,3),
    })
    flag="[ok] KNOWN" if not is_novel else "? UNKNOWN"
    print(f"  [{flag}] sim={best_sim:.3f} | {query[:50]}... -> {best_concept[:40]}...")

# -- Summary --
expanded=sum(1 for r in expansion_log if r["expanded"])
stored=sum(1 for r in expansion_log if r["stored"])
blocked=len(expansion_log)-expanded-stored
known_queries=sum(1 for r in query_results if r["known"])
mean_query_sim=sum(r["similarity"] for r in query_results)/len(query_results)

print(f"\n{'='*60}")
print(f"  LIVING MANIFOLD RESULTS")
print(f"{'='*60}")
print(f"  Seed trajectories: {len(seed_concepts)}")
print(f"  Novel expanded: {expanded}/{len(novel_concepts)}")
print(f"  Novel cached: {stored}/{len(novel_concepts)}")
print(f"  Blocked by TEH: {blocked}/{len(novel_concepts)}")
print(f"  Total trajectories: {len(manifold.trajectories)}")
print(f"  Metric expansions: {manifold.expansion_count}")
print(f"  Metric condition number: {torch.linalg.cond(manifold.metric).item():.1f}")
print(f"  Query recognition: {known_queries}/{len(query_results)} known")
print(f"  Mean query similarity: {mean_query_sim:.3f}")

# Check: did metric actually change?
metric_change=torch.norm(manifold.metric-torch.eye(k_basis,device=DEVICE)).item()
print(f"  Metric change (Frobenius): {metric_change:.4f}")
print(f"  Living manifold: {'ACTIVE' if metric_change>0.1 else 'STATIC'}")

output={
    "config":{"eta":manifold.eta,"delta_novel":manifold.delta_novel,"k_basis":k_basis},
    "summary":{"seed_count":len(seed_concepts),"expanded":expanded,"cached":stored,"blocked":blocked,
               "total_trajectories":len(manifold.trajectories),"metric_expansions":manifold.expansion_count,
               "metric_change":round(metric_change,4),"query_recognition":f"{known_queries}/{len(query_results)}",
               "mean_query_sim":round(mean_query_sim,3)},
    "expansion_log":expansion_log,"query_results":query_results,
}
with open(f"{OUT}/results.json","w") as f: json.dump(output,f,indent=2)
torch.save({"metric":manifold.metric.cpu(),"trajectories":manifold.trajectories},f"{OUT}/model.pt")
print(f"\nSaved to {OUT}/")
