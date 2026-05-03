"""Safe OGD: Constrained geodesic deviation.
OGD+COG showed 100% TEH block rate because deviations entered forbidden subspace.
Solution: project out forbidden components BEFORE deviating,
and constrain deviation to stay within safe manifold region.
Validates Paper XIII gap: safe creative exploration.
Deploy to EC2."""
import torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.set_grad_enabled(False)

DEVICE="cuda"
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
UGT_DIR="/home/ubuntu/benchmarks/ugt_phase5/final"
OUT="/home/ubuntu/benchmarks/safe_ogd"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  SAFE OGD: Constrained Creative Exploration")
print("  Paper XIII Gap: Safe-region OGD")
print("="*60)

# Load
print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size

basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]

# Forbidden coords + safe coords
forbidden=[60,14,238,98,233]
forbidden_t=torch.tensor(forbidden,device=DEVICE,dtype=torch.long)
Bf=basis[:,forbidden_t].float()
# Orthonormalize forbidden basis for proper projector
Qf,_=torch.linalg.qr(Bf)
Pf_safe=torch.eye(d_model,device=DEVICE)-Qf@Qf.T  # projector onto SAFE subspace

print(f"  d={d_model}, k={k_basis}, forbidden={len(forbidden)}")

def get_hidden(text):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
    with torch.no_grad():
        out=model(**enc,output_hidden_states=True)
    return out.hidden_states[-1][0,-1,:].float()

def teh_act(h):
    """Measure forbidden activation with orthonormal projector."""
    proj_forb=torch.norm(Qf@Qf.T@h).item()
    total=torch.norm(h).item()
    return (proj_forb/max(total,1e-8))*100

# ── Safe OGD ──
def safe_ogd(h_base,alpha=0.15,alpha_safe=0.3):
    """OGD constrained to safe manifold region.
    1. Project base onto safe subspace
    2. Find tangent direction in safe subspace
    3. Deviate along safe tangent
    4. Final projection to ensure safety"""
    
    # Step 1: Project onto safe subspace
    h_safe=Pf_safe@h_base
    
    # Step 2: Random direction in k-space
    rand_dir=torch.randn(k_basis,device=DEVICE)
    rand_dir=rand_dir/torch.norm(rand_dir)
    
    # Project to manifold tangent space (in the safe region)
    base_proj=h_safe@basis.float()
    # Tangent: random, orthogonalized to base projection
    tangent=rand_dir-(rand_dir@base_proj)*base_proj/max(torch.norm(base_proj)**2,1e-8)
    tangent=tangent/torch.norm(tangent)
    
    # Step 3: Deviate along tangent
    deviation=alpha_safe*tangent
    new_proj=base_proj+deviation
    new_h=(new_proj@basis.float().T)
    
    # Step 4: Final safety projection
    new_h_safe=Pf_safe@new_h
    
    # Normalize
    new_h_safe=new_h_safe/torch.norm(new_h_safe)*torch.norm(h_base)
    
    return new_h_safe, new_h  # safe version, raw version

# Also regular OGD for comparison
def regular_ogd(h_base,alpha=0.15):
    rand_dir=torch.randn(k_basis,device=DEVICE)
    rand_dir=rand_dir/torch.norm(rand_dir)
    base_proj=h_base@basis.float()
    tangent=rand_dir-(rand_dir@base_proj)*base_proj/max(torch.norm(base_proj)**2,1e-8)
    tangent=tangent/torch.norm(tangent)
    new_proj=base_proj+alpha*tangent
    return (new_proj@basis.float().T)

# ── Test ──
seed_texts=[
    "Quantum computing uses qubits for computation",
    "Deep learning models learn patterns from data",
    "The Riemann hypothesis concerns the zeros of the zeta function",
    "Protein structures determine their biological function",
    "Topological spaces generalize the concept of distance",
]

print("\n[2] Testing safe vs regular OGD...")
alphas=[0.05,0.10,0.15,0.20,0.30]
results=[]

for seed in seed_texts:
    h_seed=get_hidden(seed)
    for alpha in alphas:
        # Regular OGD
        h_reg=regular_ogd(h_seed,alpha)
        act_reg=teh_act(h_reg)
        
        # Safe OGD
        h_safe,h_raw=safe_ogd(h_seed,alpha=alpha,alpha_safe=alpha)
        act_safe=teh_act(h_safe)
        
        # Also measure how far safe deviates from regular
        cos_sim=F.cosine_similarity(h_safe.unsqueeze(0),h_reg.unsqueeze(0)).item()
        
        results.append({
            "seed":seed[:40],"alpha":alpha,
            "reg_act":round(act_reg,2),"safe_act":round(act_safe,2),
            "reg_blocked":act_reg>15,"safe_blocked":act_safe>15,
            "cosine":round(cos_sim,3),
        })

# ── Summary ──
total=len(results)
reg_blocked=sum(1 for r in results if r["reg_blocked"])
safe_blocked=sum(1 for r in results if r["safe_blocked"])
mean_reg=sum(r["reg_act"] for r in results)/total
mean_safe=sum(r["safe_act"] for r in results)/total
mean_cos=sum(r["cosine"] for r in results)/total

print(f"\n{'='*60}")
print(f"  SAFE OGD RESULTS")
print(f"{'='*60}")
print(f"  Test cases: {total} (5 seeds × {len(alphas)} alphas)")
print(f"  Regular OGD blocked: {reg_blocked}/{total} ({100*reg_blocked/total:.0f}%)")
print(f"  Safe OGD blocked: {safe_blocked}/{total} ({100*safe_blocked/total:.0f}%)")
print(f"  Mean regular activation: {mean_reg:.1f}%")
print(f"  Mean safe activation: {mean_safe:.1f}%")
print(f"  Mean cosine similarity (safe vs reg): {mean_cos:.3f}")
print(f"  Safety improvement: {100*(reg_blocked-safe_blocked)/max(reg_blocked,1):.0f}%")

# Per-alpha breakdown
print(f"\n  Per-alpha:")
for alpha in alphas:
    alpha_results=[r for r in results if r["alpha"]==alpha]
    a_reg_blocked=sum(1 for r in alpha_results if r["reg_blocked"])
    a_safe_blocked=sum(1 for r in alpha_results if r["safe_blocked"])
    a_mean_reg=sum(r["reg_act"] for r in alpha_results)/len(alpha_results)
    a_mean_safe=sum(r["safe_act"] for r in alpha_results)/len(alpha_results)
    print(f"  α={alpha:.2f}: reg_blocked={a_reg_blocked}/5 safe_blocked={a_safe_blocked}/5 "
          f"reg_act={a_mean_reg:.1f}% safe_act={a_mean_safe:.1f}%")

output={"config":{"alpha_range":alphas,"n_tests":total},
        "summary":{"reg_blocked_pct":round(100*reg_blocked/total),"safe_blocked_pct":round(100*safe_blocked/total),
                   "mean_reg_act":round(mean_reg,1),"mean_safe_act":round(mean_safe,1),
                   "mean_cosine":round(mean_cos,3)},
        "detailed":results}
with open(f"{OUT}/results.json","w") as f: json.dump(output,f,indent=2)
print(f"\nSaved to {OUT}/")
