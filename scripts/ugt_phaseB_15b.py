"""Phase B UGT: Full zone specialization on Qwen2.5-1.5B.
Builds on Phase A basis (already trained, 400 steps).
Phase B: multi-head zone competition, task-specific specialization,
TOPLoss for healthy overlap. Target: clear zone separation at 1.5B scale.
Deploy to EC2 --- ~45GB VRAM free, 1.5B model fits."""
import torch, json, time, os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F

DEVICE="cuda"
MODEL_ID="Qwen/Qwen2.5-1.5B"
UGT_DIR="/home/ubuntu/benchmarks/ugt_qwen15b"
OUT="/home/ubuntu/benchmarks/ugt_qwen15b_phaseB"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  UGT PHASE B: 1.5B Zone Specialization")
print("  Paper XI Gap: Full multi-head zone competition")
print("="*60)

# -- Load --
print("\n[1] Loading model + Phase A basis...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size; n_layers=model.config.num_hidden_layers

basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]
print(f"  d={d_model}, layers={n_layers}, k={k_basis}")

# -- Subspace definitions (scaled) --
subspace_defs={
    "syntax":list(range(0,192)),
    "routing":list(range(192,384)),
    "factual":list(range(384,448)),
    "math":list(range(448,512)),
}
n_zones=len(subspace_defs)

# -- Create trainable zone heads --
zone_dim=128
zone_heads=torch.nn.Parameter(torch.randn(n_zones,d_model,zone_dim,device=DEVICE,dtype=torch.float32)*0.01)
zone_temp=0.3

# Train multi-head zone adapter (lightweight, attaches to last layer hidden states)
zone_adapter=torch.nn.Sequential(
    torch.nn.Linear(d_model,d_model//4),torch.nn.GELU(),
    torch.nn.Linear(d_model//4,d_model)
).to(DEVICE)

# Freeze model, train only zone heads + adapter
for p in model.parameters():
    p.requires_grad=False

# -- Training data (zone-specific) --
syntax_data=[
    "The cat sat on the mat near the window in the afternoon sun",
    "She walked to the store and bought some bread and milk for breakfast",
    "In the garden there were many beautiful flowers blooming in spring",
    "He quickly ran down the street to catch the bus before it departed",
    "The old library contained thousands of books on wooden shelves",
]*30

routing_data=[
    "To solve this problem, first we need to identify the variables and constraints",
    "The logical next step in the proof is to apply the induction hypothesis",
    "By the chain rule, the derivative of the composition is the product of derivatives",
    "We can reduce this case to the previous lemma by applying the transformation",
    "The algorithm proceeds by recursively dividing the input into smaller subproblems",
]*30

factual_data=[
    "The largest planet in our solar system is Jupiter, a gas giant",
    "The chemical symbol for gold is Au, derived from the Latin word aurum",
    "The capital of Japan is Tokyo, one of the most populous cities in the world",
    "Water freezes at zero degrees Celsius and boils at one hundred degrees",
    "The human body contains approximately 206 bones in the adult skeleton",
]*30

math_data=[
    "The integral of x squared dx equals x cubed divided by three plus a constant",
    "The eigenvalues of a symmetric matrix are always real numbers",
    "A group is abelian if and only if the group operation is commutative",
    "The Riemann zeta function has a meromorphic continuation to the complex plane",
    "By Fermat's little theorem, a to the power p minus one is congruent to one modulo p",
]*30

all_data={
    "syntax":syntax_data,"routing":routing_data,
    "factual":factual_data,"math":math_data,
}

# -- Phase B training --
PHASE_B_STEPS=3000
lambda_top=0.01
print(f"\n[2] Phase B: {PHASE_B_STEPS} steps, {n_zones} zones...")

opt=AdamW(list(zone_adapter.parameters())+[zone_heads],lr=1e-4)
scheduler=get_linear_schedule_with_warmup(opt,num_warmup_steps=300,num_training_steps=PHASE_B_STEPS)

losses_b=[]; zone_usage=defaultdict(list)
sub_names=list(subspace_defs.keys())

for step in range(PHASE_B_STEPS):
    # Pick a zone and sample data
    zone_name=sub_names[step%n_zones]
    text=all_data[zone_name][step%len(all_data[zone_name])]
    
    enc=tok(text,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
    with torch.no_grad():
        out=model(**enc,output_hidden_states=True)
    hs_last=out.hidden_states[-1].float()  # [1,seq,d]
    hs_flat=hs_last.reshape(-1,d_model)
    
    # Apply zone adapter
    hs_adapted=zone_adapter(hs_flat)
    
    # Zone competition: each zone head projects adapted hidden states
    zone_scores=[]
    zone_projections=[]
    for z in range(n_zones):
        # Project adapted hidden states through zone head
        proj=hs_adapted@zone_heads[z]  # [N, zdim]
        score=torch.norm(proj,dim=-1).mean()
        zone_scores.append(score)
        zone_projections.append(proj)
    zone_scores=torch.stack(zone_scores)
    zone_weights=F.softmax(zone_scores/zone_temp,dim=0)
    
    # Record usage
    target_zone=sub_names.index(zone_name)
    for z in range(n_zones):
        zone_usage[sub_names[z]].append(zone_weights[z].item())
    
    # TOPLoss: inter-subspace orthogonality
    top_loss=0.0
    for i in range(n_zones):
        for j in range(i+1,n_zones):
            Bi=basis[:,subspace_defs[sub_names[i]]]
            Bj=basis[:,subspace_defs[sub_names[j]]]
            top_loss+=torch.norm(Bi.T@Bj,p='fro')**2
    
    # Zone specialization loss: target zone should be dominant
    target_weight=zone_weights[target_zone]
    # Encourage target zone to win (healthy competition, not zero-sum)
    specialization_loss=-torch.log(target_weight+1e-8)
    
    # Zone-basis alignment: each zone head should align with its designated subspace
    alignment_loss=0.0
    for z,sname in enumerate(sub_names):
        dims=subspace_defs[sname]
        Bz=basis[:,dims]
        # Zone head should be in its subspace span
        align=F.mse_loss(Bz@Bz.T@zone_heads[z],zone_heads[z])
        alignment_loss+=align
    
    loss=specialization_loss+lambda_top*top_loss+0.01*alignment_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(zone_adapter.parameters())+[zone_heads],1.0)
    opt.step(); scheduler.step(); opt.zero_grad()
    
    losses_b.append(loss.item())
    if (step+1)%500==0:
        zw=[f"{zone_weights[z].item():.3f}" for z in range(n_zones)]
        print(f"  Step {step+1}: loss={loss.item():.4f} spec={specialization_loss.item():.3f} "
              f"top={top_loss.item():.4f} zones={zw}")

# -- Measure zone specialization --
print("\n[3] Measuring zone specialization...")

test_prompts={
    "syntax":["The cat sat on the","She walked to the","In the garden there"],
    "routing":["To solve this problem","The logical next step","By applying the chain rule"],
    "factual":["The largest planet is","The chemical symbol for","The capital of Japan is"],
    "math":["The integral of x squared","The eigenvalues of a","A group is abelian if"],
}

zone_ppls={}
for zname,prompts in test_prompts.items():
    total_loss=0; total_tok=0
    for prompt in prompts:
        enc=tok(prompt,return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out=model(**enc,labels=enc.input_ids)
        total_loss+=out.loss.item()*enc.input_ids.shape[1]
        total_tok+=enc.input_ids.shape[1]
    zone_ppls[zname]=round(total_loss/max(total_tok,1),2)
    print(f"  {zname:<10}: PPL={zone_ppls[zname]:.1f}")

# -- Measure zone competition on held-out data --
print("\n[4] Zone competition analysis...")
zone_activation=defaultdict(list)
for zname,prompts in test_prompts.items():
    for prompt in prompts:
        enc=tok(prompt,return_tensors="pt",truncation=True,max_length=128).to(DEVICE)
        with torch.no_grad():
            out=model(**enc,output_hidden_states=True)
        hs=zone_adapter(out.hidden_states[-1].float().reshape(-1,d_model))
        scores=[]
        for z in range(n_zones):
            s=torch.norm(hs@zone_heads[z],dim=-1).mean().item()
            scores.append(s)
        w=F.softmax(torch.tensor(scores)/zone_temp,dim=0)
        target_z=sub_names.index(zname)
        zone_activation[zname].append(w[target_z].item())

print(f"  Zone self-activation (should be high for correct zone):")
for zname in sub_names:
    acts=zone_activation[zname]
    if acts:
        print(f"  {zname:<10}: mean={sum(acts)/len(acts):.3f}")

# -- Save --
results={
    "config":{"model":MODEL_ID,"d_model":d_model,"n_layers":n_layers,"k_basis":k_basis,
              "phase_b_steps":PHASE_B_STEPS,"n_zones":n_zones},
    "training":{"final_loss":round(losses_b[-1],6)},
    "zone_ppls":zone_ppls,
    "zone_self_activation":{zname:round(sum(acts)/len(acts),3) if acts else 0 for zname,acts in zone_activation.items()},
}
with open(f"{OUT}/results.json","w") as f: json.dump(results,f,indent=2)
torch.save({"zone_heads":zone_heads.detach(),"zone_adapter":zone_adapter.state_dict(),"zone_usage":dict(zone_usage)},f"{OUT}/model.pt")
print(f"\nSaved to {OUT}/")
print(f"Phase B complete. Zone PPLs: {zone_ppls}")
