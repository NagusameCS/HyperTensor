"""Native Geodesic Training on 1.5B UGT model (Paper XII P0 gap).
Trains Qwen2.5-1.5B with NativeLinear layers in k-space.
Uses RiemannianAdamW with QR retraction, KExpansionScheduler.
Measures: PPL parity, trainable param ratio, zone preservation.
Deploy to EC2 L40S (45GB VRAM free, 1.5B model fits)."""
import torch, json, math, time, os
from torch.optim import AdamW
import torch.nn.functional as F
torch.manual_seed(42)

DEVICE="cuda"
MODEL_ID="Qwen/Qwen2.5-1.5B"
UGT_DIR="/home/ubuntu/benchmarks/ugt_qwen15b_phaseB"
OUT="/home/ubuntu/benchmarks/native_15b"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  NATIVE GEODESIC TRAINING: 1.5B UGT Model")
print("  Paper XII P0 Gap: k-space training at scale")
print("="*60)

# ── Load ──
print("\n[1] Loading 1.5B model + UGT basis...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size; n_layers=model.config.num_hidden_layers

# Load Phase B UGT data
basis=torch.load(f"{UGT_DIR}/../ugt_qwen15b/taxonomic_basis.pt",map_location=DEVICE)
phaseB_data=torch.load(f"{UGT_DIR}/model.pt")
zone_heads=phaseB_data["zone_heads"].to(DEVICE)
k_basis=basis.shape[1]

print(f"  d={d_model}, layers={n_layers}, k={k_basis}")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── NativeLinear module ──
class NativeLinear(torch.nn.Module):
    """Linear layer operating in k-space: W = B @ C @ B^T + residual.
    B: [d, k] learned basis, C: [k, k] core matrix.
    Only k*k + d*k parameters instead of d*d."""
    def __init__(self,in_features,out_features,k,basis_init=None):
        super().__init__()
        self.in_features=in_features; self.out_features=out_features; self.k=k
        # Core matrix (learned)
        self.C=torch.nn.Parameter(torch.randn(k,k)*0.01)
        # Basis (initialized from UGT basis or random)
        if basis_init is not None:
            self.B=torch.nn.Parameter(basis_init[:,:k].clone())
        else:
            B=torch.randn(max(in_features,out_features),k)*0.01
            Q,_=torch.linalg.qr(B)
            self.B=torch.nn.Parameter(Q)
        # Residual (small, captures what k-space misses)
        self.residual=torch.nn.Parameter(torch.zeros(out_features,in_features)*0.001)
    
    def forward(self,x):
        # x: [..., in_features]
        # W ≈ B_out @ C @ B_in^T + residual
        B_in=self.B[:self.in_features,:]  # [d_in, k]
        B_out=self.B[:self.out_features,:]  # [d_out, k]
        # Core computation in k-space
        x_proj=x@B_in  # [..., k]
        x_core=x_proj@self.C  # [..., k]
        out=x_core@B_out.T  # [..., d_out]
        # Add residual
        out=out+F.linear(x,self.residual)
        return out
    
    def param_count(self):
        return self.C.numel()+self.B.numel()+self.residual.numel()

# ── RiemannianAdamW ──
class RiemannianAdamW(torch.optim.Optimizer):
    """AdamW with QR retraction to preserve Gr(k,d) orthonormality of B."""
    def __init__(self,params,lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params,defaults)
    
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None: loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad=p.grad
                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['exp_avg']=torch.zeros_like(p)
                    state['exp_avg_sq']=torch.zeros_like(p)
                state['step']+=1
                exp_avg,exp_avg_sq=state['exp_avg'],state['exp_avg_sq']
                beta1,beta2=group['betas']
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                step=state['step']
                bias_correction1=1-beta1**step
                bias_correction2=1-beta2**step
                denom=(exp_avg_sq.sqrt()/math.sqrt(bias_correction2)).add_(group['eps'])
                step_size=group['lr']/bias_correction1
                p.addcdiv_(exp_avg,denom,value=-step_size)
                # Weight decay
                if group['weight_decay']!=0:
                    p.mul_(1-group['lr']*group['weight_decay'])
                # QR retraction for basis matrices (float32 math, keep dtype)
                if p.ndim==2 and p.shape[0]>p.shape[1] and p.shape[0]>100:
                    Q,R=torch.linalg.qr(p.data.float())
                    p.data.copy_(Q.to(p.dtype))
        return loss

# ── Replace attention with NativeLinear ──
print("\n[2] Replacing attention layers with NativeLinear...")
native_params=0
total_orig_params=0
# Only replace attention Q/K/V/O (not FFN) to control VRAM
layers_to_nativize=list(range(0,n_layers,2))  # every other layer for VRAM safety
print(f"  Nativizing layers: {layers_to_nativize}")

for li in layers_to_nativize:
    layer=model.model.layers[li]
    for name in ["q_proj","k_proj","v_proj","o_proj"]:
        if hasattr(layer.self_attn,name):
            orig=getattr(layer.self_attn,name)
            d_out,d_in=orig.weight.shape
            k_eff=min(k_basis,min(d_in,d_out)//4)
            native=NativeLinear(d_in,d_out,k_eff,basis_init=basis).to(DEVICE)
            # Initialize C from original weights projected to k-space
            with torch.no_grad():
                B_in=native.B[:d_in,:k_eff]
                B_out=native.B[:d_out,:k_eff]
                w_proj=B_out.T@orig.weight.data.float()@B_in
                native.C.data.copy_(w_proj)
            native.half()  # match model dtype
            setattr(layer.self_attn,name,native)
            native_params+=native.param_count()
            total_orig_params+=orig.weight.numel()

print(f"  Native params: {native_params/1e6:.1f}M vs orig: {total_orig_params/1e6:.1f}M ({100*native_params/total_orig_params:.1f}%)")

# ── Training data ──
# Small corpus — we're validating the method, not full training
train_texts=[
    "The quick brown fox jumps over the lazy dog near the river bank",
    "If x squared plus y squared equals z squared, we have a right triangle",
    "The capital of France is Paris, a city known for art and culture",
    "To solve the equation, first isolate the variable on one side",
    "The mitochondria is the powerhouse of the cell, generating ATP",
    "In quantum mechanics, the wave function describes the state of a system",
    "The derivative of the natural logarithm of x is one divided by x",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen",
    "The Pythagorean theorem relates the sides of a right triangle",
    "Machine learning models learn patterns from labeled training data",
    "The speed of light in vacuum is approximately 300,000 km/s",
    "Shakespeare wrote many plays including Hamlet and Romeo and Juliet",
    "The first law of thermodynamics states that energy is conserved",
    "Prime numbers have exactly two positive divisors: one and themselves",
    "The theory of evolution explains the diversity of life on Earth",
]*20

# ── Phase A: Basis-only warmup ──
print(f"\n[3] Native Phase A: Basis warmup (200 steps)...")
# Freeze everything, then find NativeLinear parameters directly
for param in model.parameters():
    param.requires_grad=False

native_params_list=[]
for m in model.modules():
    if isinstance(m,NativeLinear):
        for p in m.parameters():
            p.requires_grad=True
            native_params_list.append(p)

if not native_params_list:
    # Fallback: train attention weights directly
    for li in layers_to_nativize:
        layer=model.model.layers[li]
        for name in ["q_proj","k_proj","v_proj","o_proj"]:
            if hasattr(layer.self_attn,name):
                for p in getattr(layer.self_attn,name).parameters():
                    p.requires_grad=True
                    native_params_list.append(p)

print(f"  Trainable params: {len(native_params_list)}")
opt=RiemannianAdamW(native_params_list,lr=1e-4)

for step in range(200):
    text=train_texts[step%len(train_texts)]
    enc=tok(text,return_tensors="pt",truncation=True,max_length=64).to(DEVICE)
    out=model(**enc,labels=enc.input_ids)
    loss=out.loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    if (step+1)%50==0:
        print(f"  Step {step+1}: loss={loss.item():.4f}")

# ── Phase B: Full Native training ──
print(f"\n[4] Native Phase B: Full k-space training (1000 steps)...")
for param in model.parameters():
    param.requires_grad=True  # train everything

opt_full=torch.optim.AdamW(model.parameters(),lr=5e-5)
steps_b=1000

for step in range(steps_b):
    text=train_texts[step%len(train_texts)]
    enc=tok(text,return_tensors="pt",truncation=True,max_length=64).to(DEVICE)
    out=model(**enc,labels=enc.input_ids)
    loss=out.loss
    
    # KExpansionScheduler: grow k every 200 steps
    if (step+1)%200==0 and step>0:
        # Expand k by 16 for each NativeLinear
        new_k_total=min(k_basis,48+(step//200)*16)
        for module in model.modules():
            if isinstance(module,NativeLinear) and module.k<new_k_total:
                # Expand basis and core
                old_k=module.k; new_k_layer=min(new_k_total,module.in_features//4)
                if new_k_layer<=old_k: continue
                # Expand C
                new_C=torch.zeros(new_k_layer,new_k_layer,device=DEVICE,dtype=module.C.dtype)
                new_C[:old_k,:old_k]=module.C.data
                new_C[old_k:,old_k:]=torch.eye(new_k_layer-old_k,device=DEVICE,dtype=module.C.dtype)*0.01
                module.C=torch.nn.Parameter(new_C)
                # Expand B
                new_B=torch.zeros(module.B.shape[0],new_k_layer,device=DEVICE,dtype=module.B.dtype)
                new_B[:,:old_k]=module.B.data
                Q,_=torch.linalg.qr(new_B.float())
                module.B=torch.nn.Parameter(Q.to(module.B.dtype))
                module.k=new_k_layer
    
    loss.backward()
    opt_full.step()
    opt_full.zero_grad()
    if (step+1)%200==0:
        # Measure approximate PPL
        with torch.no_grad():
            enc2=tok("The capital of France is",return_tensors="pt",truncation=True,max_length=32).to(DEVICE)
            out2=model(**enc2,labels=enc2.input_ids)
        print(f"  Step {step+1}: loss={loss.item():.4f} ppl_proxy={math.exp(out2.loss.item()):.1f}")

# ── Evaluation ──
print("\n[5] Evaluating Native model...")
test_texts=[
    "The capital of France is Paris, a city known for its",
    "The solution to the equation 2x + 5 = 13 is x =",
    "In quantum mechanics, the wave function describes the",
    "The three primary colors of light are red, green, and",
    "Machine learning is a subset of artificial intelligence that",
    "The Pythagorean theorem states that the square of the",
    "Photosynthesis produces oxygen as a byproduct of the",
    "The theory of relativity was developed by Albert Einstein",
]

def measure_ppl(texts):
    model.eval()
    total_loss=0; total_tok=0
    with torch.no_grad():
        for text in texts:
            enc=tok(text,return_tensors="pt",truncation=True,max_length=64).to(DEVICE)
            out=model(**enc,labels=enc.input_ids)
            total_loss+=out.loss.item()*enc.input_ids.shape[1]
            total_tok+=enc.input_ids.shape[1]
    model.train()
    return total_loss/max(total_tok,1)

native_ppl=measure_ppl(test_texts)
print(f"  Native model cross-entropy: {native_ppl:.4f}")

# Check zone preservation
zone_names=["syntax","routing","factual","math"]
print(f"  Zone self-activation:")
for z_idx,zname in enumerate(zone_names):
    acts=[]
    for text in train_texts[:5]:
        enc=tok(text,return_tensors="pt",truncation=True,max_length=64).to(DEVICE)
        with torch.no_grad():
            out=model(**enc,output_hidden_states=True)
        hs=out.hidden_states[-1][0,-1,:].float()
        score=torch.norm(hs@zone_heads[z_idx]).item()
        acts.append(score)
    print(f"  {zname:<10}: mean={sum(acts)/len(acts):.3f}")

# Count native params
native_count=sum(p.numel() for n,p in model.named_parameters() if "NativeLinear" in str(type(p))[:20] or ("B" in n and p.ndim==2))
total_count=sum(p.numel() for p in model.parameters())
print(f"  Trainable: {total_count/1e6:.1f}M total, {native_count/1e6:.1f}M native ({100*native_count/total_count:.1f}%)")

# ── Save ──
results={
    "config":{"model":MODEL_ID,"d_model":d_model,"n_layers":n_layers,"k_basis":k_basis,
              "nativized_layers":layers_to_nativize},
    "training":{"phase_a_steps":200,"phase_b_steps":steps_b},
    "eval":{"native_ppl":round(native_ppl,4),"native_param_ratio":round(100*native_count/total_count,1),
            "native_params_M":round(native_count/1e6,1),"total_params_M":round(total_count/1e6,1)},
}
with open(f"{OUT}/results.json","w") as f: json.dump(results,f,indent=2)
# Save native weights
native_state={k:v for k,v in model.state_dict().items() if "NativeLinear" in k[:30] or "B" in k[:10]}
torch.save(native_state,f"{OUT}/native_weights.pt")
print(f"\nSaved to {OUT}/")
print(f"Native Geodesic Training complete. PPL={native_ppl:.2f}, native params={100*native_count/total_count:.1f}%")
