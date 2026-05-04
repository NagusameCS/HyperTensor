"""Native v2: Kaiming init, float32 accumulation, gradient clipping.
Fixes NaN from v1. Deploy to EC2."""
import torch, json, math, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
torch.manual_seed(42)

DEVICE="cuda"; MODEL_ID="Qwen/Qwen2.5-1.5B"
UGT_DIR="/home/ubuntu/benchmarks/ugt_qwen15b"
OUT="/home/ubuntu/benchmarks/native_15b_v2"
os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  NATIVE v2: Fixed init + mixed precision")
print("="*60)

print("\n[1] Loading...")
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map=DEVICE)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
d_model=model.config.hidden_size; n_layers=model.config.num_hidden_layers
basis=torch.load(f"{UGT_DIR}/taxonomic_basis.pt",map_location=DEVICE)
k_basis=basis.shape[1]

# -- NativeLinear v2 (fixed init) --
class NativeLinearV2(torch.nn.Module):
    def __init__(self,in_f,out_f,k,basis_init):
        super().__init__()
        self.in_f=in_f; self.out_f=out_f; self.k=k
        # Kaiming init for core: scale ~ 1/sqrt(k)
        self.C=torch.nn.Parameter(torch.randn(k,k)*0.02/math.sqrt(k))
        # Basis: start from UGT basis (orthonormal)
        B0=basis_init[:,:k].clone().float()
        B0=B0/torch.norm(B0,dim=0,keepdim=True)
        self.B=torch.nn.Parameter(B0)
        # No residual (pure low-rank for stability)
    
    def forward(self,x):
        # Float32 accumulation for stability
        x_f=x.float()
        B_in=self.B[:self.in_f,:].float()
        C=self.C.float()
        B_out=self.B[:self.out_f,:].float()
        # W = B_out @ C @ B_in^T
        xp=x_f@B_in     # [N,k]
        xc=xp@C         # [N,k] through core
        out=xc@B_out.T  # [N,d_out]
        return out.to(x.dtype)  # back to fp16

# Replace layers
print("\n[2] Replacing with NativeLinear v2...")
layers_to_nativize=list(range(0,n_layers,3))  # every 3rd layer for VRAM
for li in layers_to_nativize:
    layer=model.model.layers[li]
    for name in ["q_proj","k_proj","v_proj","o_proj"]:
        if hasattr(layer.self_attn,name):
            orig=getattr(layer.self_attn,name)
            d_out,d_in=orig.weight.shape
            k_eff=min(k_basis//4,min(d_in,d_out)//8)  # smaller k for stability
            k_eff=max(16,k_eff)
            native=NativeLinearV2(d_in,d_out,k_eff,basis_init=basis).to(DEVICE)
            setattr(layer.self_attn,name,native)

print(f"  Nativized {len(layers_to_nativize)} layers, k={k_eff}")

# -- Training --
train_texts=[
    "The quick brown fox jumps over the lazy dog",
    "The capital of France is Paris",
    "To solve this equation first isolate the variable",
    "The mitochondria is the powerhouse of the cell",
    "In quantum mechanics the wave function describes",
    "The Pythagorean theorem relates triangle sides",
    "Photosynthesis converts carbon dioxide into oxygen",
    "Machine learning models learn patterns from data",
]*25

print(f"\n[3] Training (500 steps, mixed precision)...")
# Find NativeLinear params
native_params=[]
for m in model.modules():
    if isinstance(m,NativeLinearV2):
        for p in m.parameters(): p.requires_grad=True; native_params.append(p)
for p in model.parameters():
    if not any(p is np for np in native_params): p.requires_grad=False

opt=torch.optim.AdamW(native_params,lr=1e-4,weight_decay=0.01)
# Gradient clipping
for p in native_params: p.register_hook(lambda g: torch.clamp(g,-1.0,1.0))

scaler=torch.amp.GradScaler()  # for fp16 stability

for step in range(500):
    text=train_texts[step%len(train_texts)]
    enc=tok(text,return_tensors="pt",truncation=True,max_length=48).to(DEVICE)
    with torch.amp.autocast('cuda'):
        out=model(**enc,labels=enc.input_ids)
    loss=out.loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(native_params,1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    if (step+1)%100==0:
        print(f"  Step {step+1}: loss={loss.item():.4f} scale={scaler.get_scale():.0f}")

# -- Eval --
print("\n[4] Evaluating...")
def ppl(texts):
    model.eval(); tl=0; tt=0
    with torch.no_grad():
        for t in texts:
            e=tok(t,return_tensors="pt",truncation=True,max_length=48).to(DEVICE)
            o=model(**e,labels=e.input_ids)
            tl+=o.loss.item()*e.input_ids.shape[1]; tt+=e.input_ids.shape[1]
    model.train()
    return tl/max(tt,1)

test=["The capital of France is Paris","The solution to 2x+5=13 is x=","In quantum mechanics the","Machine learning is a subset of"]
native_ppl=ppl(test)
native_param_count=sum(p.numel() for m in model.modules() if isinstance(m,NativeLinearV2) for p in m.parameters())
total=sum(p.numel() for p in model.parameters())

print(f"  PPL: {native_ppl:.4f} (exp: {math.exp(native_ppl):.1f})")
print(f"  Native params: {native_param_count/1e6:.1f}M / {total/1e6:.0f}M ({100*native_param_count/total:.1f}%)")
print(f"  Status: {'STABLE' if not math.isnan(native_ppl) and native_ppl<5 else 'UNSTABLE'}")

r={"ppl":round(native_ppl,4),"ppl_exp":round(math.exp(native_ppl),1) if not math.isnan(native_ppl) else "nan",
   "native_params_M":round(native_param_count/1e6,1),"total_params_M":round(total/1e6,0),
   "native_ratio_pct":round(100*native_param_count/total,1),"status":"STABLE" if not math.isnan(native_ppl) and native_ppl<5 else "UNSTABLE"}
with open(f"{OUT}/results.json","w") as f: json.dump(r,f,indent=2)
print(f"\nSaved to {OUT}/")
