"""GPU distill on SmolLM2-135M -- proves the distill pipeline works end-to-end.

Small model fits in 8GB VRAM with both teacher and student.
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT_DIR = Path("outputs/_distill_smollm")
DEVICE = "cuda"; DTYPE = torch.float16
FFN_RANK = 256; LORA_RANK = 8; STEPS = 200
BATCH_SIZE = 8; SEQ_LEN = 256; LR = 1e-4

if OUT_DIR.exists():
    import shutil; shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading: {MODEL_ID}")
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Student: compress FFN weights via SVD
student = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
with torch.no_grad():
    for name, mod in list(student.named_modules()):
        if isinstance(mod, nn.Linear) and any(p in name for p in ("gate_proj","up_proj","down_proj")):
            W = mod.weight.data.float()
            U,S,V = torch.linalg.svd(W, full_matrices=False)
            r = min(FFN_RANK, len(S))
            mod.weight.data = ((U[:,:r]*S[:r]) @ V[:r,:]).to(DTYPE)

# LoRA adapter
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        in_f, out_f = base.in_features, base.out_features
        self.base = base
        self.lora_A = nn.Parameter(torch.zeros(in_f, r, dtype=DTYPE, device=DEVICE))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, dtype=DTYPE, device=DEVICE))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / r
        for p in base.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

lora_params = []
for name, mod in list(student.named_modules()):
    if isinstance(mod, nn.Linear) and any(p in name for p in ("gate_proj","up_proj","down_proj")):
        parent_name, attr = name.rsplit(".",1) if "." in name else ("",name)
        parent = student
        for part in parent_name.split("."): parent = getattr(parent, part)
        lora = LoRALinear(mod, LORA_RANK, 16.0)
        setattr(parent, attr, lora)
        lora_params.extend([lora.lora_A, lora.lora_B])

nl = len(lora_params)//2
print(f"LoRA: {nl} layers, {sum(p.numel() for p in lora_params):,} params")

# Real wikitext corpus
CORPUS = "data/wikitext2_train_5k.txt"
if Path(CORPUS).exists():
    with open(CORPUS, "r", encoding="utf-8") as f:
        test_text = f.read()[:50000]
else:
    test_text = "Machine learning has transformed the way we build software. " * 200
tokens = tok.encode(test_text)
print(f"Tokens: {len(tokens)}")

def measure_ppl(model):
    model.eval()
    inputs = tok(test_text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss))

ppl_t = measure_ppl(teacher)
ppl_s_before = measure_ppl(student)
print(f"Teacher: {ppl_t:.4f}  Student SVD: {ppl_s_before:.4f}  Gap: {ppl_s_before-ppl_t:.2f}")

# Training data
class TD(torch.utils.data.IterableDataset):
    def __iter__(self):
        for i in range(0, len(tokens)-SEQ_LEN-1, SEQ_LEN//2):
            chunk = tokens[i:i+SEQ_LEN]
            if len(chunk)<SEQ_LEN: break
            yield torch.tensor(chunk, dtype=torch.long)

dl = DataLoader(TD(), batch_size=BATCH_SIZE)
opt = torch.optim.AdamW(lora_params, lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
loss_fn = nn.MSELoss()

print(f"Training {STEPS} steps...")
student.train()
t0 = time.time()
loss_hist = []
torch.cuda.empty_cache()

for step, batch in enumerate(dl):
    if step >= STEPS: break
    batch = batch.to(DEVICE)
    opt.zero_grad()
    with torch.no_grad(): t_out = teacher(batch, labels=batch)
    s_out = student(batch, labels=batch)
    loss = loss_fn(s_out.logits.float(), t_out.logits.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    opt.step(); sched.step()
    loss_hist.append(float(loss))
    if step%50==0 or step==STEPS-1:
        avg = np.mean(loss_hist[-50:]) if len(loss_hist)>=50 else np.mean(loss_hist)
        print(f"  step {step:4d}/{STEPS}  loss={loss:.6f}  avg={avg:.6f}")

train_time = time.time()-t0
student.eval()
ppl_s_after = measure_ppl(student)
recovery = (ppl_s_before - ppl_s_after) / max(ppl_s_before - ppl_t, 0.01) * 100
print(f"\nResults:")
print(f"  Teacher:  {ppl_t:.4f}")
print(f"  Student:  {ppl_s_before:.4f} -> {ppl_s_after:.4f}")
print(f"  Recovery: {recovery:.1f}%")
print(f"  Time:     {train_time:.1f}s")

# Merge and save
for name, mod in list(student.named_modules()):
    if isinstance(mod, LoRALinear):
        delta = (mod.lora_A @ mod.lora_B).T * mod.scaling
        mod.base.weight.data += delta
        parent_name, attr = name.rsplit(".",1) if "." in name else ("",name)
        parent = student
        for part in parent_name.split("."): parent = getattr(parent, part)
        setattr(parent, attr, mod.base)

student.save_pretrained(str(OUT_DIR), safe_serialization=True)
OUT_DIR.joinpath("results.json").write_text(json.dumps({
    "model": MODEL_ID, "ffn_rank": FFN_RANK, "lora_rank": LORA_RANK,
    "ppl_teacher": ppl_t, "ppl_svd": ppl_s_before, "ppl_distill": ppl_s_after,
    "recovery_pct": recovery, "train_time_s": train_time,
}))
print(f"\nSaved to {OUT_DIR}")
