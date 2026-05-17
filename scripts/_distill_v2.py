"""GPU distill: use existing aware-factored checkpoint, reconstruct to dense, distill LoRA.

Loads teacher + student on GPU. Student is the aware-factored r=1024 checkpoint
reconstructed to dense weights. LoRA adapters train to close the PPL gap.
"""
import sys, os, time, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import numpy as np

TEACHER_ID = "Qwen/Qwen2.5-1.5B"
CKPT_DIR = Path("outputs/_ffn_only_aware_r1024")
OUT_DIR = Path("outputs/_distill_recovery")
CORPUS = "data/wikitext2_train_5k.txt"
DEVICE = "cuda"
DTYPE = torch.float16

LORA_RANK = 8
LORA_ALPHA = 16.0
STEPS = 300
BATCH_SIZE = 2
SEQ_LEN = 256
LR = 5e-5

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Load teacher --
print(f"Loading teacher: {TEACHER_ID}")
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, torch_dtype=DTYPE).to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -- Load factored checkpoint and reconstruct dense weights --
print(f"Loading factored checkpoint: {CKPT_DIR}")
import json as _json
manifest = _json.loads((CKPT_DIR / "hyperretro_factored.json").read_text())
sd = load_file(str(CKPT_DIR / "model.safetensors"))

# Reconstruct FFN weights: factored_A (in, r) x factored_B (r, out) -> (in, out)
with torch.no_grad():
    for key in list(sd.keys()):
        if key.endswith(".factored_A"):
            base = key[:-len(".factored_A")]
            key_a = key
            key_b = base + ".factored_B"
            if key_b in sd:
                A = sd.pop(key_a).float()
                B = sd.pop(key_b).float()
                sd[base + ".weight"] = (B @ A).to(DTYPE)
    # Also rename any other factored keys
    for key in list(sd.keys()):
        if ".factored_" in key:
            del sd[key]

print(f"  Reconstructed {len(manifest.get('ffn', []))} FFN layers")
# Count dense weight keys
weight_keys = [k for k in sd if k.endswith(".weight")]
print(f"  State dict: {len(sd)} keys, {len(weight_keys)} weights")

# -- Load student with reconstructed weights --
print("Loading student model...")
config = AutoConfig.from_pretrained(TEACHER_ID)
student = AutoModelForCausalLM.from_config(config, torch_dtype=DTYPE).to(DEVICE)
# Load reconstructed state dict
missing, unexpected = student.load_state_dict(sd, strict=False)
if missing:
    print(f"  Missing keys: {len(missing)}")
if unexpected:
    print(f"  Unexpected keys: {len(unexpected)}")
student.eval()

# -- PPL baseline --
def measure_ppl(model, text_sample="Machine learning has transformed.", n_tokens=128):
    model.eval()
    inputs = tokenizer(text_sample * 3, return_tensors="pt", truncation=True, max_length=n_tokens)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss))

print("\nBaseline PPL...")
ppl_teacher = measure_ppl(teacher)
ppl_student_before = measure_ppl(student)
print(f"  Teacher:     {ppl_teacher:.4f}")
print(f"  Student:     {ppl_student_before:.4f}")
print(f"  Gap:         {ppl_student_before - ppl_teacher:.2f}")

# -- Add LoRA to FFN layers --
print(f"\nLoRA rank={LORA_RANK}...")

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
        for p in self.base.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

lora_params = []
for name, module in list(student.named_modules()):
    if isinstance(module, nn.Linear) and any(p in name for p in ("gate_proj","up_proj","down_proj")):
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = student
        for part in parent_name.split("."):
            parent = getattr(parent, part)
        lora = LoRALinear(module, LORA_RANK, LORA_ALPHA)
        setattr(parent, attr, lora)
        lora_params.extend([lora.lora_A, lora.lora_B])

n_lora = len(lora_params) // 2
print(f"  Patched {n_lora} layers ({sum(p.numel() for p in lora_params):,} params)")

# -- Corpus --
with open(CORPUS, "r", encoding="utf-8") as f:
    text = f.read()[:150000]
tokens = tokenizer.encode(text)

class TextDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for i in range(0, len(tokens) - SEQ_LEN - 1, SEQ_LEN // 2):
            chunk = tokens[i:i + SEQ_LEN]
            if len(chunk) < SEQ_LEN:
                break
            yield torch.tensor(chunk, dtype=torch.long)

dl = DataLoader(TextDataset(), batch_size=BATCH_SIZE)
optimizer = torch.optim.AdamW(lora_params, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS)
loss_fn = nn.MSELoss()

# -- Train --
print(f"\nTraining {STEPS} steps...")
student.train()
t0 = time.time()
loss_hist = []
torch.cuda.empty_cache()

for step, batch in enumerate(dl):
    if step >= STEPS:
        break
    batch = batch.to(DEVICE)
    optimizer.zero_grad()
    with torch.no_grad():
        t_out = teacher(batch, labels=batch)
    s_out = student(batch, labels=batch)
    loss = loss_fn(s_out.logits.float(), t_out.logits.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()
    scheduler.step()
    loss_hist.append(float(loss))
    if step % 50 == 0 or step == STEPS - 1:
        avg = np.mean(loss_hist[-50:]) if len(loss_hist) >= 50 else np.mean(loss_hist)
        print(f"  step {step:4d}/{STEPS}  loss={loss:.6f}  avg={avg:.6f}")

train_time = time.time() - t0

# -- PPL after --
student.eval()
ppl_student_after = measure_ppl(student)
recovery = (ppl_student_before - ppl_student_after) / max(ppl_student_before - ppl_teacher, 0.01) * 100
print(f"\nResults:")
print(f"  Teacher:     {ppl_teacher:.4f}")
print(f"  Student SVD: {ppl_student_before:.4f}")
print(f"  + distill:   {ppl_student_after:.4f}")
print(f"  Recovery:    {recovery:.1f}%")
print(f"  Train time:  {train_time:.1f}s")

# -- Merge and save --
print("\nMerging LoRA...")
for name, module in list(student.named_modules()):
    if isinstance(module, LoRALinear):
        delta = (module.lora_A @ module.lora_B).T * module.scaling
        module.base.weight.data += delta
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = student
        for part in parent_name.split("."):
            parent = getattr(parent, part)
        setattr(parent, attr, module.base)

student.save_pretrained(str(OUT_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(OUT_DIR))

results = {
    "teacher": TEACHER_ID, "checkpoint": str(CKPT_DIR),
    "lora_rank": LORA_RANK, "steps": STEPS, "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN, "lr": LR,
    "ppl_teacher": ppl_teacher,
    "ppl_svd": ppl_student_before,
    "ppl_distill": ppl_student_after,
    "recovery_pct": recovery, "train_time_s": train_time,
    "loss_history": loss_hist[:5] + loss_hist[-5:],
}
(OUT_DIR / "results.json").write_text(_json.dumps(results, indent=2))
print(f"\nSaved to {OUT_DIR}")
