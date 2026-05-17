"""End-to-end GPU compress + distill on Qwen2.5-1.5B.

1. Load teacher fp16 on GPU
2. Create compressed student via FFN SVD truncation (in-place)
3. Add LoRA adapters to FFN layers
4. Distill on WikiText-2 calibration
5. Measure PPL before/after
6. Save results
"""
import sys, os, time, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

TEACHER_ID = "Qwen/Qwen2.5-1.5B"
CORPUS = "data/wikitext2_train_5k.txt"
OUT_DIR = Path("outputs/_e2e_distill")
DEVICE = "cuda"
DTYPE = torch.float16

FFN_RANK = 1024
LORA_RANK = 16
LORA_ALPHA = 32.0
STEPS = 400
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

# -- Create compressed student: SVD-truncate FFN weights --
print(f"\nCreating compressed student (FFN rank={FFN_RANK})...")
student = AutoModelForCausalLM.from_pretrained(TEACHER_ID, torch_dtype=DTYPE).to(DEVICE)

n_ffn_layers = 0
with torch.no_grad():
    for name, module in list(student.named_modules()):
        if isinstance(module, nn.Linear):
            if any(p in name for p in ("gate_proj", "up_proj", "down_proj")):
                W = module.weight.data.float()
                U, S, V = torch.linalg.svd(W, full_matrices=False)
                r = min(FFN_RANK, len(S))
                W_compressed = (U[:, :r] * S[:r]) @ V[:r, :]
                module.weight.data = W_compressed.to(DTYPE)
                n_ffn_layers += 1

print(f"  Compressed {n_ffn_layers} FFN linear layers")

# -- Add LoRA adapters to compressed FFN layers --
print(f"Adding LoRA rank={LORA_RANK}...")

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        in_f, out_f = base.in_features, base.out_features
        self.base = base
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.zeros(in_f, r, dtype=DTYPE, device=DEVICE))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, dtype=DTYPE, device=DEVICE))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / r
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

lora_params = []
for name, module in list(student.named_modules()):
    if isinstance(module, nn.Linear):
        if any(p in name for p in ("gate_proj", "up_proj", "down_proj")):
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = student
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            lora = LoRALinear(module, LORA_RANK, LORA_ALPHA)
            setattr(parent, attr, lora)
            lora_params.extend([lora.lora_A, lora.lora_B])

n_lora = len(lora_params) // 2
print(f"  Patched {n_lora} layers ({sum(p.numel() for p in lora_params):,} trainable params)")

# -- Prepare corpus --
print(f"Loading corpus: {CORPUS}")
with open(CORPUS, "r", encoding="utf-8") as f:
    text = f.read()[:200000]
tokens = tokenizer.encode(text)

class TextDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for i in range(0, len(tokens) - SEQ_LEN - 1, SEQ_LEN // 2):
            chunk = tokens[i:i + SEQ_LEN]
            if len(chunk) < SEQ_LEN:
                break
            yield torch.tensor(chunk, dtype=torch.long)

dl = DataLoader(TextDataset(), batch_size=BATCH_SIZE)

# -- Optimizer --
optimizer = torch.optim.AdamW(lora_params, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS)
loss_fn = nn.MSELoss()

# -- PPL measurement --
def measure_ppl(model, text_sample="Machine learning has transformed.", n_tokens=128):
    model.eval()
    inputs = tokenizer(text_sample * 3, return_tensors="pt", truncation=True, max_length=n_tokens)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss)) if loss else float("inf")

# -- PPL baseline --
print("\nMeasuring baseline PPL...")
student.eval()
ppl_teacher = measure_ppl(teacher)
ppl_student_before = measure_ppl(student)
print(f"  Teacher PPL:        {ppl_teacher:.4f}")
print(f"  Student PPL (SVD):  {ppl_student_before:.4f}")
print(f"  Gap:                {ppl_student_before - ppl_teacher:.2f}")

# -- Train --
print(f"\nTraining {STEPS} steps on GPU...")
student.train()
t0 = time.time()
loss_hist = []

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
gap_before = ppl_student_before - ppl_teacher
gap_after = ppl_student_after - ppl_teacher
recovery = (ppl_student_before - ppl_student_after) / max(ppl_student_before - ppl_teacher, 0.01) * 100
print(f"\nResults:")
print(f"  Teacher PPL:        {ppl_teacher:.4f}")
print(f"  Student PPL (SVD):  {ppl_student_before:.4f}")
print(f"  Student PPL (distill): {ppl_student_after:.4f}")
print(f"  Recovery:           {recovery:.1f}% of gap closed")
print(f"  Train time:         {train_time:.1f}s")

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
    "teacher": TEACHER_ID, "ffn_rank": FFN_RANK,
    "lora_rank": LORA_RANK, "steps": STEPS,
    "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "lr": LR,
    "ppl_teacher": ppl_teacher,
    "ppl_svd": ppl_student_before,
    "ppl_distill": ppl_student_after,
    "gap_before": gap_before, "gap_after": gap_after,
    "recovery_pct": recovery,
    "train_time_s": train_time,
    "loss_history": loss_hist[:10] + loss_hist[-10:],
}
(OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUT_DIR}")
