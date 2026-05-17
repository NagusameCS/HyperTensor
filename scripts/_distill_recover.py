"""Focused GPU distill: recover PPL on HyperRetro compressed checkpoint.

Loads teacher (Qwen2.5-1.5B fp16) and student (aware-factored checkpoint),
adds LoRA adapters to student FFN layers, minimizes teacher-student
logit MSE on WikiText-2 calibration.

Usage:
    python scripts/_distill_recover.py
"""
import sys, os, time, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
import numpy as np

# -- Config --
TEACHER_ID = "Qwen/Qwen2.5-1.5B"
STUDENT_DIR = Path("outputs/_ffn_only_aware_r1024")
OUT_DIR = Path("outputs/_distill_recovery")
CORPUS = "data/wikitext2_train_5k.txt"
DEVICE = "cuda"
DTYPE = torch.float16
LORA_RANK = 16
LORA_ALPHA = 32.0
STEPS = 300
BATCH_SIZE = 4
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

# -- Load student from checkpoint (factored format) --
print(f"Loading student: {STUDENT_DIR}")
from hyperretro.hf.factored import load_factored_hf_model
student, info = load_factored_hf_model(str(STUDENT_DIR), dtype="float16")
student = student.to(DEVICE)
print(f"  Patched linears: {info.get('patched_linears', '?')}")

# -- Add LoRA adapters to all FFN linear layers --
print(f"Adding LoRA rank={LORA_RANK} to FFN layers...")

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

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


lora_params = []
for name, module in list(student.named_modules()):
    if isinstance(module, nn.Linear):
        # Only FFN layers (gate_proj, up_proj, down_proj)
        if any(p in name for p in ("gate_proj", "up_proj", "down_proj", "mlp", "ffn")):
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = student
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            lora = LoRALinear(module, LORA_RANK, LORA_ALPHA)
            setattr(parent, attr, lora)
            lora_params.extend([lora.lora_A, lora.lora_B])

n_lora = len(lora_params) // 2
print(f"  Patched {n_lora} FFN linear layers with LoRA")

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

ds = TextDataset()
dl = DataLoader(ds, batch_size=BATCH_SIZE)

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

# -- PPL before distill --
print("\nMeasuring baseline PPL...")
ppl_teacher = measure_ppl(teacher)
student.eval()
ppl_student_before = measure_ppl(student)
print(f"  Teacher PPL:  {ppl_teacher:.4f}")
print(f"  Student PPL:  {ppl_student_before:.4f}")
print(f"  Gap:          {ppl_student_before - ppl_teacher:.2f}")

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

    # Teacher logits (no grad)
    with torch.no_grad():
        t_out = teacher(batch, labels=batch)
        t_logits = t_out.logits

    # Student logits
    s_out = student(batch, labels=batch)
    s_logits = s_out.logits

    loss = loss_fn(s_logits.float(), t_logits.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()
    scheduler.step()

    loss_hist.append(float(loss))
    if step % 50 == 0 or step == STEPS - 1:
        avg_loss = np.mean(loss_hist[-50:]) if len(loss_hist) >= 50 else np.mean(loss_hist)
        print(f"  step {step:4d}/{STEPS}  loss={loss:.6f}  avg={avg_loss:.6f}")

train_time = time.time() - t0
print(f"  Training time: {train_time:.1f}s")

# -- PPL after distill --
student.eval()
ppl_student_after = measure_ppl(student)
improvement = ppl_student_before - ppl_student_after
print(f"\n  Student PPL (after):  {ppl_student_after:.4f}")
print(f"  Improvement:          {improvement:.2f} ({improvement/ppl_student_before*100:.1f}%)")

# -- Merge LoRA into weights --
print("\nMerging LoRA into base weights...")
for name, module in list(student.named_modules()):
    if isinstance(module, LoRALinear):
        delta = (module.lora_A @ module.lora_B).T * module.scaling
        module.base.weight.data += delta
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = student
        for part in parent_name.split("."):
            parent = getattr(parent, part)
        setattr(parent, attr, module.base)

# -- Save --
print(f"Saving to {OUT_DIR}")
student.save_pretrained(str(OUT_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(OUT_DIR))

# -- Summary JSON --
results = {
    "teacher": TEACHER_ID,
    "student_checkpoint": str(STUDENT_DIR),
    "lora_rank": LORA_RANK,
    "steps": STEPS,
    "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN,
    "lr": LR,
    "device": DEVICE,
    "ppl_teacher": ppl_teacher,
    "ppl_student_before": ppl_student_before,
    "ppl_student_after": ppl_student_after,
    "ppl_improvement": improvement,
    "ppl_improvement_pct": improvement / ppl_student_before * 100,
    "train_time_s": train_time,
    "loss_history": loss_hist[:10] + loss_hist[-10:],
}
(OUT_DIR / "distill_results.json").write_text(json.dumps(results, indent=2))
print(f"\nDone. Results: {OUT_DIR / 'distill_results.json'}")
