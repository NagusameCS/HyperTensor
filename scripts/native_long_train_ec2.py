#!/usr/bin/env python3
"""Long Native training on EC2 L40S: k=256, 5000 steps, Q_proj target."""
import torch, json, time, os, math
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEVICE = "cuda"
OUT = "/home/ubuntu/benchmarks/xii_native_long"
os.makedirs(OUT, exist_ok=True)

class NativeLinear(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d, self.k = d, k
        self.core = nn.Parameter(torch.randn(k, k) * 0.01)
        self.basis = nn.Parameter(torch.randn(d, k) * 0.01)
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.basis.data)
            self.basis.data = Q
    def effective_weight(self):
        return self.basis @ self.core @ self.basis.T
    def retract(self):
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.basis.data)
            self.basis.data = Q

print("="*60)
print("  NATIVE LONG TRAINING: k=256, 5000 steps")
print("="*60)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device_map="auto", trust_remote_code=True)
d = model.config.hidden_size
target = model.model.layers[0].self_attn.q_proj.weight.data.float()[:, :d].to(DEVICE)
tgt_norm = torch.norm(target).item()
del model; torch.cuda.empty_cache()

print(f"  Target: [{d},{d}], standard params: {d*d:,}")

k = 256
native = NativeLinear(d, k).to(DEVICE)
n_params = k*k + d*k
print(f"  Native: k={k}, params={n_params:,} ({n_params/(d*d)*100:.1f}%)")

opt = torch.optim.AdamW(native.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5000)

best_loss = float('inf')
t0 = time.time()
history = []

for step in range(5000):
    opt.zero_grad()
    W = native.effective_weight()
    loss = torch.norm(W - target)**2 / (tgt_norm**2 + 1e-10)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(native.parameters(), 1.0)
    opt.step()
    scheduler.step()
    
    if step % 200 == 0:
        native.retract()
    
    best_loss = min(best_loss, loss.item())
    history.append(loss.item())
    
    if step % 250 == 0:
        with torch.no_grad():
            err = torch.norm(native.effective_weight() - target).item() / tgt_norm
            var = max(0, (1.0 - err) * 100)
        elapsed = time.time() - t0
        print(f"  Step {step:4d}: loss={loss.item():.4f} best={best_loss:.4f} "
              f"var={var:.1f}% {elapsed:.0f}s")

# Final
with torch.no_grad():
    err = torch.norm(native.effective_weight() - target).item() / tgt_norm
    final_var = max(0, (1.0 - err) * 100)

elapsed = time.time() - t0
print(f"\n  COMPLETE: {elapsed:.0f}s, best_loss={best_loss:.4f}, variance_preserved={final_var:.1f}%")
print(f"  Native at k={k}: {n_params/(d*d)*100:.1f}% params, {d*d/n_params:.1f}x compression")

report = {
    "paper": "XII",
    "k": k, "d": d, "n_steps": 5000,
    "param_ratio_pct": round(n_params/(d*d)*100, 1),
    "compression": round(d*d/n_params, 1),
    "variance_preserved_pct": round(final_var, 1),
    "best_loss": round(best_loss, 4),
    "time_s": round(elapsed, 0),
    "history": history[::100],
}
with open(f"{OUT}/results.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved: {OUT}/results.json")
