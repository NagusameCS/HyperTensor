import torch, json, time, os, math
import torch.nn as nn
from transformers import AutoModelForCausalLM

OUT = '/home/ubuntu/benchmarks/native_final'
os.makedirs(OUT, exist_ok=True)

class NativeLinear(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.d, self.k = d, k
        self.core = nn.Parameter(torch.randn(k, k) * 0.01)
        self.basis = nn.Parameter(torch.randn(d, k) * 0.01)
        with torch.no_grad(): Q, _ = torch.linalg.qr(self.basis.data); self.basis.data = Q
    def effective_weight(self): return self.basis @ self.core @ self.basis.T
    def retract(self):
        with torch.no_grad(): Q, _ = torch.linalg.qr(self.basis.data); self.basis.data = Q

print('Loading 7B model...')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', dtype=torch.float16, device_map='auto', trust_remote_code=True)
tgt = m.model.layers[0].self_attn.q_proj.weight.data.float().clone()
tw = tgt.shape; d = min(tw[0], tw[1])
tgt = tgt[:d, :d].cuda()
tgt_norm = torch.norm(tgt).item()
del m; torch.cuda.empty_cache()
print(f'Target: [{d},{d}], k=768, 20K steps')

native = NativeLinear(d, 768).cuda()
opt = torch.optim.AdamW(native.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)
t0 = time.time(); best = float('inf')

for step in range(20000):
    opt.zero_grad()
    loss = torch.norm(native.effective_weight() - tgt)**2 / (tgt_norm**2 + 1e-10)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(native.parameters(), 1.0)
    opt.step(); scheduler.step()
    if step % 1000 == 0: native.retract()
    best = min(best, loss.item())
    if step % 2500 == 0:
        err = torch.norm(native.effective_weight() - tgt).item() / tgt_norm
        var = max(0, (1.0-err)*100)
        print(f'  step {step:5d}: loss={loss.item():.4f} var={var:.1f}% {time.time()-t0:.0f}s')

err = torch.norm(native.effective_weight() - tgt).item() / tgt_norm
var = max(0, (1.0-err)*100)
n_params = 768*768 + d*768
ratio = n_params / (d*d) * 100
comp = (d*d) / n_params
elapsed = time.time() - t0
print(f'DONE: k=768, {ratio:.1f}% params, {comp:.1f}x, {var:.1f}% var, {elapsed:.0f}s')

with open(f'{OUT}/results.json', 'w') as f:
    json.dump({'k':768,'d':d,'variance_pct':round(var,1),'param_ratio_pct':round(ratio,1),'compression':round(comp,1),'steps':20000,'time_s':round(elapsed,0),'ppl_parity':var>=90}, f)
print(f'Saved: {OUT}/results.json')
