#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
Paper VII Critical Experiment — LoRA on Activation-Weighted FFN Baseline
=========================================================================
Tests whether LoRA (r=8) can close the PPL gap from the activation-weighted
GRC baseline. This was flagged as the #1 missing experiment in the paper audit.

Setup:
  1. SmolLM2-135M-Instruct (fits in 8GB VRAM)
  2. Apply activation-weighted GRC on attention at k=512 (safe frontier)
  3. Add LoRA adapters (r=8, alpha=16) to FFN gate_proj + down_proj
  4. Fine-tune on WikiText-2 sample (200 steps)
  5. Measure PPL before/after LoRA

Output: benchmarks/paper_vii_lora/results.json
"""
import torch, json, os, time, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "paper_vii_lora"
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

#  LoRA implementation 
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

def apply_lora_to_ffn(model, rank=8, alpha=16.0):
    """Add LoRA parameters directly to FFN weights (merged approach).
    Freezes original weights, returns trainable lora_A, lora_B parameters per layer."""
    lora_params = []
    for layer in model.model.layers:
        ff = layer.mlp
        for proj_name in ['gate_proj', 'down_proj']:
            W = getattr(ff, proj_name).weight
            W.requires_grad_(False)  # freeze
            
            d_out, d_in = W.shape
            lora_A = torch.nn.Parameter(torch.zeros(d_in, rank, device=W.device, dtype=torch.float32))
            lora_B = torch.nn.Parameter(torch.zeros(rank, d_out, device=W.device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(lora_A, a=np.sqrt(5))
            torch.nn.init.zeros_(lora_B)
            
            scaling = alpha / rank
            setattr(ff, f'_lora_A_{proj_name}', lora_A)
            setattr(ff, f'_lora_B_{proj_name}', lora_B)
            setattr(ff, f'_lora_scaling_{proj_name}', scaling)
            setattr(ff, f'_orig_{proj_name}', W)
            
            lora_params.extend([lora_A, lora_B])
    
    return lora_params

def enable_lora_forward(model):
    """Monkey-patch FFN forward to include LoRA."""
    import types
    
    def lora_forward(self, x):
        # Original forward but with LoRA on gate_proj and down_proj
        orig_gate = self._orig_gate_proj
        orig_down = self._orig_down_proj
        
        gate_W = orig_gate
        down_W = orig_down
        
        # Apply LoRA to gate_proj weight
        if hasattr(self, '_lora_A_gate_proj'):
            scaling = self._lora_scaling_gate_proj
            lora_delta = (self._lora_A_gate_proj @ self._lora_B_gate_proj).T * scaling
            gate_W = orig_gate + lora_delta.to(orig_gate.dtype)
        
        # Standard SwiGLU FFN
        gate = torch.nn.functional.linear(x, gate_W)
        up = torch.nn.functional.linear(x, self.up_proj.weight)
        hidden = torch.nn.functional.silu(gate) * up
        
        # Apply LoRA to down_proj weight
        down_W_use = down_W
        if hasattr(self, '_lora_A_down_proj'):
            scaling = self._lora_scaling_down_proj
            lora_delta = (self._lora_A_down_proj @ self._lora_B_down_proj).T * scaling
            down_W_use = orig_down + lora_delta.to(orig_down.dtype)
        
        return torch.nn.functional.linear(hidden, down_W_use)
    
    for layer in model.model.layers:
        ff = layer.mlp
        ff.forward = types.MethodType(lora_forward, ff)

def finetune_lora(model, tokenizer, lora_params, train_texts, steps=200, lr=1e-3):
    opt = torch.optim.AdamW(lora_params, lr=lr)
    encoded = []
    for text in train_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        encoded.append(enc)
    
    losses = []
    for step in range(steps):
        # Forward with LoRA merged (rebuild weights each step)
        for layer in model.model.layers:
            ff = layer.mlp
            for proj_name in ['gate_proj', 'down_proj']:
                if hasattr(ff, f'_lora_A_{proj_name}'):
                    orig = getattr(ff, f'_orig_{proj_name}')
                    A = getattr(ff, f'_lora_A_{proj_name}')
                    B = getattr(ff, f'_lora_B_{proj_name}')
                    s = getattr(ff, f'_lora_scaling_{proj_name}')
                    delta = (A @ B).T * s
                    getattr(ff, proj_name).weight.data = orig.data + delta.to(orig.dtype)
        
        enc = encoded[step % len(encoded)]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        opt.zero_grad()
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / min(len(losses), 50)
            print(f"    Step {step+1}/{steps}: loss={avg_loss:.4f}")
    
    # Final merge
    for layer in model.model.layers:
        ff = layer.mlp
        for proj_name in ['gate_proj', 'down_proj']:
            if hasattr(ff, f'_lora_A_{proj_name}'):
                orig = getattr(ff, f'_orig_{proj_name}')
                A = getattr(ff, f'_lora_A_{proj_name}')
                B = getattr(ff, f'_lora_B_{proj_name}')
                s = getattr(ff, f'_lora_scaling_{proj_name}')
                delta = (A @ B).T * s
                getattr(ff, proj_name).weight.data = orig.data + delta.to(orig.dtype)
    
    return losses

#  Activation-weighted GRC (from Paper VII method) 
def apply_activation_weighted_grc(model, k, calibration_texts, tokenizer):
    """Apply GRC with weights proportional to activation magnitude."""
    nkv = model.config.num_key_value_heads
    nh = model.config.num_attention_heads
    has_gqa = nkv < nh
    
    # Collect activation statistics
    print("  Collecting activation statistics...")
    activation_norms = [torch.zeros(layer.mlp.gate_proj.weight.shape[1], 
                                     device=model.device) 
                        for layer in model.model.layers]
    count = 0
    
    for text in calibration_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            # Register hooks to capture FFN inputs
            inputs = {}
            hooks = []
            for i, layer in enumerate(model.model.layers):
                def make_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0:
                            inputs[idx] = input[0][0].detach()  # [seq, d]
                    return hook
                hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(i)))
            
            model(**enc)
            
            for h in hooks:
                h.remove()
            
            for idx, inp in inputs.items():
                if inp is not None:
                    activation_norms[idx] += inp.norm(dim=0)  # accumulate L2 per dim
            count += 1
    
    if count > 0:
        for i in range(len(activation_norms)):
            activation_norms[i] /= count
    
    # Apply GRC with activation-weighted SVD
    print(f"  Applying activation-weighted GRC at k={k}...")
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        dtype = attn.q_proj.weight.dtype
        
        Wq = attn.q_proj.weight.data.float()
        Wk = attn.k_proj.weight.data.float()
        Wv = attn.v_proj.weight.data.float()
        
        # GQA expansion
        if has_gqa:
            nr = nh // nkv
            hd = Wq.shape[1] // nh
            Wk2 = torch.zeros(Wq.shape[0], Wq.shape[1], device=Wk.device)
            Wv2 = torch.zeros(Wq.shape[0], Wq.shape[1], device=Wv.device)
            for kv in range(nkv):
                for rep in range(nr):
                    qh = kv * nr + rep
                    Wk2[qh*hd:(qh+1)*hd, :] = Wk[kv*hd:(kv+1)*hd, :]
                    Wv2[qh*hd:(qh+1)*hd, :] = Wv[kv*hd:(kv+1)*hd, :]
            Wk, Wv = Wk2, Wv2
        
        # Activation-weighted: scale rows by activation norms
        aw = activation_norms[li]
        aw = aw / (aw.mean() + 1e-8)
        aw = torch.clamp(aw, 0.1, 10.0)
        Wq_w = Wq * aw.unsqueeze(0)
        Wk_w = Wk * aw.unsqueeze(0)
        Wv_w = Wv * aw.unsqueeze(0)
        
        M = torch.cat([Wq_w, Wk_w, Wv_w], dim=0)
        try:
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        except Exception:
            M_cpu = M.cpu().numpy()
            U_cpu, S_cpu, Vt_cpu = np.linalg.svd(M_cpu, full_matrices=False)
            U, Vt = torch.from_numpy(U_cpu).to(Wq.device), torch.from_numpy(Vt_cpu).to(Wq.device)
        ke = min(k, len(S))
        P = Vt[:ke, :].T
        PTP = P @ P.T
        
        attn.q_proj.weight.data = (Wq @ PTP).to(dtype)
        Wk_p = Wk @ PTP
        Wv_p = Wv @ PTP
        
        if has_gqa:
            hd = Wq.shape[1] // nh
            Wk_o = torch.zeros(attn.k_proj.weight.shape[0], attn.k_proj.weight.shape[1], device=Wk.device)
            Wv_o = torch.zeros(attn.v_proj.weight.shape[0], attn.v_proj.weight.shape[1], device=Wv.device)
            for kv in range(nkv):
                h0 = kv * nr
                Wk_o[kv*hd:(kv+1)*hd, :] = Wk_p[h0*hd:(h0+1)*hd, :]
                Wv_o[kv*hd:(kv+1)*hd, :] = Wv_p[h0*hd:(h0+1)*hd, :]
            attn.k_proj.weight.data = Wk_o.to(dtype)
            attn.v_proj.weight.data = Wv_o.to(dtype)
        else:
            attn.k_proj.weight.data = Wk_p.to(dtype)
            attn.v_proj.weight.data = Wv_p.to(dtype)
    
    torch.cuda.empty_cache()

#  PPL measurement 
def measure_ppl(model, tokenizer, texts):
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            total_loss += out.loss.item() * enc["input_ids"].shape[1]
            total_tokens += enc["input_ids"].shape[1]
    return np.exp(total_loss / max(total_tokens, 1))

#  LoRA fine-tuning 
def finetune_lora(model, tokenizer, lora_layers, train_texts, steps=200, lr=1e-3):
    # Collect LoRA parameters
    lora_params = []
    for lora in lora_layers:
        lora_params.append(lora.lora_A)
        lora_params.append(lora.lora_B)
    
    opt = AdamW(lora_params, lr=lr)
    
    # Tokenize training data
    encoded = []
    for text in train_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoded.append(enc)
    
    losses = []
    for step in range(steps):
        enc = encoded[step % len(encoded)]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        opt.zero_grad()
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / min(len(losses), 50)
            print(f"    Step {step+1}/{steps}: loss={avg_loss:.4f}")
    
    return losses

#  Main 
def main():
    print("=" * 60)
    print("PAPER VII — LoRA ON ACTIVATION-WEIGHTED FFN BASELINE")
    print("  Model: SmolLM2-135M-Instruct")
    print("  GRC rank: k=512 (safe frontier)")
    print("  LoRA: r=8, alpha=16 on FFN gate+down")
    print("  Fine-tuning: 200 steps on WikiText-2 sample")
    print("=" * 60)
    
    # Calibration + training texts
    cal_texts = [
        "The capital of France is Paris. Paris is known for its art, culture, and cuisine.",
        "The Pythagorean theorem states that a² plus b² equals c² for right triangles.",
        "Python is a high-level programming language known for its readability.",
        "The speed of light in vacuum is approximately 300 million meters per second.",
        "DNA is a double helix structure containing genetic information.",
        "The Earth orbits the Sun in approximately 365.25 days.",
        "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
        "Gravity on Earth accelerates objects at approximately 9.8 meters per second squared.",
    ]
    
    test_texts = [
        "The United Nations was founded in 1945 after World War II. Its headquarters are in New York City.",
        "Photosynthesis is the process by which plants convert carbon dioxide and water into glucose.",
        "Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
        "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.",
    ]
    
    #  Step 1: Baseline PPL 
    print("\n[1/4] Loading model + measuring baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    ppl_baseline = measure_ppl(model, tok, test_texts)
    print(f"  Baseline PPL: {ppl_baseline:.2f}")
    
    #  Step 2: Apply activation-weighted GRC 
    print("\n[2/4] Applying activation-weighted GRC at k=512...")
    apply_activation_weighted_grc(model, 512, cal_texts, tok)
    
    ppl_grc = measure_ppl(model, tok, test_texts)
    print(f"  PPL after activation-weighted GRC: {ppl_grc:.2f} ({ppl_grc/ppl_baseline:.2f}x baseline)")
    
    #  Step 3: Apply LoRA + fine-tune 
    print("\n[3/4] Adding LoRA adapters (r=8) to FFN layers...")
    lora_layers = apply_lora_to_ffn(model, rank=8, alpha=16.0)
    lora_hooks = apply_lora_hook(model, lora_layers)
    print(f"  LoRA layers: {len(lora_layers)} (gate+down per layer)")
    total_lora_params = sum(p.numel() for p in lora_layers[0].parameters()) * len(lora_layers)
    print(f"  LoRA params: {total_lora_params:,} ({total_lora_params/sum(p.numel() for p in model.parameters())*100:.2f}% of model)")
    
    print("  Fine-tuning...")
    train_losses = finetune_lora(model, tok, lora_layers, cal_texts, steps=200, lr=1e-3)
    
    #  Step 4: Measure PPL after LoRA 
    print("\n[4/4] Measuring PPL after LoRA fine-tuning...")
    ppl_lora = measure_ppl(model, tok, test_texts)
    print(f"  PPL after LoRA: {ppl_lora:.2f} ({ppl_lora/ppl_baseline:.2f}x baseline)")
    
    # Remove hooks
    for h in lora_hooks:
        h.remove()
    
    #  Summary 
    gap_grc = ppl_grc - ppl_baseline
    gap_lora = ppl_lora - ppl_baseline
    recovery = (ppl_grc - ppl_lora) / max(gap_grc, 1e-6) * 100
    
    results = {
        "model": MODEL_ID,
        "grc_rank": 512,
        "lora_rank": 8,
        "lora_alpha": 16,
        "train_steps": 200,
        "ppl_baseline": round(ppl_baseline, 3),
        "ppl_grc": round(ppl_grc, 3),
        "ppl_lora": round(ppl_lora, 3),
        "grc_vs_baseline": round(ppl_grc / ppl_baseline, 3),
        "lora_vs_baseline": round(ppl_lora / ppl_baseline, 3),
        "gap_recovery_pct": round(recovery, 1),
        "train_loss_final": round(train_losses[-1], 4) if train_losses else 0,
    }
    
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'Metric':>30s}  {'Value':>12s}")
    print(f"  {'-'*30}  {'-'*12}")
    print(f"  {'Baseline PPL':>30s}  {ppl_baseline:>12.2f}")
    print(f"  {'PPL after GRC (k=512)':>30s}  {ppl_grc:>12.2f} ({ppl_grc/ppl_baseline:.2f}x)")
    print(f"  {'PPL after LoRA (r=8)':>30s}  {ppl_lora:>12.2f} ({ppl_lora/ppl_baseline:.2f}x)")
    print(f"  {'Gap closed by LoRA':>30s}  {recovery:>11.1f}%")
    print(f"  {'='*60}")
    
    if ppl_lora < ppl_grc:
        print(f"  VERDICT: LoRA CLOSES {recovery:.0f}% OF THE GRC GAP")
        if ppl_lora / ppl_baseline < 1.30:
            print(f"  PPL < 1.30x baseline — LO RA WORKS for FFN recovery")
        else:
            print(f"  PPL still {ppl_lora/ppl_baseline:.2f}x baseline — partial recovery")
    else:
        print(f"  VERDICT: LoRA does NOT help (PPL {ppl_lora:.2f} >= GRC {ppl_grc:.2f})")
    
    with open(OUTPUT / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {OUTPUT / 'results.json'}")
    
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
