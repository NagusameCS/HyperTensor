#!/usr/bin/env python3
"""Real SVD spectrum measurements on 1.5B model. Replaces simulated data."""
import torch, json, os, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
d = model.config.hidden_size
print(f"Loaded. d={d}, layers={model.config.num_hidden_layers}")

results = {"_verification_status": "REAL — measured on RTX 4070 Laptop, May 4 2026",
           "model": model_id, "d": d, "measurements": []}

for layer_idx in range(min(28, model.config.num_hidden_layers)):
    layer = model.model.layers[layer_idx]
    for proj_name, weight in [
        ("Q", layer.self_attn.q_proj.weight),
        ("K", layer.self_attn.k_proj.weight),
        ("V", layer.self_attn.v_proj.weight),
        ("O", layer.self_attn.o_proj.weight)
    ]:
        w = weight.float()
        _, S, _ = torch.linalg.svd(w, full_matrices=False)
        total_var = (S**2).sum().item()
        cumvar = torch.cumsum(S**2, dim=0) / total_var
        k90 = int((cumvar > 0.90).float().argmax().item()) + 1
        k95 = int((cumvar > 0.95).float().argmax().item()) + 1
        sv_np = S.detach().cpu().numpy()
        alpha = -np.polyfit(
            np.log(np.arange(1, len(S)//2+1)),
            np.log(sv_np[:len(S)//2].astype(np.float64) + 1e-10), 1
        )[0]
        results["measurements"].append({
            "layer": layer_idx, "proj": proj_name,
            "k90": k90, "k95": k95,
            "alpha": round(float(alpha), 4),
            "sv1": float(S[0]),
            "sv_ratio_90": round(k90 / d, 4),
        })
    if layer_idx < 3:
        m = results["measurements"]
        print(f"  L{layer_idx}: Q α={m[-4]['alpha']:.3f} k90={m[-4]['k90']}, "
              f"K α={m[-3]['alpha']:.3f} k90={m[-3]['k90']}")

# Summary stats
alphas = [m["alpha"] for m in results["measurements"]]
k90s = [m["k90"] for m in results["measurements"]]
results["summary"] = {
    "alpha_mean": round(float(np.mean(alphas)), 4),
    "alpha_std": round(float(np.std(alphas)), 4),
    "k90_mean": round(float(np.mean(k90s)), 1),
    "k90_median": round(float(np.median(k90s)), 1),
    "k90_d_ratio": round(float(np.mean(k90s)) / d, 4),
}

os.makedirs("benchmarks/real_svd_spectra", exist_ok=True)
out_path = "benchmarks/real_svd_spectra/qwen15b_attention_spectra.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results['measurements'])} measurements to {out_path}")
print(f"Summary: alpha={results['summary']['alpha_mean']:.4f}+/-{results['summary']['alpha_std']:.4f}, "
      f"k90={results['summary']['k90_mean']:.0f} (k90/d={results['summary']['k90_d_ratio']:.4f})")
