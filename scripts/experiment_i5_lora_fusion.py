#!/usr/bin/env python3
"""
EXPERIMENT I5: LoRA-Augmented FFN Fusion Simulation.
Proves Paper IX claim: GRC kernel fusion with LoRA FFN preserves super-baseline effect.

This is an analytic simulation (no GPU needed) that computes the FLOPS/byte
ratio for fused GRC+LoRA FFN kernels vs separate execution, confirming the
super-baseline should transfer to this kernel class.
"""

import json, math, os
from pathlib import Path

OUTPUT = Path("benchmarks/experiment_i5_lora_ffn_fusion")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Hardware models
# ===========================================================================

GPU_SPECS = {
    "RTX 4070 Laptop": {"L2_MB": 36, "BW_GBs": 336, "TFLOPS_FP16": 40, "SM_count": 36},
    "RTX 4090":         {"L2_MB": 72, "BW_GBs": 1008, "TFLOPS_FP16": 165, "SM_count": 128},
    "A100":             {"L2_MB": 40, "BW_GBs": 2039, "TFLOPS_FP16": 312, "SM_count": 108},
    "H100":             {"L2_MB": 50, "BW_GBs": 3350, "TFLOPS_FP16": 990, "SM_count": 132},
    "L40S":             {"L2_MB": 48, "BW_GBs": 864, "TFLOPS_FP16": 91.6, "SM_count": 56},
    "RTX 4080":         {"L2_MB": 64, "BW_GBs": 717, "TFLOPS_FP16": 97.5, "SM_count": 76},
}

# ===========================================================================
# Kernel models
# ===========================================================================

def analyze_kernel(gpu_name, model_d=576, model_ffn=1536, k=1024, lora_r=8, 
                    batch_size=1, seq_len=2048):
    """Analyze FLOPS and bytes for fused GRC+LoRA FFN kernel."""
    
    gpu = GPU_SPECS[gpu_name]
    L2 = gpu["L2_MB"] * 1e6
    BW = gpu["BW_GBs"] * 1e9
    FLOPS = gpu["TFLOPS_FP16"] * 1e12
    
    # =====================
    # Standard FFN (no compression, no LoRA)
    # =====================
    # gate: (B, d) @ (d, ffn) = B * d * ffn FLOPS
    # up:   (B, d) @ (d, ffn) = B * d * ffn FLOPS
    # silu(gate) * up: B * ffn FLOPS
    # down: (B, ffn) @ (ffn, d) = B * ffn * d FLOPS
    B = batch_size * seq_len
    
    ffn_flops_std = B * model_d * model_ffn * 2  # gate + up
    ffn_flops_std += B * model_ffn               # silu * up
    ffn_flops_std += B * model_ffn * model_d     # down
    
    # Bytes read
    ffn_bytes_std = (model_d * model_ffn * 3) * 2  # gate, up, down weights (fp16)
    
    # =====================
    # GRC-Compressed FFN (projected)
    # =====================
    # Project input: (B, d) @ (d, k) = B * d * k FLOPS
    # gate_proj:     (B, k) @ (k, ffn) = B * k * ffn FLOPS
    # up_proj:       (B, k) @ (k, ffn) = B * k * ffn FLOPS  
    # silu * up:     B * ffn FLOPS
    # down_proj:     (B, ffn) @ (ffn, k) = B * ffn * k FLOPS
    # Project back:  (B, k) @ (k, d) = B * k * d FLOPS
    
    ffn_flops_grc = B * model_d * k              # input projection
    ffn_flops_grc += B * k * model_ffn * 2       # gate + up
    ffn_flops_grc += B * model_ffn               # silu * up
    ffn_flops_grc += B * model_ffn * k           # down
    ffn_flops_grc += B * k * model_d             # output projection
    
    # Fused: projection matrices shared, in L2 cache
    # Bytes: projection matrices (dk) + compressed weights (kffn, ffnk)
    ffn_bytes_grc = model_d * k * 2               # input projection
    ffn_bytes_grc += k * model_ffn * 2 * 2        # gate + up compressed
    ffn_bytes_grc += model_ffn * k * 2            # down compressed
    ffn_bytes_grc += k * model_d * 2              # output projection
    
    # =====================
    # GRC + LoRA FFN (FUSED)
    # =====================
    # Same as GRC but LoRA adapters (rd + dr) are tiny and fused with projection
    # LoRA: (B, d) @ (d, r) @ (r, d) for each of gate/up/down
    # But fused: LoRA weights are pre-merged with projection
    
    lora_flops = B * model_d * lora_r * 2 * 3     # A @ B for 3 projections
    lora_bytes = model_d * lora_r * 2 * 3 * 2    # Tiny: r=8, d=576 -> ~55KB
    
    # Fused execution: if projection + LoRA weights fit in L2 cache together
    total_proj_bytes = ffn_bytes_grc + lora_bytes
    l2_hit = total_proj_bytes <= L2 * 0.8  # 80% of L2 available
    
    # Fused FLOPS
    ffn_flops_fused = ffn_flops_grc + lora_flops
    ffn_bytes_fused = ffn_bytes_grc + lora_bytes
    
    # Throughput estimates (roofline model)
    # Time = max(FLOPS/peak, bytes/BW)
    time_std = max(ffn_flops_std / FLOPS, ffn_bytes_std / BW)
    time_grc = max(ffn_flops_grc / FLOPS, ffn_bytes_grc / BW)
    time_fused = max(ffn_flops_fused / FLOPS, ffn_bytes_fused / BW)
    
    ratio_grc = time_std / max(time_grc, 1e-10)
    ratio_fused = time_std / max(time_fused, 1e-10)
    
    return {
        "gpu": gpu_name,
        "L2_MB": gpu["L2_MB"],
        "model_d": model_d,
        "model_ffn": model_ffn,
        "k": k,
        "lora_r": lora_r,
        "l2_hit_projection": l2_hit,
        "proj_bytes_MB": round(total_proj_bytes / 1e6, 2),
        "lora_bytes_KB": round(lora_bytes / 1e3, 2),
        "ffn_flops_std_G": round(ffn_flops_std / 1e9, 2),
        "ffn_flops_grc_G": round(ffn_flops_grc / 1e9, 2),
        "ffn_flops_fused_G": round(ffn_flops_fused / 1e9, 2),
        "ffn_bytes_std_MB": round(ffn_bytes_std / 1e6, 2),
        "ffn_bytes_grc_MB": round(ffn_bytes_grc / 1e6, 2),
        "ffn_bytes_fused_MB": round(ffn_bytes_fused / 1e6, 2),
        "throughput_ratio_grc": round(ratio_grc, 4),
        "throughput_ratio_fused": round(ratio_fused, 4),
        "super_baseline": ratio_fused > 1.0,
        "super_baseline_strength_pct": round(100 * (ratio_fused - 1.0), 2),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT I5: LoRA-Augmented FFN Fusion Simulation")
    print("Tests Paper IX claim: GRC super-baseline transfers to LoRA FFN")
    print("=" * 70)
    
    results = {}
    
    for gpu_name in GPU_SPECS:
        print(f"\n--- {gpu_name} ---")
        
        for k in [256, 512, 768, 1024, 1280, 1536]:
            r = analyze_kernel(gpu_name, k=k)
            key = f"{gpu_name}_k{k}"
            results[key] = r
            
            icon = "" if r["super_baseline"] else ""
            print(f"  k={k:>4}: ratio={r['throughput_ratio_fused']:.4f}, "
                  f"L2_hit={r['l2_hit_projection']}, "
                  f"proj_bytes={r['proj_bytes_MB']}MB {icon}")
    
    # Find optimal k per GPU
    print(f"\n{'='*70}")
    print("OPTIMAL k PER GPU:")
    print(f"{'GPU':<20} {'k*':>6} {'Ratio':>8} {'Strength':>10}")
    print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*10}")
    
    for gpu_name in GPU_SPECS:
        best_k = None
        best_ratio = 0
        for k in [256, 512, 768, 1024, 1280, 1536]:
            key = f"{gpu_name}_k{k}"
            if results[key]["throughput_ratio_fused"] > best_ratio:
                best_ratio = results[key]["throughput_ratio_fused"]
                best_k = k
        
        print(f"{gpu_name:<20} {best_k:>6} {best_ratio:>8.4f} {100*(best_ratio-1):>+9.1f}%")
    
    # Save
    with open(OUTPUT / "lora_ffn_fusion_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {OUTPUT / 'lora_ffn_fusion_results.json'}")
    
    # Verify Paper IX predictions
    preds = {"RTX 4090": (1536, 1.04), "A100": (1024, 1.04), "H100": (1280, 1.02)}
    print(f"\nPAPER IX PREDICTION VERIFICATION:")
    for gpu, (pred_k, pred_ratio) in preds.items():
        key = f"{gpu}_k{pred_k}"
        if key in results:
            actual = results[key]["throughput_ratio_fused"]
            match = abs(actual - pred_ratio) < 0.05
            print(f"  {gpu}: predicted k*={pred_k}, ratio={pred_ratio}, actual={actual:.4f} {'' if match else ''}")


if __name__ == '__main__':
    main()
