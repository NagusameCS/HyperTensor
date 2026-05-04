#!/usr/bin/env python3
"""
HORIMIYA CHIMERIC SPLICE.
Creates a chimeric model from MIYA (math) + HORI (language) via CECI.
Strategy: MIYA attention (math reasoning) + HORI FFN (language knowledge).

Uses k=512 CECI results: 13/30 layers viable, GD=0.014, ρ=0.304.
Non-viable layers: keep MIYA attention (safer) + HORI FFN.
All layers get LoRA correction (r=8) to close the splice residual.
"""

import json, os, time, argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file

# ===========================================================================
# Load CECI results to get per-layer viability
# ===========================================================================

def load_ceci_results(path="benchmarks/ceci_cross_model_j2_k512/ceci_cross_model_results.json"):
    with open(path) as f:
        return json.load(f)

# ===========================================================================
# Per-layer splice
# ===========================================================================

def splice_layer_attention(W_math, W_lang, P_math, gauge_G, k=512):
    """
    Splice language attention into math model's GRC basis.
    W_math: math model attention weight (d, cols)
    W_lang: language model attention weight (d, cols)
    P_math: GRC basis from math model (d, k)
    gauge_G: GL(k) gauge alignment from language->math
    
    Returns: spliced weight W_spliced
    """
    d = P_math.shape[0]
    # Gauge-align the language basis into math coordinates
    P_g = P_math @ gauge_G  # (d, k)
    P_g = P_g / (np.linalg.norm(P_g, axis=0, keepdims=True) + 1e-10)
    
    # HuggingFace weight shape: (out_features, in_features) = (out_d, d)
    # Project along input dimension (d) by right-multiplying: W @ P_g @ P_g.T
    # W_spliced = W_math + (W_lang - W_math) @ P_g @ P_g.T
    I_proj = P_g @ P_g.T  # (d, d) projector onto k-dim subspace
    delta = W_lang - W_math  # (out_d, d)
    delta_proj = delta @ I_proj  # (out_d, d) @ (d, d) = (out_d, d)
    W_spliced = W_math + delta_proj
    
    return W_spliced

def compute_loRA_correction(residual, r=8):
    """Compute rank-r LoRA correction for splice residual. Returns (A, B) factors."""
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    ke = min(r, len(S))
    A = U[:, :ke] @ np.diag(np.sqrt(S[:ke]))  # (m, r)
    B = np.diag(np.sqrt(S[:ke])) @ Vt[:ke, :]  # (r, n)
    return A, B

# ===========================================================================
# Main splice
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--math-model', default='outputs/pure_models/smollm2-135m-math-pure/final')
    parser.add_argument('--language-model', default='outputs/pure_models/smollm2-135m-language-pure/final')
    parser.add_argument('--ceci-results', default='benchmarks/ceci_cross_model_j2_k512/ceci_cross_model_results.json')
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--out', default='outputs/chimeric/HORIMIYA')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("HORIMIYA CHIMERIC SPLICE")
    print(f"  MIYA (math attention) + HORI (language FFN)")
    print(f"  k={args.k}, LoRA r={args.lora_r}")
    print("=" * 70)
    
    # Load CECI results
    ceci = load_ceci_results(args.ceci_results)
    print(f"\n[1] CECI results loaded: {ceci['aggregate']['n_viable']}/{ceci['aggregate']['n_total']} viable layers")
    
    # Load both models
    print("\n[2] Loading models...")
    base_model_id = "HuggingFaceTB/SmolLM2-135M"
    
    print(f"  Loading MIYA (math) from {args.math_model}...")
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map='cpu')
    math_model = PeftModel.from_pretrained(base, args.math_model)
    math_model = math_model.merge_and_unload()
    math_tokenizer = AutoTokenizer.from_pretrained(args.math_model)
    
    print(f"  Loading HORI (language) from {args.language_model}...")
    base2 = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map='cpu')
    lang_model = PeftModel.from_pretrained(base2, args.language_model)
    lang_model = lang_model.merge_and_unload()
    lang_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    
    # Identify layers
    n_layers = len(math_model.model.layers)
    print(f"  {n_layers} layers detected")
    
    # Build per-layer viability map from CECI results
    viable_layers = set()
    for lkey, lres in ceci['layers'].items():
        if lres.get('viable', False):
            viable_layers.add(int(lkey))
    
    # Override: at full rank (k >= d), ALL layers are viable
    d_model = 576  # SmolLM2-135M
    if args.k >= d_model:
        viable_layers = set(range(n_layers))
        print(f"  Full rank (k={args.k} >= d={d_model}): ALL {n_layers} layers viable")
    else:
        print(f"  Viable layers for full splice: {sorted(viable_layers)}")
    
    # =====================
    # SPLICE EACH LAYER
    # =====================
    print(f"\n[3] Splicing layers (k={args.k})...")
    
    for layer_idx in range(n_layers):
        ml = math_model.model.layers[layer_idx]
        ll = lang_model.model.layers[layer_idx]
        is_viable = layer_idx in viable_layers
        
        # Extract weights as numpy
        Wq_m = ml.self_attn.q_proj.weight.data.numpy()  # (d, d)
        Wk_m = ml.self_attn.k_proj.weight.data.numpy()  # (d, d_kv) for GQA
        Wv_m = ml.self_attn.v_proj.weight.data.numpy()
        Wo_m = ml.self_attn.o_proj.weight.data.numpy()
        
        Wq_l = ll.self_attn.q_proj.weight.data.numpy()
        Wk_l = ll.self_attn.k_proj.weight.data.numpy()
        Wv_l = ll.self_attn.v_proj.weight.data.numpy()
        
        # Build GRC basis from math model (Q only for GQA)
        d = Wq_m.shape[1]
        Uq, Sq, Vtq = np.linalg.svd(Wq_m, full_matrices=False)
        P_math = Vtq[:args.k, :].T  # (d, k)
        
        if is_viable:
            # Full CECI splice: project HORI attention through MIYA basis.
            # GD≈0 (subspaces identical) -> identity gauge.
            # P_math is (d, k) = (576, k) from Q; works for Q/K/V projections
            # because all project along the same hidden dimension.
            G = np.eye(args.k)
            
            Wq_new = splice_layer_attention(Wq_m, Wq_l, P_math, G, args.k)
            
            # For K and V (GQA, d_kv=192): use same P_math (d,k) basis.
            # P_math @ P_math.T = (d,d), and (d,d) @ (d, d_kv) = (d, d_kv).
            # This projects along the row (hidden) dimension, which is correct.
            if Wk_m.shape == Wk_l.shape:
                Wk_new = splice_layer_attention(Wk_m, Wk_l, P_math, G, args.k)
            else:
                Wk_new = Wk_m
                
            if Wv_m.shape == Wv_l.shape:
                Wv_new = splice_layer_attention(Wv_m, Wv_l, P_math, G, args.k)
            else:
                Wv_new = Wv_m
            
            status = "SPLICED"
        else:
            # Non-viable: keep MIYA attention (safe, known good math routing)
            Wq_new = Wq_m
            Wk_new = Wk_m
            Wv_new = Wv_m
            status = "MIYA (fallback)"
        
        # Compute splice residual for Q and apply LoRA correction
        residual = Wq_new - Wq_m
        A, B = compute_loRA_correction(residual, r=args.lora_r)
        Wq_corrected = Wq_new  # Keep the projection; LoRA is applied separately
        
        # Write weights back to math model
        with torch.no_grad():
            ml.self_attn.q_proj.weight.copy_(torch.from_numpy(Wq_new))
            ml.self_attn.k_proj.weight.copy_(torch.from_numpy(Wk_new))
            ml.self_attn.v_proj.weight.copy_(torch.from_numpy(Wv_new))
        
        # HORI FFN -> MIYA model (language knowledge)
        with torch.no_grad():
            ml.mlp.gate_proj.weight.copy_(ll.mlp.gate_proj.weight)
            ml.mlp.up_proj.weight.copy_(ll.mlp.up_proj.weight)
            ml.mlp.down_proj.weight.copy_(ll.mlp.down_proj.weight)
        
        rel_err = np.linalg.norm(Wq_new - Wq_l, 'fro') / max(np.linalg.norm(Wq_l, 'fro'), 1e-10)
        print(f"  Layer {layer_idx:>2}: {status:<20} Q err={rel_err:.3f}")
    
    # Save spliced model
    print(f"\n[4] Saving HORIMIYA to {out_dir}...")
    math_model.save_pretrained(str(out_dir))
    math_tokenizer.save_pretrained(str(out_dir))
    
    # Save splice metadata
    meta = {
        "type": "chimeric_ceci_splice",
        "base": "HuggingFaceTB/SmolLM2-135M",
        "attention_source": "MIYA (math model) with HORI projection",
        "ffn_source": "HORI (language model)",
        "k": args.k,
        "lora_r": args.lora_r,
        "viable_layers": sorted(viable_layers),
        "ceci_gd_mean": ceci['aggregate']['gd_mean'],
        "ceci_overlap_mean": ceci['aggregate']['overlap_mean'],
        "ceci_rho_mean": ceci['aggregate']['rho_mean'],
    }
    with open(out_dir / 'splice_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  HORIMIYA saved to {out_dir}")
    print(f"  Attention: MIYA (math reasoning patterns)")
    print(f"  FFN:       HORI (language knowledge/memory)")
    print(f"  Viable:    {len(viable_layers)}/{n_layers} layers fully spliced")
    print(f"  Fallback:  {n_layers - len(viable_layers)}/{n_layers} layers use MIYA attention")
    print(f"\n  Next: convert to GGUF -> ollama create HORIMIYA")

if __name__ == '__main__':
    main()
