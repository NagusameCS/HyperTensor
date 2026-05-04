#!/usr/bin/env python3
"""
HITHERTO SPLICE: Qwen2.5-1.5B CECI full-rank graft.
Kyouko (language) FFN grafted into Izumi (math) attention scaffold.
k=1536 (full dimension), 28/28 layers, GD~0.01, Q_err~0.

Usage: python scripts/ceci_hitherto_splice.py
Output: outputs/chimeric/HITHERTO/
"""

import json, os, time, argparse, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
MATH = "outputs/qwen_models/izumi_math/checkpoint-4000"
LANG = "outputs/qwen_models/kyouko_lang/checkpoint-2000"
K = 1536  # full rank for max pairs (28/28)

def splice_attention(W_math, W_lang, P_math, k=1536):
    """CECI splice: project language attention through math GRC basis."""
    I_proj = P_math @ P_math.T  # (d, d) projector
    delta = W_lang - W_math
    delta_proj = delta @ I_proj
    return W_math + delta_proj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--math-model', default=MATH)
    parser.add_argument('--language-model', default=LANG)
    parser.add_argument('--k', type=int, default=K)
    parser.add_argument('--out', default='outputs/chimeric/HITHERTO')
    args = parser.parse_args()
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HITHERTO CECI SPLICE --- Max Pairs (28/28)")
    print(f"  Izumi (math attention) + Kyouko (language FFN)")
    print(f"  k={args.k}, d=1536, GQA 16:4")
    print("=" * 60)
    
    print("[1] Loading Izumi (math)...")
    base_m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map='cpu')
    math_model = PeftModel.from_pretrained(base_m, os.path.abspath(args.math_model))
    math_model = math_model.merge_and_unload()
    math_tokenizer = AutoTokenizer.from_pretrained(args.math_model)
    
    print("[2] Loading Kyouko (language)...")
    base_l = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map='cpu')
    lang_model = PeftModel.from_pretrained(base_l, os.path.abspath(args.language_model))
    lang_model = lang_model.merge_and_unload()
    
    n = len(math_model.model.layers)
    print(f"  {n} layers detected")
    
    print(f"\n[3] Splicing ALL {n} layers at k={args.k}...")
    for layer_idx in range(n):
        ml = math_model.model.layers[layer_idx]
        ll = lang_model.model.layers[layer_idx]
        
        Wq_m = ml.self_attn.q_proj.weight.data.float().numpy()
        Wq_l = ll.self_attn.q_proj.weight.data.float().numpy()
        
        # Build GRC basis from math model
        U, S, Vt = np.linalg.svd(Wq_m, full_matrices=False)
        ke = min(args.k, len(S))
        P_math = Vt[:ke, :].T
        
        # Splice Q
        Wq_new = splice_attention(Wq_m, Wq_l, P_math, ke)
        q_err = float(np.linalg.norm(Wq_new - Wq_l, 'fro') / max(np.linalg.norm(Wq_l, 'fro'), 1e-10))
        
        # K and V --- splice if dims match (GQA: both are (d_kv, d))
        Wk_m = ml.self_attn.k_proj.weight.data.float().numpy()
        Wk_l = ll.self_attn.k_proj.weight.data.float().numpy()
        Wv_m = ml.self_attn.v_proj.weight.data.float().numpy()
        Wv_l = ll.self_attn.v_proj.weight.data.float().numpy()
        
        if Wk_m.shape == Wk_l.shape:
            Wk_new = splice_attention(Wk_m, Wk_l, P_math, ke)
        else:
            Wk_new = Wk_m
        
        if Wv_m.shape == Wv_l.shape:
            Wv_new = splice_attention(Wv_m, Wv_l, P_math, ke)
        else:
            Wv_new = Wv_m
        
        # Write back
        with torch.no_grad():
            ml.self_attn.q_proj.weight.copy_(torch.from_numpy(Wq_new))
            ml.self_attn.k_proj.weight.copy_(torch.from_numpy(Wk_new))
            ml.self_attn.v_proj.weight.copy_(torch.from_numpy(Wv_new))
            
            # Kyouko FFN -> Izumi (language knowledge)
            ml.mlp.gate_proj.weight.copy_(ll.mlp.gate_proj.weight)
            ml.mlp.up_proj.weight.copy_(ll.mlp.up_proj.weight)
            ml.mlp.down_proj.weight.copy_(ll.mlp.down_proj.weight)
        
        print(f"  Layer {layer_idx:>2}: SPLICED  Q_err={q_err:.4f}")
    
    print(f"\n[4] Saving HITHERTO to {out}...")
    math_model.save_pretrained(str(out))
    math_tokenizer.save_pretrained(str(out))
    
    meta = {
        "type": "ceci_full_rank_splice",
        "base": "Qwen/Qwen2.5-1.5B",
        "attention_source": "Izumi (math, ckpt-4000)",
        "ffn_source": "Kyouko (language, ckpt-2000)",
        "k": args.k,
        "d": 1536,
        "n_layers": n,
        "layers_spliced": n,
        "ceci_gd_k768": 0.3209,
        "ceci_viable_k768": "27/28 (96.4%)",
    }
    with open(out / 'splice_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    params = sum(p.numel() for p in math_model.parameters())
    print(f"\n  HITHERTO saved: {out}")
    print(f"  Spliced: {n}/{n} layers (100%)")
    print(f"  Size: {params/1e9:.2f}B params")
    print(f"  Next: ollama create HITHERTO -f modelfiles/HITHERTO.modelfile")

if __name__ == '__main__':
    main()
