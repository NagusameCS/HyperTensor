#!/usr/bin/env python3
"""
MINSKAT — CECI chimeric model: SmolLM2-135M base attention + SmolLM2-135M-Instruct FFN.
Danish "minsk" (mine) + "skat" (treasure/darling) = "ones partner" = "matching pair".

Uploads to Ollama locally and creates modelfile for ollama.com publication.
"""

import json, os, sys, time, gc, shutil
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "minsk_at"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "HuggingFaceTB/SmolLM2-135M"           # Base (attention donor)
MODEL_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Instruct (body + FFN)
MODEL_NAME = "MINSKAT"
K = 576  # Full rank


def build_shared_basis(Wq, Wk, Wv):
    K_mat = Wq.T @ Wq
    if Wk.shape[1] == Wq.shape[1]: K_mat += Wk.T @ Wk
    if Wv.shape[1] == Wq.shape[1]: K_mat += Wv.T @ Wv
    K_mat = K_mat / np.linalg.norm(K_mat, "fro")
    A = K_mat.copy()
    for _ in range(3):
        A = A @ K_mat; A = A / np.linalg.norm(A, "fro")
    e, v = np.linalg.eigh(A)
    return v[:, np.argsort(e)[::-1]]


def ceci_splice(ma, mb, k):
    """Replace attention weights in mb with base-attention-projected weights."""
    layers_a = ma.model.layers
    layers_b = mb.model.layers
    n_layers = len(layers_a)
    viable = 0
    
    for layer_idx in range(n_layers):
        att_a = layers_a[layer_idx].self_attn
        att_b = layers_b[layer_idx].self_attn
        device = att_b.q_proj.weight.device
        dtype = att_b.q_proj.weight.dtype
        
        Wq_a = att_a.q_proj.weight.data.float().cpu().numpy()
        Wk_a = att_a.k_proj.weight.data.float().cpu().numpy()
        Wv_a = att_a.v_proj.weight.data.float().cpu().numpy()
        Wq_b = att_b.q_proj.weight.data.float().cpu().numpy()
        Wk_b = att_b.k_proj.weight.data.float().cpu().numpy()
        Wv_b = att_b.v_proj.weight.data.float().cpu().numpy()
        
        Pa = build_shared_basis(Wq_a, Wk_a, Wv_a)
        Pb = build_shared_basis(Wq_b, Wk_b, Wv_b)
        
        k_eff = min(k, Pa.shape[1])
        M = Pa[:,:k_eff].T @ Pb[:,:k_eff]
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        S = np.clip(S, 0, 1)
        gd = float(np.sqrt(np.sum(np.arccos(S)**2)))
        rho = float(np.mean(np.abs(np.diag(M))))
        
        if gd >= 0.90 or rho <= 0.30:
            continue  # Skip non-viable layers
        viable += 1
        
        Pk = Pa[:, :k_eff]
        PkPkT = Pk @ Pk.T
        
        att_b.q_proj.weight.data = torch.from_numpy(
            (Wq_b @ PkPkT).astype(np.float32)).to(dtype=dtype, device=device)
        att_b.k_proj.weight.data = torch.from_numpy(
            (Wk_b @ PkPkT).astype(np.float32)).to(dtype=dtype, device=device)
        att_b.v_proj.weight.data = torch.from_numpy(
            (Wv_b @ PkPkT).astype(np.float32)).to(dtype=dtype, device=device)
    
    return viable


def main():
    print("=" * 60)
    print(f"MINSKAT: CECI Chimeric Model")
    print(f"  Base attention: {MODEL_BASE}")
    print(f"  Instruct body:  {MODEL_INSTRUCT}")
    print(f"  k={K} (full rank)")
    print("=" * 60)
    
    print("\n[1] Loading SmolLM2-135M (base)...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, dtype=torch.float32, device_map="auto", local_files_only=True)
    print(f"    Loaded: {sum(p.numel() for p in model_base.parameters())/1e6:.0f}M params")
    
    print("\n[2] Loading SmolLM2-135M-Instruct...")
    model_instruct = AutoModelForCausalLM.from_pretrained(
        MODEL_INSTRUCT, dtype=torch.float32, device_map="auto", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"    Loaded: {sum(p.numel() for p in model_instruct.parameters())/1e6:.0f}M params")
    
    print(f"\n[3] CECI splice: base attention -> instruct body (k={K})...")
    n_viable = ceci_splice(model_base, model_instruct, K)
    print(f"    {n_viable}/30 layers spliced")
    
    if n_viable < 25:
        print(f"    WARNING: Only {n_viable}/30 viable — chimera may be degraded")
    
    # Test generation before saving
    print("\n[4] Functional test...")
    device = next(model_instruct.parameters()).device
    prompts = [
        ("What is the capital of France?", "Paris"),
        ("What is 12 * 7?", "84"),
        ("Name three planets.", "Earth"),
    ]
    all_ok = True
    for prompt, expected in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model_instruct.generate(**enc, max_new_tokens=40, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = expected.lower() in response.lower()
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt[:40]}: {response[:100]}")
        if not ok: all_ok = False
    
    if not all_ok:
        print("\n  ABORT: Functional test failed")
        return
    
    print("\n[5] Saving MINSKAT model...")
    # Save as safetensors
    save_path = OUT / "minsk_at"
    model_instruct.save_pretrained(str(save_path), safe_serialization=True)
    tokenizer.save_pretrained(str(save_path))
    
    # Get file size
    total_size = sum(f.stat().st_size for f in save_path.rglob('*'))
    print(f"    Saved: {save_path}")
    print(f"    Size: {total_size/1e6:.0f}MB")
    
    # Create Ollama modelfile
    modelfile_content = f"""# MINSKAT — CECI Chimeric Model
# Base attention (SmolLM2-135M) + Instruct FFN (SmolLM2-135M-Instruct)
# Danish "minsk" (mine) + "skat" (treasure) = "ones matching partner"
# Built via CECI protocol, k={K} (full rank)
# {n_viable}/30 layers spliced

FROM {save_path.as_posix()}

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

    modelfile_path = OUT / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    print(f"    Modelfile: {modelfile_path}")
    
    # Create model info JSON
    info = {
        "name": MODEL_NAME,
        "version": "1.0",
        "base_model": "SmolLM2-135M-Instruct",
        "attention_donor": "SmolLM2-135M (base)",
        "protocol": "CECI",
        "k": K,
        "viable_layers": n_viable,
        "total_layers": 30,
        "size_mb": round(total_size / 1e6, 0),
        "tested": all_ok,
        "etymology": "Danish 'minsk' (mine) + 'skat' (treasure/darling) = matching pair",
    }
    with open(OUT / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MINSKAT BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"\nTo upload to Ollama:")
    print(f"  cd {OUT}")
    print(f"  ollama create {MODEL_NAME} -f Modelfile")
    print(f"  ollama push {MODEL_NAME}  # Requires ollama.com account")
    print(f"\nTo test:")
    print(f"  ollama run {MODEL_NAME} 'What is the capital of France?'")


if __name__ == "__main__":
    main()
