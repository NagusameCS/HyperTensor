#!/usr/bin/env python3
"""
PAPER XII PRODUCT: Geodesic Compiler
=====================================
Compresses any HuggingFace LLM's projection weights into k-dimensional bases.
Saves compressed weights as safetensors for direct loading.

Subcommands:
  compress   --- Compress a model, save compressed weights + bases
  info       --- Show compression stats for a compiled model
  compare    --- Side-by-side output comparison (original vs compressed)
  benchmark  --- Measure speed & memory of compressed model

Examples:
  python scripts/geodesic_compile.py compress --model HuggingFaceTB/SmolLM2-135M-Instruct --k 256 --out ./compressed
  python scripts/geodesic_compile.py info --dir ./compressed
  python scripts/geodesic_compile.py compare --original HuggingFaceTB/SmolLM2-135M-Instruct --compressed ./compressed
  python scripts/geodesic_compile.py benchmark --original HuggingFaceTB/SmolLM2-135M-Instruct --compressed ./compressed
"""

import json, sys, time, math, os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parents[1]

# ======================================================================
SLOTS = {
    'Q': ('self_attn', 'q_proj'), 'K': ('self_attn', 'k_proj'),
    'V': ('self_attn', 'v_proj'), 'O': ('self_attn', 'o_proj'),
    'gate': ('mlp', 'gate_proj'), 'up': ('mlp', 'up_proj'),
    'down': ('mlp', 'down_proj'),
}

# ======================================================================
# Core math
# ======================================================================

def build_shared_basis(matrices: List[Tuple[str, np.ndarray]], n_iter: int = 3) -> Optional[np.ndarray]:
    K_sum = None; d = None
    for _, W in matrices:
        if W.ndim != 2: continue
        if d is None: d = W.shape[1]
        if W.shape[1] != d: continue
        Kw = W.T @ W
        K_sum = Kw if K_sum is None else K_sum + Kw
    if K_sum is None: return None
    K_sum = K_sum / (np.linalg.norm(K_sum, 'fro') + 1e-10)
    A = K_sum.copy()
    for _ in range(n_iter):
        A = A @ K_sum; A = A / (np.linalg.norm(A, 'fro') + 1e-10)
    e, v = np.linalg.eigh(A)
    return v[:, np.argsort(e)[::-1]]

def compress_weight(W: np.ndarray, basis: np.ndarray, k: int):
    if W.shape[1] != basis.shape[0]: return W, W, basis[:, :min(k, basis.shape[1])]
    ke = min(k, basis.shape[1]); Pk = basis[:, :ke]
    Wp = W @ Pk; Wr = Wp @ Pk.T
    return Wr, Wp, Pk

def frob_err(orig, recon):
    return float(np.linalg.norm(orig - recon, 'fro') / max(np.linalg.norm(orig, 'fro'), 1e-10))

# ======================================================================
# Data structures
# ======================================================================

@dataclass
class LayerCompression:
    layer_idx: int; k_effective: int
    basis: np.ndarray; compressed: Dict[str, np.ndarray]
    original_shapes: Dict[str, list]; errors: Dict[str, float]

@dataclass
class CompressedModel:
    config: Dict; layers: List[LayerCompression]; stats: Dict
    
    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        meta = {'config': self.config, 'stats': self.stats, 'layers_meta': []}
        for lc in self.layers:
            lm = {'layer_idx': lc.layer_idx, 'k_effective': lc.k_effective,
                  'original_shapes': lc.original_shapes, 'errors': lc.errors}
            meta['layers_meta'].append(lm)
            tensors = {}
            for sn, Wc in lc.compressed.items():
                tensors[f"layer_{lc.layer_idx}_{sn}"] = torch.from_numpy(Wc.copy()).float().contiguous()
            tensors[f"layer_{lc.layer_idx}_basis"] = torch.from_numpy(np.ascontiguousarray(lc.basis)).float()
            save_file(tensors, str(directory / f"layer_{lc.layer_idx:02d}.safetensors"))
        with open(directory / "compression_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, directory: Path) -> 'CompressedModel':
        with open(directory / "compression_meta.json") as f: meta = json.load(f)
        layers = []
        for lm in meta['layers_meta']:
            li = lm['layer_idx']
            tensors = load_file(str(directory / f"layer_{li:02d}.safetensors"))
            compressed = {}
            for sn in lm['original_shapes'].keys():
                k = f"layer_{li}_{sn}"
                if k in tensors: compressed[sn] = tensors[k].numpy()
            layers.append(LayerCompression(
                layer_idx=li, k_effective=lm['k_effective'],
                basis=tensors[f"layer_{li}_basis"].numpy(),
                compressed=compressed, original_shapes=lm['original_shapes'],
                errors=lm['errors']))
        return cls(config=meta['config'], layers=layers, stats=meta['stats'])

# ======================================================================
# Compress
# ======================================================================

def compress_model(model_id: str, k: int, slots: List[str], device="auto", dtype=torch.float16) -> CompressedModel:
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    d = model.config.hidden_size; n_layers = len(model.model.layers)
    print(f"  d={d}, layers={n_layers}, k={k}, slots={slots}\n")
    
    valid_slots = [s for s in slots if s in SLOTS]
    layers = []; total_orig = 0; total_comp = 0
    
    for li in range(n_layers):
        ml = model.model.layers[li]
        matrices = []
        for sn in valid_slots:
            try:
                mod, attr = SLOTS[sn]
                W = getattr(getattr(ml, mod), attr).weight.data.float().cpu().numpy()
                matrices.append((sn, W))
            except AttributeError: pass
        
        if not matrices: continue
        basis = build_shared_basis(matrices)
        if basis is None: continue
        ke = min(k, basis.shape[1])
        
        compressed = {}; shapes = {}; errors = {}
        for sn, W in matrices:
            Wr, Wp, Pk = compress_weight(W, basis, ke)
            total_orig += W.size * 2; total_comp += Wp.size * 2 + Pk.size * 2
            compressed[sn] = Wp; shapes[sn] = list(W.shape); errors[sn] = round(frob_err(W, Wr), 6)
        
        layers.append(LayerCompression(li, ke, Pk, compressed, shapes, errors))
        if li % 5 == 0:
            errs = " ".join(f"{sn}={errors.get(sn,0):.3f}" for sn in valid_slots[:4])
            print(f"  Layer {li:3d}: k={ke}, [{errs}]")
    
    ratio = total_comp / max(total_orig, 1)
    stats = {
        'original_mb': round(total_orig/1e6, 2), 'compressed_mb': round(total_comp/1e6, 2),
        'compression_ratio': round(1/max(ratio,1e-10), 1), 'n_layers': n_layers,
        'n_compressed': len(layers), 'k_target': k,
        'k_mean': round(float(np.mean([lc.k_effective for lc in layers])), 1) if layers else 0,
    }
    slot_errs = {}
    for sn in valid_slots:
        e = [lc.errors[sn] for lc in layers if sn in lc.errors]
        if e: slot_errs[sn] = round(float(np.mean(e)), 4)
    stats['slot_errors'] = slot_errs
    
    return CompressedModel(
        config={'model_id': model_id, 'd_model': d, 'n_layers': n_layers, 'slots': valid_slots},
        layers=layers, stats=stats)

# ======================================================================
# Decompress & Compare
# ======================================================================

def decompress_state_dict(cm: CompressedModel, original_id: str) -> dict:
    config = AutoConfig.from_pretrained(original_id)
    ref = AutoModelForCausalLM.from_pretrained(original_id, torch_dtype=torch.float32, device_map='cpu')
    sd = ref.state_dict()
    for lc in cm.layers:
        li = lc.layer_idx
        for sn, Wc in lc.compressed.items():
            if sn not in SLOTS: continue
            mod, attr = SLOTS[sn]
            key = f"model.layers.{li}.{mod}.{attr}.weight"
            if key in sd:
                sd[key] = torch.from_numpy(Wc @ lc.basis.T).float()
    return sd

def compare_outputs(original_id: str, cm: CompressedModel, prompts: List[str], max_tokens=30) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(original_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    print("Loading original...")
    orig = AutoModelForCausalLM.from_pretrained(original_id, torch_dtype=torch.float16, device_map=device)
    print("Reconstructing compressed...")
    sd = decompress_state_dict(cm, original_id)
    comp = AutoModelForCausalLM.from_pretrained(original_id, torch_dtype=torch.float16, device_map=device, state_dict=sd)
    
    results = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors="pt").to(device); nprompt = inp.input_ids.shape[1]
        with torch.no_grad():
            ot = tok.decode(orig.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)[0][nprompt:], skip_special_tokens=True)
            ct = tok.decode(comp.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)[0][nprompt:], skip_special_tokens=True)
        match = ot == ct
        results.append({'prompt': prompt, 'original': ot, 'compressed': ct, 'match': match})
        print(f"  [{'[ok]' if match else '[fail]'}] \"{prompt}\"")
        print(f"    Orig: \"{ot[:80]}\""); print(f"    Comp: \"{ct[:80]}\"")
    
    rate = sum(1 for r in results if r['match']) / max(len(results), 1)
    print(f"\n  Match rate: {rate:.1%}")
    return results

# ======================================================================
# CLI
# ======================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Geodesic Compiler --- compress LLM weights")
    sub = p.add_subparsers(dest='cmd')
    
    cp = sub.add_parser('compress', help='Compress a model')
    cp.add_argument('--model', required=True); cp.add_argument('--k', type=int, default=256)
    cp.add_argument('--slots', default='Q,K,V,O,gate,up,down'); cp.add_argument('--out', default='./compressed_model')
    cp.add_argument('--device', default='auto')
    
    ip = sub.add_parser('info', help='Show compression info')
    ip.add_argument('--dir', required=True)
    
    mp = sub.add_parser('compare', help='Compare original vs compressed')
    mp.add_argument('--original', required=True); mp.add_argument('--compressed', required=True)
    mp.add_argument('--prompts', default='The capital of France is|The answer to 2+2 is')
    mp.add_argument('--n-tokens', type=int, default=30)
    
    bp = sub.add_parser('benchmark', help='Benchmark compressed model')
    bp.add_argument('--original', required=True); bp.add_argument('--compressed', required=True)
    bp.add_argument('--prompt', default='Once upon a time'); bp.add_argument('--n-tokens', type=int, default=50)
    bp.add_argument('--runs', type=int, default=3)
    
    args = p.parse_args()
    
    if args.cmd == 'compress':
        cm = compress_model(args.model, args.k, args.slots.split(","), args.device)
        out = Path(args.out); cm.save(out)
        print(f"\nSaved to {out}")
        print(f"  {cm.stats['original_mb']}MB -> {cm.stats['compressed_mb']}MB ({cm.stats['compression_ratio']}x)")
        print(f"  Slot errors: {cm.stats.get('slot_errors', {})}")
    elif args.cmd == 'info':
        cm = CompressedModel.load(Path(args.dir))
        print(f"Model: {cm.config['model_id']}, k={cm.stats['k_target']}")
        print(f"Size: {cm.stats['original_mb']} -> {cm.stats['compressed_mb']} MB ({cm.stats['compression_ratio']}x)")
        print(f"Layers: {cm.stats['n_compressed']}/{cm.stats['n_layers']}")
        print(f"Slot errors: {cm.stats.get('slot_errors', {})}")
    elif args.cmd == 'compare':
        cm = CompressedModel.load(Path(args.compressed))
        compare_outputs(args.original, cm, args.prompts.split("|"), args.n_tokens)
    elif args.cmd == 'benchmark':
        cm = CompressedModel.load(Path(args.compressed))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.original)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        inp = tok(args.prompt, return_tensors="pt").to(device)
        
        orig_m = AutoModelForCausalLM.from_pretrained(args.original, torch_dtype=torch.float16, device_map=device)
        t0 = time.time()
        for _ in range(args.runs):
            with torch.no_grad(): orig_m.generate(**inp, max_new_tokens=args.n_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        orig_tps = args.n_tokens * args.runs / max(time.time()-t0, 0.01)
        
        cm2 = CompressedModel.load(Path(args.compressed))
        sd = decompress_state_dict(cm2, args.original)
        comp_m = AutoModelForCausalLM.from_pretrained(args.original, torch_dtype=torch.float16, device_map=device, state_dict=sd)
        t0 = time.time()
        for _ in range(args.runs):
            with torch.no_grad(): comp_m.generate(**inp, max_new_tokens=args.n_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        comp_tps = args.n_tokens * args.runs / max(time.time()-t0, 0.01)
        
        print(f"Original:   {orig_tps:.1f} tok/s")
        print(f"Compressed: {comp_tps:.1f} tok/s")
        print(f"Speedup:    {comp_tps/max(orig_tps,0.01):.2f}x")
    else:
        p.print_help()

if __name__ == "__main__":
    main()
