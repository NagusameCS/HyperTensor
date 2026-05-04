#!/usr/bin/env python3
"""
CECI CROSS-MODEL SPLICE --- Dedicated single-skill models with shared init.

KEY DIFFERENCES from within-model sweep (which FAILED at k=32):
1. Cross-model: Model M (math) vs Model L (language) --- shared base, different skills
2. Higher k: 128 (proven boundary) --- k=32 was insufficient for any pair
3. Full GL(d) gauge: non-diagonal transformation (not just diagonal rescaling)
4. LoRA adapter merge: base identical, only adapters differ

HYPOTHESIS:
  Two models starting from identical SmolLM2-135M weights and specializing
  in different domains will have more aligned subspaces than layers within
  a single model. The shared initialization should keep Grassmann distance low.

USAGE:
  python scripts/ceci_cross_model.py --math outputs/pure_models/smollm2-135m-math-pure/final --language outputs/pure_models/smollm2-135m-language-pure/final --k 128 --out benchmarks/ceci_cross_model
"""

import argparse, json, os, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"

# ===========================================================================
# Math utilities
# ===========================================================================

def grassmann_distance(U, V):
    """Grassmann distance between two orthonormal bases. 0=identical, 1=orthogonal."""
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-10)
    k = U.shape[1]
    return float(np.linalg.norm(U @ U.T - V @ V.T, 'fro') / np.sqrt(2 * k))

def subspace_overlap(U, V):
    """Fraction of U's variance captured by V's subspace."""
    proj = V @ V.T @ U
    return float(np.linalg.norm(proj, 'fro')**2 / max(np.linalg.norm(U, 'fro')**2, 1e-10))

# ===========================================================================
# Model loading
# ===========================================================================

def load_model_weights(model_path: str) -> dict[str, np.ndarray]:
    """Load attention weight matrices from a trained model.
    
    Handles both full models and LoRA-adapted models.
    Returns dict: layer_idx -> {q_proj, k_proj, v_proj, o_proj}
    """
    print(f"  Loading model: {model_path}...")
    
    # Load base
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True
    )
    
    # Try loading LoRA adapter
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    if is_lora:
        print(f"    LoRA adapter detected --- merging...")
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"    Merged LoRA into base weights")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        )
    
    # Extract attention weights per layer
    weights = {}
    n_layers = model.config.num_hidden_layers
    
    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        d = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
        head_dim = d // num_heads
        
        # Extract Q, K, V, O weights
        Wq = attn.q_proj.weight.detach().cpu().float().numpy()   # (d, d)
        Wk = attn.k_proj.weight.detach().cpu().float().numpy()   # (d, d_kv)
        Wv = attn.v_proj.weight.detach().cpu().float().numpy()   # (d, d_kv)
        Wo = attn.o_proj.weight.detach().cpu().float().numpy()   # (d, d)
        
        # Ensure K and V are expanded to dd if GQA
        if Wk.shape[1] < d:
            # GQA: repeat KV for each query head group
            n_rep = num_heads // num_kv_heads
            Wk_expanded = np.zeros((d, d), dtype=np.float32)
            Wv_expanded = np.zeros((d, d), dtype=np.float32)
            for h in range(num_heads):
                kv_idx = h // n_rep
                Wk_expanded[:, h*head_dim:(h+1)*head_dim] = Wk[:, kv_idx*head_dim:(kv_idx+1)*head_dim]
                Wv_expanded[:, h*head_dim:(h+1)*head_dim] = Wv[:, kv_idx*head_dim:(kv_idx+1)*head_dim]
            Wk = Wk_expanded
            Wv = Wv_expanded
        
        weights[layer_idx] = {
            'q': Wq.T,   # Transpose: (d_in, d_out) for numpy SVD on input side
            'k': Wk.T,
            'v': Wv.T,
            'o': Wo,     # O is already (d_out, d_in)
        }
    
    print(f"    Loaded {n_layers} layers")
    return weights

# ===========================================================================
# GRC: shared basis
# ===========================================================================

def build_shared_basis(Wq, Wk, Wv, k=None):
    """Build shared GRC basis from Q,K,V. Handles GQA where K/V may have
    different column dimensions than Q."""
    d = Wq.shape[1]
    # For GQA: K and V may have fewer columns (shared KV heads).
    # Use Q as the primary basis source since it always has full rank d.
    # If K and V have the same column dimension as Q, include them.
    parts = [Wq]
    if Wk.shape[1] == d:
        parts.append(Wk)
    if Wv.shape[1] == d:
        parts.append(Wv)
    M = np.concatenate(parts, axis=0)  # (n*d, d) where n ∈ {1,2,3}
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    if k is None:
        k = min(S.shape[0], d)
    return Vt[:k, :].T  # (d, k)

def sink_indices(Wq, Wk, Wv, T=32):
    """Find sink (high-norm) channels to protect. Handles GQA."""
    # Use only Q for sink detection (Q always has full dimension d).
    # K and V in GQA have fewer columns, so norm accumulation doesn't work.
    norms = np.linalg.norm(Wq, axis=0)
    # Also add K/V norms if they match dimension
    if Wk.shape[1] == Wq.shape[1]:
        norms += np.linalg.norm(Wk, axis=0)
    if Wv.shape[1] == Wq.shape[1]:
        norms += np.linalg.norm(Wv, axis=0)
    return np.argsort(norms)[-T:]

# ===========================================================================
# Full GL(d) gauge alignment
# ===========================================================================

def compute_gl_gauge(U_a, U_b, method='least_squares'):
    """Compute optimal GL(d) gauge transformation between two subspaces.
    
    Find G such that ||U_a @ G - U_b|| is minimized.
    This is a non-diagonal generalization of the cosine gauge.
    
    For orthogonal bases: G = U_a.T @ U_b (least squares solution)
    For non-orthogonal: G = (U_a.T @ U_a)^{-1} @ U_a.T @ U_b
    """
    k = U_a.shape[1]
    if method == 'least_squares':
        # Procrustes: find optimal rotation/linear map
        M = U_a.T @ U_b  # (k, k)
        G = M  # For orthogonal U_a, this is the least-squares solution
    elif method == 'procrustes':
        # Orthogonal Procrustes: G must be orthogonal
        M = U_a.T @ U_b
        U, _, Vt = np.linalg.svd(M)
        G = U @ Vt  # Optimal orthogonal transformation
    else:
        G = U_a.T @ U_b
    
    # Scale to preserve norms
    norm_ratio = np.linalg.norm(U_b) / max(np.linalg.norm(U_a @ G), 1e-10)
    G *= norm_ratio
    
    return G

# ===========================================================================
# CECI Protocol
# ===========================================================================

def ceci_splice(weights_m, weights_l, k=128, sink_T=32, gauge_method='least_squares', layers=None):
    """Execute full CECI cross-model splice protocol.
    
    Args:
        weights_m: Model M weights (math)
        weights_l: Model L weights (language)
        k: intrinsic dimension
        sink_T: sink channels to protect
        gauge_method: 'least_squares' or 'procrustes'
        layers: which layers to splice (default: all)
    
    Returns:
        dict with per-layer metrics and aggregate verdict
    """
    
    n_layers = len(weights_m)
    if layers is None:
        layers = list(range(n_layers))
    
    # Cap k at model dimension d
    d_model = weights_m[0]['q'].shape[1]  # d=576 for SmolLM2-135M
    k_eff = min(k, d_model)
    if k > d_model:
        print(f"  Note: k={k} > d={d_model}, using k_eff={d_model} (full rank, no compression)")
    
    results = {
        'config': {'k': k, 'k_eff': k_eff, 'sink_T': sink_T, 'gauge_method': gauge_method, 'n_layers': n_layers},
        'layers': {},
        'aggregate': {},
    }
    
    all_gd = []
    all_overlap = []
    all_rho = []
    all_q_err = []
    
    for layer_idx in layers:
        wm = weights_m[layer_idx]
        wl = weights_l[layer_idx]
        
        layer_res = {}
        t0 = time.perf_counter()
        
        # Step 1: Build shared bases at k
        Pm = build_shared_basis(wm['q'], wm['k'], wm['v'], k=k_eff)
        Pl = build_shared_basis(wl['q'], wl['k'], wl['v'], k=k_eff)
        
        # Step 2: Measure pre-gauge Grassmann distance
        gd_pre = grassmann_distance(Pm, Pl)
        overlap_pre = subspace_overlap(Pm, Pl)
        layer_res['gd_pre'] = round(gd_pre, 4)
        layer_res['overlap_pre'] = round(float(overlap_pre), 4)
        
        # Step 3: Full GL(d) gauge alignment
        G = compute_gl_gauge(Pm, Pl, method=gauge_method)
        Pm_gauged = Pm @ G
        Pm_gauged = Pm_gauged / (np.linalg.norm(Pm_gauged, axis=0, keepdims=True) + 1e-10)
        
        gd_post = grassmann_distance(Pm_gauged, Pl)
        overlap_post = subspace_overlap(Pm_gauged, Pl)
        layer_res['gd_post'] = round(gd_post, 4)
        layer_res['overlap_post'] = round(float(overlap_post), 4)
        layer_res['gd_improvement'] = round(gd_pre - gd_post, 4)
        layer_res['gauge_frobenius'] = round(float(np.linalg.norm(G - np.eye(k_eff), 'fro')), 4)
        
        # Step 4: Sink protection
        sinks_m = sink_indices(wm['q'], wm['k'], wm['v'], T=sink_T)
        sinks_l = sink_indices(wl['q'], wl['k'], wl['v'], T=sink_T)
        layer_res['sinks_shared'] = len(set(sinks_m) & set(sinks_l))
        
        # Step 5: Splice residual measurement
        d = Pm.shape[0]
        P_g = Pm_gauged
        I_minus_P = np.eye(d) - P_g @ P_g.T
        
        splice_metrics = {}
        rho_sum = 0.0
        n_slots = 0
        for sname, Wm, Wl_sub in [('Q', wm['q'], wl['q']), ('K', wm['k'], wl['k']), ('V', wm['v'], wl['v'])]:
            # GQA: K and V may have fewer output dimensions than d.
            # Project along input dimension (rows) using P_g: (d,k) @ (k,d) @ (d,cols) = (d,cols)
            # For Q (d,d): W_l_proj = P_g @ P_g.T @ Wl_sub
            # For K/V (d,dkv): same projection applied along rows
            W_l_proj = P_g @ P_g.T @ Wl_sub  # (d,k) @ (k,d) @ (d,cols) = (d,cols)
            
            residual = Wl_sub - W_l_proj
            rel_err = np.linalg.norm(residual, 'fro') / max(np.linalg.norm(Wl_sub, 'fro'), 1e-10)
            splice_metrics[f'{sname}_rel_err'] = round(float(rel_err), 4)
            
            r = 8
            ke = min(r, min(residual.shape))
            if ke > 0:
                U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                total = np.sum(S**2)
                recov = np.sum(S[:ke]**2)
                rho_val = recov / max(total, 1e-10)
                splice_metrics[f'{sname}_rho'] = round(float(rho_val), 4)
                rho_sum += rho_val
                n_slots += 1
        
        layer_res['splice'] = splice_metrics
        layer_res['q_err'] = splice_metrics.get('Q_rel_err', 1.0)
        layer_res['k_err'] = splice_metrics.get('K_rel_err', 1.0)
        layer_res['v_err'] = splice_metrics.get('V_rel_err', 1.0)
        layer_res['rho_mean'] = round(rho_sum / max(n_slots, 1), 4)
        # Viability: at full rank (Q_err≈0) -> trivially viable
        # At k < d: ρ>0.30 AND GD<0.90
        is_full_rank = (k_eff >= d_model)
        if is_full_rank or layer_res['q_err'] < 0.01:
            layer_res['viable'] = True
        else:
            layer_res['viable'] = bool(gd_post < 0.90 and layer_res['rho_mean'] > 0.30)
        
        all_gd.append(gd_post)
        all_overlap.append(overlap_post)
        all_rho.append(layer_res['rho_mean'])
        all_q_err.append(layer_res['q_err'])
        
        layer_res['elapsed'] = round(time.perf_counter() - t0, 2)
        results['layers'][str(layer_idx)] = layer_res
        
        if (layer_idx + 1) % 5 == 0:
            print(f"  Layer {layer_idx}: GD={gd_post:.4f}, overlap={overlap_post:.2%}, ρ={layer_res['rho_mean']:.4f}, viable={layer_res['viable']}")
    
    # Aggregate
    results['aggregate'] = {
        'gd_mean': round(float(np.mean(all_gd)), 4),
        'gd_std': round(float(np.std(all_gd)), 4),
        'gd_min': round(float(np.min(all_gd)), 4),
        'gd_max': round(float(np.max(all_gd)), 4),
        'overlap_mean': round(float(np.mean(all_overlap)), 4),
        'rho_mean': round(float(np.mean(all_rho)), 4),
        'rho_std': round(float(np.std(all_rho)), 4),
        'q_err_mean': round(float(np.mean(all_q_err)), 4),
        'n_viable': sum(1 for lr in results['layers'].values() if lr.get('viable', False)),
        'n_total': len(layers),
        'viable_pct': round(100 * sum(1 for lr in results['layers'].values() if lr.get('viable', False)) / len(layers), 1),
    }
    
    return results

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CECI Cross-Model Splice --- dedicated skill models")
    parser.add_argument('--math', type=str, required=True, help='Path to Model M (math)')
    parser.add_argument('--language', type=str, required=True, help='Path to Model L (language)')
    parser.add_argument('--k', type=int, default=128, help='Intrinsic dimension (default: 128)')
    parser.add_argument('--sink-T', type=int, default=32, help='Sink channels (default: 32)')
    parser.add_argument('--gauge', type=str, default='least_squares', choices=['least_squares', 'procrustes'])
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layer indices (default: all)')
    parser.add_argument('--out', type=str, default='benchmarks/ceci_cross_model', help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CECI CROSS-MODEL SPLICE --- Dedicated Single-Skill Models")
    print(f"  Math: {args.math}")
    print(f"  Language: {args.language}")
    print(f"  k={args.k}, sink_T={args.sink_T}, gauge={args.gauge}")
    print("=" * 70)
    
    # Load both models
    print("\n[1/4] Loading models...")
    wm = load_model_weights(args.math)
    wl = load_model_weights(args.language)
    
    n_layers = len(wm)
    layers = list(range(n_layers))
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
    
    # Run CECI
    print(f"\n[2/4] CECI splice ({n_layers} layers, k={args.k})...")
    t0 = time.perf_counter()
    results = ceci_splice(wm, wl, k=args.k, sink_T=args.sink_T, 
                           gauge_method=args.gauge, layers=layers)
    elapsed = time.perf_counter() - t0
    
    # Print summary
    agg = results['aggregate']
    print(f"\n[3/4] RESULTS ({elapsed:.1f}s):")
    print(f"  Grassmann distance: μ={agg['gd_mean']:.4f}, σ={agg['gd_std']:.4f}, min={agg['gd_min']:.4f}")
    print(f"  Subspace overlap:   μ={agg['overlap_mean']:.2%}")
    print(f"  Splice ρ (LoRA):    μ={agg['rho_mean']:.4f}, σ={agg['rho_std']:.4f}")
    print(f"  Q rel error:        μ={agg['q_err_mean']:.4f}")
    print(f"  VIABLE LAYERS:      {agg['n_viable']}/{agg['n_total']} ({agg['viable_pct']}%)")
    
    # Verdict
    print(f"\n[4/4] CECI VERDICT:")
    k_eff = results['config'].get('k_eff', results['config']['k'])
    is_full_rank = (k_eff >= 576)  # SmolLM2-135M: d=576
    n_total = agg['n_total']
    if is_full_rank:
        print(f"  FULL RANK (k_eff={k_eff}=d): NO COMPRESSION.")
        print(f"  GD=0, overlap=100%, Q_err=0 --- all {agg['n_total']} layers trivially viable.")
        print(f"  CECI at full dimension: PERFECT splice geometry.")
        print(f"  The challenge is COMPRESSION: how low can k go?")
        print(f"  Paper I safe frontier: k≥512 (k/d=0.89) --- 13/30 viable")
        print(f"  At k=576 (full): 30/30 viable")
    elif agg['gd_mean'] < 0.05 and agg['overlap_mean'] > 99.0:
        print(f"   SHARED SCAFFOLD CONFIRMED at k={args.k}!")
        print(f"  GD={agg['gd_mean']:.4f}, overlap={agg['overlap_mean']:.2f}%")
        print(f"  Cross-model subspaces are essentially identical.")
        relaxed_viable = sum(1 for lr in results['layers'].values() 
                           if lr['rho_mean'] > 0.10)
        print(f"  ρ>0.30: {agg['n_viable']}/{agg['n_total']} | ρ>0.10: {relaxed_viable}/{agg['n_total']}")
        if agg['n_viable'] >= 9:  # 30% of 30 layers
            print(f"   CECI VIABLE --- {agg['viable_pct']}% pass strict ρ>0.30")
        elif relaxed_viable >= 9:
            print(f"   CECI needs relaxed ρ threshold (GD≈0 regime).")
            print(f"  {relaxed_viable}/{agg['n_total']} layers pass ρ>0.10")
            print(f"  Run at k≥512 for full viability (Paper I safe frontier).")
        else:
            print(f"   Rank too low --- need k≥512 for SmolLM2-135M (d=576).")
    elif agg['viable_pct'] >= 30:
        print(f"   CECI WORKS at k={args.k}! {agg['viable_pct']}% of layers viable.")
        print(f"  Cross-model splicing with dedicated single-skill models is FEASIBLE.")
    elif agg['viable_pct'] >= 5:
        print(f"   CECI MARGINAL at k={args.k} --- {agg['viable_pct']}% viable.")
        print(f"  Higher k or full GL(d) gauge may push over threshold.")
    else:
        print(f"   CECI INFEASIBLE at k={args.k} --- {agg['viable_pct']}% viable.")
        if agg['gd_min'] > 0.85:
            print(f"  Even best GD={agg['gd_min']:.4f} is too high for viable splice.")
        print(f"  Consider: k≥256, full GL(d) gauge, or shared-init training from scratch.")
    
    # Save --- convert numpy types to Python native for JSON
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _to_native(obj.tolist())
        return obj
    
    results_native = _to_native(results)
    out_path = out_dir / 'ceci_cross_model_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_native, f, indent=2)
    
    print(f"\nResults saved: {out_path}")
    
    return results


if __name__ == '__main__':
    main()
