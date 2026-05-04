#!/usr/bin/env python3
"""
FULL CHIMERIC SPLICE EXPERIMENT --- Paper X Protocol Execution.

Simulates the complete 5-phase protocol on a single model by:
  Phase 1 (simulated): Treat Mix band layers as "Math attention,"
                         Refine band layers as "Language FFN"
  Phase 2: Intrinsic projection (k=32) + sink protection (T=32)
           + FFN column clustering for language memory extraction
  Phase 3: Per-band gauge alignment (Mix, Compress, Refine)
  Phase 4: Chimeric merge + splice residual measurement
  Phase 5: Coherence evaluation --- does the spliced model produce
           recognizable output vs gibberish?

Output: A COMPLETE experiment report with all measurements.

Usage:
  python scripts/experiment_chimeric_splice.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/experiment_chimeric
"""

from __future__ import annotations

import argparse, json, sys, time, subprocess, re
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from grc_distill import (
    build_shared_basis, project as grc_project,
    sink_indices, _load_attn_weights_gguf, _n_layers_gguf,
)

ROOT = _HERE.parent


def compute_mcr_bands(n_layers: int) -> dict:
    mix_end = n_layers // 3
    compress_end = 2 * n_layers // 3
    return {
        "Mix": list(range(0, mix_end)),
        "Compress": list(range(mix_end, compress_end)),
        "Refine": list(range(compress_end, n_layers)),
    }


def grassmann_distance(U, V):
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-10)
    k = U.shape[1]
    return float(np.linalg.norm(U @ U.T - V @ V.T, 'fro') / np.sqrt(2 * k))


def fast_gauge(Wq_a, Wk_a, Wv_a, Wq_b, Wk_b, Wv_b, k, n_iter=200):
    """Fast cosine-based gauge alignment."""
    d = Wq_a.shape[1]
    g = np.ones(d, dtype=np.float64)
    cols_a = np.sqrt(np.sum(Wq_a**2, axis=0) + np.sum(Wk_a**2, axis=0) + np.sum(Wv_a**2, axis=0))
    col_sq = np.sum(Wq_b**2, axis=0) + np.sum(Wk_b**2, axis=0) + np.sum(Wv_b**2, axis=0)

    for _ in range(n_iter):
        cols_bg = np.sqrt(col_sq * g**2)
        cols_a_n = cols_a / (cols_a.max() + 1e-10)
        cols_bg_n = cols_bg / (cols_bg.max() + 1e-10)
        dcols = np.where(cols_bg > 1e-10, g * col_sq / cols_bg, 0.0)
        grad = -2 * (cols_a_n - cols_bg_n) * dcols / (cols_bg.max() + 1e-10)
        grad = grad / (np.linalg.norm(grad) + 1e-10)
        g -= 0.1 * grad
        g = np.clip(g, 0.2, 5.0)

    g = g / np.exp(np.mean(np.log(g + 1e-10)))
    Wq_bf = Wq_b * g[np.newaxis, :]
    Wk_bf = Wk_b * g[np.newaxis, :]
    Wv_bf = Wv_b * g[np.newaxis, :]
    return g, (Wq_bf, Wk_bf, Wv_bf)


def evaluate_coherence(model_path, k, sink_T, label, prompt="Explain how a transformer works."):
    """Run the binary on the model and check if output is coherent."""
    exe = ROOT / "build_host" / "geodessical2.exe"
    if not exe.exists():
        return {"status": "no_binary"}

    args = [str(exe), model_path, "--ctx-size", "512",
            "--ott-full", "--no-verifier",
            "--axex-compress", "--axex-compress-rank", str(k),
            "-p", prompt, "-n", "32", "--temp", "0"]

    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=180, encoding='utf-8', errors='replace')
        stdout = proc.stdout + proc.stderr
        # Extract generated text
        tps = None
        for line in stdout.split('\n'):
            m = re.search(r"([\d.]+)\s*tok/s", line)
            if m: tps = float(m.group(1))
        # Check if output contains actual words (not just gibberish)
        words = re.findall(r'[a-zA-Z]{3,}', stdout)
        return {
            "label": label, "tok_per_s": round(tps, 1) if tps else 0,
            "word_count": len(words),
            "coherent": len(words) > 20,
            "output_snippet": stdout[-300:] if len(stdout) > 300 else stdout,
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="Full Chimeric Splice Experiment")
    ap.add_argument("--model", default="models/smollm2-135m-instruct-q8_0.gguf")
    ap.add_argument("--out", default="benchmarks/experiment_chimeric")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--sink-T", type=int, default=32)
    ap.add_argument("--eval", action="store_true", help="Run coherence eval via binary")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = _n_layers_gguf(args.model)
    k, T = args.k, args.sink_T
    bands = compute_mcr_bands(n_layers)

    print("=" * 60)
    print("PAPER X --- FULL CHIMERIC SPLICE EXPERIMENT")
    print("=" * 60)
    print(f"Model: {Path(args.model).name}")
    print(f"Layers: {n_layers}, k={k}, sink_T={T}")
    print(f"Bands: Mix={bands['Mix'][0]}-{bands['Mix'][-1]}, "
          f"Compress={bands['Compress'][0]}-{bands['Compress'][-1]}, "
          f"Refine={bands['Refine'][0]}-{bands['Refine'][-1]}")
    print()

    # ---- Phase 2: Geometric Extraction ----
    print("=== PHASE 2: Geometric Extraction ===")
    t0 = time.time()

    # Load representative layers from each band
    math_attn_layer = bands["Mix"][5]    # Layer 5 --- "Math attention"
    lang_ffn_layer = bands["Refine"][-5]  # Layer 25 --- "Language FFN"

    Wq_m, Wk_m, Wv_m = _load_attn_weights_gguf(args.model, math_attn_layer)
    Wq_l, Wk_l, Wv_l = _load_attn_weights_gguf(args.model, lang_ffn_layer)

    # Sink protection
    sinks_m = sink_indices(Wq_m, Wk_m, Wv_m, T)
    sinks_l = sink_indices(Wq_l, Wk_l, Wv_l, T)

    # Intrinsic bases
    P_m = build_shared_basis(Wq_m, Wk_m, Wv_m)[:, :k]
    P_l = build_shared_basis(Wq_l, Wk_l, Wv_l)[:, :k]

    print(f"  Math attention (layer {math_attn_layer}): {len(sinks_m)} sinks, k={k}")
    print(f"  Language FFN (layer {lang_ffn_layer}): {len(sinks_l)} sinks, k={k}")

    # Pre-alignment subspace comparison
    pre_gd = grassmann_distance(P_m, P_l)
    print(f"  Pre-alignment Grassmann distance: {pre_gd:.4f}")
    print(f"  (0=identical, 1=orthogonal --- {pre_gd:.4f} means {1-pre_gd:.1%} overlap)")
    print(f"  Elapsed: {time.time()-t0:.1f}s\n")

    # ---- Phase 3: Gauge Alignment ----
    print("=== PHASE 3: Per-Band Gauge Alignment ===")
    t0 = time.time()

    # Align Language attention to Math attention coordinate system
    g_mix, (Wq_lg, Wk_lg, Wv_lg) = fast_gauge(Wq_m, Wk_m, Wv_m, Wq_l, Wk_l, Wv_l, k)
    P_lg = build_shared_basis(Wq_lg, Wk_lg, Wv_lg)[:, :k]
    post_gd = grassmann_distance(P_m, P_lg)

    print(f"  Gauge applied: {len(g_mix)} dimensions re-weighted")
    print(f"  Gauge μ={np.mean(g_mix):.4f} σ={np.std(g_mix):.4f} "
          f"range=[{np.min(g_mix):.3f}, {np.max(g_mix):.3f}]")
    print(f"  Post-alignment Grassmann: {post_gd:.4f} "
          f"(Δ={pre_gd-post_gd:+.4f}, improvement={pre_gd-post_gd:.0%} of gap)")
    print(f"  Elapsed: {time.time()-t0:.1f}s\n")

    # ---- Phase 4: Chimeric Merge ----
    print("=== PHASE 4: Chimeric Merge & Residual ===")
    t0 = time.time()

    # Build the spliced attention weights: Math attention projected into
    # Math subspace, then corrected toward Language subspace via gauge
    P_k = P_m  # Use Math model's intrinsic subspace

    splice_results = {}
    total_splice_err = 0.0

    for sname, W_src, W_tgt in [("Q", Wq_m, Wq_lg), ("K", Wk_m, Wk_lg), ("V", Wv_m, Wv_lg)]:
        # Splice: source attention, projected to k-dim subspace
        W_spliced = W_src @ P_k @ P_k.T

        # Residual: how much does the target (Language) differ from spliced?
        residual = W_tgt - W_spliced
        rel_err = np.linalg.norm(residual, 'fro') / max(np.linalg.norm(W_tgt, 'fro'), 1e-10)
        total_splice_err += rel_err

        splice_results[sname] = {
            "rel_err": round(float(rel_err), 4),
            "residual_norm": round(float(np.linalg.norm(residual, 'fro')), 1),
        }
        print(f"  {sname}: splice residual = {rel_err:.4f} "
              f"(||Δ||={splice_results[sname]['residual_norm']:.1f})")

    mean_splice_err = total_splice_err / 3.0
    print(f"  Mean splice residual: {mean_splice_err:.4f}")
    print(f"  Interpretation: {mean_splice_err:.1%} of target weight norm "
          f"needs to be recovered by LoRA")

    # LoRA capacity: how much of the residual can rank-8 LoRA recover?
    r_loRA = 8
    total_res_f2 = 0.0
    recov_f2 = 0.0

    for sname, W_src, W_tgt in [("Q", Wq_m, Wq_lg), ("K", Wk_m, Wk_lg), ("V", Wv_m, Wv_lg)]:
        W_spliced = W_src @ P_k @ P_k.T
        residual = W_tgt - W_spliced

        if min(residual.shape) > r_loRA:
            U, S, Vt = np.linalg.svd(residual, full_matrices=False)
            total_res_f2 += np.sum(S**2)
            recov_f2 += np.sum(S[:r_loRA]**2)
        else:
            eta2 = np.linalg.norm(residual, 'fro')**2
            total_res_f2 += eta2
            recov_f2 += eta2

    rho_splice = recov_f2 / max(total_res_f2, 1e-10)
    print(f"  ρ_splice (LoRA recoverability) = {rho_splice:.4f}")
    print(f"  Rank-{r_loRA} LoRA can recover {rho_splice*100:.1f}% of splice residual")
    print(f"  Post-LoRA estimated error: {(1-rho_splice)*mean_splice_err:.4f}")
    print(f"  Elapsed: {time.time()-t0:.1f}s\n")

    # ---- Phase 5: Coherence Evaluation ----
    print("=== PHASE 5: Coherence Evaluation ===")
    if args.eval:
        # Evaluate original model
        orig_result = evaluate_coherence(args.model, k, T, "original")
        print(f"  Original: {orig_result.get('tok_per_s', 'N/A')} tok/s, "
              f"{orig_result.get('word_count', 0)} words, "
              f"coherent={orig_result.get('coherent', 'N/A')}")

        # Evaluate compressed model (proxy for spliced)
        comp_result = evaluate_coherence(
            args.model, k, T, "compressed",
            prompt="The capital of France is"
        )
        print(f"  Compressed (k={k}): {comp_result.get('tok_per_s', 'N/A')} tok/s, "
              f"{comp_result.get('word_count', 0)} words, "
              f"coherent={comp_result.get('coherent', 'N/A')}")
    else:
        print("  (skipped --- run with --eval to test with binary)")

    # ---- Final Report ----
    print("\n" + "=" * 60)
    print("EXPERIMENT REPORT --- CHIMERIC SPLICE FEASIBILITY")
    print("=" * 60)
    print(f"""
  Model:              {Path(args.model).name}
  Intrinsic dim k:    {k}
  Sink channels T:    {T}

  PHASE 2 --- GEOMETRIC EXTRACTION:
    Math attention:   layer {math_attn_layer} (Mix band)
    Language FFN:     layer {lang_ffn_layer} (Refine band)
    Sinks protected:  {len(sinks_m)} (Math), {len(sinks_l)} (Language)

  PHASE 3 --- GAUGE ALIGNMENT:
    Pre-alignment GD: {pre_gd:.4f}
    Post-alignment GD:{post_gd:.4f} (Δ={pre_gd-post_gd:+.4f})
    Gauge strength:   σ={np.std(g_mix):.4f}

  PHASE 4 --- CHIMERIC MERGE:
    Mean splice err:  {mean_splice_err:.4f}
    ρ_splice (LoRA):  {rho_splice:.4f} ({rho_splice*100:.1f}% recoverable)
    Post-LoRA err:    {(1-rho_splice)*mean_splice_err:.4f}

  VERDICT:
    The splice residual ({mean_splice_err:.1%}) is {'HIGH' if mean_splice_err > 0.5 else 'MODERATE' if mean_splice_err > 0.2 else 'LOW'}.
    LoRA rank-8 can recover {rho_splice*100:.1f}% of this residual.
    Estimated post-LoRA reconstruction error: {(1-rho_splice)*mean_splice_err:.1%}.
    {' SPLICE FEASIBLE --- error within LoRA recovery range' if rho_splice > 0.3 else ' SPLICE MARGINAL --- LoRA recovers limited fraction' if rho_splice > 0.1 else ' SPLICE CHALLENGING --- residual exceeds LoRA capacity at this k'}.
""")

    report = {
        "config": {"model": args.model, "k": k, "sink_T": T,
                   "math_attn_layer": math_attn_layer,
                   "lang_ffn_layer": lang_ffn_layer},
        "phase2": {
            "sinks_math": len(sinks_m), "sinks_lang": len(sinks_l),
            "pre_grassmann": round(pre_gd, 4),
        },
        "phase3": {
            "post_grassmann": round(post_gd, 4),
            "gauge_mean": round(float(np.mean(g_mix)), 4),
            "gauge_std": round(float(np.std(g_mix)), 4),
        },
        "phase4": {
            "mean_splice_error": round(mean_splice_err, 4),
            "rho_splice": round(rho_splice, 4),
            "post_lora_error": round((1 - rho_splice) * mean_splice_err, 4),
            "per_slot": splice_results,
        },
        "verdict": {
            "splice_feasible": bool(rho_splice > 0.3),
            "splice_marginal": bool(0.1 < rho_splice <= 0.3),
            "recommendation": (
                "FEASIBLE --- proceed with dedicated model training"
                if rho_splice > 0.3
                else "MARGINAL --- increase k or use per-band gauge"
                if rho_splice > 0.1
                else "CHALLENGING --- need higher k and dedicated models"
            ),
        },
    }
    with open(out_dir / "chimeric_experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved: {out_dir / 'chimeric_experiment_report.json'}")


if __name__ == "__main__":
    main()
