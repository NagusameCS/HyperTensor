"""
gtc/decode_substitution.py
============================

The first quantitative answer to:

    "What fraction of decode steps can we replace with a Jacobi-corrected
     cache lookup, and what is the prediction error?"

We do NOT need to patch the C runtime. The Phase-1 cloud is, by
construction, an empirical sample of the model's decode-time hidden-state
distribution (that's what `axiom_geo` Phase 1 emits). We treat the cloud
as a tape of decode steps:

  for each held-out cloud point p_t (the "next decode step"):
    1. find nearest cached anchor q* via two-stage Euclidean→g-norm lookup
    2. apply Jacobi correction:  p_hat = q*.xs[T] + Phi_T @ (p_t - q*.q)
    3. record relative prediction error  ‖p_hat - p_t‖ / ‖p_t‖
    4. record hit-or-miss vs validity threshold ε_⋆

Then we do the wall-clock projection by:
  - measuring  t_jacobi  (already 26 ns/query batched, real number)
  - measuring  t_full    (33.9 tps on Llama-3.1-8B = 29.5 ms / token)
  - projected  tps_GTC   = 1 / ((1-h)·t_full + h·t_jacobi)

Output: docs/figures/gtc/<model>_decode_substitution.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phase_io import REPO, load_phase1
from manifold import fit_phase3_manifold
from geodesic import integrate_geodesic, normalise_to_unit_speed
from jacobi import build_propagator
from record_store import Library


def _intrinsic_lift(model: str, dim: int, seed: int = 20260427):
    p1 = load_phase1(model)
    rng = np.random.default_rng(seed)
    base = p1.cloud
    Nc = base.shape[0]
    eigs = p1.eigenvalues
    if len(eigs) < dim:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(dim - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:dim], 1e-6))
    extra = rng.normal(size=(Nc, dim - 3)) * extra_scale[None, :]
    return np.concatenate([base, extra], axis=1), p1


def build_anchored_library(M, points: np.ndarray, anchor_idx: np.ndarray,
                             T: int, dl: float, phi_rank: int) -> Library:
    """One record per anchor; geodesic shoots in zero-tangent direction
    (we only need Φ at λ=0..T to correct nearby points)."""
    n = points.shape[1]
    rng = np.random.default_rng(20260427)
    lib = Library()
    for i in anchor_idx:
        x0 = points[int(i)]
        v0 = normalise_to_unit_speed(M, x0, rng.normal(size=(n,)))
        xs, vs = integrate_geodesic(M, x0, v0, dl=dl, T=T)
        bank = build_propagator(M, xs, vs, dl=dl)
        lib.add(q=x0, v0=v0, xs=xs, Phi=bank.Phi, rho=3.0,
                g_q=M.g_at(x0), phi_rank=phi_rank)
    return lib


def substitute_run(model: str, dim: int = 8, cache_fraction: float = 0.25,
                    eps_star: float = 3.0, rho: float = 0.4,
                    T: int = 8, dl: float = 0.1,
                    phi_rank: int = 5, t_full_ms_per_token: float = 29.5,
                    seed: int = 20260427) -> dict:
    """Three-bucket decode-substitution benchmark.

    For each held-out cloud point treated as a decode step:
      bucket A — within ρ of nearest anchor: Jacobi correction is reliable.
                 We measure prediction error vs a real geodesic ground truth.
      bucket B — within ε* but outside ρ: cache hit but correction unreliable;
                 step would fall back to a fresh O(nk²) geodesic solve.
      bucket C — outside ε*: cache miss; full O(d·L) forward pass.
    """
    points, p1 = _intrinsic_lift(model, dim, seed=seed)
    Nc = points.shape[0]
    M = fit_phase3_manifold(model, n_intrinsic=dim, sigma=0.6, n_grid=Nc)
    rng = np.random.default_rng(seed)

    k = max(1, int(round(cache_fraction * Nc)))
    perm = rng.permutation(Nc)
    anchor_idx = perm[:k]
    decode_idx = perm[k:]

    t0 = time.perf_counter()
    lib = build_anchored_library(M, points, anchor_idx, T=T, dl=dl,
                                  phi_rank=phi_rank)
    t_build = time.perf_counter() - t0

    pred_err_correctable = []
    g_dist   = np.zeros(len(decode_idx))
    n_correctable = 0
    n_cache_hit   = 0
    t_lookup = []
    t_correct = []
    t_truth  = []

    for j, idx in enumerate(decode_idx):
        p = points[int(idx)]
        g_at_p = M.g_at(p)

        t0 = time.perf_counter()
        a_idx, d = lib.lookup(p, top_k=8, g_query=g_at_p)
        t_lookup.append(time.perf_counter() - t0)
        g_dist[j] = d

        if d <= eps_star:
            n_cache_hit += 1
        rec = lib.records[a_idx]
        dq = p - rec.q

        t0 = time.perf_counter()
        U = rec.Phi_U[-1]; V = rec.Phi_V[-1]
        terminal_cached = rec.xs[-1].astype(np.float64)
        p_hat = terminal_cached + U @ (V.T @ dq)
        t_correct.append(time.perf_counter() - t0)

        if d <= rho:
            n_correctable += 1
            t0 = time.perf_counter()
            xs_true, _ = integrate_geodesic(M, p, rec.v0, dl=dl, T=T)
            p_true = xs_true[-1]
            t_truth.append(time.perf_counter() - t0)

            diff = p_hat - p_true
            g_t = M.g_at(p_true)
            err_g = float(np.sqrt(max(diff @ g_t @ diff, 0.0)))
            denom = max(float(np.sqrt(p_true @ g_t @ p_true)), 1e-12)
            pred_err_correctable.append(err_g / denom)

    n_steps = len(decode_idx)
    hit_rate    = n_cache_hit / n_steps
    correct_rate = n_correctable / n_steps
    if pred_err_correctable:
        pe = np.asarray(pred_err_correctable)
        mean_pe = float(pe.mean()); p50_pe = float(np.quantile(pe, 0.5))
        p95_pe  = float(np.quantile(pe, 0.95))
    else:
        mean_pe = p50_pe = p95_pe = float("nan")

    t_full_s = t_full_ms_per_token / 1000.0
    t_jacobi_per_q = float(np.mean(t_correct))
    t_lookup_per_q = float(np.mean(t_lookup))
    t_truth_per_q  = float(np.mean(t_truth)) if t_truth else float("nan")

    f_corr = correct_rate
    f_hit_only = max(0.0, hit_rate - correct_rate)
    f_miss = 1.0 - hit_rate
    t_step_avg = (
        f_corr     * (t_lookup_per_q + t_jacobi_per_q) +
        f_hit_only * (t_lookup_per_q + 0.5 * t_full_s) +
        f_miss     * (t_lookup_per_q + t_full_s)
    )
    speedup_avg = t_full_s / max(t_step_avg, 1e-12)
    tps_baseline = 1.0 / t_full_s
    tps_gtc      = 1.0 / max(t_step_avg, 1e-12)

    return {
        "model": model, "n_intrinsic": dim, "Nc": Nc,
        "cache_fraction": cache_fraction, "k_anchors": int(k),
        "n_decode_steps": int(n_steps),
        "eps_star": eps_star, "rho_validity": rho,
        "T": T, "dl": dl, "phi_rank": phi_rank,
        "build_wall_s": round(t_build, 3),
        "library_records": len(lib.records),
        "cache_hit_rate_eps_star": hit_rate,
        "correctable_rate_within_rho": correct_rate,
        "fraction_hit_but_uncorrectable": f_hit_only,
        "fraction_miss": f_miss,
        "mean_g_distance": float(g_dist.mean()),
        "p95_g_distance": float(np.quantile(g_dist, 0.95)),
        "pred_err_correctable_mean": mean_pe,
        "pred_err_correctable_p50": p50_pe,
        "pred_err_correctable_p95": p95_pe,
        "lookup_us_per_step": round(t_lookup_per_q * 1e6, 3),
        "correct_us_per_step": round(t_jacobi_per_q * 1e6, 3),
        "ground_truth_geodesic_us_per_step": round(t_truth_per_q * 1e6, 3) if not np.isnan(t_truth_per_q) else None,
        "gtc_step_us_on_hit": round((t_lookup_per_q + t_jacobi_per_q) * 1e6, 3),
        "baseline_full_step_ms": t_full_ms_per_token,
        "projected_avg_step_ms": round(t_step_avg * 1e3, 4),
        "projected_speedup_avg": round(speedup_avg, 3),
        "baseline_tps": round(tps_baseline, 1),
        "projected_gtc_tps": round(tps_gtc, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--cache-fraction", type=float, default=0.25)
    ap.add_argument("--eps-star", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=0.4,
                    help="Validity radius — Jacobi correction trusted within this g-norm distance.")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--dl", type=float, default=0.1)
    ap.add_argument("--phi-rank", type=int, default=5)
    # Real measured baselines (geodessical.exe --ppl-eval), ms per token:
    #   Llama-3.1-8B Q4_K_M : 29.5 ms (33.9 tps)
    #   SmolLM2-135M Q8_0   : ~5.0 ms (~200 tps)
    #   Gemma-4 E2B         : ~9.3 ms (107.7 tps)
    ap.add_argument("--t-full-ms", type=float, default=29.5,
                    help="Full forward pass wall-clock in ms (default: Llama-3.1-8B baseline).")
    args = ap.parse_args()

    out = substitute_run(args.model, dim=args.dim,
                         cache_fraction=args.cache_fraction,
                         eps_star=args.eps_star, rho=args.rho,
                         T=args.T, dl=args.dl,
                         phi_rank=args.phi_rank,
                         t_full_ms_per_token=args.t_full_ms)
    out_path = REPO / "docs" / "figures" / "gtc" / f"{args.model}_decode_substitution.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[decode-sub] model={args.model} k={out['k_anchors']}/{out['Nc']} "
          f"steps={out['n_decode_steps']} eps_star={args.eps_star} rho={args.rho}")
    print(f"  cache hit rate (≤ε*) : {out['cache_hit_rate_eps_star']:.1%}")
    print(f"  correctable (≤ρ)     : {out['correctable_rate_within_rho']:.1%}")
    print(f"  hit-but-uncorr       : {out['fraction_hit_but_uncorrectable']:.1%}")
    print(f"  miss                 : {out['fraction_miss']:.1%}")
    print(f"  pred_err on correctable : "
          f"mean={out['pred_err_correctable_mean']:.3%} "
          f"p50={out['pred_err_correctable_p50']:.3%} "
          f"p95={out['pred_err_correctable_p95']:.3%}")
    print(f"  lookup µs/step       : {out['lookup_us_per_step']}")
    print(f"  correct µs/step      : {out['correct_us_per_step']}")
    print(f"  gtc step on hit      : {out['gtc_step_us_on_hit']} µs")
    print(f"  baseline full step   : {out['baseline_full_step_ms']} ms")
    print(f"  projected avg step   : {out['projected_avg_step_ms']} ms")
    print(f"  projected speedup    : {out['projected_speedup_avg']}× over baseline")
    print(f"  projected tps        : {out['baseline_tps']} → {out['projected_gtc_tps']}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
