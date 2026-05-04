"""
gtc/decode_substitution_dense.py
=================================

Companion to ``decode_substitution.py``. The 64-point Phase-1 cloud is
too sparse: the median nearest-neighbour g-distance (~2-3) is well
above the empirical validity radius ρ≈0.4. This is itself a finding
(GTC needs ~10 denser sampling than the runtime currently exports),
but it doesn't test the actual GTC contract.

This script tests the contract directly: place synthetic queries
*within* ρ of cache anchors (the regime Paper 5 §4.3 specifies), and
measure correction accuracy + wall-clock substitution.

For each anchor q*:
  - sample N queries  q_i = q* + δq_i  with  ‖δq_i‖_g ∈ U(0, ρ)
  - measure prediction error  ‖p_hat - p_true‖_g  /  ‖p_true‖_g
    where p_true is the geodesic of (q_i, v0) and p_hat is q*.xs[T] +
    Φ_T · δq_i.
  - measure wall-clock per substitution.

This is the operational test: "given a query within the validity radius,
what is the actual error and speedup?"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    return np.concatenate([base, extra], axis=1)


def run(model: str, dim: int = 8, n_anchors: int = 12, n_queries_per: int = 32,
         rho_grid=(0.05, 0.10, 0.20, 0.40), T: int = 8, dl: float = 0.05,
         phi_rank: int = 5, t_full_ms_per_token: float = 5.0,
         seed: int = 20260427) -> dict:
    points = _intrinsic_lift(model, dim, seed=seed)
    Nc = points.shape[0]
    M = fit_phase3_manifold(model, n_intrinsic=dim, sigma=0.6, n_grid=Nc)
    rng = np.random.default_rng(seed)
    anchor_idx = rng.choice(Nc, size=min(n_anchors, Nc), replace=False)

    # Build records & propagators
    records = []
    t_build0 = time.perf_counter()
    for i in anchor_idx:
        x0 = points[int(i)]
        v0 = normalise_to_unit_speed(M, x0, rng.normal(size=(dim,)))
        xs, vs = integrate_geodesic(M, x0, v0, dl=dl, T=T)
        bank = build_propagator(M, xs, vs, dl=dl)
        records.append({
            "q": x0, "v0": v0, "xs": xs, "Phi_T": bank.Phi[-1],
            "g_q": M.g_at(x0),
        })
    t_build = time.perf_counter() - t_build0

    rho_results = []
    for rho in rho_grid:
        all_err = []
        all_jacobi_us = []
        all_truth_us  = []
        for rec in records:
            g_q = rec["g_q"]
            # Eigendecomposition of g for sampling unit g-vectors
            wvals, wvecs = np.linalg.eigh(g_q)
            wvals = np.clip(wvals, 1e-9, None)
            inv_sqrt = wvecs @ np.diag(1.0 / np.sqrt(wvals)) @ wvecs.T

            for _ in range(n_queries_per):
                # uniform direction in g-norm: unit ambient vec then scale by inv_sqrt(g)
                u = rng.normal(size=(dim,))
                u /= max(np.linalg.norm(u), 1e-12)
                # radius uniform in [0, rho] so that g-norm magnitude is rho_q
                r = rho * float(rng.random())
                dq = inv_sqrt @ u * r

                # Ground truth at parameter T
                t0 = time.perf_counter()
                xs_true, _ = integrate_geodesic(M, rec["q"] + dq, rec["v0"], dl=dl, T=T)
                p_true = xs_true[-1]
                all_truth_us.append((time.perf_counter() - t0) * 1e6)

                # Jacobi prediction
                t0 = time.perf_counter()
                p_hat = rec["xs"][-1] + rec["Phi_T"] @ dq
                all_jacobi_us.append((time.perf_counter() - t0) * 1e6)

                diff = p_hat - p_true
                g_t = M.g_at(p_true)
                err_g = float(np.sqrt(max(diff @ g_t @ diff, 0.0)))
                denom = max(float(np.sqrt(p_true @ g_t @ p_true)), 1e-12)
                all_err.append(err_g / denom)

        err = np.asarray(all_err)
        jt  = np.asarray(all_jacobi_us)
        tt  = np.asarray(all_truth_us)
        rho_results.append({
            "rho": rho,
            "n_samples": int(err.size),
            "pred_err_mean": float(err.mean()),
            "pred_err_p50": float(np.quantile(err, 0.5)),
            "pred_err_p95": float(np.quantile(err, 0.95)),
            "pred_err_max": float(err.max()),
            "jacobi_us_mean": float(jt.mean()),
            "ground_truth_us_mean": float(tt.mean()),
            "speedup_vs_geodesic_solve": float(tt.mean() / max(jt.mean(), 1e-12)),
        })

    # Wall-clock projection assuming 100% correctable (within ρ) and a real
    # measured baseline forward pass time.
    t_full_s = t_full_ms_per_token / 1000.0
    best_jacobi_us = min(r["jacobi_us_mean"] for r in rho_results)
    speedup_vs_baseline = (t_full_s * 1e6) / best_jacobi_us

    out = {
        "model": model, "n_intrinsic": dim,
        "n_anchors": int(len(records)),
        "n_queries_per_anchor": n_queries_per,
        "T": T, "dl": dl, "phi_rank": phi_rank,
        "build_wall_s": round(t_build, 3),
        "rho_sweep": rho_results,
        "best_jacobi_us": best_jacobi_us,
        "baseline_full_step_ms": t_full_ms_per_token,
        "projected_speedup_per_correctable_step":
            round(speedup_vs_baseline, 1),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--queries-per", type=int, default=32)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--dl", type=float, default=0.05)
    ap.add_argument("--phi-rank", type=int, default=5)
    ap.add_argument("--t-full-ms", type=float, default=5.0)
    args = ap.parse_args()

    out = run(args.model, dim=args.dim, n_anchors=args.anchors,
              n_queries_per=args.queries_per, T=args.T, dl=args.dl,
              phi_rank=args.phi_rank, t_full_ms_per_token=args.t_full_ms)

    out_path = REPO / "docs" / "figures" / "gtc" / f"{args.model}_decode_substitution_dense.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[decode-sub-dense] model={args.model} anchors={out['n_anchors']} "
          f"q/anchor={args.queries_per} T={args.T} dl={args.dl}")
    print(f"  {'rho':>6}  {'mean.err':>10} {'p50':>10} {'p95':>10} "
          f"{'jacobi_µs':>10} {'truth_µs':>10} {'spd vs solve':>12}")
    for r in out["rho_sweep"]:
        print(f"  {r['rho']:>6.3f}  {r['pred_err_mean']:>10.3%} "
              f"{r['pred_err_p50']:>10.3%} {r['pred_err_p95']:>10.3%} "
              f"{r['jacobi_us_mean']:>10.3f} {r['ground_truth_us_mean']:>10.3f} "
              f"{r['speedup_vs_geodesic_solve']:>11.1f}")
    print(f"  best Jacobi µs : {out['best_jacobi_us']:.3f}")
    print(f"  baseline ms    : {out['baseline_full_step_ms']}")
    print(f"  per-step gain  : {out['projected_speedup_per_correctable_step']} "
          f"on correctable steps")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
