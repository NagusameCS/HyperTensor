"""
gtc/batch_jacobi.py
====================

Replicates Paper 5 §4.5 "Resonance" Tests 4a-4c on the real SmolLM2-135M
manifold. The claim is that the Jacobi correction over a batch of B
queries is

    [δx_1 | … | δx_B] = J · [δq_1 | … | δq_B]

a single (n  n) by (n  B) matmul of cost O(n² B), which on a vectorised
backend should equal the cost of B sequential matvecs but with much better
memory access pattern.

We measure:
  - sequential time:  sum over i of  (n  n) @ (n,)
  - batched time:     single (n  n) @ (n  B)
  - reconstruction error: should be ≤ float32 roundoff

Output: docs/figures/gtc/<model>_batch_jacobi.json.
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


def bench_batch(M, x0: np.ndarray, T: int = 16, dl: float = 0.1,
                B_grid=(1, 10, 100, 1000, 10000), n_warmup: int = 3,
                seed: int = 20260427):
    n = x0.shape[0]
    rng = np.random.default_rng(seed)

    v0 = normalise_to_unit_speed(M, x0, rng.normal(size=(n,)))
    xs, vs = integrate_geodesic(M, x0, v0, dl=dl, T=T)
    bank = build_propagator(M, xs, vs, dl=dl)
    Phi_T = bank.Phi[-1]  # (n, n) at final waypoint

    results = []
    for B in B_grid:
        # Same random perturbations for both modes
        Dq = rng.normal(scale=0.05, size=(n, B))

        # Warmup
        for _ in range(n_warmup):
            _ = Phi_T @ Dq
            for j in range(min(B, 8)):
                _ = Phi_T @ Dq[:, j]

        # Sequential (one matvec per query)
        t0 = time.perf_counter()
        Dx_seq = np.zeros_like(Dq)
        for j in range(B):
            Dx_seq[:, j] = Phi_T @ Dq[:, j]
        t_seq = time.perf_counter() - t0

        # Batched (single matmul)
        t0 = time.perf_counter()
        Dx_bat = Phi_T @ Dq
        t_bat = time.perf_counter() - t0

        rel_err = float(np.linalg.norm(Dx_seq - Dx_bat) /
                        max(np.linalg.norm(Dx_seq), 1e-12))
        speedup = t_seq / max(t_bat, 1e-12)

        results.append({
            "B": int(B),
            "t_sequential_s": t_seq,
            "t_batched_s": t_bat,
            "speedup": round(speedup, 3),
            "matmul_relerr": rel_err,
            "us_per_query_seq": round(t_seq / B * 1e6, 3),
            "us_per_query_bat": round(t_bat / B * 1e6, 4),
        })
    return results


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    args = ap.parse_args()

    points = _intrinsic_lift(args.model, args.dim)
    M = fit_phase3_manifold(args.model, n_intrinsic=args.dim, sigma=0.6,
                            n_grid=points.shape[0])

    t0 = time.time()
    rows = bench_batch(M, points[0], B_grid=(1, 10, 100, 1000, 10000))
    wall = time.time() - t0

    summary = {
        "model": args.model, "n_intrinsic": args.dim, "wall_s": round(wall, 2),
        "results": rows,
    }
    out = REPO / "docs" / "figures" / "gtc" / f"{args.model}_batch_jacobi.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[batch-jacobi] model={args.model} dim={args.dim} wall={wall:.2f}s")
    print(f"  {'B':>6} {'t_seq(ms)':>10} {'t_bat(ms)':>10} {'speedup':>8} "
          f"{'us/q seq':>10} {'us/q bat':>10} {'rel.err':>10}")
    for r in rows:
        print(f"  {r['B']:>6} {r['t_sequential_s']*1e3:>10.3f} "
              f"{r['t_batched_s']*1e3:>10.3f} {r['speedup']:>8.2f} "
              f"{r['us_per_query_seq']:>10.3f} {r['us_per_query_bat']:>10.4f} "
              f"{r['matmul_relerr']:>10.2e}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
