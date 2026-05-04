"""
gtc/validity_radius.py
=======================

Sweep the seed-perturbation magnitude ε on a chosen ``Manifold`` and report

    ε⋆(τ) := max{ ε : mean_relF_endpoint_error(ε) ≤ τ }

for τ ∈ {0.05, 0.10, 0.20}. The "true" trajectory is RK4 on the manifold's
own Christoffel field (``geodesic.integrate_geodesic``); the "predicted"
trajectory is the seed endpoint plus the Jacobi-propagated correction
``Φ(λ_T) · δx₀``. Error is measured in the manifold-induced norm
``√(g(x)(δ, δ))`` at the endpoint, normalised by the geodesic length.

Run::

    .venv\\Scripts\\python.exe scripts/gtc/validity_radius.py --case sphere
    .venv\\Scripts\\python.exe scripts/gtc/validity_radius.py --case smollm2-135m
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _phase_io import REPO
from geodesic import integrate_geodesic, normalise_to_unit_speed
from jacobi import apply_correction, build_propagator
from manifold import Manifold, build_sphere_manifold, fit_phase3_manifold


def _g_norm(g: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(max(v @ g @ v, 0.0)))


def sweep(M: Manifold, n_seeds: int, T: int, dl: float, n_perturb: int,
          eps_grid, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = M.dim

    if n_seeds > M.sample_points.shape[0]:
        n_seeds = M.sample_points.shape[0]
    idx = rng.choice(M.sample_points.shape[0], size=n_seeds, replace=False)
    seeds = M.sample_points[idx]

    summary = {f"{eps:.4f}": [] for eps in eps_grid}

    t0 = time.time()
    for i in range(n_seeds):
        x0 = seeds[i]
        v0 = rng.normal(size=n)
        v0 = normalise_to_unit_speed(M, x0, v0)

        xs, vs = integrate_geodesic(M, x0, v0, dl=dl, T=T)
        jb = build_propagator(M, xs, vs, dl=dl)
        ref_endpoint = xs[-1]

        for eps in eps_grid:
            for _ in range(n_perturb):
                d0 = rng.normal(size=n)
                d0 *= eps / max(np.linalg.norm(d0), 1e-12)
                xs_p, _ = integrate_geodesic(M, x0 + d0, v0, dl=dl, T=T)
                true_end = xs_p[-1]
                pred_end = ref_endpoint + apply_correction(jb.Phi[-1], d0)
                g_end = M.g_at(true_end)
                num = _g_norm(g_end, true_end - pred_end)
                den = max(_g_norm(g_end, true_end - x0), 1e-9)
                summary[f"{eps:.4f}"].append(num / den)
    wall = time.time() - t0

    out = {"manifold": M.name, "dim": M.dim, "n_seeds": n_seeds, "T": T,
           "dl": dl, "n_perturb": n_perturb, "eps_grid": list(eps_grid),
           "wall_s": round(wall, 2), "stats": {}}

    for eps in eps_grid:
        arr = np.asarray(summary[f"{eps:.4f}"])
        out["stats"][f"{eps:.4f}"] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.quantile(arr, 0.95)),
            "n": int(arr.size),
        }

    star = {}
    for tau in (0.05, 0.10, 0.20):
        best = 0.0
        for eps in eps_grid:
            if out["stats"][f"{eps:.4f}"]["mean"] <= tau:
                best = max(best, eps)
        star[f"tau_{tau}"] = best
    out["epsilon_star"] = star

    return out


def make_manifold(case: str, dim: int) -> Manifold:
    if case == "sphere":
        return build_sphere_manifold(n_intrinsic=dim, n_grid=48, radius=1.0)
    return fit_phase3_manifold(case, n_intrinsic=dim, sigma=0.6, n_grid=48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="sphere",
                    help="'sphere' for the sanity case, or a model name like smollm2-135m.")
    ap.add_argument("--dim", type=int, default=0,
                    help="Intrinsic dimension; 0 -> defaults (3 for sphere, 8 for models).")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20, dest="T")
    ap.add_argument("--dl", type=float, default=0.05)
    ap.add_argument("--n-perturb", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    if args.dim <= 0:
        args.dim = 3 if args.case == "sphere" else 6

    M = make_manifold(args.case, args.dim)
    eps_grid = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0, 2.0, 3.0, 5.0)
    out = sweep(M, args.n_seeds, args.T, args.dl, args.n_perturb, eps_grid, args.seed)

    out_dir = REPO / "docs" / "figures" / "gtc"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.case}_validity_radius.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[gtc/validity] case={args.case} dim={args.dim} wall={out['wall_s']}s -> {p}")
    for eps in eps_grid:
        s = out["stats"][f"{eps:.4f}"]
        print(f"  eps={eps:>6.3f}  mean={s['mean']:.3f}  p95={s['p95']:.3f}  (n={s['n']})")
    print(f"  epsilon_star: {out['epsilon_star']}")


if __name__ == "__main__":
    main()
