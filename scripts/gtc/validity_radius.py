"""
gtc/validity_radius.py
======================

Sweep perturbation magnitude ``epsilon`` in seed-tangent space and measure the
endpoint error of the **Jacobi-corrected** lookup against the **true** geodesic
re-integration. Reports::

    epsilon_star = max{ epsilon : mean_relative_endpoint_error(epsilon) <= tau }

This is the falsifiable headline number for the GTC prototype. We report
``tau in {0.05, 0.10, 0.20}`` so the reader can pick a tolerance.

Run::

    .venv\\Scripts\\python.exe scripts/gtc/validity_radius.py --model smollm2-135m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _phase_io import REPO, load_phase3
from bake_trajectories import bake, geodesic_step
from jacobi_propagator import apply_correction, build_propagators


def integrate_perturbed(p3, x0, v0, dl, T):
    x, v = x0.copy(), v0.copy()
    path = np.zeros((T + 1, 3))
    path[0] = x
    for t in range(T):
        x, v = geodesic_step(p3, x, v, dl)
        path[t + 1] = x
    return path


def sweep(model: str, n_seeds: int, T: int, dl: float, n_perturb: int,
          eps_grid: list[float], seed: int) -> dict:
    bank = bake(model, n_seeds=n_seeds, T=T, dl=dl, seed=seed)
    p3 = load_phase3(model)

    jb = build_propagators(bank["paths"], bank["veloc"], bank["R_path"],
                           dim=bank["dim"], dl=dl)

    rng = np.random.default_rng(seed + 1)

    results = {str(eps): [] for eps in eps_grid}
    # Sample n_perturb directions per (seed, epsilon).
    for i in range(n_seeds):
        gamma_true_endpoint = bank["paths"][i, -1]
        for eps in eps_grid:
            errs = []
            for _ in range(n_perturb):
                d0 = rng.normal(size=3)
                d0 *= eps / max(np.linalg.norm(d0), 1e-12)
                # True endpoint with perturbed seed
                true_path = integrate_perturbed(
                    p3,
                    bank["seeds"][i] + d0,
                    bank["veloc"][i, 0],
                    dl, T,
                )
                true_endpoint = true_path[-1]
                # Jacobi-corrected lookup endpoint
                corrected_endpoint = gamma_true_endpoint + apply_correction(jb.Phi[i, -1], d0)
                num = np.linalg.norm(true_endpoint - corrected_endpoint)
                den = max(np.linalg.norm(true_endpoint - bank["seeds"][i]), 1e-9)
                errs.append(num / den)
            results[str(eps)].append(errs)

    summary = {}
    for eps in eps_grid:
        arr = np.array(results[str(eps)])
        summary[str(eps)] = dict(
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            p95=float(np.quantile(arr, 0.95)),
        )

    # epsilon_star at three thresholds
    star = {}
    for tau in (0.05, 0.10, 0.20):
        best = 0.0
        for eps in eps_grid:
            if summary[str(eps)]["mean"] <= tau:
                best = max(best, eps)
        star[f"tau={tau}"] = best

    return dict(model=model, n_seeds=n_seeds, T=T, dl=dl, n_perturb=n_perturb,
                eps_grid=eps_grid, summary=summary, epsilon_star=star,
                fisher_trace_mean=p3.fisher_trace_mean,
                mean_R=p3.mean_R, dim=p3.dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--n-seeds", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64, dest="T")
    ap.add_argument("--dl", type=float, default=0.05)
    ap.add_argument("--n-perturb", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    eps_grid = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

    out = sweep(args.model, args.n_seeds, args.T, args.dl, args.n_perturb,
                eps_grid, args.seed)

    out_dir = REPO / "docs" / "figures" / "gtc"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.model}_validity_radius.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[gtc/validity] wrote {p}")
    for eps in eps_grid:
        s = out["summary"][str(eps)]
        print(f"  eps={eps:>5}  mean={s['mean']:.3f}  p95={s['p95']:.3f}")
    print(f"  epsilon_star: {out['epsilon_star']}")


if __name__ == "__main__":
    main()
