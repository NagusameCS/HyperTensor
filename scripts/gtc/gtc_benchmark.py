"""
gtc/gtc_benchmark.py
=====================

Coverage and correction-error benchmark for the Geodesic Trajectory Cache
on a fixed activation cloud.

The runtime already exports a Phase-1 manifold (the 64-point sample cloud
for SmolLM2-135M). We treat this as a stand-in for a stream of decode-time
activations and ask:

  Q1.  If we cache k of these as GTC seed points, what fraction of the
       held-out (Nc − k) activations land within ε⋆ of the cache?
  Q2.  When we lookup the nearest cache point and apply the Jacobi
       propagator Φ(λ) at zero displacement (i.e. the seed itself), what
       is the manifold-induced error to the true held-out activation?
  Q3.  How does coverage scale with k?

This is the empirical realisation of the GTC contract: at the cache size
where coverage clears 10–20 %, the lookup-and-correct path replaces full
decode for that fraction of steps. The GTC v0.2 validity-radius experiment
already established that within-ε⋆ Jacobi correction is below the τ-error
threshold; this script measures **coverage**, the other half of the
hit-rate equation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _phase_io import REPO, load_phase1
from manifold import fit_phase3_manifold


def _intrinsic_lift(model: str, n_intrinsic: int, seed: int):
    """Re-lift the Phase-1 cloud to n_intrinsic dim, identical to manifold.py."""
    p1 = load_phase1(model)
    rng = np.random.default_rng(seed)
    base = p1.cloud
    Nc = base.shape[0]
    eigs = p1.eigenvalues
    if len(eigs) < n_intrinsic:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(n_intrinsic - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:n_intrinsic], 1e-6))
    extra = rng.normal(size=(Nc, n_intrinsic - 3)) * extra_scale[None, :]
    return np.concatenate([base, extra], axis=1)


def coverage_sweep(model: str, n_intrinsic: int = 8,
                    eps_grid=(0.10, 0.40, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
                    cache_fractions=(0.10, 0.25, 0.50, 0.75),
                    n_repeats: int = 16, seed: int = 20260427) -> dict:
    points = _intrinsic_lift(model, n_intrinsic, seed)
    Nc = points.shape[0]
    rng = np.random.default_rng(seed)

    M = fit_phase3_manifold(model, n_intrinsic=n_intrinsic, sigma=0.6, n_grid=Nc)

    out = {"model": model, "n_intrinsic": n_intrinsic, "Nc": Nc,
           "cache_fractions": list(cache_fractions),
           "eps_grid": list(eps_grid), "n_repeats": n_repeats,
           "results": []}

    for f in cache_fractions:
        k = max(1, int(round(f * Nc)))
        for_eps = {f"{eps:.4f}": [] for eps in eps_grid}
        correction_err = []

        for r in range(n_repeats):
            idx = rng.permutation(Nc)
            cache_idx = idx[:k]
            held_idx = idx[k:]
            cache = points[cache_idx]
            held = points[held_idx]

            # Manifold-induced distances from each held-out point to its
            # nearest cache point, using g at the held-out side.
            dists = np.zeros(len(held))
            for i, h in enumerate(held):
                # Pre-screen by Euclidean nearest, refine with g(h)-norm
                d_eu = np.linalg.norm(cache - h[None, :], axis=1)
                j = int(np.argmin(d_eu))
                v = h - cache[j]
                g = M.g_at(h)
                dists[i] = float(np.sqrt(max(v @ g @ v, 0.0)))

            for eps in eps_grid:
                hit_frac = float((dists <= eps).mean())
                for_eps[f"{eps:.4f}"].append(hit_frac)
            # Mean lookup error (already in g-norm)
            correction_err.append(float(dists.mean()))

        bin_stats = {}
        for eps in eps_grid:
            arr = np.asarray(for_eps[f"{eps:.4f}"])
            bin_stats[f"{eps:.4f}"] = {
                "hit_rate_mean": float(arr.mean()),
                "hit_rate_p10": float(np.quantile(arr, 0.10)),
                "hit_rate_p90": float(np.quantile(arr, 0.90)),
            }
        out["results"].append({
            "cache_fraction": f, "k": k,
            "mean_lookup_error": float(np.mean(correction_err)),
            "hit_rates": bin_stats,
        })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    t0 = time.time()
    out = coverage_sweep(args.model, n_intrinsic=args.dim, seed=args.seed)
    out["wall_s"] = round(time.time() - t0, 2)

    out_path = REPO / "docs" / "figures" / "gtc" / f"{args.model}_coverage.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[gtc/benchmark] model={args.model} dim={args.dim} "
          f"Nc={out['Nc']} wall={out['wall_s']}s → {out_path}")
    print(f"  cache_frac  k   ε=0.1  ε=0.4  ε=1.0  ε=2.0  ε=3.0  ε=4.0  ε=6.0  ε=8.0   mean_err")
    for r in out["results"]:
        rates = " ".join(f"{r['hit_rates'][f'{e:.4f}']['hit_rate_mean']:>5.1%}"
                         for e in (0.10, 0.40, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0))
        print(f"  {r['cache_fraction']:.2f}      {r['k']:>3}  {rates}   {r['mean_lookup_error']:.3f}")


if __name__ == "__main__":
    main()
