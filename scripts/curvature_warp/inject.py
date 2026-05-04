"""
curvature_warp/inject.py
=========================

Knowledge injection by Christoffel warp.

Setup. We have a fitted Manifold M with metric g(x) and Christoffel field
Γ(x). A "fact" is a pair (x_query, x_target) where the model's current
decode activation x_query *should* flow toward x_target under one
geodesic step but currently does not (i.e. the geodesic from x_query in
the model's preferred decode direction misses x_target).

Operation. We inject the fact by locally adding a Gaussian "warp" to
``g`` centred at x_query that lowers the metric distance toward x_target
by a controllable factor α:

    g'(x) = g(x) − α · (1 − exp(−|x − x_query|² / 2σ²)) · w wᵀ

where ``w`` is the (g-normalised) direction toward x_target. The Γ
field is recomputed from g' by the standard formula on the affected
ball.

Test protocol (paper §3 success criterion):

    For 50 synthetic facts on the fitted SmolLM2 manifold:
      pre-injection geodesic-step error to x_target  -> mean ε_pre
      post-injection geodesic-step error             -> mean ε_post
    Success: ε_post < 0.5 · ε_pre and the unperturbed-region p95 error
    delta is below 5 % (no catastrophic spillover).

This file is the runnable harness for that protocol. It is not yet wired
to a live model; that requires a runtime hook to evaluate decode
activations on demand. The synthetic test exercises the geometry
end-to-end and produces a falsifiable number.
"""
from __future__ import annotations

import sys
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gtc"))

from _phase_io import REPO  # noqa: E402
from geodesic import integrate_geodesic, normalise_to_unit_speed  # noqa: E402
from manifold import Manifold, christoffel_from_metric, fit_phase3_manifold, _smooth_metric_field  # noqa: E402


#  Warp 

@dataclass
class Warp:
    centre:    np.ndarray   # (n,)
    direction: np.ndarray   # (n,) g-unit vector at centre (toward target)
    strength:  float        # α ∈ [0, 1]
    sigma:     float        # spatial extent


def apply_warp(M: Manifold, warps: list[Warp]) -> Manifold:
    """Return a new Manifold with each warp summed into g and Γ recomputed."""
    n = M.dim
    sample = M.sample_points
    base_seeds = sample.copy()
    base_g = M.g_field.copy()

    # Build a metric-of-x function that subtracts each warp at x.
    base_g_of_x = _smooth_metric_field(base_seeds, base_g, sigma=0.6)

    def g_warped(x: np.ndarray) -> np.ndarray:
        g = base_g_of_x(x).copy()
        for w in warps:
            d2 = float((x - w.centre) @ (x - w.centre))
            shape = 1.0 - np.exp(-d2 / (2.0 * w.sigma ** 2))
            g -= w.strength * shape * np.outer(w.direction, w.direction)
        # Project to SPD if a warp drove it negative-definite
        s, V = np.linalg.eigh(0.5 * (g + g.T))
        s = np.clip(s, 1e-3, None)
        return (V * s) @ V.T

    g_field = np.stack([g_warped(p) for p in sample])
    gamma_field = christoffel_from_metric(sample, g_field, h=1e-2, metric_fn=g_warped)
    return Manifold(sample_points=sample, g_field=g_field, gamma_field=gamma_field,
                    R_scalar=M.R_scalar, dim=n, name=M.name + "+warp")


def make_warp(M: Manifold, x_query: np.ndarray, x_target: np.ndarray,
               strength: float = 0.4, sigma: float = 0.5) -> Warp:
    direction = x_target - x_query
    g = M.g_at(x_query)
    s2 = float(direction @ g @ direction)
    if s2 < 1e-12:
        return Warp(centre=x_query, direction=np.zeros_like(direction),
                    strength=0.0, sigma=sigma)
    direction = direction / np.sqrt(s2)
    return Warp(centre=x_query, direction=direction,
                 strength=strength, sigma=sigma)


#  Test harness 

def _geodesic_endpoint(M: Manifold, x0: np.ndarray, target: np.ndarray,
                        T: int = 16, dl: float = 0.1) -> np.ndarray:
    """Integrate a geodesic in the direction of target; return endpoint."""
    v0 = target - x0
    v0 = normalise_to_unit_speed(M, x0, v0)
    xs, _ = integrate_geodesic(M, x0, v0, dl=dl, T=T)
    return xs[-1]


def _g_dist(M: Manifold, x: np.ndarray, y: np.ndarray) -> float:
    g = M.g_at(0.5 * (x + y))
    v = y - x
    return float(np.sqrt(max(v @ g @ v, 0.0)))


def run_protocol(model: str = "smollm2-135m", n_intrinsic: int = 8,
                  n_facts: int = 50, strength: float = 0.4, sigma: float = 0.5,
                  T: int = 16, dl: float = 0.1, seed: int = 20260427) -> dict:
    rng = np.random.default_rng(seed)
    M = fit_phase3_manifold(model, n_intrinsic=n_intrinsic, sigma=0.6, n_grid=64)
    pts = M.sample_points
    Nc = pts.shape[0]

    # Build n_facts queries: pick a query point, pick a target that is NOT
    # the geodesic neighbour from the query.
    queries = rng.choice(Nc, size=n_facts, replace=True)
    pre_err = np.zeros(n_facts)
    post_err = np.zeros(n_facts)
    spillover = []  # for each fact, error change at unrelated points

    holdout_idx = rng.choice(Nc, size=min(16, Nc), replace=False)

    t0 = time.time()
    for f in range(n_facts):
        q_idx = int(queries[f])
        x_query = pts[q_idx]
        # Target = a random other cloud point at distance > median g-norm
        d2 = np.einsum("ij,ij->i", pts - x_query, pts - x_query)
        order = np.argsort(d2)
        # Target is the median-far point (gives non-trivial geodesic)
        t_idx = int(order[Nc // 2])
        x_target = pts[t_idx]

        # Pre-injection
        end_pre = _geodesic_endpoint(M, x_query, x_target, T=T, dl=dl)
        pre_err[f] = _g_dist(M, end_pre, x_target)

        # Apply single warp
        w = make_warp(M, x_query, x_target, strength=strength, sigma=sigma)
        M_warped = apply_warp(M, [w])
        end_post = _geodesic_endpoint(M_warped, x_query, x_target, T=T, dl=dl)
        post_err[f] = _g_dist(M_warped, end_post, x_target)

        # Spillover: for each holdout point not in the warp ball, did its
        # geodesic to its own neighbour change?
        sp = []
        for h_idx in holdout_idx:
            if h_idx == q_idx or h_idx == t_idx:
                continue
            x_h = pts[int(h_idx)]
            # If holdout is close enough to the warp centre, skip (expected change)
            if float((x_h - x_query) @ (x_h - x_query)) < (2.0 * sigma) ** 2:
                continue
            x_hn = pts[int(order[Nc // 4])]  # arbitrary other point
            e_b = _g_dist(M, _geodesic_endpoint(M, x_h, x_hn, T=T, dl=dl), x_hn)
            e_a = _g_dist(M_warped, _geodesic_endpoint(M_warped, x_h, x_hn, T=T, dl=dl), x_hn)
            if e_b > 1e-9:
                sp.append(abs(e_a - e_b) / e_b)
        spillover.extend(sp)
    wall = time.time() - t0

    pre_mean = float(pre_err.mean()); post_mean = float(post_err.mean())
    out = {
        "model": model, "n_intrinsic": n_intrinsic, "n_facts": n_facts,
        "strength": strength, "sigma": sigma, "T": T, "dl": dl,
        "wall_s": round(wall, 2),
        "pre_err_mean":  pre_mean,
        "post_err_mean": post_mean,
        "improvement":   1.0 - post_mean / max(pre_mean, 1e-12),
        "pre_err_p95":   float(np.quantile(pre_err, 0.95)),
        "post_err_p95":  float(np.quantile(post_err, 0.95)),
        "spillover_p95": float(np.quantile(spillover, 0.95)) if spillover else None,
        "spillover_mean": float(np.mean(spillover)) if spillover else None,
        "n_spillover":   len(spillover),
        "success_50pct_reduction":      bool(post_mean < 0.5 * pre_mean),
        "success_spillover_under_5pct": bool((np.mean(spillover) if spillover else 1.0) < 0.05),
    }
    return out


def main():
    out = run_protocol()
    out_path = REPO / "docs" / "figures" / "curvature_warp" / "smollm2-135m_protocol.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[curvature_warp/inject] result:")
    for k, v in out.items():
        print(f"  {k:>30} : {v}")


if __name__ == "__main__":
    main()
