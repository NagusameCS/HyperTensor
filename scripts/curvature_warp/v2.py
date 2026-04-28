"""
curvature_warp/v2.py
=====================

Curvature-warp v2: compact-support, tangent-projected, SPD-safe metric warp.

v1 failed because the Gaussian rank-1 perturbation bled globally and produced
spillover/NaNs at high strength. v2 changes three things:

1) Compact support bump (C1): zero effect outside radius R.
2) Tangent projection: perturbation direction is projected onto the tangent
   complement of local radial axis (reduces global pull).
3) SPD-safe update in log-Euclidean space: update log(g) then expm back.

The warp at x is
    log g'(x) = log g(x) - alpha * b(r) * P_t(w) P_t(w)^T,
where r = ||x-c||/R, b(r)=(1-r^2)^2 for r<1 else 0, and P_t projects away
local radial direction from centre c.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gtc"))

from _phase_io import REPO  # noqa: E402
from geodesic import integrate_geodesic, normalise_to_unit_speed  # noqa: E402
from manifold import (  # noqa: E402
    Manifold,
    christoffel_from_metric,
    fit_phase3_manifold,
    _smooth_metric_field,
    _spd_log,
    _spd_exp,
)


@dataclass
class WarpV2:
    centre: np.ndarray
    direction: np.ndarray
    alpha: float
    radius: float


def _bump_compact(r: float) -> float:
    if r >= 1.0:
        return 0.0
    t = 1.0 - r * r
    return t * t


def _project_tangent(vec: np.ndarray, radial: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Project vec to tangent of sphere around centre under metric g."""
    rr = float(radial @ g @ radial)
    if rr < 1e-12:
        return vec
    coeff = float(vec @ g @ radial) / rr
    return vec - coeff * radial


def apply_warp_v2(M: Manifold, warps: list[WarpV2], smooth_sigma: float = 0.6) -> Manifold:
    n = M.dim
    sample = M.sample_points
    base_seeds = sample.copy()
    base_g = M.g_field.copy()
    base_g_of_x = _smooth_metric_field(base_seeds, base_g, sigma=smooth_sigma)

    def g_warped(x: np.ndarray) -> np.ndarray:
        g = base_g_of_x(x)
        L = _spd_log(g)

        for w in warps:
            diff = x - w.centre
            d = float(np.linalg.norm(diff))
            r = d / max(w.radius, 1e-9)
            b = _bump_compact(r)
            if b <= 0.0:
                continue

            v = _project_tangent(w.direction, diff, g)
            s2 = float(v @ g @ v)
            if s2 < 1e-12:
                continue
            v = v / np.sqrt(s2)

            # Log-domain symmetric update, compact support
            L = L - (w.alpha * b) * np.outer(v, v)

        gp = _spd_exp(L)
        # final sym/spd cleanup
        s, V = np.linalg.eigh(0.5 * (gp + gp.T))
        s = np.clip(s, 1e-6, None)
        return (V * s) @ V.T

    g_field = np.stack([g_warped(p) for p in sample])
    gamma_field = christoffel_from_metric(sample, g_field, h=1e-2, metric_fn=g_warped)
    return Manifold(sample_points=sample, g_field=g_field, gamma_field=gamma_field,
                    R_scalar=M.R_scalar, dim=n, name=M.name + "+warp_v2")


def make_warp_v2(M: Manifold, x_query: np.ndarray, x_target: np.ndarray,
                 alpha: float = 0.35, radius: float = 1.2) -> WarpV2:
    direction = x_target - x_query
    g = M.g_at(x_query)
    s2 = float(direction @ g @ direction)
    if s2 < 1e-12:
        direction = np.zeros_like(direction)
    else:
        direction = direction / np.sqrt(s2)
    return WarpV2(centre=x_query, direction=direction, alpha=alpha, radius=radius)


def _geodesic_endpoint(M: Manifold, x0: np.ndarray, target: np.ndarray,
                       T: int = 16, dl: float = 0.1) -> np.ndarray:
    v0 = target - x0
    v0 = normalise_to_unit_speed(M, x0, v0)
    xs, _ = integrate_geodesic(M, x0, v0, dl=dl, T=T)
    return xs[-1]


def _g_dist(M: Manifold, x: np.ndarray, y: np.ndarray) -> float:
    g = M.g_at(0.5 * (x + y))
    v = y - x
    return float(np.sqrt(max(v @ g @ v, 0.0)))


def run_protocol_v2(model: str = "smollm2-135m", n_intrinsic: int = 8,
                    n_facts: int = 50, alpha: float = 0.35, radius: float = 1.2,
                    T: int = 16, dl: float = 0.1, seed: int = 20260427) -> dict:
    rng = np.random.default_rng(seed)
    M = fit_phase3_manifold(model, n_intrinsic=n_intrinsic, sigma=0.6, n_grid=64)
    pts = M.sample_points
    Nc = pts.shape[0]

    queries = rng.choice(Nc, size=n_facts, replace=True)
    pre_err = np.zeros(n_facts)
    post_err = np.zeros(n_facts)
    spillover = []

    holdout_idx = rng.choice(Nc, size=min(16, Nc), replace=False)

    t0 = time.time()
    for f in range(n_facts):
        q_idx = int(queries[f])
        x_query = pts[q_idx]
        d2 = np.einsum("ij,ij->i", pts - x_query, pts - x_query)
        order = np.argsort(d2)
        t_idx = int(order[Nc // 2])
        x_target = pts[t_idx]

        end_pre = _geodesic_endpoint(M, x_query, x_target, T=T, dl=dl)
        pre_err[f] = _g_dist(M, end_pre, x_target)

        w = make_warp_v2(M, x_query, x_target, alpha=alpha, radius=radius)
        M_warped = apply_warp_v2(M, [w])
        end_post = _geodesic_endpoint(M_warped, x_query, x_target, T=T, dl=dl)
        post_err[f] = _g_dist(M_warped, end_post, x_target)

        sp = []
        for h_idx in holdout_idx:
            if h_idx == q_idx or h_idx == t_idx:
                continue
            x_h = pts[int(h_idx)]
            if float(np.linalg.norm(x_h - x_query)) < 1.5 * radius:
                continue
            x_hn = pts[int(order[Nc // 4])]
            e_b = _g_dist(M, _geodesic_endpoint(M, x_h, x_hn, T=T, dl=dl), x_hn)
            e_a = _g_dist(M_warped, _geodesic_endpoint(M_warped, x_h, x_hn, T=T, dl=dl), x_hn)
            if e_b > 1e-9:
                sp.append(abs(e_a - e_b) / e_b)
        spillover.extend(sp)

    wall = time.time() - t0
    pre_mean = float(pre_err.mean()); post_mean = float(post_err.mean())
    sp_mean = float(np.mean(spillover)) if spillover else 1.0

    return {
        "model": model, "n_intrinsic": n_intrinsic, "n_facts": n_facts,
        "alpha": alpha, "radius": radius, "T": T, "dl": dl,
        "wall_s": round(wall, 2),
        "pre_err_mean": pre_mean,
        "post_err_mean": post_mean,
        "improvement": 1.0 - post_mean / max(pre_mean, 1e-12),
        "pre_err_p95": float(np.quantile(pre_err, 0.95)),
        "post_err_p95": float(np.quantile(post_err, 0.95)),
        "spillover_mean": sp_mean,
        "spillover_p95": float(np.quantile(spillover, 0.95)) if spillover else None,
        "n_spillover": len(spillover),
        "success_50pct_reduction": bool(post_mean < 0.5 * pre_mean),
        "success_spillover_under_5pct": bool(sp_mean < 0.05),
        "success": bool((post_mean < 0.5 * pre_mean) and (sp_mean < 0.05)),
    }


def main():
    out = run_protocol_v2()
    out_path = REPO / "docs" / "figures" / "curvature_warp" / "smollm2-135m_protocol_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[curvature_warp/v2] result:")
    for k, v in out.items():
        print(f"  {k:>32} : {v}")


if __name__ == "__main__":
    main()
