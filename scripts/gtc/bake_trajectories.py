"""
gtc/bake_trajectories.py
========================

Integrate ``N`` geodesics from seed points sampled out of the Phase-1 cloud,
under the isotropic-curvature proxy

    ds^2 = sum_a g_diag_a(x) dx_a^2,    R(x) := nearest-neighbour scalar curv.

The geodesic equation in the diagonal-metric, isotropic-curvature regime
reduces to

    ddot x_a + 0.5 * d_a log g_a(x) * dot x_a^2 = 0   (a = 1..3 vis dims)

We approximate ``d_a log g_a`` as a finite difference between the metric at
the current point and at its nearest Phase-3 neighbour, which is what the
Phase-3 export gives us today. RK4 over ``T`` steps, step size ``dl``.

Output: ``docs/figures/gtc/<model>_bank.npz`` with::

    seeds  : (N, 3)       seed positions
    paths  : (N, T+1, 3)  trajectory positions
    veloc  : (N, T+1, 3)  trajectory velocities
    R_path : (N, T+1)     scalar curvature along path
    g_path : (N, T+1, 3)  diagonal metric along path

This is a deliberately small, deterministic baseline. It is not a faithful
forward of the LLM; it is a faithful forward of the *manifold the runtime
caches*.

Run::

    .venv\\Scripts\\python.exe scripts/gtc/bake_trajectories.py --model smollm2-135m
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _phase_io import REPO, Phase3, load_phase1, load_phase3, nearest_point


def metric_at(p3: Phase3, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (g_diag(x), R(x)) using nearest Phase-3 neighbour."""
    pt = nearest_point(p3, x)
    return pt.g_diag.copy(), pt.R


def grad_log_g_diag(p3: Phase3, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Central-difference approximation to d_a log g_a(x), per axis."""
    out = np.zeros(3)
    for a in range(3):
        ep = x.copy(); ep[a] += h
        em = x.copy(); em[a] -= h
        gp, _ = metric_at(p3, ep)
        gm, _ = metric_at(p3, em)
        # log-derivative of the diagonal entry along its own axis
        out[a] = (np.log(max(gp[a], 1e-12)) - np.log(max(gm[a], 1e-12))) / (2 * h)
    return out


def geodesic_step(p3: Phase3, x: np.ndarray, v: np.ndarray, dl: float) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step of the diagonal-metric geodesic ODE.

    Christoffel terms for a diagonal g with weak cross-coupling are
    approximated by the dominant ``-0.5 * d_a log g_a * v_a^2`` term, which is
    the leading contribution at low speed.
    """
    def accel(xx, vv):
        g_log_grad = grad_log_g_diag(p3, xx)
        return -0.5 * g_log_grad * vv * vv  # Hadamard squared

    k1v = accel(x, v); k1x = v
    k2v = accel(x + 0.5 * dl * k1x, v + 0.5 * dl * k1v); k2x = v + 0.5 * dl * k1v
    k3v = accel(x + 0.5 * dl * k2x, v + 0.5 * dl * k2v); k3x = v + 0.5 * dl * k2v
    k4v = accel(x + dl * k3x, v + dl * k3v); k4x = v + dl * k3v

    x_next = x + (dl / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (dl / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_next, v_next


def bake(model: str, n_seeds: int, T: int, dl: float, seed: int) -> dict:
    p1 = load_phase1(model)
    p3 = load_phase3(model)
    rng = np.random.default_rng(seed)

    cloud = p1.cloud
    if n_seeds > cloud.shape[0]:
        raise ValueError(f"only {cloud.shape[0]} cloud points available for {model}")
    idx = rng.choice(cloud.shape[0], size=n_seeds, replace=False)
    seeds = cloud[idx]

    paths  = np.zeros((n_seeds, T + 1, 3))
    veloc  = np.zeros((n_seeds, T + 1, 3))
    R_path = np.zeros((n_seeds, T + 1))
    g_path = np.zeros((n_seeds, T + 1, 3))

    for i, s in enumerate(seeds):
        # Initial velocity: unit-norm, sampled uniformly in tangent space (3-D vis).
        v0 = rng.normal(size=3); v0 /= max(np.linalg.norm(v0), 1e-12)
        x, v = s.copy(), v0
        paths[i, 0] = x; veloc[i, 0] = v
        g0, R0 = metric_at(p3, x)
        g_path[i, 0] = g0; R_path[i, 0] = R0
        for t in range(T):
            x, v = geodesic_step(p3, x, v, dl)
            paths[i, t + 1] = x
            veloc[i, t + 1] = v
            g_t, R_t = metric_at(p3, x)
            g_path[i, t + 1] = g_t
            R_path[i, t + 1] = R_t

    return dict(seeds=seeds, paths=paths, veloc=veloc, R_path=R_path, g_path=g_path,
                model=model, dim=p3.dim, dl=dl, T=T, n_seeds=n_seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--steps", type=int, default=128, dest="T")
    ap.add_argument("--dl", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    bank = bake(args.model, args.n_seeds, args.T, args.dl, args.seed)

    out_dir = REPO / "docs" / "figures" / "gtc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model}_bank.npz"
    np.savez_compressed(out, **{k: v for k, v in bank.items() if isinstance(v, np.ndarray)},
                        meta=np.array([bank["model"], bank["dim"], bank["dl"], bank["T"], bank["n_seeds"]],
                                      dtype=object))
    print(f"[gtc/bake] wrote {out} "
          f"(N={bank['n_seeds']}, T={bank['T']}, dl={bank['dl']}, "
          f"mean|R|={float(np.mean(np.abs(bank['R_path']))):.3f})")


if __name__ == "__main__":
    main()
