"""
gtc/geodesic.py
================

RK4 integrator for the full geodesic ODE

    ddot x^k + Γ^k_ij(x) dot x^i dot x^j = 0

on a sampled ``Manifold`` with per-point Christoffel symbols. We evaluate
Γ at the nearest sample point — the fields are smooth so this is acceptable
for short trajectories at the validity-radius scale (we measure the error
introduced by *that* approximation in ``validity_radius.py``).

This replaces the toy diagonal-isotropic step from v0.1.
"""
from __future__ import annotations

import numpy as np

from manifold import Manifold


def _accel(M: Manifold, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ddot x^k = − Γ^k_ij v^i v^j."""
    G = M.gamma_at(x)  # shape (n, n, n) with k as last axis
    return -np.einsum("ijk,i,j->k", G, v, v)


def rk4_step(M: Manifold, x: np.ndarray, v: np.ndarray, dl: float):
    a1 = _accel(M, x, v); k1x = v
    a2 = _accel(M, x + 0.5 * dl * k1x, v + 0.5 * dl * a1); k2x = v + 0.5 * dl * a1
    a3 = _accel(M, x + 0.5 * dl * k2x, v + 0.5 * dl * a2); k3x = v + 0.5 * dl * a2
    a4 = _accel(M, x + dl * k3x, v + dl * a3); k4x = v + dl * a3

    x_next = x + (dl / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (dl / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return x_next, v_next


def integrate_geodesic(M: Manifold, x0: np.ndarray, v0: np.ndarray, dl: float,
                       T: int):
    """Return (T+1, n) array of positions and (T+1, n) array of velocities."""
    n = x0.shape[0]
    xs = np.zeros((T + 1, n)); vs = np.zeros((T + 1, n))
    xs[0] = x0; vs[0] = v0
    x, v = x0.copy(), v0.copy()
    for t in range(T):
        x, v = rk4_step(M, x, v, dl)
        xs[t + 1] = x; vs[t + 1] = v
    return xs, vs


def normalise_to_unit_speed(M: Manifold, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rescale v so that g(x)(v, v) = 1."""
    g = M.g_at(x)
    s2 = float(v @ g @ v)
    if s2 < 1e-12:
        return v
    return v / np.sqrt(s2)
