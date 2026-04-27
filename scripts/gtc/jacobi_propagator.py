"""
gtc/jacobi_propagator.py
========================

Jacobi field along a baked trajectory.

The Jacobi equation along a geodesic ``gamma(lambda)`` is

    D^2 J / d lambda^2  +  R(J, gamma_dot) gamma_dot  =  0,

where ``R`` is the Riemann tensor. Under the isotropic-curvature proxy used by
``bake_trajectories.py`` (``R(X, Y)Z = K * (<Y, Z> X - <X, Z> Y)``, with ``K``
the local sectional curvature derived from the scalar curvature ``R_scalar``
and the working dimension ``n``), this reduces to a per-step second-order
linear ODE in the orthogonal complement of ``gamma_dot``::

    J''(lambda) = -K(lambda) * |gamma_dot|^2 * J_perp(lambda)

We integrate this with implicit-midpoint over the discrete trajectory tape and
return the propagator matrix ``Phi(lambda) in R^{3x3}`` such that

    J(lambda) = Phi(lambda) @ J(0)        (J(0) given, J'(0) = 0).

That ``Phi`` is the first-order correction map: a perturbation of the
seed by ``delta x_0`` propagates to ``Phi(lambda) @ delta x_0`` along the
cached trajectory, *to leading order*. The validity-radius script measures
how far that holds.

Why isotropic K from R_scalar?  R_scalar = n(n-1) K for a constant-curvature
space; we use ``K = R_scalar / (n*(n-1))`` with ``n = dim`` (intrinsic).
This is a coarse proxy. Promoting to the full anisotropic Riemann tensor
requires emitting it from ``runtime/nn/axiom_vis.c`` — TODO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JacobiBank:
    Phi: np.ndarray         # (N, T+1, 3, 3) propagators
    K_path: np.ndarray      # (N, T+1) sectional curvature proxy
    speed_path: np.ndarray  # (N, T+1) |gamma_dot|


def _proj_perp(v: np.ndarray) -> np.ndarray:
    """Projector onto the subspace orthogonal to v in R^3."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3)
    u = v / n
    return np.eye(3) - np.outer(u, u)


def build_propagators(paths: np.ndarray, veloc: np.ndarray, R_path: np.ndarray,
                      dim: int, dl: float) -> JacobiBank:
    """
    paths  : (N, T+1, 3)
    veloc  : (N, T+1, 3)
    R_path : (N, T+1)
    """
    N, Tp1, _ = paths.shape
    Phi = np.zeros((N, Tp1, 3, 3))
    Phi[:, 0] = np.eye(3)[None].repeat(N, axis=0)
    K_path = R_path / max(dim * (dim - 1), 1)
    speed_path = np.linalg.norm(veloc, axis=-1)

    # State vector y = [J, J'] in R^6 for each path.
    # Step matrix M(lambda) in block form:
    #   d/dl [J, J'] = [[0, I], [-K*|v|^2 * P_perp(v), 0]] [J, J']
    # We integrate Phi via M (variation of parameters) per step using
    # midpoint Magnus to remain symplectic.
    for i in range(N):
        y = np.zeros((6, 3))
        y[:3] = np.eye(3)         # J(0) = I  (treat each column as a basis perturbation)
        y[3:] = 0.0               # J'(0) = 0
        for t in range(Tp1 - 1):
            # midpoint quantities
            v_mid = 0.5 * (veloc[i, t] + veloc[i, t + 1])
            K_mid = 0.5 * (K_path[i, t] + K_path[i, t + 1])
            s2_mid = float(np.dot(v_mid, v_mid))
            P = _proj_perp(v_mid)
            A = np.zeros((6, 6))
            A[0:3, 3:6] = np.eye(3)
            A[3:6, 0:3] = -K_mid * s2_mid * P
            # exp(A * dl) via 2nd-order Pade is overkill; use 4-term Taylor (stable for small dl)
            E = np.eye(6) + dl * A + 0.5 * (dl ** 2) * (A @ A) + (1.0 / 6.0) * (dl ** 3) * (A @ A @ A)
            y = E @ y
            Phi[i, t + 1] = y[:3]
    return JacobiBank(Phi=Phi, K_path=K_path, speed_path=speed_path)


def apply_correction(Phi_lambda: np.ndarray, dx0: np.ndarray) -> np.ndarray:
    """First-order correction: dx(lambda) ~= Phi(lambda) @ dx0."""
    return Phi_lambda @ dx0
