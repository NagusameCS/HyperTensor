"""
gtc/jacobi.py
==============

Jacobi field along a discretised geodesic, in arbitrary intrinsic dimension.

Given a baked geodesic γ(λ) with positions x_t and velocities v_t, the
Jacobi equation is

    (∇_v ∇_v J)^k + R^k_iml v^i J^m v^l = 0

We discretise this in the *coordinate* basis using the sampled Christoffel
field Γ^k_ij and a finite-difference Riemann tensor

    R^a_bcd ≈ ∂_c Γ^a_bd − ∂_d Γ^a_bc + Γ^a_ce Γ^e_bd − Γ^a_de Γ^e_bc.

The propagator Φ(λ) is the (n × n) matrix sending J(0) to J(λ) at zero
initial covariant velocity. Used by ``validity_radius.py`` and
``gtc_benchmark.py`` as the first-order correction map for off-trajectory
lookups.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifold import Manifold


@dataclass
class JacobiBank:
    Phi:        np.ndarray      # (T+1, n, n) at each tape step
    R_path:     np.ndarray      # (T+1, n, n, n, n) Riemann tensor along path
    speeds:     np.ndarray      # (T+1,) g(v, v)


def riemann_tensor(M: Manifold, x: np.ndarray, h: float = 5e-3) -> np.ndarray:
    """Riemann tensor R^a_bcd at x, in coordinate basis. Shape (n, n, n, n)."""
    n = M.dim
    G_x = M.gamma_at(x)  # (n, n, n) with k last
    # Need ∂_c Γ^a_bd — recompute Γ at perturbed nearest neighbours.
    dG = np.zeros((n, n, n, n))  # (c, i, j, k) with k last (= a)
    for c in range(n):
        ep = x.copy(); ep[c] += h
        em = x.copy(); em[c] -= h
        dG[c] = (M.gamma_at(ep) - M.gamma_at(em)) / (2.0 * h)
    R = np.zeros((n, n, n, n))  # R^a_bcd
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    # Convention: G_x[i, j, k] = Γ^k_ij; dG[c, i, j, k] = ∂_c Γ^k_ij
                    term1 = dG[c, b, d, a]   # ∂_c Γ^a_bd
                    term2 = dG[d, b, c, a]   # ∂_d Γ^a_bc
                    cross = 0.0
                    for e in range(n):
                        cross += G_x[c, e, a] * G_x[b, d, e]
                        cross -= G_x[d, e, a] * G_x[b, c, e]
                    R[a, b, c, d] = term1 - term2 + cross
    return R


def build_propagator(M: Manifold, xs: np.ndarray, vs: np.ndarray, dl: float) -> JacobiBank:
    """Integrate Φ(λ) along the discrete tape using midpoint Magnus."""
    Tp1, n = xs.shape
    Phi = np.zeros((Tp1, n, n))
    Phi[0] = np.eye(n)
    R_path = np.zeros((Tp1, n, n, n, n))
    speeds = np.zeros(Tp1)

    for t in range(Tp1):
        R_path[t] = riemann_tensor(M, xs[t])
        g_t = M.g_at(xs[t])
        speeds[t] = float(vs[t] @ g_t @ vs[t])

    # State y = [J, J'] in R^{2n}; J' here is the coordinate (not covariant)
    # derivative — sufficient for the leading-order propagator.
    state = np.eye(2 * n)[:, :n].copy()  # columns = perturbation directions
    state_full = np.zeros((2 * n, n))
    state_full[:n] = np.eye(n)  # J(0) = I
    # state_full[n:] = 0          # J'(0) = 0

    for t in range(Tp1 - 1):
        v_mid = 0.5 * (vs[t] + vs[t + 1])
        R_mid = 0.5 * (R_path[t] + R_path[t + 1])
        # Build coupling A in 2n×2n: d/dλ [J, J'] = [[0, I], [−K, 0]] [J, J']
        # where K^a_m = R^a_iml v^i v^l (acts on J^m).
        K = np.einsum("aiml,i,l->am", R_mid, v_mid, v_mid)
        A = np.zeros((2 * n, 2 * n))
        A[0:n, n:2 * n] = np.eye(n)
        A[n:2 * n, 0:n] = -K
        E = np.eye(2 * n) + dl * A + 0.5 * (dl ** 2) * (A @ A) \
            + (1.0 / 6.0) * (dl ** 3) * (A @ A @ A)
        state_full = E @ state_full
        Phi[t + 1] = state_full[:n]

    return JacobiBank(Phi=Phi, R_path=R_path, speeds=speeds)


def apply_correction(Phi_lambda: np.ndarray, dx0: np.ndarray) -> np.ndarray:
    return Phi_lambda @ dx0
