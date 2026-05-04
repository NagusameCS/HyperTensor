"""
gtc/manifold.py
================

Lift the Phase-1 / Phase-3 exports into a continuous Riemannian manifold in
the *intrinsic* dimension (e.g. n=17 for SmolLM2-135M), with a fitted metric
tensor field g(x) and a Christoffel-symbol field Γ(x) computed by finite
differences. This replaces the toy isotropic-diagonal proxy in the v0.1
prototype.

Concretely:

  * The Phase-1 cloud lives in R^3 (visualisation projection). We rebuild a
    pseudo-intrinsic representation by sampling points uniformly on the
    convex hull of the cloud and assigning each a metric tensor that is the
    (smoothed) outer product of local PCA directions. This is a faithful
    Riemannian object even though it is not the same one the runtime caches.
    For the GTC validity experiment that is enough --- we are measuring whether
    Jacobi linearisation is valid on a non-trivially-curved manifold, not
    whether the manifold is *the* one inside Llama.

  * For the synthetic sanity case we construct a unit n-sphere with the round
    metric, where the Jacobi field has a closed-form expression and the
    validity radius can be cross-checked against theory.

Public API:

    Manifold(g, gamma, sample_points, dim) --- frozen dataclass
    fit_phase3_manifold(model, n_intrinsic, sigma, n_grid)
    build_sphere_manifold(n_intrinsic, n_grid, radius=1.0)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from _phase_io import REPO, load_phase1, load_phase3


#  Data class 

@dataclass(frozen=True)
class Manifold:
    """Sampled Riemannian manifold with metric and Christoffel field."""
    sample_points: np.ndarray   # (M, n)
    g_field:        np.ndarray  # (M, n, n) symmetric positive-definite
    gamma_field:    np.ndarray  # (M, n, n, n)  Γ^k_ij with k in last axis
    R_scalar:       np.ndarray  # (M,)
    dim:            int
    name:           str = ""
    interp_sigma:   float = 0.0

    def nearest(self, x: np.ndarray) -> int:
        d = self.sample_points - x[None, :]
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def g_at(self, x: np.ndarray) -> np.ndarray:
        if self.interp_sigma <= 0.0:
            return self.g_field[self.nearest(x)]

        # Smooth local metric interpolant (log-Euclidean RBF blend of SPD samples).
        d = self.sample_points - x[None, :]
        d2 = np.einsum("ij,ij->i", d, d)
        inv_2s2 = 1.0 / max(2.0 * self.interp_sigma * self.interp_sigma, 1e-12)
        w = np.exp(-d2 * inv_2s2)
        ws = float(w.sum())
        if ws < 1e-12:
            return self.g_field[self.nearest(x)]

        accum = np.zeros((self.dim, self.dim), dtype=np.float64)
        for i in range(self.g_field.shape[0]):
            wi = float(w[i])
            if wi < 1e-12:
                continue
            try:
                lg = _spd_log(self.g_field[i] + 1e-9 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                continue
            accum += (wi / ws) * lg
        return _spd_exp(accum)

    def gamma_at(self, x: np.ndarray) -> np.ndarray:
        return self.gamma_field[self.nearest(x)]


#  Christoffel symbols from a metric field 

def christoffel_from_metric(points: np.ndarray, g: np.ndarray, h: float = 1e-2,
                             metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
                             ) -> np.ndarray:
    """
    Γ^k_ij(x) = 1/2 g^{kl}(x) [ ∂_i g_jl + ∂_j g_il − ∂_l g_ij ]

    Computed by central differences. ``metric_fn`` is required: it must accept
    ``x : (n,)`` and return ``g(x) : (n, n)``. We re-evaluate it at perturbed
    points, which is the only way to take partial derivatives consistently
    when the field is given as a non-uniform sample.
    """
    M, n = points.shape
    out = np.zeros((M, n, n, n))
    for p_idx in range(M):
        x = points[p_idx]
        g_x = g[p_idx]
        try:
            g_inv = np.linalg.inv(g_x + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            g_inv = np.linalg.pinv(g_x)
        # Build ∂_a g(x) tensor: (n, n, n) where first axis = direction
        dg = np.zeros((n, n, n))
        for a in range(n):
            ep = x.copy(); ep[a] += h
            em = x.copy(); em[a] -= h
            dg[a] = (metric_fn(ep) - metric_fn(em)) / (2.0 * h)
        # Γ^k_ij = 1/2 g^{kl} (∂_i g_jl + ∂_j g_il − ∂_l g_ij)
        # Index convention here: axis 0 of dg is the differentiation direction.
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    s = 0.0
                    for l in range(n):
                        s += g_inv[k, l] * (dg[i, j, l] + dg[j, i, l] - dg[l, i, j])
                    out[p_idx, i, j, k] = 0.5 * s
    return out


def riemann_scalar_from_christoffel(g_inv: np.ndarray, gamma: np.ndarray,
                                     dgamma: np.ndarray) -> float:
    """
    Compute the scalar curvature R from Christoffel symbols and their derivatives.

    This is the contracted Riemann tensor:
        R^k_ijl = ∂_i Γ^k_jl − ∂_j Γ^k_il + Γ^k_im Γ^m_jl − Γ^k_jm Γ^m_il
        R_jl   = R^i_jil
        R      = g^{jl} R_jl

    ``dgamma`` has shape (n, n, n, n) where axis 0 = differentiation direction
    and the remaining axes are the (i, j, k) of Γ^k_ij.
    """
    n = gamma.shape[0]
    Ricci = np.zeros((n, n))
    for j in range(n):
        for l in range(n):
            for i in range(n):
                # ∂_i Γ^i_jl − ∂_j Γ^i_il
                term1 = dgamma[i, j, l, i]
                term2 = dgamma[j, i, l, i]  # ∂_j Γ^i_il
                # Γ^i_im Γ^m_jl − Γ^i_jm Γ^m_il
                term3 = 0.0
                term4 = 0.0
                for m in range(n):
                    term3 += gamma[i, m, i] * gamma[m, j, l]
                    term4 += gamma[j, m, i] * gamma[m, i, l]
                Ricci[j, l] += term1 - term2 + term3 - term4
    return float(np.einsum("jl,jl->", g_inv, Ricci))


#  Phase-3 backed manifold (intrinsic-dimensional) 

def _smooth_metric_field(seed_points: np.ndarray, seed_g: np.ndarray,
                          sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """RBF-smoothed metric tensor field. Returns g(x) = sum_i w_i(x) g_i / sum w_i."""
    M, n = seed_points.shape
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    def g_of_x(x: np.ndarray) -> np.ndarray:
        d2 = np.einsum("ij,ij->i", seed_points - x[None, :], seed_points - x[None, :])
        w = np.exp(-d2 * inv_2s2)
        ws = w.sum()
        if ws < 1e-12:
            return np.eye(n)
        # Weighted Frechet mean of SPD matrices via affine-invariant log-Euclidean
        # approximation: log-mean (cheap and stable for nearby SPDs).
        accum = np.zeros((n, n))
        for k in range(M):
            if w[k] < 1e-12:
                continue
            try:
                Lk = _spd_log(seed_g[k] + 1e-9 * np.eye(n))
            except np.linalg.LinAlgError:
                continue
            accum += (w[k] / ws) * Lk
        return _spd_exp(accum)

    return g_of_x


def _spd_log(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)) @ V.T


def _spd_exp(L: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    return (V * np.exp(w)) @ V.T


def fit_phase3_manifold(model: str, n_intrinsic: Optional[int] = None,
                         sigma: float = 0.6, n_grid: int = 64,
                         seed: int = 20260427, k_neighbours: int = 8) -> Manifold:
    """Construct an intrinsic-dim manifold from the Phase-1/3 exports.

    The Phase-3 export is in 3-D vis space and the runtime emits only a
    diagonal of g per point with one global Γ --- too thin to give a
    non-trivial manifold directly. We rebuild a faithful Riemannian object
    in three steps:

      1. Lift the Phase-1 cloud to ``n_intrinsic`` dimensions, padding with
         coordinates drawn from the tail of the PCA eigenvalue spectrum.
      2. At each lifted seed point, estimate the metric tensor from the
         **local k-nearest-neighbour covariance** of the cloud --- i.e.
         ``g(x) = Cov_local(x)^{-1}``. This is the classical Mahalanobis
         metric on the embedded data, which is a Fisher-information proxy
         when the data are model activations.
      3. Smooth the resulting tensor field with a log-Euclidean RBF and
         derive Christoffel symbols by central differences.

    The resulting manifold is *not identical* to the one the runtime caches,
    but it is non-trivially curved and constructed only from the Phase-1
    cloud --- which the runtime *does* emit faithfully. This is what makes the
    GTC validity-radius experiment meaningful.
    """
    p1 = load_phase1(model)
    p3 = load_phase3(model)
    n = int(n_intrinsic or p3.dim)
    if n < 3:
        raise ValueError("intrinsic dimension must be >= 3")

    rng = np.random.default_rng(seed)

    base = p1.cloud  # (Nc, 3)
    Nc = base.shape[0]
    eigs = p1.eigenvalues
    if len(eigs) < n:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(n - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:n], 1e-6))
    extra = rng.normal(size=(Nc, n - 3)) * extra_scale[None, :]
    seeds = np.concatenate([base, extra], axis=1)  # (Nc, n)

    # Local k-NN covariance -> metric.
    seed_g = np.zeros((Nc, n, n))
    k_nn = min(k_neighbours, Nc - 1)
    for i in range(Nc):
        diffs = seeds - seeds[i:i + 1]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        nn = np.argsort(d2)[1:1 + k_nn]
        local = seeds[nn] - seeds[i]
        cov = local.T @ local / max(k_nn - 1, 1) + 1e-3 * np.eye(n)
        try:
            g_local = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            g_local = np.linalg.pinv(cov)
        # Symmetrise & ensure SPD
        g_local = 0.5 * (g_local + g_local.T)
        # Clamp eigenvalues for numerical stability
        w, V = np.linalg.eigh(g_local)
        w = np.clip(w, 1e-3, 1e3)
        seed_g[i] = (V * w) @ V.T

    g_of_x = _smooth_metric_field(seeds, seed_g, sigma=sigma)

    # Sample manifold on a quasi-uniform grid (subset of the seeds + interpolated).
    if n_grid >= Nc:
        sample_points = seeds.copy()
    else:
        idx = rng.choice(Nc, size=n_grid, replace=False)
        sample_points = seeds[idx]

    M = sample_points.shape[0]
    g_field = np.stack([g_of_x(sample_points[i]) for i in range(M)])
    gamma_field = christoffel_from_metric(sample_points, g_field, h=1e-2,
                                           metric_fn=g_of_x)

    # Scalar curvature requires ∂Γ; cheap proxy: trace of Γ^2 contracted with g^{-1}.
    R_scalar = np.zeros(M)
    for p_idx in range(M):
        try:
            g_inv = np.linalg.inv(g_field[p_idx] + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            g_inv = np.linalg.pinv(g_field[p_idx])
        Gamma2 = np.einsum("ijk,jik->", gamma_field[p_idx], gamma_field[p_idx])
        R_scalar[p_idx] = float(np.trace(g_inv) * Gamma2 / max(n * n, 1))

    return Manifold(
        sample_points=sample_points, g_field=g_field, gamma_field=gamma_field,
        R_scalar=R_scalar, dim=n, name=f"{model}-intrinsic{n}", interp_sigma=sigma,
    )


#  Constant-curvature sphere (sanity case) 

def build_sphere_manifold(n_intrinsic: int = 3, n_grid: int = 64,
                          radius: float = 1.0, seed: int = 20260427) -> Manifold:
    """
    Round metric on the unit n-sphere in stereographic coordinates from the
    south pole. The metric is

        g_ij(x) = (4 r^4 / (r^2 + |x|^2)^2) δ_ij,

    which has constant sectional curvature K = 1/r^2 and scalar curvature
    R = n(n-1)/r^2.

    This is the closed-form sanity case for Jacobi field validation.
    """
    rng = np.random.default_rng(seed)
    n = int(n_intrinsic)
    # Sample points in a ball of radius 0.6 r in stereographic chart (avoids the
    # north-pole singularity at |x| -> ∞).
    pts = rng.normal(size=(n_grid, n))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= rng.uniform(0.0, 0.6 * radius, size=(n_grid, 1))

    def g_of_x(x: np.ndarray) -> np.ndarray:
        denom = (radius * radius + float(x @ x)) ** 2
        coef = 4.0 * radius ** 4 / max(denom, 1e-12)
        return coef * np.eye(n)

    g_field = np.stack([g_of_x(p) for p in pts])
    gamma_field = christoffel_from_metric(pts, g_field, h=1e-3, metric_fn=g_of_x)

    R_scalar = np.full(n_grid, n * (n - 1) / (radius * radius))

    return Manifold(
        sample_points=pts, g_field=g_field, gamma_field=gamma_field,
        R_scalar=R_scalar, dim=n, name=f"sphere{n}-r{radius}", interp_sigma=0.15 * radius,
    )
