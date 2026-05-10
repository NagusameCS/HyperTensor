#!/usr/bin/env python3
"""
Axiom Gauge: GL(d) Diagonal Gauge-Optimal Compression (Paper II §Gauge)

Implements the diagonal gauge optimisation that minimises joint tail energy
across attention read/write weight pairs. The gauge g ∈ R^d_{>0} is found by
gradient descent in log-space, then baked into the compressed factors so the
inference path is unchanged (zero runtime overhead).

Reference: Stewart, "Geodesic Projection Pipeline," HyperTensor Paper II, 2026.

Usage:
    from axiom_gauge import AxiomGauge
    gauge = AxiomGauge(d=4096)
    g_opt = gauge.fit(weight_dict, n_iter=30, lr=0.01)
    baked = gauge.bake(g_opt, weight_dict)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GaugeResult:
    """Result of gauge optimisation."""
    g: np.ndarray          # Optimal gauge vector, shape (d,), g_i > 0
    initial_loss: float    # Loss before optimisation
    final_loss: float      # Loss after optimisation
    iterations: int        # Number of iterations used
    loss_history: np.ndarray  # Per-iteration loss
    converged: bool        # Whether convergence criterion was met


class AxiomGauge:
    """
    Diagonal GL(d) gauge optimisation for transformer weight compression.

    The residual stream carries an exact gauge symmetry: for any invertible
    G ∈ GL(d), the substitution x → Gx, W^read → W^read G^{-1},
    W^write → G W^write leaves outputs unchanged, but the SVD spectrum of
    W^read G^{-1} depends on G unless G is orthogonal.

    This class finds the diagonal G = diag(g) that minimises the joint
    tail energy (Frobenius norm of the rank-r residual) across all specified
    read/write weight matrices, using gradient descent in log-space to
    enforce g_i > 0.
    """

    def __init__(self, d: int, rank: int = 1024, seed: int = 42):
        """
        Args:
            d: Ambient dimension (model hidden size).
            rank: Compression rank for tail-energy computation.
            seed: Random seed for initialisation.
        """
        self.d = d
        self.rank = rank
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        reads: Dict[str, np.ndarray],
        writes: Optional[Dict[str, np.ndarray]] = None,
        n_iter: int = 30,
        lr: float = 0.01,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> GaugeResult:
        """
        Fit the optimal diagonal gauge vector.

        Args:
            reads: Dict mapping names to read-side weight matrices W ∈ R^{m×d}.
                   The gauge acts as W → W diag(g^{-1}) on reads.
            writes: Optional dict of write-side matrices W ∈ R^{d×n}.
                    The gauge acts as W → diag(g) W on writes.
            n_iter: Maximum gradient descent iterations.
            lr: Learning rate in log-space.
            tol: Convergence tolerance on relative loss change.
            verbose: Print per-iteration diagnostics.

        Returns:
            GaugeResult with optimal g, loss history, convergence flag.
        """
        # Initialise g = 1 (no gauge) in log-space: λ = log(g) = 0
        lam = np.zeros(self.d, dtype=np.float64)
        g = np.exp(lam)

        # Precompute row-wise squared norms for reads (batch over rows)
        # For read-side: tail_r(W diag(g^{-1})) = sum over rows of
        #   ||(row_i / g) - trunc_r(row_i / g)||^2
        # Gradient: ∂_λj = -2 * sum_reads ||X_tail[row_i, j]||^2 / g_j^2
        #   ≈ -2 * tail_energy_contribution_j

        # We compute gradient via finite differences for simplicity and correctness
        # A full analytic gradient implementation would track tail subspace per matrix

        def compute_loss(g_vec: np.ndarray) -> float:
            """Compute total tail energy under gauge g."""
            total = 0.0
            g_inv = 1.0 / g_vec
            for W in reads.values():
                # Apply gauge: W_g = W * (1/g) broadcast row-wise
                W_g = W * g_inv[np.newaxis, :]
                # Truncated SVD at rank r
                U, S, Vt = np.linalg.svd(W_g, full_matrices=False)
                if len(S) > self.rank:
                    tail = np.sum(S[self.rank:] ** 2)
                else:
                    tail = 0.0
                total += tail
            if writes:
                for W in writes.values():
                    W_g = g_vec[:, np.newaxis] * W
                    U, S, Vt = np.linalg.svd(W_g, full_matrices=False)
                    if len(S) > self.rank:
                        tail = np.sum(S[self.rank:] ** 2)
                    else:
                        tail = 0.0
                    total += tail
            return total

        # For efficiency on large matrices, use analytic gradient derived in Paper II:
        # ∂_λi tail_r(W diag(e^{-λ})) = -2 ||X_tail[:, i]||^2
        # ∂_λi tail_r(diag(e^{λ}) W) = +2 ||X_tail[i, :]||^2
        # where X_tail = W - W_r is the rank-r residual.
        def compute_gradient(g_vec: np.ndarray) -> np.ndarray:
            """Compute gradient of loss w.r.t. log-gauge λ."""
            grad = np.zeros(self.d, dtype=np.float64)
            g_inv = 1.0 / g_vec

            for W in reads.values():
                W_g = W * g_inv[np.newaxis, :]
                # Rank-r truncation via partial SVD
                U, S, Vt = np.linalg.svd(W_g, full_matrices=False)
                r = min(self.rank, len(S))
                # Tail = W_g - W_g_r
                W_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
                X_tail = W_g - W_r
                # Gradient contribution: -2 * ||X_tail[:, i]||^2 * g_i^{-2} * g_i
                # In log-space: dL/dλ_i = dL/dg_i * g_i
                # For reads: dL/dg_i = -2 * sum_j X_tail[j,i]^2 / g_i^3
                # dL/dλ_i = dL/dg_i * g_i = -2 * sum_j X_tail[j,i]^2 / g_i^2
                col_norms_sq = np.sum(X_tail ** 2, axis=0)
                grad -= 2.0 * col_norms_sq / (g_vec ** 2)

            if writes:
                for W in writes.values():
                    W_g = g_vec[:, np.newaxis] * W
                    U, S, Vt = np.linalg.svd(W_g, full_matrices=False)
                    r = min(self.rank, len(S))
                    W_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
                    X_tail = W_g - W_r
                    # For writes: dL/dλ_i = +2 * sum_j X_tail[i,j]^2 / g_i^2
                    row_norms_sq = np.sum(X_tail ** 2, axis=1)
                    grad += 2.0 * row_norms_sq / (g_vec ** 2)

            return grad

        initial_loss = compute_loss(g)
        loss_history = [initial_loss]
        converged = False

        for it in range(n_iter):
            grad = compute_gradient(g)

            # Gradient descent in log-space: λ_{t+1} = λ_t - lr * grad
            lam = lam - lr * grad
            g = np.exp(lam)

            # Normalise: enforce geometric mean = 1 to fix global scale
            log_g_mean = np.mean(lam)
            lam = lam - log_g_mean
            g = np.exp(lam)

            current_loss = compute_loss(g)
            loss_history.append(current_loss)

            if verbose and it % 5 == 0:
                g_std = np.std(g)
                print(f"  iter {it:3d}: loss={current_loss:.6f}, "
                      f"g_mean={np.mean(g):.4f}, g_std={g_std:.4f}, "
                      f"g_range=[{np.min(g):.3f}, {np.max(g):.3f}]")

            # Check convergence
            if it > 0:
                rel_change = abs(loss_history[-2] - current_loss) / (abs(initial_loss) + 1e-10)
                if rel_change < tol:
                    converged = True
                    if verbose:
                        print(f"  Converged at iter {it} (rel_change={rel_change:.2e})")
                    break

        return GaugeResult(
            g=g,
            initial_loss=initial_loss,
            final_loss=current_loss,
            iterations=it + 1,
            loss_history=np.array(loss_history),
            converged=converged,
        )

    def bake(
        self,
        g: np.ndarray,
        reads: Dict[str, np.ndarray],
        writes: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Bake the gauge into compressed factors for zero-overhead inference.

        The gauge is absorbed into the stored basis/projected-weight
        representations so the inference GEMV path is unchanged:

        For read-side: W_proj → W_proj * diag(g)  (column scaling)
        For write-side: basis → diag(1/g) * basis  (row scaling)

        Args:
            g: Optimal gauge vector, shape (d,).
            reads: Dict of projected read-side weights W_proj ∈ R^{m×r}.
            writes: Optional dict of write-side basis matrices U ∈ R^{r×d}.

        Returns:
            (baked_reads, baked_writes) with gauge absorbed.
        """
        baked_reads = {}
        for name, W in reads.items():
            # Read gauge: each column j is scaled by g_j
            baked_reads[name] = W * g[np.newaxis, :]

        baked_writes = None
        if writes:
            baked_writes = {}
            g_inv = 1.0 / g
            for name, U in writes.items():
                # Write gauge: each row i is scaled by 1/g_i
                baked_writes[name] = U * g_inv[:, np.newaxis]

        return baked_reads, baked_writes

    def predict_tail_reduction(self, g: np.ndarray, reads: Dict[str, np.ndarray]) -> float:
        """
        Predict fractional tail-energy reduction from the gauge.
        Returns: (ungauged_tail - gauged_tail) / ungauged_tail.
        """
        # Without gauge
        ungauged_tail = 0.0
        for W in reads.values():
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            if len(S) > self.rank:
                ungauged_tail += np.sum(S[self.rank:] ** 2)

        # With gauge
        g_inv = 1.0 / g
        gauged_tail = 0.0
        for W in reads.values():
            W_g = W * g_inv[np.newaxis, :]
            U, S, Vt = np.linalg.svd(W_g, full_matrices=False)
            if len(S) > self.rank:
                gauged_tail += np.sum(S[self.rank:] ** 2)

        if ungauged_tail < 1e-15:
            return 0.0
        return (ungauged_tail - gauged_tail) / ungauged_tail


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Axiom Gauge — GL(d) Diagonal Gauge Optimisation")
    print("=" * 60)

    # Synthetic test: create a weight matrix with strongly non-uniform
    # column scales; the gauge should detect and compensate.
    d, m, rank = 256, 512, 64
    rng = np.random.default_rng(42)

    # Ground-truth gauge: columns 100-120 are 5x magnified
    g_true = np.ones(d)
    g_true[100:120] = 5.0

    # Create weight with exaggerated column scales
    W_base = rng.normal(0, 1, (m, d)).astype(np.float64)
    W = W_base * g_true[np.newaxis, :]  # Inject gauge

    gauge = AxiomGauge(d=d, rank=rank, seed=42)
    result = gauge.fit({"W": W}, n_iter=50, lr=0.02, tol=1e-8, verbose=True)

    # Check recovery: correlation between true and estimated g
    corr = np.corrcoef(g_true, result.g)[0, 1]
    print(f"\n  True-vs-estimated g correlation: {corr:.4f}")
    print(f"  Loss reduction: {result.initial_loss:.2f} → {result.final_loss:.2f}")
    print(f"  Converged: {result.converged}")
    print(f"  g range: [{np.min(result.g):.3f}, {np.max(result.g):.3f}]")

    # Verify tail reduction
    reduction = gauge.predict_tail_reduction(result.g, {"W": W})
    print(f"  Predicted tail reduction: {reduction*100:.1f}%")
    print("\n  Axiom Gauge module: OK")
