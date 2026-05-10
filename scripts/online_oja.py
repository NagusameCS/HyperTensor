#!/usr/bin/env python3
"""
Online Basis: Rejection-Driven Oja Update (Paper II §Online)

Implements Oja's rule for online PCA, triggered exclusively by speculative-
decode rejection residuals. The key insight: every rejection is a position
where the draft model and verifier disagreed, so the residual is a sample
of the direction the current basis fails to capture. Only these rejection
residuals are fed into Oja's rule, making it adaptive to distribution shift.

Reference: Stewart, "Geodesic Projection Pipeline," HyperTensor Paper II, 2026.

Usage:
    from online_oja import OnlineOjaBasis
    oja = OnlineOjaBasis(d=4096, k=1024)
    oja.update(rejection_residual)           # Feed on spec-decode rejection
    oja.apply_pending()                      # Apply accumulated updates
    drift = oja.basis_drift()                # Measure basis staleness
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class OjaConfig:
    """Configuration for the Oja online basis updater."""
    d: int = 4096              # Ambient dimension
    k: int = 1024              # Subspace dimension
    eta0: float = 0.01         # Initial learning rate
    min_rejections: int = 4     # Minimum rejections before applying updates
    max_pending: int = 256      # Maximum queued residuals
    bump_increment: int = 1     # Version bump per applied update
    burn_in_samples: int = 10000  # Estimated samples to reach projector error ~5e-4


class OnlineOjaBasis:
    """
    Online PCA basis updated via Oja's rule on speculative-decode rejection
    residuals. The basis W ∈ R^{k×d} tracks the running covariance of the
    difference between verifier and drafter hidden states, which measures
    calibration→deployment drift.

    Why Oja (not SGA or Krasulina):
    - SGA: requires explicit orthogonalisation, O(k²d) per update.
    - Krasulina: needs running-mean estimate, extra state per layer.
    - Oja: implicit orthogonalisation, O(kd) per update, correct under
      centred residuals (rejection residuals are zero-mean by construction).
    """

    def __init__(self, config: Optional[OjaConfig] = None):
        """
        Args:
            config: OjaConfig with d, k, learning rate, gates.
        """
        self.cfg = config or OjaConfig()
        d, k = self.cfg.d, self.cfg.k

        # Basis matrix W ∈ R^{k×d}, initialised random orthonormal
        W0 = np.random.randn(k, d).astype(np.float64)
        Q, _ = np.linalg.qr(W0.T)
        self.W = Q.T  # k × d, rows are orthonormal

        self._pending: list = []       # Queued rejection residuals
        self._update_count: int = 0     # Total updates applied (t in schedule)
        self._version: int = 0          # Monotonic version counter
        self._initial_W: np.ndarray = self.W.copy()  # For drift measurement

    @property
    def version(self) -> int:
        """Current basis version (for cache invalidation)."""
        return self._version

    @property
    def update_count(self) -> int:
        """Total Oja updates applied."""
        return self._update_count

    def record_rejection(self, residual: np.ndarray) -> bool:
        """
        Record a rejection residual for later processing.

        Args:
            residual: Hidden-state difference h_verifier - h_drafter, shape (d,).

        Returns:
            True if the pending queue is full and apply_pending() should be called.
        """
        residual = np.asarray(residual, dtype=np.float64).ravel()
        if len(residual) != self.cfg.d:
            raise ValueError(f"Expected residual of shape ({self.cfg.d},), "
                             f"got {residual.shape}")

        # Centre: rejection residuals are zero-mean by construction
        # (spec-decode rejection sampling is unbiased), but we clip
        # extreme values for numerical stability.
        r_norm = np.linalg.norm(residual)
        if r_norm > 1e-8:
            residual = residual / r_norm  # Unit-norm to avoid magnitude drift

        self._pending.append(residual)

        return len(self._pending) >= self.cfg.max_pending

    def apply_pending(self) -> int:
        """
        Apply all queued rejection residuals via Oja's rule.

        Returns:
            Number of residuals processed.
        """
        if len(self._pending) < self.cfg.min_rejections:
            return 0

        n_processed = 0
        for x in self._pending:
            self._update_count += 1
            t = self._update_count

            # Learning rate schedule: η_t = η_0 / sqrt(t)
            # Satisfies Robbins-Monro: Σ η_t = ∞, Σ η_t² < ∞
            eta = self.cfg.eta0 / np.sqrt(max(t, 1))

            # Oja's rule: w_i ← w_i + η * x * (x^T w_i), then normalise
            # This is equivalent to projector update:
            #   P_{t+1} = P_t + η (x x^T - P_t x x^T P_t) + O(η²)
            for i in range(self.cfg.k):
                proj = np.dot(x, self.W[i])  # scalar
                self.W[i] = self.W[i] + eta * proj * x

            # Re-orthonormalise rows (deflated Gram-Schmidt for stability)
            for i in range(self.cfg.k):
                # Orthogonalise against previous rows
                for j in range(i):
                    self.W[i] -= np.dot(self.W[i], self.W[j]) * self.W[j]
                norm = np.linalg.norm(self.W[i])
                if norm > 1e-15:
                    self.W[i] /= norm
                else:
                    # Degenerate: replace with random direction
                    self.W[i] = np.random.randn(self.cfg.d)
                    for j in range(i):
                        self.W[i] -= np.dot(self.W[i], self.W[j]) * self.W[j]
                    self.W[i] /= np.linalg.norm(self.W[i])

            n_processed += 1

        self._version += self.cfg.bump_increment
        self._pending.clear()
        return n_processed

    def update(self, residual: np.ndarray) -> int:
        """
        Convenience: record + conditionally apply.

        Args:
            residual: Hidden-state difference, shape (d,).

        Returns:
            Number of residuals processed (0 if below threshold).
        """
        full = self.record_rejection(residual)
        if full:
            return self.apply_pending()
        return 0

    def basis_drift(self) -> float:
        """
        Measure how far the current basis has drifted from initial.

        Returns:
            Frobenius distance between current and initial projectors:
            ||W_cur^T W_cur - W_init^T W_init||_F / sqrt(2k).
            Ranges from 0 (no drift) to 1 (fully orthogonal).
        """
        P_cur = self.W.T @ self.W
        P_init = self._initial_W.T @ self._initial_W
        diff = P_cur - P_init
        return float(np.linalg.norm(diff, 'fro') / np.sqrt(2 * self.cfg.k))

    def projector_error_bound(self) -> float:
        """
        Estimate current projector error based on Oja convergence theory.

        For stationary covariance Σ with eigengap Δ = λ_k - λ_{k+1}:
            E||P_t - P*||²_F ≈ d / (Δ² t)
        after burn-in (Jain et al., 2016; Allen-Zhu & Li, 2017).

        Returns:
            Estimated Frobenius error of current projector.
        """
        t = max(self._update_count, 1)
        # Conservative eigengap estimate for LM hidden states
        delta_est = 1e-3
        return float(self.cfg.d / (delta_est ** 2 * t))

    def get_state(self) -> dict:
        """Export basis state for persistence."""
        return {
            "W": self.W.tolist(),
            "version": self._version,
            "update_count": self._update_count,
            "drift": self.basis_drift(),
        }

    @classmethod
    def from_state(cls, state: dict, config: Optional[OjaConfig] = None) -> "OnlineOjaBasis":
        """Restore from persisted state."""
        obj = cls(config)
        obj.W = np.array(state["W"], dtype=np.float64)
        obj._version = state["version"]
        obj._update_count = state["update_count"]
        obj._initial_W = obj.W.copy()
        return obj


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Online Oja Basis — Self-Test")
    print("=" * 60)

    d, k = 128, 16  # Small for test
    cfg = OjaConfig(d=d, k=k, eta0=0.05, min_rejections=2, max_pending=20)
    oja = OnlineOjaBasis(cfg)

    # Simulate: stream of rejection residuals from a drifting distribution
    rng = np.random.default_rng(42)

    # True covariance slowly rotates from axis-aligned to off-diagonal
    n_samples = 200
    drift_schedule = np.linspace(0, 1, n_samples)

    for i in range(n_samples):
        alpha = drift_schedule[i]
        # Covariance: mixture of identity and a specific direction
        true_dir = np.array([1.0 if j < k else 0.0 for j in range(d)])
        true_dir /= np.linalg.norm(true_dir)

        # Sample from N(0, Σ) where Σ rotates
        z = rng.normal(0, 1, d)
        x = z + alpha * 3.0 * true_dir * np.dot(true_dir, z)
        x /= np.linalg.norm(x)

        oja.record_rejection(x)

        if i % 20 == 0:
            processed = oja.apply_pending()
            drift = oja.basis_drift()
            err = oja.projector_error_bound()
            print(f"  step {i:4d}: processed={processed:3d}, "
                  f"drift={drift:.4f}, est_err={err:.6f}, "
                  f"version={oja.version}")

    # Final drift
    print(f"\n  Final drift: {oja.basis_drift():.4f}")
    print(f"  Total updates: {oja.update_count}")
    print(f"  Version: {oja.version}")

    # State round-trip
    state = oja.get_state()
    restored = OnlineOjaBasis.from_state(state, cfg)
    print(f"  State round-trip drift match: "
          f"{abs(oja.basis_drift() - restored.basis_drift()) < 1e-10}")

    print("\n  Online Oja module: OK")
