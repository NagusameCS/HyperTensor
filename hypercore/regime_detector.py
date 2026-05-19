#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................:::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
hypercore/regime_detector.py — RegimeDetector: Geometric Regime Change Detection
=================================================================================
HyperTensor v1.1 | May 18, 2026

Principle 6: Regime Shift Detection.  Monitors whether the manifold itself
is deforming by computing five independent geometric signals from a learned
manifold.  A geometric jury aggregates them into a single Regime Change
Index (RCI) that fires when the manifold's shape changes.

The five signals:
  1. Manifold Deviation  — reconstruction error from the learned basis
  2. Curvature Anomaly    — abrupt change in Riemannian curvature
  3. Neighbor Instability — Jaccard distance between KNN graphs
  4. Spectral Drift       — shift in the graph Laplacian eigenvalue spectrum
  5. Geodesic Misalignment— deviation of actual motion from geodesic prediction

Jury aggregation (Theorem 1, Aggregation Uniqueness):
    RCI  = Σ w_i · s_i
    J    = 1 − ∏(1 − w_i · s_i)          (confidence)

Usage:
    from hypercore.regime_detector import RegimeDetector

    rd = RegimeDetector(intrinsic_dim=12, window_size=252)
    rd.fit(training_trajectories)         # (N, T, D) tensor

    # On each new observation:
    result = rd.check(new_observations)   # (N, D) tensor
    if result.regime_change:
        print(f"Regime change detected! RCI={result.rci:.3f}, J={result.confidence:.3f}")
        for s in result.signals:
            print(f"  {s.name}: {s.normalized:.3f}")
"""

import math, warnings, time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

# ── optional torch support ──────────────────────────────────────────────────
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeSignal:
    """One of the five geometric regime-detection signals.

    Attributes:
        name: human-readable signal name
        raw: un-normalised scalar value (before [0,1] clamping)
        normalized: value in [0, 1] after clamping to baseline range
        weight: jury weight (default 0.2 for uniform)
        fired: whether this signal exceeded its individual threshold
    """
    name: str
    raw: float = 0.0
    normalized: float = 0.0
    weight: float = 0.2
    fired: bool = False

    def __repr__(self) -> str:
        flag = " *** FIRED ***" if self.fired else ""
        return (f"RegimeSignal({self.name}: raw={self.raw:.4f}, "
                f"norm={self.normalized:.4f}, w={self.weight:.3f}){flag}")


@dataclass
class RegimeAssessment:
    """Full regime-detection output for one observation window.

    Attributes:
        rci: Regime Change Index ∈ [0, 1]
        confidence: jury confidence J ∈ [0, 1]
        regime_change: True when RCI > threshold
        signals: list of the five RegimeSignal objects
        timestamp: optional wall-clock or step timestamp
        description: human-readable summary
    """
    rci: float = 0.0
    confidence: float = 0.0
    regime_change: bool = False
    signals: List[RegimeSignal] = field(default_factory=list)
    timestamp: Optional[float] = None
    description: str = ""

    def __repr__(self) -> str:
        status = "REGIME CHANGE" if self.regime_change else "normal"
        return (f"RegimeAssessment(rci={self.rci:.4f}, J={self.confidence:.4f}, "
                f"status={status}, n_signals_fired={sum(1 for s in self.signals if s.fired)})")


# ═══════════════════════════════════════════════════════════════════════════════
#  RegimeDetector
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """Geometric regime-change detector based on manifold deformation.

    Learns a low-dimensional manifold from training trajectories, then
    monitors five independent geometric signals that each capture a
    different aspect of manifold deformation.  A geometric jury (Theorem 1
    aggregation) combines them into a single Regime Change Index.

    Parameters:
        intrinsic_dim: target dimension k for the learned manifold (SVD basis)
        window_size: number of recent observations kept for rolling baselines
        knn_k: number of nearest neighbours for the KNN graph (signal 3)
        threshold: RCI value above which a regime change is declared
        weights: optional 5-vector of signal weights (default: uniform 0.2)
        device: torch device string (ignored if torch unavailable)
    """

    _SIGNAL_NAMES = [
        "manifold_deviation",
        "curvature_anomaly",
        "neighbor_instability",
        "spectral_drift",
        "geodesic_misalignment",
    ]

    def __init__(
        self,
        intrinsic_dim: int = 12,
        window_size: int = 252,
        knn_k: int = 5,
        threshold: float = 0.55,
        weights: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        self.k = intrinsic_dim
        self.window_size = window_size
        self.knn_k = knn_k
        self.threshold = threshold

        # jury weights (default uniform)
        if weights is None:
            self.weights = np.full(5, 0.2, dtype=np.float64)
        else:
            if len(weights) != 5:
                raise ValueError(f"weights must have length 5, got {len(weights)}")
            w = np.asarray(weights, dtype=np.float64)
            self.weights = w / w.sum()  # normalise to sum=1

        self.device = device

        # ── fitted state ──
        self._fitted = False
        self.basis: Optional[np.ndarray] = None          # V ∈ ℝ^{D×k}
        self.mu: Optional[np.ndarray] = None             # μ ∈ ℝ^D
        self.sigma: Optional[np.ndarray] = None          # σ ∈ ℝ^D
        self.baseline_curvature: float = 0.0
        self.baseline_spectrum: Optional[np.ndarray] = None
        self.baseline_neighbors: Optional[np.ndarray] = None  # (N, K) indices

        # ── rolling windows ──
        self._proj_history: deque = deque(maxlen=window_size)   # each entry: (N, k) ndarray
        self._raw_history: deque = deque(maxlen=window_size)    # each entry: (N, D) ndarray
        self._curvature_history: deque = deque(maxlen=window_size)

        # ── signal calibration ranges ──
        # Per-signal [P5, P95] percentiles from training data.
        # Raw values below P5 map to 0, above P95 map to 1, linear in between.
        self._sig_p05: Dict[str, float] = {}
        self._sig_p95: Dict[str, float] = {}

    #  Public API  .........................................................

    def fit(self, trajectories: np.ndarray) -> "RegimeDetector":
        """Learn the normal-regime manifold from training trajectories.

        Args:
            trajectories: (N, T, D) array of N entity trajectories,
                          each of length T in D-dimensional ambient space.

        Returns:
            self (for chaining)
        """
        N, T, D = trajectories.shape
        if N < 3:
            raise ValueError(f"Need at least 3 entities, got {N}")
        if T < 3:
            raise ValueError(f"Need at least 3 timesteps, got {T}")

        # Pool all feature vectors: (N*T, D)
        pooled = trajectories.reshape(-1, D).astype(np.float64)

        # ── normalisation parameters ──
        self.mu = pooled.mean(axis=0)
        self.sigma = pooled.std(axis=0) + 1e-10

        # ── SVD basis ──
        centred = pooled - self.mu
        U, S, Vt = np.linalg.svd(centred.T, full_matrices=False)
        k_eff = min(self.k, len(S) - 1, D)
        if k_eff < self.k:
            warnings.warn(f"Intrinsic dim reduced from {self.k} → {k_eff} due to data rank")
            self.k = k_eff
        self.basis = U[:, :self.k]  # (D, k)

        # Clamp KNN K to N-1
        if self.knn_k >= N:
            self.knn_k = max(1, N - 1)
            warnings.warn(f"knn_k reduced to {self.knn_k} (N={N})")

        # ── project all training points ──
        proj = self._project(pooled)  # (N*T, k)

        # ── baseline curvature (median turning angle over training) ──
        turnings = []
        for t in range(2, T):
            z_pp = proj[t - 2::T]   # (N, k)  ← t-2
            z_p  = proj[t - 1::T]   # (N, k)  ← t-1
            z_c  = proj[t::T]       # (N, k)  ← t
            v1 = z_p - z_pp
            v2 = z_c - z_p
            dot = np.sum(v1 * v2, axis=1)
            n1 = np.linalg.norm(v1, axis=1) + 1e-10
            n2 = np.linalg.norm(v2, axis=1) + 1e-10
            cos_th = np.clip(dot / (n1 * n2), -1.0, 1.0)
            turnings.extend((1.0 - cos_th).tolist())
        self.baseline_curvature = float(np.median(turnings)) if turnings else 0.0

        # ── baseline Laplacian spectrum (from last T steps) ──
        recent = proj[-T * N:].reshape(N, T, self.k) if proj.shape[0] >= T * N else proj.reshape(N, -1, self.k)
        self.baseline_spectrum = self._laplacian_spectrum(recent[:, -1, :])

        # ── baseline neighbour graph (from last timestep) ──
        self.baseline_neighbors = self._knn_indices(proj[-N:].reshape(N, self.k))

        # ── calibrate signal ranges on training data ──
        self._calibrate_signals(trajectories)

        self._fitted = True
        return self

    def check(self, observations: np.ndarray,
              timestamp: Optional[float] = None) -> RegimeAssessment:
        """Evaluate all five signals on a new observation and return assessment.

        Args:
            observations: (N, D) array — one feature vector per entity
            timestamp: optional wall-clock / step timestamp

        Returns:
            RegimeAssessment with RCI, confidence, signal breakdown
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before check()")

        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            observations = observations[np.newaxis, :]
        N, D = observations.shape

        # Normalise and project
        z = self._project(observations)  # (N, k)

        # ── update rolling history ──
        self._proj_history.append(z.copy())
        self._raw_history.append(observations.copy())

        # ── compute all five signals ──
        signals: List[RegimeSignal] = []

        # Signal 1: Manifold Deviation
        s1 = self._signal_manifold_deviation(observations, z)

        # Signal 2: Curvature Anomaly
        s2 = self._signal_curvature_anomaly(z)

        # Signal 3: Neighbor Instability
        s3 = self._signal_neighbor_instability(z)

        # Signal 4: Spectral Drift
        s4 = self._signal_spectral_drift(z)

        # Signal 5: Geodesic Misalignment
        s5 = self._signal_geodesic_misalignment(z)

        signals = [s1, s2, s3, s4, s5]

        # ── jury aggregation ──
        rci, confidence = self._aggregate(signals)

        regime_change = rci >= self.threshold

        # ── human-readable description ──
        fired_names = [s.name for s in signals if s.fired]
        if regime_change:
            desc = (f"Regime change detected (RCI={rci:.3f}, J={confidence:.3f}). "
                    f"Firing signals: {', '.join(fired_names) if fired_names else 'none'}.")
        else:
            desc = f"Normal regime (RCI={rci:.3f}, J={confidence:.3f})."

        return RegimeAssessment(
            rci=rci,
            confidence=confidence,
            regime_change=regime_change,
            signals=signals,
            timestamp=timestamp,
            description=desc,
        )

    def reset(self) -> None:
        """Clear rolling windows (keep fitted manifold)."""
        self._proj_history.clear()
        self._raw_history.clear()
        self._curvature_history.clear()

    #  Signal 1: Manifold Deviation  .......................................

    def _signal_manifold_deviation(self, raw: np.ndarray, proj: np.ndarray) -> RegimeSignal:
        """How far observations lie from the learned manifold."""
        recon = self._reconstruct(proj)          # (N, D)
        raw_norms = np.linalg.norm(raw, axis=1) + 1e-10
        errors = np.linalg.norm(raw - recon, axis=1) / raw_norms
        mean_err = float(np.mean(errors))
        normed = self._normalize("manifold_deviation", mean_err)
        return RegimeSignal(
            name="manifold_deviation", raw=mean_err,
            normalized=normed, weight=self.weights[0],
            fired=normed > 0.5,
        )

    #  Signal 2: Curvature Anomaly  ........................................

    def _signal_curvature_anomaly(self, z: np.ndarray) -> RegimeSignal:
        """Abrupt change in Riemannian sectional curvature.

        Instead of the numerically fragile κ = ||a|| / ||v||², we use the
        **turning angle** between consecutive velocity vectors — a
        scale-invariant measure of how sharply the trajectory is bending:

            cos θ = (v1 · v2) / (||v1|| · ||v2||)
            turning = 1 − cos θ  ∈ [0, 2]

        Anomaly = |mean(turning) − baseline_turning| / baseline_turning.
        """
        if len(self._proj_history) < 3:
            return RegimeSignal(
                name="curvature_anomaly", raw=0.0,
                normalized=0.0, weight=self.weights[1], fired=False,
            )

        proj_hist = list(self._proj_history)
        z_prev2 = proj_hist[-3]   # (N, k)
        z_prev1 = proj_hist[-2]   # (N, k)
        z_curr  = proj_hist[-1]   # (N, k)

        v1 = z_prev1 - z_prev2    # (N, k)
        v2 = z_curr  - z_prev1    # (N, k)

        # Cosine of turning angle per entity
        dot = np.sum(v1 * v2, axis=1)
        n1 = np.linalg.norm(v1, axis=1) + 1e-10
        n2 = np.linalg.norm(v2, axis=1) + 1e-10
        cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
        turning = 1.0 - cos_theta  # 0 = straight, 2 = complete reversal

        mean_turning = float(np.mean(turning))
        self._curvature_history.append(mean_turning)

        if self.baseline_curvature < 1e-10:
            anomaly = mean_turning
        else:
            anomaly = abs(mean_turning - self.baseline_curvature) / (self.baseline_curvature + 1e-10)

        normed = self._normalize("curvature_anomaly", anomaly)
        return RegimeSignal(
            name="curvature_anomaly", raw=float(anomaly),
            normalized=normed, weight=self.weights[1],
            fired=normed > 0.5,
        )

    #  Signal 3: Neighbor Instability  .....................................

    def _signal_neighbor_instability(self, z: np.ndarray) -> RegimeSignal:
        """How much the KNN graph has changed from baseline."""
        if self.baseline_neighbors is None:
            return RegimeSignal(
                name="neighbor_instability", raw=0.0,
                normalized=0.0, weight=self.weights[2], fired=False,
            )

        current_neighbors = self._knn_indices(z)   # (N, K)
        if current_neighbors.shape != self.baseline_neighbors.shape:
            # Different N — skip
            return RegimeSignal(
                name="neighbor_instability", raw=0.0,
                normalized=0.0, weight=self.weights[2], fired=False,
            )

        N = current_neighbors.shape[0]
        jaccards = []
        for i in range(N):
            inter = len(set(current_neighbors[i]) & set(self.baseline_neighbors[i]))
            jaccards.append(inter / self.knn_k)

        mean_jaccard = float(np.mean(jaccards))
        instability = 1.0 - mean_jaccard
        normed = self._normalize("neighbor_instability", instability)
        return RegimeSignal(
            name="neighbor_instability", raw=instability,
            normalized=normed, weight=self.weights[2],
            fired=normed > 0.5,
        )

    #  Signal 4: Spectral Drift  ...........................................

    def _signal_spectral_drift(self, z: np.ndarray) -> RegimeSignal:
        """Shift in the graph Laplacian eigenvalue spectrum."""
        if self.baseline_spectrum is None:
            return RegimeSignal(
                name="spectral_drift", raw=0.0,
                normalized=0.0, weight=self.weights[3], fired=False,
            )

        current_spectrum = self._laplacian_spectrum(z)

        # Align lengths (take min)
        min_len = min(len(current_spectrum), len(self.baseline_spectrum))
        cur = current_spectrum[:min_len]
        base = self.baseline_spectrum[:min_len]

        base_norm = np.linalg.norm(base) + 1e-10
        drift = float(np.linalg.norm(cur - base) / base_norm)
        normed = self._normalize("spectral_drift", drift)
        return RegimeSignal(
            name="spectral_drift", raw=drift,
            normalized=normed, weight=self.weights[3],
            fired=normed > 0.5,
        )

    #  Signal 5: Geodesic Misalignment  ....................................

    def _signal_geodesic_misalignment(self, z: np.ndarray) -> RegimeSignal:
        """Deviation of actual velocity from geodesic prediction."""
        if len(self._proj_history) < 3:
            return RegimeSignal(
                name="geodesic_misalignment", raw=0.0,
                normalized=0.0, weight=self.weights[4], fired=False,
            )

        proj_hist = list(self._proj_history)
        z_prev2 = proj_hist[-3]  # (N, k)
        z_prev1 = proj_hist[-2]  # (N, k)
        z_curr  = proj_hist[-1]  # (N, k)

        # predicted velocity (linear extrapolation)
        v_pred = z_prev1 - z_prev2            # (N, k)
        # actual velocity
        v_actual = z_curr - z_prev1           # (N, k)

        # cosine similarity per entity, then mean
        dot = np.sum(v_pred * v_actual, axis=1)
        norm_p = np.linalg.norm(v_pred, axis=1) + 1e-10
        norm_a = np.linalg.norm(v_actual, axis=1) + 1e-10
        cos_sim = dot / (norm_p * norm_a)

        mean_cos = float(np.mean(np.clip(cos_sim, -1.0, 1.0)))
        misalignment = 1.0 - mean_cos
        normed = self._normalize("geodesic_misalignment", misalignment)
        return RegimeSignal(
            name="geodesic_misalignment", raw=misalignment,
            normalized=normed, weight=self.weights[4],
            fired=normed > 0.5,
        )

    #  Jury Aggregation  ...................................................

    def _aggregate(self, signals: List[RegimeSignal]) -> Tuple[float, float]:
        """Compute RCI and jury confidence from the five signals.

        RCI = Σ w_i · s_i                     (weighted sum)
        J   = 1 − ∏(1 − w_i · s_i)           (Theorem 1 aggregation)
        """
        vals = np.array([s.weight * s.normalized for s in signals])
        rci = float(np.sum(vals))
        confidence = float(max(0.0, min(1.0, 1.0 - np.prod(1.0 - vals + 1e-12))))
        return rci, confidence

    #  Internal helpers  ...................................................

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Project from ambient space ℝ^D → intrinsic space ℝ^k."""
        normed = (x - self.mu) / self.sigma
        return normed @ self.basis   # (..., k)

    def _reconstruct(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct from intrinsic space ℝ^k → ambient space ℝ^D."""
        return (z @ self.basis.T) * self.sigma + self.mu

    def _knn_indices(self, z: np.ndarray) -> np.ndarray:
        """Return (N, K) array of nearest-neighbour indices in intrinsic space."""
        N = z.shape[0]
        k_eff = min(self.knn_k, N - 1)
        if k_eff < 1:
            return np.zeros((N, 0), dtype=np.int64)

        # pairwise distances
        dists = np.sum(z ** 2, axis=1, keepdims=True) \
                + np.sum(z ** 2, axis=1, keepdims=True).T \
                - 2 * (z @ z.T)
        # exclude self
        np.fill_diagonal(dists, np.inf)

        # top-k nearest (smallest distance)
        indices = np.argpartition(dists, k_eff, axis=1)[:, :k_eff]
        return indices

    def _laplacian_spectrum(self, z: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of the normalised graph Laplacian.

        Builds adjacency from RBF kernel on intrinsic distances.
        """
        N = z.shape[0]
        if N < 3:
            return np.zeros(min(N, self.k))

        # RBF adjacency
        dists_sq = np.sum(z ** 2, axis=1, keepdims=True) \
                   + np.sum(z ** 2, axis=1, keepdims=True).T \
                   - 2 * (z @ z.T)
        sigma = np.median(np.sqrt(dists_sq[dists_sq > 0])) if np.any(dists_sq > 0) else 1.0
        if sigma < 1e-10:
            sigma = 1.0
        A = np.exp(-dists_sq / (2.0 * sigma ** 2))
        np.fill_diagonal(A, 0.0)

        # degree matrix
        d = A.sum(axis=1) + 1e-10
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))

        # normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
        L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

        # eigenvalues (symmetric → real)
        eigvals = np.linalg.eigvalsh(L)

        # return top k (smallest) eigenvalues
        return eigvals[:min(self.k, len(eigvals))]

    def _calibrate_signals(self, trajectories: np.ndarray) -> None:
        """Run all signals on training data to compute [P5, P95] ranges."""
        N, T, D = trajectories.shape
        all_raw: Dict[str, List[float]] = {name: [] for name in self._SIGNAL_NAMES}

        # slide through training data
        for t in range(T):
            obs = trajectories[:, t, :]   # (N, D)
            z = self._project(obs)
            self._proj_history.append(z.copy())
            self._raw_history.append(obs.copy())

            if t >= 2:
                s1 = self._signal_manifold_deviation(obs, z)
                s2 = self._signal_curvature_anomaly(z)
                s3 = self._signal_neighbor_instability(z)
                s4 = self._signal_spectral_drift(z)
                s5 = self._signal_geodesic_misalignment(z)
                for s in [s1, s2, s3, s4, s5]:
                    all_raw[s.name].append(s.raw)

        # clear history after calibration
        self._proj_history.clear()
        self._raw_history.clear()
        self._curvature_history.clear()

        for name in self._SIGNAL_NAMES:
            vals = np.array(all_raw[name])
            if len(vals) > 0:
                self._sig_p05[name] = float(np.percentile(vals, 5))
                self._sig_p95[name] = float(np.percentile(vals, 95))
                # Ensure P95 > P05 to avoid division by zero
                if self._sig_p95[name] <= self._sig_p05[name]:
                    self._sig_p95[name] = self._sig_p05[name] + 1e-6
            else:
                self._sig_p05[name] = 0.0
                self._sig_p95[name] = 1.0

    def _normalize(self, name: str, raw: float) -> float:
        """Map raw signal to [0, 1] using percentile-based linear clamping.

        Training-data P5 → 0, P95 → 1, linear in between.
        Values outside [P5, P95] are clamped to [0, 1].
        """
        p05 = self._sig_p05.get(name, 0.0)
        p95 = self._sig_p95.get(name, 1.0)
        if p95 - p05 < 1e-12:
            return 0.0
        normed = (raw - p05) / (p95 - p05)
        return float(max(0.0, min(1.0, normed)))

    #  Convenience: fit + check from torch tensors  ........................

    def fit_torch(self, trajectories: "torch.Tensor") -> "RegimeDetector":
        """Torch-friendly fit (converts to numpy internally)."""
        if not _HAS_TORCH:
            raise ImportError("torch is not installed")
        return self.fit(trajectories.detach().cpu().numpy())

    def check_torch(self, observations: "torch.Tensor",
                    timestamp: Optional[float] = None) -> RegimeAssessment:
        """Torch-friendly check (converts to numpy internally)."""
        if not _HAS_TORCH:
            raise ImportError("torch is not installed")
        return self.check(observations.detach().cpu().numpy(), timestamp=timestamp)

    def __repr__(self) -> str:
        fit_status = "fitted" if self._fitted else "not fitted"
        return (f"RegimeDetector(k={self.k}, window={self.window_size}, "
                f"K={self.knn_k}, threshold={self.threshold}, {fit_status})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic test helpers
# ═══════════════════════════════════════════════════════════════════════════════

def generate_coupled_dynamics(
    N: int = 20,
    T: int = 500,
    D: int = 64,
    regime_change_at: int = 350,
    coupling_strength: float = 0.3,
    noise: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic trajectories from a stable coupled dynamical system.

    Uses a VAR(1) process with spectral radius < 1 to keep trajectories
    bounded.  Entities evolve according to:

        x(t+1) = φ · A · tanh(x(t)) + ε_t

    where φ = coupling_strength < 1 ensures stationarity, and
    ε_t ~ N(0, noise²·I).  At ``regime_change_at``, the coupling matrix A
    is replaced with a new random one, inducing a structural regime change.

    Returns:
        trajectories: (N, T, D) float64 array
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (N, D)).astype(np.float64)

    # Normalise coupling so spectral radius ≈ coupling_strength (stable < 1)
    A1_raw = rng.normal(0, 1, (N, N)).astype(np.float64)
    rho1 = max(abs(np.linalg.eigvals(A1_raw)))
    A1 = (coupling_strength / max(rho1, 1e-10)) * A1_raw

    A2_raw = rng.normal(0, 1, (N, N)).astype(np.float64)
    rho2 = max(abs(np.linalg.eigvals(A2_raw)))
    A2 = (coupling_strength / max(rho2, 1e-10)) * A2_raw

    traj = np.zeros((N, T, D), dtype=np.float64)
    for t in range(T):
        traj[:, t, :] = x
        A = A1 if t < regime_change_at else A2
        # Stable VAR(1): x_{t+1} = φ·A·tanh(x_t) + noise
        drive = A @ np.tanh(x)
        dx = coupling_strength * drive + rng.normal(0, noise, (N, D))
        x = x + dx
        # Gentle re-normalisation to prevent drift
        row_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(row_norms, 1.0)  # cap at unit ball

    return traj


def generate_volatility_regime(
    N: int = 20,
    T: int = 500,
    D: int = 64,
    regime_change_at: int = 350,
    base_vol: float = 0.02,
    high_vol: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Generate trajectories with a volatility regime shift.

    Uses the same stable VAR(1) dynamics as ``generate_coupled_dynamics``
    but keeps the coupling matrix fixed and instead increases the noise
    standard deviation at ``regime_change_at``.

    Returns:
        trajectories: (N, T, D) float64 array
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (N, D)).astype(np.float64)

    # Fixed coupling matrix (stable)
    A_raw = rng.normal(0, 1, (N, N)).astype(np.float64)
    rho = max(abs(np.linalg.eigvals(A_raw)))
    A = (0.3 / max(rho, 1e-10)) * A_raw

    traj = np.zeros((N, T, D), dtype=np.float64)
    for t in range(T):
        traj[:, t, :] = x
        vol = high_vol if t >= regime_change_at else base_vol
        drive = A @ np.tanh(x)
        dx = 0.3 * drive + rng.normal(0, vol, (N, D))
        x = x + dx
        row_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(row_norms, 1.0)

    return traj
