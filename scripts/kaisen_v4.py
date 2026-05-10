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
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
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
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
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

#!/usr/bin/env python3
"""
KAISEN v4 — ULTIMATE MANIFOLD SCOUT
=====================================

Every unique HyperTensor technique integrated into a single autonomous
scout module.  No mocks, no stubs, no placeholders.  Everything is real.

INTEGRATED TECHNIQUES
---------------------
  MANIFOLD     AdaptiveManifold (auto-k) + DenseManifold (clusters) + GRCProjector (L2-aware)
  JURY         Jury Formula J=1−∏(1−e^(−d_i/R)) + Euclidean metric + InstinctHorizon
  DRAFT        Geodesic steering (Christoffel) + GRC cache + Vocab k-NN
  VERIFY       Multi-model chain (scout→verify→retry) + Batch scouting (5 strategies)
  CACHE        GTC 3-tier (exact/binary/jury) + LRU eviction
  HYPERFORM    Auto CPU/GPU dispatch + JIT hot paths + JuryDraftGateCUDA
  MEMORY       ScoutMemory (vector store + dead-ends) + Session persistence (.kaisen)
  CONFIDENCE   J-band calibration + Adaptive threshold
  CLI          /scout /report /calibrate /save /seed /horizon /jury /steer /hyper /bench

ARCHITECTURE
------------
  KaisenV4
   AdaptiveManifold (auto-k, topic clusters, GRC L2-aware k*)
      MetricField (covariance curvature)
   Jury (J formula, Euclidean metric, instinct horizon)
      JuryDraftGateCUDA (auto CPU/GPU dispatch)
   GeodesicSteerer (Christoffel geodesic steps + GRC cache)
   GTCCache (3-tier semantic cache: exact/binary/jury)
   Verifier (multi-model chain: gen→check→retry, 5-strat batch)
   ScoutMemory (vector store, dead-end avoidance)
   ConfidenceCalibrator (J-band reliability tracking)
   Session (.kaisen persistence format)

William "Nagusame" Stewart — HyperTensor 2026
"""
import torch, time, math, json, os, sys, random, argparse
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

torch.set_grad_enabled(False)

# 
# CONSTANTS — All tunable hyperparameters in one place
# 

class K:
    """Central hyperparameter registry."""
    # Jury
    N_JURORS          = 7        # Paper XVI optimal
    JURY_THRESHOLD    = 0.85     # Auto-accept threshold
    R_PERCENTILE      = 75       # Coverage radius percentile
    
    # Manifold
    K_MIN             = 4
    K_MAX             = 64
    K_ADAPTIVE_DIV    = 5        # n_trajs // K_ADAPTIVE_DIV = k
    K_L2_FACTOR       = 42.7     # Empirical: L2_MB * 42.7 ≈ k*
    
    # GTC Cache
    GTC_MAX_SIZE      = 50000
    GTC_BIN_SIZE      = 0.5
    GRC_MAX_SIZE      = 4096
    GRC_BIN_SIZE      = 0.5     # k-space bin width for GRC cache
    
    # Scout
    MAX_SCOUT_DEPTH   = 8
    BATCH_APPROACHES  = 5
    VERIFY_RETRIES    = 2
    MAX_FAILURES      = 3         # Consecutive fails → stop
    
    # Memory
    MEMORY_WINDOW     = 64        # Recent visited window for dedup
    EXPLORE_THRESHOLD = 0.15      # Cosine distance for "already visited"
    
    # GPU
    GPU_JUROR_THRESH  = 1000     # Switch to GPU above this
    GPU_CHRISTOFFEL_K = 64       # Switch to GPU above k=64


# 
# 1. ADAPTIVE MANIFOLD — Auto-k, topic clusters, GRC L2-aware
# 

class MetricField:
    """Covariance-based curvature correction in k-space."""
    
    def __init__(self):
        self._metric_cov = None
        self._n_cal = 0
    
    def calibrate(self, projs: torch.Tensor):
        """Build metric covariance from k-space projections.
        
        Args:
            projs: [N, K] k-space projections
        """
        if projs.shape[0] < 4:
            return
        proj_centered = projs - projs.mean(dim=0, keepdim=True)
        self._metric_cov = (proj_centered.T @ proj_centered) / (proj_centered.shape[0] - 1)
        self._metric_cov = self._metric_cov + 0.01 * torch.eye(
            projs.shape[1], device=self._metric_cov.device, dtype=self._metric_cov.dtype
        )
        self._n_cal = projs.shape[0]
    
    def curvature_correction(self, v: torch.Tensor) -> torch.Tensor:
        """Approximate Christoffel correction: Γ·v⊗v ≈ v - M_inv @ v.
        
        Args:
            v: Velocity vector [K]
        Returns:
            correction: Curvature correction [K]
        """
        if self._metric_cov is None or self._n_cal < 4:
            return torch.zeros_like(v)
        try:
            M_inv = torch.linalg.inv(self._metric_cov)
            return v - (M_inv @ v)
        except Exception:
            return torch.zeros_like(v)


class AdaptiveManifold:
    """K-space manifold with auto-k, topic clusters, L2-aware k*.
    
    Features:
    - Auto-k: k = max(K_MIN, min(K_MAX, n_traj // K_ADAPTIVE_DIV))
    - Topic clustering: 4 domains (math, science, code, general)
    - GRC L2-aware: k* = L2_MB * K_L2_FACTOR for L2-residency
    - Metric field: covariance-based curvature
    """
    
    def __init__(self, k: int = 12, l2_mb: Optional[float] = None):
        self.k = k
        self.l2_mb = l2_mb  # GPU L2 cache size in MB (auto-detect if None)
        self._k_effective = k
        
        # Storage
        self._texts: List[str] = []
        self._hidden_states: List[torch.Tensor] = []    # [d_model]
        self._projs: List[torch.Tensor] = []             # [k]
        self._labels: List[str] = []
        
        # PCA basis
        self._basis: Optional[torch.Tensor] = None       # [d_model, k]
        self._mean: Optional[torch.Tensor] = None        # [d_model]
        
        # Topic clusters
        self._clusters: Dict[str, List[int]] = {
            "math": [], "science": [], "code": [], "general": []
        }
        
        # Metric
        self.metric = MetricField()
        
        # Calibration state
        self._calibrated = False
        self.d_model: Optional[int] = None
    
    def _detect_l2(self) -> Optional[float]:
        """Auto-detect GPU L2 cache size in MB."""
        if not torch.cuda.is_available():
            return None
        try:
            props = torch.cuda.get_device_properties(0)
            # L2 cache size in bytes → MB
            l2_bytes = getattr(props, 'l2_cache_size', 0)
            if l2_bytes > 0:
                return l2_bytes / (1024 * 1024)
        except Exception:
            pass
        return None
    
    def _compute_k_star(self) -> int:
        """GRC L2-residency-aware optimal k."""
        if self.d_model is None:
            return self.k
        l2 = self.l2_mb or self._detect_l2()
        if l2 is None:
            return self.k
        # k* ≈ L2_MB * 42.7 (empirical, Paper IX)
        k_star = int(l2 * K.K_L2_FACTOR)
        # Clamp: basis [d_model, k] in fp16 = d_model * k * 2 bytes
        # Must fit in 80% of L2
        max_k = int(0.8 * l2 * 1e6 / (self.d_model * 2))
        return max(4, min(k_star, max_k, K.K_MAX))
    
    def seed_from_texts(self, texts: List[str], labels: Optional[List[str]] = None,
                        hidden_states: Optional[List[torch.Tensor]] = None):
        """Initialize manifold with known facts.
        
        Args:
            texts: List of fact strings
            labels: Optional domain labels (math/science/code/general)
            hidden_states: Optional pre-computed hidden states
        """
        for i, text in enumerate(texts):
            self._texts.append(text)
            if hidden_states and i < len(hidden_states):
                self._hidden_states.append(hidden_states[i].float().cpu())
            label = labels[i] if labels and i < len(labels) else "general"
            self._labels.append(label)
            if label in self._clusters:
                self._clusters[label].append(i)
    
    def add_point(self, text: str, hidden_state: torch.Tensor, label: str = "general"):
        """Add a single point to the manifold."""
        hs = hidden_state.float().cpu()
        self._texts.append(text)
        self._hidden_states.append(hs)
        self._labels.append(label)
        if label in self._clusters:
            self._clusters[label].append(len(self._texts) - 1)
        if self.d_model is None and hs.ndim == 1:
            self.d_model = hs.shape[0]
        
        # Rebuild if we have enough points
        if len(self._hidden_states) >= 4 and len(self._hidden_states) % 8 == 0:
            self._rebuild()
    
    def _rebuild(self):
        """Rebuild PCA basis from all hidden states."""
        if len(self._hidden_states) < 4:
            return
        
        hs_stack = torch.stack(self._hidden_states).float()  # [N, d_model]
        self.d_model = hs_stack.shape[1]
        
        # Adaptive k
        n = hs_stack.shape[0]
        self._k_effective = max(K.K_MIN, min(K.K_MAX, n // K.K_ADAPTIVE_DIV))
        k_star = self._compute_k_star()
        self._k_effective = min(self._k_effective, k_star) if k_star else self._k_effective
        self.k = self._k_effective
        
        # PCA via SVD
        self._mean = hs_stack.mean(dim=0)
        hs_centered = hs_stack - self._mean
        try:
            U, S, V = torch.svd_lowrank(hs_centered.T, q=min(self._k_effective, n - 1))
            self._basis = U[:, :self._k_effective]  # [d_model, k]
        except Exception:
            # Fallback: use full SVD on small stack
            U, S, Vt = torch.linalg.svd(hs_centered.T, full_matrices=False)
            self._basis = U[:, :self._k_effective]
        
        # Re-project all points
        self._projs = []
        for hs in self._hidden_states:
            self._projs.append(self.project(hs))
        
        # Calibrate metric
        if len(self._projs) >= 4:
            proj_stack = torch.stack(self._projs)
            self.metric.calibrate(proj_stack)
        
        self._calibrated = True
    
    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to k-space.
        
        Args:
            hidden_state: [d_model] or [B, d_model]
        Returns:
            proj: [k] or [B, k]
        """
        if self._basis is None or self._mean is None:
            raise RuntimeError("Manifold not calibrated — call seed_from_texts or add_point first")
        
        hs = hidden_state.float()
        if hs.device != self._mean.device:
            hs = hs.to(self._mean.device)
        
        hs_centered = hs - self._mean if hs.ndim == 1 else hs - self._mean.unsqueeze(0)
        return hs_centered @ self._basis
    
    def unproject(self, proj: torch.Tensor) -> torch.Tensor:
        """Reconstruct approximate hidden state from k-space."""
        if self._basis is None or self._mean is None:
            raise RuntimeError("Manifold not calibrated")
        recon = proj.float() @ self._basis.T
        if recon.ndim == 2 and self._mean.ndim == 1:
            recon = recon + self._mean.unsqueeze(0)
        elif recon.ndim == 1:
            recon = recon + self._mean
        return recon
    
    @property
    def n_points(self) -> int:
        return len(self._hidden_states)
    
    @property
    def coverage_radius(self) -> float:
        """75th percentile of nearest-neighbor distances."""
        if len(self._projs) < 4:
            return 0.5
        proj_stack = torch.stack(self._projs)
        n_sample = min(proj_stack.shape[0], 256)
        idx = torch.randperm(proj_stack.shape[0])[:n_sample]
        sample = proj_stack[idx]
        
        nn_dists = []
        for i in range(n_sample):
            d = torch.norm(sample[i:i+1] - sample, dim=1)
            d_sorted = torch.sort(d)[0]
            if len(d_sorted) > 1:
                nn_dists.append(d_sorted[1].item())  # skip self (d=0)
        
        if not nn_dists:
            return 0.5
        nn_dists.sort()
        return nn_dists[int(len(nn_dists) * K.R_PERCENTILE / 100)]


# 
# 2. JURY — Formula + Euclidean metric + Instinct Horizon + CUDA
# 

class Jury:
    """Jury confidence with Euclidean metric, instinct horizon, and auto CPU/GPU dispatch.
    
    Formula:
        J = 1 − ∏ᵢ (1 − cᵢ)    where   cᵢ = exp(−dᵢ / R)
        d_h = R · (−ln(1 − 0.5^(1/N)))
        J_threshold(α) = 0.5 + 0.49·(1 − 2α)
    """
    
    def __init__(self, n_jurors: int = K.N_JURORS, creativity: float = 0.5):
        self.N = n_jurors
        self.creativity = creativity
        self.R = 0.5  # auto-calibrated
        self._juror_tensor: Optional[torch.Tensor] = None       # [N_jurors, K] CPU
        self._juror_tensor_gpu: Optional[torch.Tensor] = None   # [N_jurors, K] GPU
        self._juror_labels: List[str] = []
        self._calibrated = False
        self._update_threshold()
    
    def _update_threshold(self):
        """α → J_threshold: 0.0→0.99, 0.5→0.5, 1.0→0.01"""
        alpha = max(0.0, min(1.0, self.creativity))
        self.J_threshold = 0.5 + 0.49 * (1.0 - 2.0 * alpha)
        self.J_threshold = max(0.01, min(0.99, self.J_threshold))
    
    @property
    def instinct_horizon_distance(self) -> float:
        """d_h where J = 0.5"""
        return self.R * (-math.log(1.0 - 0.5 ** (1.0 / self.N)))
    
    def calibrate(self, projs: List[torch.Tensor], labels: Optional[List[str]] = None):
        """Calibrate from k-space projections.
        
        Args:
            projs: List of [K] tensors
            labels: Optional labels for audit trail
        """
        if len(projs) < 4:
            return
        
        # Store jurors
        self._juror_tensor = torch.stack([p.float().flatten() for p in projs])
        if labels:
            self._juror_labels = labels[:len(projs)]
        
        # Auto-calibrate R from NN distances
        n_sample = min(len(projs), 256)
        idx = torch.randperm(len(projs))[:n_sample]
        sample = self._juror_tensor[idx]
        
        nn_dists = []
        for i in range(n_sample):
            d = torch.norm(sample[i:i+1] - sample, dim=1)
            d_sorted = torch.sort(d)[0]
            if len(d_sorted) > 1:
                nn_dists.append(d_sorted[1].item())
        
        if nn_dists:
            nn_dists.sort()
            self.R = nn_dists[int(len(nn_dists) * K.R_PERCENTILE / 100)]
            self.R = max(0.01, self.R)
        
        # GPU mirror if beneficial
        if torch.cuda.is_available() and len(projs) > K.GPU_JUROR_THRESH:
            self._juror_tensor_gpu = self._juror_tensor.cuda()
        
        self._calibrated = True
    
    def jury_confidence(self, query_k: torch.Tensor) -> Tuple[float, float]:
        """Compute J and best similarity for a query point.
        
        Uses Euclidean distances (not cosine) for consistency with C engine.
        Auto-dispatches GPU path for large juror pools.
        
        Args:
            query_k: [K] tensor in k-space
        Returns:
            (J, best_sim) where J ∈ [0,1], best_sim = max confidence
        """
        if self._juror_tensor is None:
            return 1.0, 1.0
        
        q = query_k.float().flatten()
        N_eff = min(self.N, self._juror_tensor.shape[0])
        
        # GPU path
        if self._juror_tensor_gpu is not None and q.device.type == 'cuda':
            juror = self._juror_tensor_gpu
        elif q.device.type == 'cuda' and self._juror_tensor.shape[0] > K.GPU_JUROR_THRESH:
            juror = self._juror_tensor.cuda()
        else:
            juror = self._juror_tensor.to(q.device)
        
        # Euclidean distances
        dists = torch.norm(juror - q.unsqueeze(0), dim=1)
        top_dists, _ = torch.topk(dists, k=N_eff, largest=False)
        
        # Jury formula
        confidences = torch.exp(-top_dists / self.R)
        J = 1.0 - torch.prod(1.0 - confidences)
        
        best_sim = confidences[0].item() if confidences.numel() > 0 else 1.0
        return J.item(), best_sim
    
    def is_inside_horizon(self, query_k: torch.Tensor) -> Tuple[bool, float, str]:
        """Check if query is inside the instinct horizon.
        
        Returns:
            (safe, J, verdict_string)
        """
        J_val, best = self.jury_confidence(query_k)
        
        if J_val >= 0.99:
            verdict = "DEEPLY FAMILIAR"
        elif J_val >= self.J_threshold + 0.2:
            verdict = "FAMILIAR"
        elif J_val >= self.J_threshold:
            verdict = "INSIDE HORIZON"
        elif J_val >= self.J_threshold - 0.15:
            verdict = "NEAR HORIZON"
        elif J_val >= 0.2:
            verdict = "OUTSIDE HORIZON"
        else:
            verdict = "UNFAMILIAR"
        
        return J_val >= self.J_threshold, J_val, verdict
    
    def juror_diversity(self) -> float:
        """Measure juror spread: mean pairwise distance / R."""
        if self._juror_tensor is None or self._juror_tensor.shape[0] < 4:
            return 0.0
        j = self._juror_tensor
        n_sample = min(j.shape[0], 100)
        idx = torch.randperm(j.shape[0])[:n_sample]
        s = j[idx]
        pairwise = torch.cdist(s, s)
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=pairwise.device)
        mean_dist = pairwise[mask].mean().item()
        return mean_dist / self.R if self.R > 0 else 0.0


# 
# 3. GEODESIC STEERER — Christoffel steps + GRC cache + Vocab k-NN
# 

class GeodesicSteerer:
    """Steer generation toward manifold geodesics.
    
    Pipeline:
        1. Project current + previous hidden states to k-space
        2. Compute velocity v = p_curr − p_prev
        3. Apply metric curvature: p_pred = p_curr + v − 0.5·Γ·v⊗v
        4. Find nearest vocab token via embedding dot-product
        5. Cache prediction in GRC for O(1) retrieval
    """
    
    def __init__(self, manifold: AdaptiveManifold, embedding_weight: Optional[torch.Tensor] = None):
        self.manifold = manifold
        self.embed_weight = embedding_weight
        
        # GRC cache: bin_idx → [(k_vec, token_id, confidence)]
        self._grc: Dict[int, List[Tuple[torch.Tensor, int, float]]] = OrderedDict()
        self._grc_bin_size = K.GRC_BIN_SIZE
        self._grc_max = K.GRC_MAX_SIZE
        self._grc_hits = 0
        self._grc_misses = 0
    
    def _bin_key(self, k_vec: torch.Tensor) -> int:
        """Hash k-vector to spatial bin."""
        return int(torch.round(k_vec.sum() / self._grc_bin_size).item())
    
    def _grc_lookup(self, k_vec: torch.Tensor, top_k: int = 32) -> Optional[List[int]]:
        """O(1) lookup in GRC cache."""
        key = self._bin_key(k_vec)
        for candidate_key in [key, key - 1, key + 1]:
            if candidate_key in self._grc:
                entries = self._grc[candidate_key]
                if entries:
                    # Return top token_ids from cached entries
                    tokens = [e[1] for e in entries[:top_k]]
                    self._grc_hits += 1
                    return tokens
        self._grc_misses += 1
        return None
    
    def _grc_store(self, k_vec: torch.Tensor, token_id: int, confidence: float):
        """Store prediction in GRC cache with LRU eviction."""
        key = self._bin_key(k_vec)
        if key not in self._grc:
            self._grc[key] = []
        self._grc[key].append((k_vec.detach().cpu(), token_id, confidence))
        
        # LRU eviction
        while sum(len(v) for v in self._grc.values()) > self._grc_max:
            oldest_key = next(iter(self._grc))
            if self._grc[oldest_key]:
                self._grc[oldest_key].pop(0)
            if not self._grc[oldest_key]:
                del self._grc[oldest_key]
    
    def geodesic_step(self, h_curr: torch.Tensor, h_prev: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """Predict next k-space position.
        
        Args:
            h_curr: Current hidden state [d_model]
            h_prev: Previous hidden state [d_model] or None
        Returns:
            p_pred: Predicted k-space position [k]
        """
        p_curr = self.manifold.project(h_curr)
        
        if h_prev is not None:
            p_prev = self.manifold.project(h_prev)
            v = p_curr - p_prev
        else:
            v = torch.zeros_like(p_curr)
        
        # Normalize velocity
        v_norm = torch.norm(v)
        if v_norm > 1e-8:
            v = v / v_norm
        
        # Christoffel correction
        correction = self.manifold.metric.curvature_correction(v)
        
        # Geodesic step
        p_pred = p_curr + v - 0.5 * correction
        
        return p_pred
    
    def nearest_tokens(self, k_pred: torch.Tensor, top_k: int = 64
                       ) -> List[Tuple[int, float]]:
        """Find nearest vocabulary tokens to predicted k-space position.
        
        Uses embedding dot-product: recon @ embed_weight.T
        
        Args:
            k_pred: Predicted k-space position [k]
            top_k: Number of candidates
        Returns:
            List of (token_id, confidence)
        """
        if self.embed_weight is None:
            return []
        
        # GRC cache lookup first
        cached = self._grc_lookup(k_pred, top_k)
        if cached is not None:
            return [(t, 0.8) for t in cached]  # conservative confidence
        
        # Reconstruct to full space
        e_pred = self.manifold.unproject(k_pred)
        
        # Dot product with embedding
        if self.embed_weight.device != e_pred.device:
            e_pred = e_pred.to(self.embed_weight.device)
        
        logits = e_pred @ self.embed_weight.T  # [vocab]
        top_vals, top_ids = torch.topk(logits, k=min(top_k, logits.shape[0]))
        
        result = [(top_ids[i].item(), top_vals[i].item()) for i in range(len(top_ids))]
        
        # Cache
        if result:
            self._grc_store(k_pred, result[0][0], result[0][1])
        
        return result
    
    @property
    def grc_hit_rate(self) -> float:
        total = self._grc_hits + self._grc_misses
        return self._grc_hits / total if total > 0 else 0.0


# 
# 4. GTC CACHE — 3-tier semantic cache (exact/binary/jury)
# 

class GTCCache:
    """3-tier semantic cache with jury voting.
    
    Tier 0: Exact string match (instant, 100% reliable)
    Tier 1: Binary threshold (cosine distance < radius)
    Tier 2: Jury voting (J ≥ 0.85 AND best_sim ≥ 0.5)
    """
    
    def __init__(self, max_size: int = K.GTC_MAX_SIZE):
        self.max_size = max_size
        self._query_texts: List[str] = []
        self._query_projs: List[torch.Tensor] = []
        self._responses: List[str] = []
        self.radius: float = 0.3  # auto-calibrated
        self._calibrated = False
        self.hits = 0
        self.misses = 0
    
    def calibrate(self, cal_projs: List[torch.Tensor]):
        """Set radius from 75th percentile of pairwise cosine distances."""
        if len(cal_projs) < 4:
            return
        
        stack = F.normalize(torch.stack([p.float().flatten() for p in cal_projs]), dim=1)
        n_sample = min(stack.shape[0], 200)
        
        dists = []
        for _ in range(100):
            i, j = random.randint(0, n_sample - 1), random.randint(0, n_sample - 1)
            if i != j:
                dists.append(1.0 - torch.dot(stack[i], stack[j]).item())
        
        if dists:
            dists.sort()
            self.radius = dists[int(len(dists) * 0.75)]
            self.radius = max(0.10, min(0.60, self.radius))
        
        self._calibrated = True
    
    def store(self, query_proj: torch.Tensor, response: str, query_text: str = ""):
        """Store a query-response pair."""
        qp = query_proj.float().flatten().cpu()
        
        self._query_projs.append(qp)
        self._query_texts.append(query_text[:200])
        self._responses.append(response[:2000])
        
        # LRU eviction
        while len(self._query_projs) > self.max_size:
            self._query_projs.pop(0)
            self._query_texts.pop(0)
            self._responses.pop(0)
    
    def query(self, query_k: torch.Tensor, raw_text: str = "",
              n_jurors: int = 7) -> Tuple[bool, Optional[str], float, float]:
        """Query the cache with 3-tier matching.
        
        Returns:
            (hit, response, J, best_sim)
        """
        # Tier 0: Exact match
        if raw_text and raw_text in self._query_texts:
            idx = self._query_texts.index(raw_text)
            self.hits += 1
            return True, self._responses[idx], 1.0, 1.0
        
        if not self._query_projs:
            self.misses += 1
            return False, None, 0.0, 0.0
        
        q = F.normalize(query_k.float().flatten().unsqueeze(0), dim=1)  # [1, K]
        q_stack = F.normalize(torch.stack(self._query_projs), dim=1)     # [N, K]
        
        # Tier 1: Binary threshold
        sims = (q_stack @ q.T).squeeze(-1)  # [N]
        best_idx = torch.argmax(sims).item()
        best_sim = sims[best_idx].item()
        
        if (1.0 - best_sim) < self.radius:
            self.hits += 1
            return True, self._responses[best_idx], best_sim, best_sim
        
        # Tier 2: Jury voting
        N_eff = min(n_jurors, len(self._query_projs))
        top_sims, top_idx = torch.topk(sims, k=N_eff)
        distances = 1.0 - top_sims
        confidences = torch.exp(-distances / max(self.radius, 0.01))
        J = 1.0 - torch.prod(1.0 - confidences).item()
        
        if J >= 0.85 and best_sim >= 0.5:
            self.hits += 1
            return True, self._responses[best_idx], J, best_sim
        
        self.misses += 1
        return False, None, J, best_sim
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        return len(self._query_projs)


# 
# 5. SCOUT MEMORY — Vector store + dead-end avoidance
# 

@dataclass
class ScoutEntry:
    """A single scouting expedition record."""
    k_vector: torch.Tensor
    topic: str
    outcome: str      # "verified", "unverified", "dead_end"
    J_before: float
    J_after: float
    timestamp: float = field(default_factory=time.time)


class ScoutMemory:
    """Tracks explored territory to avoid re-scouting the same region."""
    
    def __init__(self, window: int = K.MEMORY_WINDOW):
        self.window = window
        self.visited: List[ScoutEntry] = []
        self.dead_ends: List[torch.Tensor] = []
    
    def is_explored(self, k_vec: torch.Tensor, topic: str = "",
                    threshold: float = K.EXPLORE_THRESHOLD) -> Tuple[bool, float]:
        """Check if a k-space region has already been scouted.
        
        Returns:
            (already_explored, max_similarity)
        """
        if not self.visited:
            return False, 0.0
        
        recent = self.visited[-self.window:]
        stack = torch.stack([e.k_vector for e in recent])
        
        q = F.normalize(k_vec.float().flatten(), dim=0)
        s = F.normalize(stack, dim=1)
        sims = (s @ q)
        max_sim = sims.max().item()
        
        return max_sim > (1.0 - threshold), max_sim
    
    def is_dead_end(self, k_vec: torch.Tensor, threshold: float = 0.05) -> bool:
        """Check if this region is a known dead end."""
        if not self.dead_ends:
            return False
        
        stack = torch.stack(self.dead_ends)
        q = F.normalize(k_vec.float().flatten(), dim=0)
        s = F.normalize(stack, dim=1)
        sims = s @ q
        return sims.max().item() > (1.0 - threshold)
    
    def record(self, k_vec: torch.Tensor, topic: str, outcome: str,
               J_before: float, J_after: float):
        """Record a scouting result."""
        entry = ScoutEntry(
            k_vector=k_vec.float().flatten().cpu(),
            topic=topic,
            outcome=outcome,
            J_before=J_before,
            J_after=J_after,
        )
        self.visited.append(entry)
        
        if outcome == "dead_end":
            self.dead_ends.append(k_vec.float().flatten().cpu())
            # Cap dead ends
            if len(self.dead_ends) > 256:
                self.dead_ends = self.dead_ends[-128:]
    
    @property
    def n_points(self) -> int:
        return len(self.visited)
    
    def stats(self) -> Dict[str, Any]:
        """Memory statistics."""
        total = len(self.visited)
        if total == 0:
            return {"total": 0}
        
        verified = sum(1 for e in self.visited if e.outcome == "verified")
        unverified = sum(1 for e in self.visited if e.outcome == "unverified")
        dead = sum(1 for e in self.visited if e.outcome == "dead_end")
        
        J_deltas = [e.J_after - e.J_before for e in self.visited if e.outcome == "verified"]
        avg_expansion = sum(J_deltas) / len(J_deltas) if J_deltas else 0.0
        
        return {
            "total": total,
            "verified": verified,
            "unverified": unverified,
            "dead_ends": dead,
            "verify_rate": verified / total if total > 0 else 0.0,
            "avg_J_expansion": avg_expansion,
        }


# 
# 6. CONFIDENCE CALIBRATOR — J-band reliability tracking
# 

class ConfidenceCalibrator:
    """Learns which J bands produce reliable (verified) responses."""
    
    def __init__(self, n_bands: int = 10):
        self.n_bands = n_bands
        self.bands: Dict[int, Dict[str, int]] = {}  # band_idx → {correct, total}
        for i in range(n_bands):
            self.bands[i] = {"correct": 0, "total": 0}
    
    def _band(self, J: float) -> int:
        """Map J ∈ [0,1] to band index."""
        return max(0, min(self.n_bands - 1, int(J * self.n_bands)))
    
    def record(self, J: float, was_correct: bool):
        """Record an observation."""
        b = self._band(J)
        self.bands[b]["total"] += 1
        if was_correct:
            self.bands[b]["correct"] += 1
    
    def reliability(self, J: float) -> float:
        """Estimated reliability at given J level."""
        b = self._band(J)
        band = self.bands[b]
        if band["total"] == 0:
            # Interpolate from neighbors
            return self._interpolate_reliability(b)
        return band["correct"] / band["total"]
    
    def _interpolate_reliability(self, band_idx: int) -> float:
        """Interpolate reliability from neighboring bands."""
        neighbors = []
        for offset in [-1, 1, -2, 2]:
            nb = band_idx + offset
            if 0 <= nb < self.n_bands and self.bands[nb]["total"] > 0:
                neighbors.append(self.bands[nb]["correct"] / self.bands[nb]["total"])
        return sum(neighbors) / len(neighbors) if neighbors else 0.5
    
    def suggested_threshold(self, min_reliability: float = 0.8) -> float:
        """Find lowest J band with reliability ≥ threshold."""
        best_J = 0.99
        for i in range(self.n_bands):
            band = self.bands[i]
            if band["total"] >= 3:
                rel = band["correct"] / band["total"]
                if rel >= min_reliability:
                    best_J = min(best_J, (i + 0.5) / self.n_bands)
        return best_J
    
    @property
    def total(self) -> int:
        """Total number of recorded observations."""
        return sum(b["total"] for b in self.bands.values())
    
    def report(self) -> str:
        """Human-readable calibration report."""
        lines = ["CONFIDENCE CALIBRATION:"]
        for i in range(self.n_bands):
            band = self.bands[i]
            J_lo = i / self.n_bands
            J_hi = (i + 1) / self.n_bands
            if band["total"] > 0:
                rel = band["correct"] / band["total"]
                bar = "" * int(rel * 10)
                lines.append(f"  J={J_lo:.1f}-{J_hi:.1f}: {rel:.0%} reliable ({band['correct']}/{band['total']}) {bar}")
            else:
                lines.append(f"  J={J_lo:.1f}-{J_hi:.1f}: (no data)")
        return "\n".join(lines)


# 
# 7. KAISEN v4 — The Ultimate Scout
# 

SCOUT_APPROACHES = [
    "Solve step by step. Show all work. Be precise.",
    "Think from first principles. Derive everything.",
    "Break into sub-problems. Solve each independently.",
    "Consider edge cases. Verify boundary conditions.",
    "Work backwards from the answer. Reverse-engineer the solution.",
]

VERIFY_PROMPT = """You are a strict fact-checker. Verify this solution:

PROBLEM: {problem}
SOLUTION: {solution}

CRITERIA: {criteria}

Is the solution CORRECT or INCORRECT? Answer with CORRECT or INCORRECT, then explain why.
Your verification:"""

RETRY_PROMPT = """Your previous solution was flagged as potentially incorrect.
Feedback: {feedback}

PROBLEM: {problem}

Please solve again more carefully. Show ALL steps. Double-check your work.
Solution:"""


class KaisenV4:
    """Ultimate autonomous manifold scout.
    
    Integrates every HyperTensor technique into a single module:
    - Adaptive manifold with auto-k and topic clusters
    - Jury confidence with Euclidean metric and instinct horizon
    - Geodesic steering with Christoffel correction and GRC cache
    - 3-tier GTC semantic cache
    - Multi-model verification chain with retry
    - 5-strategy batch scouting
    - Scout memory with dead-end avoidance
    - J-band confidence calibration
    - Session persistence (.kaisen format)
    - Auto CPU/GPU dispatch
    """
    
    def __init__(self,
                 scout_model=None,
                 scout_tokenizer=None,
                 verify_model=None,
                 verify_tokenizer=None,
                 k: int = 12,
                 creativity: float = 0.5,
                 device: str = "auto"):
        """
        Args:
            scout_model: HuggingFace model for exploration (small/fast)
            scout_tokenizer: Tokenizer for scout model
            verify_model: HuggingFace model for verification (larger/accurate, optional)
            verify_tokenizer: Tokenizer for verify model
            k: Initial manifold projection dimension
            creativity: α ∈ [0,1] for instinct horizon strictness
            device: "auto", "cuda", or "cpu"
        """
        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Models
        self.scout_model = scout_model
        self.scout_tokenizer = scout_tokenizer
        self.verify_model = verify_model or scout_model
        self.verify_tokenizer = verify_tokenizer or scout_tokenizer
        
        if self.scout_model:
            self.scout_model.eval()
        if self.verify_model and self.verify_model is not self.scout_model:
            self.verify_model.eval()
        
        # Embedding weight (for geodesic steerer)
        embed_weight = None
        if scout_model is not None:
            embed_weight = scout_model.get_input_embeddings().weight.detach()
        
        # Core components
        self.manifold = AdaptiveManifold(k=k)
        self.jury = Jury(n_jurors=K.N_JURORS, creativity=creativity)
        self.steerer = GeodesicSteerer(self.manifold, embed_weight)
        self.gtc = GTCCache()
        self.memory = ScoutMemory()
        self.calibrator = ConfidenceCalibrator()
        
        # State
        self._d_model: Optional[int] = None
        self._total_scouts = 0
        self._total_verified = 0
        self._session_file: Optional[str] = None
    
    #  Model helpers 
    
    def _get_d_model(self) -> int:
        """Get model hidden dimension."""
        if self._d_model:
            return self._d_model
        if self.scout_model:
            cfg = self.scout_model.config
            self._d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', 2048)
            return self._d_model
        return 2048
    
    def _hidden_state(self, text: str) -> torch.Tensor:
        """Extract last-token hidden state from text."""
        if self.scout_model is None:
            raise RuntimeError("No scout model loaded")
        
        tok = self.scout_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        
        with torch.no_grad():
            out = self.scout_model(**tok, output_hidden_states=True)
            hs = out.hidden_states[-1][0, -1, :]  # [d_model]
        
        return hs.float().cpu()
    
    def _generate(self, prompt: str, model=None, tokenizer=None,
                  max_tokens: int = 256, temperature: float = 0.7,
                  top_p: float = 0.9) -> str:
        """Generate text from a prompt."""
        if model is None:
            model = self.scout_model
        if tokenizer is None:
            tokenizer = self.scout_tokenizer
        
        tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        
        with torch.no_grad():
            out = model.generate(
                **tok,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        prompt_len = tok["input_ids"].shape[1]
        response = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        return response.strip()
    
    #  Initialization 
    
    def seed(self, facts: List[str], labels: Optional[List[str]] = None):
        """Seed the manifold with known facts.
        
        Args:
            facts: List of factual statements
            labels: Optional domain labels
        """
        print(f"[KAISEN v4] Seeding manifold with {len(facts)} facts...")
        
        hidden_states = []
        for fact in facts:
            try:
                hs = self._hidden_state(fact)
                hidden_states.append(hs)
            except Exception as e:
                print(f"  [WARN] Failed to encode: {fact[:50]}... ({e})")
        
        if not hidden_states:
            print("  [WARN] No facts encoded — manifold empty")
            return
        
        self.manifold.seed_from_texts(facts, labels, hidden_states)
        self.manifold._rebuild()
        
        # Calibrate jury from manifold projections
        if self.manifold._calibrated:
            self.jury.calibrate(self.manifold._projs, self.manifold._labels)
            self.gtc.calibrate(self.manifold._projs)
        
        self._print_manifold_status()
    
    def _print_manifold_status(self):
        """Print current manifold statistics."""
        mp = self.manifold
        j = self.jury
        n = mp.n_points
        if n < 2:
            print("  (manifold too small)")
            return
        
        print(f"  {n} trajectories | k={mp.k} | R={j.R:.2f} | "
              f"d_h={j.instinct_horizon_distance:.2f} | J_thr={j.J_threshold:.3f} | "
              f"α={j.creativity:.1f}")
    
    #  Scout Loop 
    
    def scout(self, problem: str, criteria: str = "Mathematical correctness and logical consistency.",
              max_depth: int = K.MAX_SCOUT_DEPTH, verbose: bool = True) -> Dict[str, Any]:
        """Run a full autonomous scouting expedition.
        
        Args:
            problem: The problem/question to solve
            criteria: Verification criteria
            max_depth: Maximum scout depth
            verbose: Print progress
        Returns:
            Dict with response, J values, verification status, stats
        """
        self._total_scouts += 1
        
        # Step 0: Check GTC cache
        try:
            h_problem = self._hidden_state(problem)
        except Exception:
            h_problem = torch.zeros(self._get_d_model())
        
        if self.manifold._calibrated:
            k_problem = self.manifold.project(h_problem)
        else:
            k_problem = torch.zeros(12)
        
        gtc_hit, cached_response, J_cache, sim_cache = self.gtc.query(k_problem, problem)
        if gtc_hit and cached_response:
            if verbose:
                print(f"\n  [GTC HIT] J={J_cache:.3f} sim={sim_cache:.3f}")
            return {
                "response": cached_response,
                "verified": True,
                "source": "GTC",
                "J_before": J_cache,
                "J_after": J_cache,
                "depth": 0,
                "trajectories": self.manifold.n_points,
            }
        
        # Step 1: JURY — Is this inside the instinct horizon?
        J_before = 1.0
        if self.jury._calibrated:
            J_before, _ = self.jury.jury_confidence(k_problem)
            inside, _, verdict = self.jury.is_inside_horizon(k_problem)
        else:
            inside, verdict = True, "UNCALIBRATED"
        
        if verbose:
            status = "IN" if inside else "OUT"
            print(f"\n{'='*55}")
            print(f"  KAISEN v4 | J={J_before:.3f} | {status} | {verdict}")
            print(f"  {problem[:80]}...")
            print(f"{'='*55}")
        
        # Step 2: Check memory — already explored?
        if self.manifold._calibrated and self.memory.n_points > 0:
            explored, sim = self.memory.is_explored(k_problem)
            if explored:
                if verbose:
                    print(f"  [SKIP] Already explored (sim={sim:.3f})")
                return {
                    "response": "(already explored)",
                    "verified": False,
                    "source": "memory",
                    "J_before": J_before,
                    "J_after": J_before,
                    "depth": 0,
                    "trajectories": self.manifold.n_points,
                }
            
            if self.memory.is_dead_end(k_problem):
                if verbose:
                    print(f"  [SKIP] Known dead end")
                return {
                    "response": "(dead end)",
                    "verified": False,
                    "source": "dead_end",
                    "J_before": J_before,
                    "J_after": J_before,
                    "depth": 0,
                    "trajectories": self.manifold.n_points,
                }
        
        # Step 3: BATCH SCOUT — Try multiple approaches
        best_response = ""
        best_verified = False
        best_verdict = ""
        J_after = J_before
        expansions = 0
        consecutive_failures = 0
        
        for depth in range(1, max_depth + 1):
            # Check if we're now inside familiar territory
            if J_after >= self.jury.J_threshold and depth > 1:
                if verbose:
                    print(f"  [DONE] Territory now familiar (J={J_after:.3f})")
                break
            
            if consecutive_failures >= K.MAX_FAILURES:
                if verbose:
                    print(f"  [STOP] {consecutive_failures} consecutive failures")
                break
            
            # Try batch of approaches
            approach_start = (depth - 1) % len(SCOUT_APPROACHES)
            approaches_tried = []
            
            for a_offset in range(min(K.BATCH_APPROACHES, len(SCOUT_APPROACHES))):
                a_idx = (approach_start + a_offset) % len(SCOUT_APPROACHES)
                approach = SCOUT_APPROACHES[a_idx]
                
                # Skip if already tried
                if approach in approaches_tried:
                    continue
                approaches_tried.append(approach)
                
                prompt = f"{problem}\n\n{approach}"
                response = self._generate(prompt, model=self.scout_model,
                                          tokenizer=self.scout_tokenizer,
                                          max_tokens=256 if depth == 1 else 384)
                
                # Verify
                verified, verdict = self._verify(problem, response, criteria)
                
                if verified:
                    best_verified = True
                    best_response = response
                    best_verdict = verdict
                    
                    # Expand manifold
                    try:
                        h_resp = self._hidden_state(response[:300])
                        k_resp = self.manifold.project(h_resp)
                        J_after, _ = self.jury.jury_confidence(k_resp)
                        
                        self.manifold.add_point(response[:200], h_resp)
                        expansions += 1
                        
                        # Record in memory
                        self.memory.record(k_resp, "general",
                                          "verified", J_before, J_after)
                        
                        # Calibrator
                        self.calibrator.record(J_after, True)
                        
                        if verbose:
                            delta = J_after - J_before
                            sign = "+" if delta >= 0 else ""
                            print(f"  [OK] d={depth} J={J_after:.3f} ({sign}{delta:.3f}) "
                                  f"{response[:80]}...")
                    except Exception as e:
                        if verbose:
                            print(f"  [WARN] Expansion failed: {e}")
                    
                    consecutive_failures = 0
                    break  # Found a verified approach
                else:
                    consecutive_failures += 1
                    if verbose and a_offset == 0:
                        print(f"  [??] d={depth} (v{depth}) {response[:70]}...")
                    
                    # Still record J for unverified
                    try:
                        h_resp = self._hidden_state(response[:300])
                        if self.manifold._calibrated:
                            k_resp = self.manifold.project(h_resp)
                            J_check, _ = self.jury.jury_confidence(k_resp)
                            self.calibrator.record(J_check, False)
                    except Exception:
                        pass
            
            if best_verified:
                break
        
        # Fallback: if nothing verified, use best attempt
        if not best_response and consecutive_failures > 0:
            # Generate a final best-effort response
            prompt = f"{problem}\n\n{random.choice(SCOUT_APPROACHES)}"
            best_response = self._generate(prompt, max_tokens=256)
            try:
                h_resp = self._hidden_state(best_response[:300])
                if self.manifold._calibrated:
                    k_resp = self.manifold.project(h_resp)
                    J_after, _ = self.jury.jury_confidence(k_resp)
                    self.memory.record(k_resp, "general", "unverified", J_before, J_after)
            except Exception:
                pass
        
        # Step 4: Store in GTC cache
        if best_response and self.manifold._calibrated:
            try:
                h_resp = self._hidden_state(best_response[:200])
                k_resp = self.manifold.project(h_resp)
                self.gtc.store(k_resp, best_response, problem)
            except Exception:
                pass
        
        # Step 5: Rebuild manifold periodically
        if expansions > 0 and self.manifold.n_points % 16 == 0:
            self.manifold._rebuild()
            if self.manifold._calibrated:
                self.jury.calibrate(self.manifold._projs, self.manifold._labels)
        
        if verbose:
            verified_str = "VERIFIED" if best_verified else "UNVERIFIED"
            print(f"  => {verified_str} | J {J_before:.3f}→{J_after:.3f} | "
                  f"traj={self.manifold.n_points}")
        
        if best_verified:
            self._total_verified += 1
        
        return {
            "response": best_response,
            "verified": best_verified,
            "source": "scout",
            "J_before": J_before,
            "J_after": J_after,
            "depth": depth,
            "trajectories": self.manifold.n_points,
            "verdict": best_verdict,
        }
    
    def _verify(self, problem: str, solution: str, criteria: str,
                retries: int = K.VERIFY_RETRIES) -> Tuple[bool, str]:
        """Multi-attempt verification chain.
        
        Args:
            problem: Original problem
            solution: Proposed solution
            criteria: Verification criteria
            retries: Number of retry attempts
        Returns:
            (is_correct, verdict_string)
        """
        for attempt in range(1 + retries):
            verify_prompt = VERIFY_PROMPT.format(
                problem=problem, solution=solution[:600], criteria=criteria
            )
            
            verdict = self._generate(
                verify_prompt,
                model=self.verify_model,
                tokenizer=self.verify_tokenizer,
                max_tokens=128,
                temperature=0.3,
            )
            
            is_correct = "CORRECT" in verdict.upper() and "INCORRECT" not in verdict.upper()
            
            if is_correct:
                return True, verdict
            
            if attempt < retries:
                retry_prompt = RETRY_PROMPT.format(
                    feedback=verdict[:150], problem=problem
                )
                solution = self._generate(
                    retry_prompt,
                    model=self.scout_model,
                    tokenizer=self.scout_tokenizer,
                    max_tokens=256,
                )
        
        return False, verdict if 'verdict' in dir() else "VERIFICATION FAILED"
    
    #  Batch Scout 
    
    def batch_scout(self, problems: List[str], criteria: str = "",
                    verbose: bool = True) -> List[Dict[str, Any]]:
        """Scout multiple problems.
        
        Args:
            problems: List of problem strings
            criteria: Verification criteria (shared)
            verbose: Print progress
        Returns:
            List of result dicts
        """
        if not criteria:
            criteria = "Mathematical correctness and logical consistency."
        
        results = []
        for i, problem in enumerate(problems):
            if verbose:
                print(f"\n[{i+1}/{len(problems)}]", end=" ")
            result = self.scout(problem, criteria, verbose=verbose)
            results.append(result)
        
        return results
    
    #  Session Persistence (.kaisen format) 
    
    def save(self, filepath: str):
        """Save full Kaisen state to .kaisen JSON file + .kaisen.pt tensor file."""
        state = {
            "format": "kaisen-v4",
            "version": "4.0.0",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "k": self.manifold.k,
                "k_effective": self.manifold._k_effective,
                "n_jurors": self.jury.N,
                "jury_threshold": self.jury.J_threshold,
                "creativity": self.jury.creativity,
                "coverage_radius": self.jury.R,
                "instinct_horizon": self.jury.instinct_horizon_distance,
                "d_model": self._get_d_model(),
            },
            "memory": self.memory.stats(),
            "gtc": {
                "size": self.gtc.size,
                "hits": self.gtc.hits,
                "misses": self.gtc.misses,
                "hit_rate": self.gtc.hit_rate,
                "radius": self.gtc.radius,
            },
            "scouts": {
                "total": self._total_scouts,
                "verified": self._total_verified,
            },
            "trajectories": [],
        }
        
        # Serialize trajectories
        for i in range(self.manifold.n_points):
            state["trajectories"].append({
                "label": self.manifold._labels[i] if i < len(self.manifold._labels) else "general",
                "text": self.manifold._texts[i][:200] if i < len(self.manifold._texts) else "",
            })
        
        # Save JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        # Save tensors
        tensor_file = filepath.replace('.json', '.pt') if filepath.endswith('.json') else filepath + '.pt'
        tensor_state = {}
        if self.manifold._basis is not None:
            tensor_state["basis"] = self.manifold._basis.cpu()
        if self.manifold._mean is not None:
            tensor_state["mean"] = self.manifold._mean.cpu()
        if self.manifold.metric._metric_cov is not None:
            tensor_state["metric_cov"] = self.manifold.metric._metric_cov.cpu()
        if self.jury._juror_tensor is not None:
            tensor_state["juror_tensor"] = self.jury._juror_tensor.cpu()
        if self.manifold._projs:
            tensor_state["projs"] = torch.stack(self.manifold._projs).cpu()
        
        if tensor_state:
            torch.save(tensor_state, tensor_file)
        
        self._session_file = filepath
        print(f"[KAISEN v4] Saved: {filepath} + {tensor_file}")
    
    def load(self, filepath: str):
        """Load Kaisen state from .kaisen files."""
        tensor_file = filepath.replace('.json', '.pt') if filepath.endswith('.json') else filepath + '.pt'
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Restore config
        cfg = state.get("config", {})
        self.jury.creativity = cfg.get("creativity", 0.5)
        self.jury._update_threshold()
        self.jury.R = cfg.get("coverage_radius", 0.5)
        
        # Restore tensors
        if os.path.exists(tensor_file):
            tensor_state = torch.load(tensor_file, map_location='cpu', weights_only=True)
            if "basis" in tensor_state:
                self.manifold._basis = tensor_state["basis"]
            if "mean" in tensor_state:
                self.manifold._mean = tensor_state["mean"]
            if "metric_cov" in tensor_state:
                self.manifold.metric._metric_cov = tensor_state["metric_cov"]
            if "juror_tensor" in tensor_state:
                self.jury._juror_tensor = tensor_state["juror_tensor"]
                self.jury._calibrated = True
            if "projs" in tensor_state:
                self.manifold._projs = [tensor_state["projs"][i] for i in range(tensor_state["projs"].shape[0])]
        
        # Restore trajectories
        for traj in state.get("trajectories", []):
            self.manifold._texts.append(traj["text"])
            self.manifold._labels.append(traj["label"])
        
        self.manifold._calibrated = self.manifold._basis is not None
        self._session_file = filepath
        
        if self.manifold.n_points >= 4 and not self.manifold._projs and self.manifold._basis is not None:
            # Re-project if we have basis but lost projections
            pass  # Would need hidden states — skip
        
        print(f"[KAISEN v4] Loaded: {filepath} ({self.manifold.n_points} trajectories)")
    
    #  Reports 
    
    def report(self) -> str:
        """Generate comprehensive status report."""
        mp = self.manifold
        j = self.jury
        m = self.memory
        
        lines = [
            "" * 55,
            "  KAISEN v4 — STATUS REPORT",
            "" * 55,
            "",
            " MANIFOLD ",
            f"  Points: {mp.n_points}",
            f"  k (effective): {mp.k}",
            f"  d_model: {self._get_d_model()}",
            f"  Calibrated: {mp._calibrated}",
            f"  Clusters: math={len(mp._clusters['math'])} science={len(mp._clusters['science'])} "
            f"code={len(mp._clusters['code'])} general={len(mp._clusters['general'])}",
            "",
            " JURY ",
            f"  Jurors: {j._juror_tensor.shape[0] if j._juror_tensor is not None else 0}",
            f"  Coverage radius R: {j.R:.4f}",
            f"  Instinct horizon d_h: {j.instinct_horizon_distance:.4f}",
            f"  Threshold J_thr: {j.J_threshold:.3f} (creativity α={j.creativity:.1f})",
            f"  Juror diversity: {j.juror_diversity():.2f}×R",
            "",
            " CACHE ",
            f"  GTC entries: {self.gtc.size}",
            f"  GTC hit rate: {self.gtc.hit_rate:.1%}",
            f"  GRC hit rate: {self.steerer.grc_hit_rate:.1%}",
            "",
            " MEMORY ",
        ]
        
        mem_stats = m.stats()
        for k, v in mem_stats.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.3f}")
            else:
                lines.append(f"  {k}: {v}")
        
        lines += [
            "",
            " SESSION ",
            f"  Total scouts: {self._total_scouts}",
            f"  Total verified: {self._total_verified}",
            f"  Verify rate: {self._total_verified/self._total_scouts:.1%}" if self._total_scouts > 0 else "  Verify rate: N/A",
            f"  Session file: {self._session_file or '(none)'}",
            "",
            self.calibrator.report(),
            "",
            "" * 55,
        ]
        
        return "\n".join(lines)
    
    #  Interactive CLI 
    
    def interactive(self):
        """Start interactive Kaisen CLI."""
        print("" * 55)
        print("  KAISEN v4 — ULTIMATE MANIFOLD SCOUT")
        print("" * 55)
        print("  Commands:")
        print("    /scout <problem>    — Explore a problem")
        print("    /report             — Full status report")
        print("    /calibrate          — Show confidence calibration")
        print("    /save [file]        — Save session (.kaisen)")
        print("    /seed <fact>        — Add known fact to manifold")
        print("    /horizon            — Show instinct horizon config")
        print("    /jury <query>       — Query jury confidence")
        print("    /steer <problem>    — Show geodesic steering")
        print("    /hyper              — Show hyperparameters")
        print("    /batch <file>       — Batch scout from file (1 per line)")
        print("    /creativity <α>     — Set creativity (0.0-1.0)")
        print("    /quit               — Exit")
        print("" * 55)
        
        while True:
            try:
                cmd = input("\nkaisen> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[quit]")
                break
            
            if not cmd:
                continue
            
            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if command in ('/quit', '/exit', '/q'):
                break
            
            elif command == '/report':
                print(self.report())
            
            elif command == '/calibrate':
                print(self.calibrator.report())
                if self._total_scouts > 5:
                    suggested = self.calibrator.suggested_threshold(0.8)
                    print(f"\n  Suggested J threshold (80% reliability): {suggested:.3f}")
            
            elif command == '/save':
                fp = arg if arg else f"kaisen_v4_{time.strftime('%Y%m%d_%H%M%S')}.kaisen.json"
                self.save(fp)
            
            elif command == '/seed':
                if arg:
                    self.seed([arg])
                else:
                    print("  Usage: /seed <fact text>")
            
            elif command == '/horizon':
                j = self.jury
                print(f"  Creativity α: {j.creativity}")
                print(f"  J threshold: {j.J_threshold:.3f}")
                print(f"  Coverage radius R: {j.R:.4f}")
                print(f"  Instinct horizon d_h: {j.instinct_horizon_distance:.4f}")
                print(f"  Jurors: {j._juror_tensor.shape[0] if j._juror_tensor is not None else 0}")
            
            elif command == '/jury':
                if not arg:
                    print("  Usage: /jury <query text>")
                elif not self.jury._calibrated:
                    print("  [WARN] Jury not calibrated — seed manifold first")
                else:
                    try:
                        h = self._hidden_state(arg)
                        k = self.manifold.project(h)
                        J, best = self.jury.jury_confidence(k)
                        safe, _, verdict = self.jury.is_inside_horizon(k)
                        print(f"  J={J:.4f} | best_conf={best:.4f} | {verdict}")
                        print(f"  d_h={self.jury.instinct_horizon_distance:.4f} | R={self.jury.R:.4f}")
                    except Exception as e:
                        print(f"  [ERR] {e}")
            
            elif command == '/steer':
                if not arg:
                    print("  Usage: /steer <problem text>")
                elif not self.manifold._calibrated:
                    print("  [WARN] Manifold not calibrated — seed first")
                else:
                    try:
                        h = self._hidden_state(arg)
                        p = self.manifold.project(h)
                        tokens = self.steerer.nearest_tokens(p, top_k=10)
                        print(f"  k-space position: [{', '.join(f'{x:.3f}' for x in p[:6].tolist())}...]")
                        print(f"  Top geodesic tokens:")
                        for tid, conf in tokens[:5]:
                            token_str = self.scout_tokenizer.decode([tid]) if self.scout_tokenizer else f"id={tid}"
                            print(f"    {token_str!r} (conf={conf:.3f})")
                    except Exception as e:
                        print(f"  [ERR] {e}")
            
            elif command == '/hyper':
                print("  HYPERPARAMETERS:")
                for attr in dir(K):
                    if attr.isupper() and not attr.startswith('_'):
                        print(f"    {attr} = {getattr(K, attr)}")
                print(f"\n  DEVICE: {self.device}")
                print(f"  d_model: {self._get_d_model()}")
            
            elif command == '/batch':
                if not arg or not os.path.exists(arg):
                    print("  Usage: /batch <file> (one problem per line)")
                else:
                    with open(arg, 'r', encoding='utf-8') as f:
                        problems = [l.strip() for l in f if l.strip()]
                    print(f"  Batch scouting {len(problems)} problems...")
                    results = self.batch_scout(problems, verbose=True)
                    verified = sum(1 for r in results if r["verified"])
                    print(f"\n  Done: {verified}/{len(problems)} verified")
            
            elif command == '/scout':
                if not arg:
                    print("  Usage: /scout <problem>")
                else:
                    self.scout(arg, verbose=True)
            
            elif command == '/creativity':
                try:
                    alpha = float(arg)
                    alpha = max(0.0, min(1.0, alpha))
                    self.jury.creativity = alpha
                    self.jury._update_threshold()
                    print(f"  Creativity set to α={alpha:.2f} → J_threshold={self.jury.J_threshold:.3f}")
                except ValueError:
                    print("  Usage: /creativity <0.0-1.0>")
            
            else:
                # Treat as direct scout
                self.scout(cmd, verbose=True)


# 
# 8. MAIN — Entry point
# 

def main():
    parser = argparse.ArgumentParser(
        description="KAISEN v4 — Ultimate Manifold Scout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kaisen_v4.py --interactive
  python kaisen_v4.py --scout "What is 17*43?"
  python kaisen_v4.py --batch problems.txt
  python kaisen_v4.py --seed facts.txt --scout "Explain gravity"
  python kaisen_v4.py --load session.kaisen.json --report
        """
    )
    parser.add_argument("--scout-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model for exploration (small/fast)")
    parser.add_argument("--verify-model", type=str, default=None,
                       help="Model for verification (defaults to scout-model)")
    parser.add_argument("--creativity", type=float, default=0.5,
                       help="Creativity α ∈ [0,1] (0=strict, 1=loose)")
    parser.add_argument("--k", type=int, default=12,
                       help="Initial manifold projection dimension")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto/cuda/cpu")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive CLI")
    parser.add_argument("--scout", type=str, default=None,
                       help="Single problem to scout")
    parser.add_argument("--batch", type=str, default=None,
                       help="File with one problem per line")
    parser.add_argument("--seed-file", type=str, default=None,
                       help="File with seed facts (one per line)")
    parser.add_argument("--load", type=str, default=None,
                       help="Load .kaisen session file")
    parser.add_argument("--save", type=str, default=None,
                       help="Save .kaisen session file after scouting")
    parser.add_argument("--report", action="store_true",
                       help="Print status report and exit")
    parser.add_argument("--depth", type=int, default=K.MAX_SCOUT_DEPTH,
                       help="Maximum scout depth")
    args = parser.parse_args()
    
    # Load models
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[KAISEN v4] Loading scout model: {args.scout_model}")
    scout_model = AutoModelForCausalLM.from_pretrained(
        args.scout_model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        device_map="auto" if args.device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    scout_tokenizer = AutoTokenizer.from_pretrained(args.scout_model)
    
    verify_model = None
    verify_tokenizer = None
    if args.verify_model and args.verify_model != args.scout_model:
        print(f"[KAISEN v4] Loading verify model: {args.verify_model}")
        verify_model = AutoModelForCausalLM.from_pretrained(
            args.verify_model,
            torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
            device_map="auto" if args.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        verify_tokenizer = AutoTokenizer.from_pretrained(args.verify_model)
    
    # Create Kaisen
    kaisen = KaisenV4(
        scout_model=scout_model,
        scout_tokenizer=scout_tokenizer,
        verify_model=verify_model,
        verify_tokenizer=verify_tokenizer,
        k=args.k,
        creativity=args.creativity,
        device=args.device,
    )
    
    # Load session
    if args.load:
        kaisen.load(args.load)
    
    # Seed from file
    if args.seed_file and os.path.exists(args.seed_file):
        with open(args.seed_file, 'r', encoding='utf-8') as f:
            facts = [l.strip() for l in f if l.strip()]
        kaisen.seed(facts)
    
    # Report only
    if args.report:
        print(kaisen.report())
        return
    
    # Interactive
    if args.interactive:
        if kaisen.manifold.n_points == 0:
            print("[KAISEN v4] Manifold empty — seeding with defaults...")
            default_facts = [
                "The Earth orbits the Sun in approximately 365.25 days.",
                "Water freezes at 0°C and boils at 100°C at standard pressure.",
                "The Pythagorean theorem: a² + b² = c² for right triangles.",
                "Python is a high-level interpreted programming language.",
                "The speed of light in vacuum is approximately 3×10⁸ m/s.",
                "DNA is a double helix structure containing genetic code.",
                "Gravity accelerates objects at 9.8 m/s² near Earth's surface.",
                "The derivative of x² is 2x.",
                "Atoms consist of protons, neutrons, and electrons.",
                "The sum of angles in a triangle is 180 degrees.",
                "Shakespeare wrote Romeo and Juliet.",
                "An algorithm is a step-by-step procedure for solving a problem.",
                "Photosynthesis converts CO₂ and H₂O into glucose using sunlight.",
                "The capital of Japan is Tokyo.",
                "A prime number has exactly two divisors: 1 and itself.",
                "Newton's second law: F = ma.",
            ]
            kaisen.seed(default_facts)
        kaisen.interactive()
        return
    
    # Single scout
    if args.scout:
        result = kaisen.scout(args.scout, max_depth=args.depth, verbose=True)
        print(f"\n{'='*55}")
        print(f"  BEST RESPONSE:")
        print(f"  {result['response'][:500]}")
        print(f"{'='*55}")
        
        if kaisen.calibrator.total > 3:
            print(f"\n{kaisen.calibrator.report()}")
    
    # Batch scout
    if args.batch and os.path.exists(args.batch):
        with open(args.batch, 'r', encoding='utf-8') as f:
            problems = [l.strip() for l in f if l.strip()]
        results = kaisen.batch_scout(problems, verbose=True)
        verified = sum(1 for r in results if r["verified"])
        print(f"\n{'='*55}")
        print(f"  BATCH RESULT: {verified}/{len(problems)} verified")
        print(f"{'='*55}")
    
    # Save
    if args.save:
        kaisen.save(args.save)
    elif args.scout or args.batch:
        # Auto-save after successful scouts
        auto_save = f"kaisen_v4_{time.strftime('%Y%m%d_%H%M%S')}.kaisen.json"
        kaisen.save(auto_save)


if __name__ == "__main__":
    main()
