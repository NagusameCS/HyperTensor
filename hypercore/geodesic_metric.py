"""
hypercore/geodesic_metric.py — GeodesicMetric: Unified Geometric Reasoning
============================================================================
HyperTensor v1.0 | May 7, 2026

A drop-in module implementing all five geometric principles:
  1. Token Collapse measurement
  2. Gravitational Mass (jury-gated metric updates)
  3. Geodesic Half-Lives (time-based decay with usage reinforcement)
  4. Topological Tear detection (hallucination guard)
  5. Topological Compression (trajectory summarization)

Usage:
    from hypercore.geodesic_metric import GeodesicMetric, HallucinationGuard
    
    gm = GeodesicMetric(dim=64)
    guard = HallucinationGuard(gm, model, tokenizer)
    
    # Instead of model.generate():
    output, metrics = guard.safe_generate("What is the capital of France?")
    # metrics contains: jury_confidence, collapse_loss, is_hallucination, safe_to_generate
    
    # Track knowledge:
    gm.add_trajectory(hidden_state, jury_approved=True)
    gm.step_time()  # Apply half-life decay
"""

import torch, numpy as np, math, warnings, time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

@dataclass
class GenerationMetrics:
    """Metrics collected during a single generation."""
    jury_confidence: float = 0.0
    collapse_loss: float = 0.0
    effective_choices: int = 0
    is_hallucination: bool = False
    safe_to_generate: bool = True
    tokens_generated: int = 0
    total_collapse: float = 0.0
    trajectory_nn_distance: float = float('inf')
    coverage_radius: float = 1.0

class GeodesicMetric:
    """Unified geometric metric implementing all five principles.
    
    The metric tensor M (k×k) evolves through three mechanisms:
      - Jury-gated mass accumulation (truth adds weight, lies don't)
      - Half-life decay (unused knowledge fades)
      - Usage reinforcement (frequently accessed directions persist)
    
    Parameters:
        dim: dimension of the metric space (default 64, or auto-calibrated)
        decay_rate: per-step decay factor (0 = no decay, 0.05 = aggressive)
        reinforce_strength: weight of new trajectories on the metric
        jury_threshold: minimum confidence to accept a trajectory
    """
    
    def __init__(self, dim: int = 64, decay_rate: float = 0.01,
                 reinforce_strength: float = 0.5, jury_threshold: float = 0.5):
        self.dim = dim
        self.M = torch.eye(dim) * 0.1  # Start near-flat
        self.decay_rate = decay_rate
        self.reinforce_strength = reinforce_strength
        self.jury_threshold = jury_threshold
        
        self.trajectories: List[torch.Tensor] = []
        self.labels: List[str] = []
        self.access_counts = torch.zeros(dim)
        self.time = 0
        self.collapse_history: List[float] = []
        self.basis: Optional[torch.Tensor] = None
        self._coverage_radius: float = 1.0
        
    #  Core Operations 
    
    def add_trajectory(self, hidden_state: torch.Tensor, jury_approved: bool = True,
                       label: str = "") -> bool:
        """Add knowledge to the metric. Rejected trajectories contribute nothing.
        
        Returns True if the trajectory was accepted and added.
        """
        if not jury_approved:
            return False
        
        h = hidden_state.detach().float().cpu()
        while h.dim() > 1:
            h = h.squeeze(0)
        if h.dim() == 0:
            h = h.unsqueeze(0)
        
        # Auto-calibrate basis on first call if dim < hidden_dim
        if h.shape[-1] > self.dim and self.basis is None:
            self._calibrate_basis(h.unsqueeze(0))
        
        if self.basis is not None and h.shape[-1] == self.basis.shape[0]:
            h = (h @ self.basis.float()).squeeze()
        
        h = h.float()
        if h.dim() == 0:
            h = h.unsqueeze(0)
        norm = h.norm()
        if norm < 1e-10:
            return False
        
        v = h / norm
        
        # Gravitational Mass: add outer product to metric
        self.M += self.reinforce_strength * torch.outer(v, v)
        self.trajectories.append(v.detach().clone())
        self.labels.append(label)
        
        # Update coverage radius
        if len(self.trajectories) >= 2:
            traj_stack = torch.stack(self.trajectories)
            dists = torch.cdist(traj_stack, traj_stack)
            dists.fill_diagonal_(float('inf'))
            self._coverage_radius = dists.min(dim=1).values.mean().item()
        
        return True
    
    def _calibrate_basis(self, hidden_states: torch.Tensor):
        """Build UGT-style basis from calibration hidden states."""
        hs = hidden_states.float().cpu()
        if hs.dim() == 1:
            hs = hs.unsqueeze(0)
        hs_c = hs - hs.mean(0, keepdim=True)
        U, S, V = torch.linalg.svd(hs_c.T, full_matrices=False)
        k = min(self.dim, len(hs) - 1, hs.shape[-1])
        self.basis = U[:, :k]
        # Resize metric to match new dimension
        if k != self.dim:
            self.dim = k
            self.M = torch.eye(k) * 0.1
            self.access_counts = torch.zeros(k)
    
    def step_time(self) -> None:
        """Apply one step of half-life decay with usage reinforcement."""
        self.time += 1
        # Base decay
        self.M *= math.exp(-self.decay_rate)
        # Reinforcement from access patterns
        if self.access_counts.sum() > 0:
            reinforce_dir = self.access_counts / self.access_counts.sum()
            self.M += self.reinforce_strength * 0.1 * torch.outer(reinforce_dir, reinforce_dir)
    
    def record_access(self, direction: torch.Tensor) -> None:
        """Mark a direction as recently used (resists decay)."""
        d = direction.detach().float().cpu()
        while d.dim() > 1:
            d = d.squeeze(0)
        if d.dim() == 0:
            d = d.unsqueeze(0)
        if d.shape[0] <= self.dim:
            self.access_counts[:d.shape[0]] += d.abs()
    
    def measure_collapse(self, logits: torch.Tensor) -> Tuple[float, int]:
        """Measure information loss from token generation.
        
        Returns:
            collapse_loss: fraction of probability mass destroyed (0-1)
            effective_choices: number of viable alternatives before collapse
        """
        probs = torch.softmax(logits.float().cpu(), dim=-1)
        top_prob = probs.max().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        effective = int(torch.exp(torch.tensor(entropy)).item())
        collapse = 1.0 - top_prob
        self.collapse_history.append(collapse)
        return collapse, effective
    
    def safe_to_generate(self, query_hs: torch.Tensor) -> Tuple[bool, float, float]:
        """Check if a query is within the safe region of the manifold.
        
        Returns:
            safe: whether generation is likely to be grounded
            confidence: jury confidence score
            nn_distance: distance to nearest stored trajectory
        """
        if not self.trajectories:
            return False, 0.0, float('inf')
        
        q = query_hs.detach().float().cpu()
        while q.dim() > 1:
            q = q.squeeze(0)
        if q.dim() == 0:
            q = q.unsqueeze(0)
        
        if self.basis is not None and q.shape[-1] == self.basis.shape[0]:
            q = (q @ self.basis.float()).squeeze()
        
        q = q.float()
        if q.dim() == 0:
            q = q.unsqueeze(0)
        
        v_q = q / (q.norm() + 1e-10)
        
        # Compute jury confidence from all stored trajectories
        confidences = []
        min_dist = float('inf')
        for t in self.trajectories:
            d = (v_q - t).norm().item()
            min_dist = min(min_dist, d)
            c = math.exp(-d / max(self._coverage_radius, 1e-10))
            confidences.append(c)
        
        J = 1.0 - np.prod([1.0 - c for c in confidences])
        safe = J >= self.jury_threshold and min_dist <= 2.362 * self._coverage_radius
        
        return safe, J, min_dist
    
    def compress_trajectory(self, points: torch.Tensor, segment_size: int = 8
                           ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compress a sequence of points into waypoints + direction vectors.
        
        Returns:
            waypoints: key positions along the trajectory
            directions: dominant direction per segment (rank-1 propagator)
        """
        pts = points.detach().float().cpu()
        n_pts = len(pts)
        n_seg = max(1, n_pts // segment_size)
        waypoints = []
        directions = []
        
        for seg in range(n_seg):
            start = seg * segment_size
            segment_pts = pts[start:start + segment_size]
            wp = segment_pts[0].clone()
            delta = segment_pts[-1] - segment_pts[0]
            waypoints.append(wp)
            if delta.norm() > 1e-10:
                directions.append(delta / delta.norm())
            else:
                directions.append(torch.zeros_like(delta))
        
        return waypoints, directions
    
    def get_stats(self) -> Dict:
        """Return current metric statistics."""
        return {
            "metric_norm": self.M.norm().item(),
            "n_trajectories": len(self.trajectories),
            "time_steps": self.time,
            "coverage_radius": self._coverage_radius,
            "avg_collapse_loss": np.mean(self.collapse_history) if self.collapse_history else 0.0,
            "dim": self.dim,
            "decay_rate": self.decay_rate,
            "jury_threshold": self.jury_threshold,
        }


class HallucinationGuard:
    """Safe generation wrapper using GeodesicMetric.
    
    Wraps a HuggingFace model's generate() method with:
      - Pre-generation tear check (refuse if outside safe region)
      - Token-level collapse measurement
      - Automatic jury-gated trajectory storage
      - Half-life time stepping
    
    Usage:
        guard = HallucinationGuard(metric, model, tokenizer)
        output, metrics = guard.safe_generate("What is the capital of France?")
        if metrics.is_hallucination:
            print("Refused to generate — query outside knowledge boundary")
    """
    
    def __init__(self, metric: GeodesicMetric, model, tokenizer,
                 max_new_tokens: int = 50, refuse_on_hallucination: bool = True):
        self.metric = metric
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.refuse_on_hallucination = refuse_on_hallucination
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def safe_generate(self, prompt: str, **generate_kwargs) -> Tuple[str, GenerationMetrics]:
        """Generate text with full geometric safety checks."""
        metrics = GenerationMetrics()
        
        # 1. Encode prompt and extract hidden state
        enc = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                            max_length=256).to(self.model.device)
        
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        
        # 2. Pre-generation safety check
        safe, confidence, nn_dist = self.metric.safe_to_generate(h)
        metrics.jury_confidence = confidence
        metrics.safe_to_generate = safe
        metrics.trajectory_nn_distance = nn_dist
        metrics.coverage_radius = self.metric._coverage_radius
        
        if not safe and self.refuse_on_hallucination:
            metrics.is_hallucination = True
            return "[Generation refused: query outside knowledge boundary]", metrics
        
        # 3. Measure collapse on the initial token
        logits = out.logits[0, -1, :]
        collapse, effective = self.metric.measure_collapse(logits)
        metrics.collapse_loss = collapse
        metrics.effective_choices = effective
        metrics.total_collapse = collapse
        
        # 4. Generate
        gen_kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        gen_kwargs.update(generate_kwargs)
        
        with torch.no_grad():
            gen_out = self.model.generate(**enc, **gen_kwargs)
        
        n_new = gen_out.shape[1] - enc['input_ids'].shape[1]
        metrics.tokens_generated = n_new
        
        # 5. Measure collapse: re-run forward pass on generated tokens
        # (generate() doesn't return logits in newer transformers)
        full_ids = gen_out[0, :]
        with torch.no_grad():
            full_out = self.model(full_ids.unsqueeze(0), output_hidden_states=True)
        
        gen_logits = full_out.logits[0]
        input_len = enc['input_ids'].shape[1]
        for i in range(min(n_new, 10)):
            pos_logits = gen_logits[input_len + i - 1, :]
            c, _ = self.metric.measure_collapse(pos_logits)
            metrics.total_collapse += c
        
        # 6. Store trajectory (jury self-approves for factual prompts)
        final_h = full_out.hidden_states[-1][0, -1, :].float().cpu()
        self.metric.add_trajectory(final_h, jury_approved=confidence > 0.3,
                                   label=prompt[:50])
        
        # 7. Step time
        self.metric.step_time()
        self.metric.record_access(h)
        
        # 8. Decode
        output_text = self.tokenizer.decode(gen_out[0], skip_special_tokens=True)
        
        return output_text, metrics
    
    def calibrate_from_prompts(self, prompts: List[str]) -> int:
        """Build initial knowledge base from a list of factual prompts."""
        hs_list = []
        for p in prompts:
            enc = self.tokenizer(p, return_tensors='pt', truncation=True,
                                max_length=48).to(self.model.device)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            hs_list.append(out.hidden_states[-1][0, -1, :].float().cpu())
        
        hs_stack = torch.stack(hs_list)
        
        # Build basis
        hs_c = hs_stack - hs_stack.mean(0, keepdim=True)
        U, S, V = torch.linalg.svd(hs_c.T, full_matrices=False)
        k = min(self.metric.dim, len(prompts) - 1, hs_stack.shape[-1])
        self.metric.basis = U[:, :k]
        # Resize metric if basis changed dimension
        if k != self.metric.dim:
            self.metric.dim = k
            self.metric.M = torch.eye(k) * 0.1
            self.metric.access_counts = torch.zeros(k)
        
        # Add all prompts as approved trajectories
        for h in hs_stack:
            self.metric.add_trajectory(h, jury_approved=True)
        
        return k
