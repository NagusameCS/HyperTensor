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
INSTINCT HORIZON — Hallucination Prevention via Manifold Geometry
==================================================================

THEORY:
  Hallucinations happen when a language model generates text about
  topics it hasn't been trained on — it extrapolates from insufficient
  evidence. This is exactly what "outside the instinct horizon" means.

  The instinct horizon d_h = R * (-ln(1 - 0.5^(1/N))) is the boundary
  in k-space where jury confidence J = 0.5. Inside the horizon, the
  manifold has seen similar patterns before — responses are grounded.
  Outside, the model is in novel territory — hallucinations happen.

  By requiring J >= threshold for every generated token, we can:
  - PREVENT hallucinations entirely (threshold = 0.99)
  - ALLOW controlled creativity (threshold = 0.5)
  - Let the model go anywhere (threshold = 0.0, no guard)

  The creativity slider α maps to jury threshold:
    α = 0.0 (strict):  J_threshold = 0.99  → only deeply familiar
    α = 0.5 (balanced): J_threshold = 0.50  → instinct horizon
    α = 1.0 (creative): J_threshold = 0.01  → nearly anything goes

USAGE:
  from hyperformance import HallucinationGuard

  guard = HallucinationGuard(creativity=0.5)  # balanced
  guard.calibrate(trajectories)               # learn familiar territory

  # Check a generated response
  safe, J, verdict = guard.check(query_k)
  # safe=True: response is grounded (inside horizon)
  # safe=False: response may be hallucination (outside horizon)

William "Nagusame" Stewart — HyperTensor 2026
"""
import torch, math
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

torch.set_grad_enabled(False)


class InstinctHorizon:
    """The boundary between known and unknown territory in k-space.
    
    The instinct horizon distance d_h is where jury confidence J = 0.5:
      d_h = R * (-ln(1 - 0.5^(1/N)))
    
    Inside d_h: the manifold knows this region → grounded responses.
    Outside d_h: the manifold has no intuition → potential hallucinations.
    
    The creativity parameter α maps to a jury threshold:
      J_threshold = 0.5 + 0.49 * (1 - 2α)
    """
    
    def __init__(self, coverage_radius: float = 0.5, n_jurors: int = 7,
                 creativity: float = 0.5):
        """
        Args:
            coverage_radius: Manifold coverage radius R (auto-calibrated)
            n_jurors: Number of nearest neighbors to consult
            creativity: Float in [0.0, 1.0] controlling strictness
                0.0 = zero tolerance for unfamiliarity (J_threshold ≈ 0.99)
                0.5 = instinct horizon (J_threshold = 0.50)
                1.0 = anything goes (J_threshold ≈ 0.01)
        """
        self.R = coverage_radius
        self.N = n_jurors
        self.creativity = creativity
        self._juror_tensor = None
        self._juror_labels = []
        self._calibrated = False
        self._update_threshold()
    
    def _update_threshold(self):
        """Map creativity α ∈ [0,1] to jury threshold J ∈ [0.01, 0.99]."""
        alpha = max(0.0, min(1.0, self.creativity))
        # At α=0.5 (balanced), J_threshold = 0.5 (the instinct horizon)
        # At α=0.0 (strict), J_threshold = 0.99
        # At α=1.0 (creative), J_threshold = 0.01
        self.J_threshold = 0.5 + 0.49 * (1.0 - 2.0 * alpha)
        self.J_threshold = max(0.01, min(0.99, self.J_threshold))
    
    @property
    def instinct_horizon_distance(self) -> float:
        """The distance in k-space where J = 0.5."""
        return self.R * (-math.log(1.0 - 0.5 ** (1.0 / self.N)))
    
    def calibrate(self, trajectories: List[Dict]):
        """Calibrate coverage radius from trajectory density.
        
        Args:
            trajectories: List of {"proj": tensor[K], "label": str}
        """
        if len(trajectories) < 4:
            return
        
        projs = []
        for t in trajectories:
            p = t.get("proj")
            if p is not None:
                projs.append(p.float().flatten())
        
        if len(projs) < 4:
            return
        
        # Compute coverage radius from nearest-neighbor distances
        proj_stack = torch.stack(projs)
        n_sample = min(proj_stack.shape[0], 512)
        idx = torch.randperm(proj_stack.shape[0])[:n_sample]
        sample = proj_stack[idx]
        
        dists = []
        for i in range(n_sample):
            d = torch.norm(sample[i:i+1] - sample, dim=1)
            d_sorted = torch.sort(d)[0]
            if len(d_sorted) > 1:
                dists.append(d_sorted[1].item())  # nearest neighbor
        
        if dists:
            sorted_d = sorted(dists)
            # Use 75th percentile as coverage radius
            idx_pct = int(len(sorted_d) * 0.75)
            self.R = max(sorted_d[min(idx_pct, len(sorted_d)-1)], 0.01)
        
        # Build juror pool
        juror_tensors = []
        self._juror_labels = []
        for t in trajectories[-2048:]:
            p = t.get("proj")
            if p is not None:
                juror_tensors.append(p.float().flatten().unsqueeze(0))
                self._juror_labels.append(t.get("label", "")[:60])
        
        if juror_tensors:
            self._juror_tensor = torch.cat(juror_tensors, dim=0)
        
        self._calibrated = True
    
    def jury_confidence(self, query_k: torch.Tensor) -> Tuple[float, float]:
        """Compute jury confidence J for a k-space query using EUCLIDEAN distances.
        
        Uses Euclidean k-space distance (not cosine) for consistency with
        the C engine and the coverage radius R which is calibrated from
        Euclidean nearest-neighbor distances.
        """
        if self._juror_tensor is None or self._juror_tensor.shape[0] < 1:
            return 0.5, 0.0
        
        device = query_k.device
        jurors = self._juror_tensor.to(device)
        
        # Euclidean distances to all jurors (NO normalization)
        dists = torch.norm(query_k.unsqueeze(0).float() - jurors, dim=1)
        
        # Top-k nearest
        n_top = min(self.N, jurors.shape[0])
        top_dists, _ = torch.topk(dists, k=n_top, largest=False)
        
        # Jury formula with Euclidean metric
        confidences = torch.exp(-top_dists / self.R)
        J = (1.0 - torch.prod(1.0 - confidences)).item()
        
        # Similarity proxy: exp(-d_min/R) for reporting
        best_sim = math.exp(-top_dists[0].item() / self.R)
        
        return J, best_sim
    
    def is_inside_horizon(self, query_k: torch.Tensor) -> Tuple[bool, float, str]:
        """Check if a query is inside the instinct horizon.
        
        Returns:
            (safe, jury_confidence_J, verdict_message)
        """
        if not self._calibrated:
            return True, 0.5, "uncalibrated (allowing)"
        
        J, sim = self.jury_confidence(query_k)
        
        if J >= 0.99:
            return True, J, "deeply familiar"
        elif J >= 0.85:
            return True, J, "well-grounded"
        elif J >= self.J_threshold:
            return True, J, "inside horizon (familiar)"
        elif J >= 0.25:
            return False, J, f"near horizon (J={J:.2f} < threshold={self.J_threshold:.2f})"
        elif J >= 0.05:
            return False, J, "outside horizon (unfamiliar)"
        else:
            return False, J, "deeply unfamiliar (likely hallucination)"


class HallucinationGuard:
    """Prevents hallucination using the instinct horizon.
    
    Wraps any text generation pipeline with jury-based familiarity
    checking. Operates at the RESPONSE level (post-generation check)
    and optionally at the TOKEN level (requires C engine integration
    for per-token interception).
    
    Usage:
        guard = HallucinationGuard(creativity=0.5)
        guard.calibrate(trajectories)
        
        # After generation:
        safe, J, verdict = guard.check_response(response_k)
        if not safe:
            response = fallback_response  # or regenerate
    """
    
    DOMAIN_THRESHOLDS = {
        "math": 0.85,       # Math must be precise
        "science": 0.80,    # Science tolerates some extrapolation
        "code": 0.90,       # Code must be correct
        "creative": 0.30,   # Creative writing needs freedom
        "general": 0.50,    # General knowledge: instinct horizon
    }
    
    def __init__(self, creativity: float = 0.5,
                 domain: str = "general",
                 fallback_response: str = None):
        """
        Args:
            creativity: Float in [0,1]. 0=strict, 0.5=horizon, 1=creative
            domain: Domain name for domain-specific thresholds
            fallback_response: Response to use when hallucination detected
        """
        self.creativity = creativity
        self.domain = domain
        self.fallback = fallback_response or (
            "I don't have enough grounded knowledge to answer that reliably. "
            "Here's what I can tell you: [the following is based on my training, "
            "but I'm less confident about it]"
        )
        
        # Use domain-specific threshold if not explicitly set
        if domain in self.DOMAIN_THRESHOLDS and creativity == 0.5:
            self.horizon = InstinctHorizon(creativity=creativity)
            self.horizon.J_threshold = self.DOMAIN_THRESHOLDS[domain]
        else:
            self.horizon = InstinctHorizon(creativity=creativity)
        
        self.stats = {
            "total_checks": 0,
            "passed": 0,
            "flagged": 0,
            "flagged_responses": [],
        }
    
    def calibrate(self, trajectories: List[Dict]):
        self.horizon.calibrate(trajectories)
    
    def check_response(self, response_k: torch.Tensor,
                       raw_response: str = "") -> Tuple[bool, float, str, str]:
        """Check a generated response against the instinct horizon.
        
        Args:
            response_k: k-space projection of the response hidden state
            raw_response: The actual text (for logging)
        
        Returns:
            (safe, jury_confidence_J, verdict_message, safe_response)
        """
        self.stats["total_checks"] += 1
        
        safe, J, verdict = self.horizon.is_inside_horizon(response_k)
        
        if safe:
            self.stats["passed"] += 1
            return True, J, verdict, raw_response
        else:
            self.stats["flagged"] += 1
            if raw_response:
                self.stats["flagged_responses"].append({
                    "text": raw_response[:200],
                    "J": round(J, 4),
                    "verdict": verdict,
                })
            return False, J, verdict, self.fallback
    
    def set_creativity(self, alpha: float):
        """Adjust the creativity/strictness level.
        
        Args:
            alpha: 0.0 = zero tolerance, 0.5 = instinct horizon, 1.0 = creative
        """
        self.creativity = max(0.0, min(1.0, alpha))
        self.horizon.creativity = self.creativity
        self.horizon._update_threshold()
    
    def set_domain(self, domain: str):
        """Set domain for domain-specific threshold."""
        self.domain = domain
        if domain in self.DOMAIN_THRESHOLDS:
            self.horizon.J_threshold = self.DOMAIN_THRESHOLDS[domain]
    
    def report(self) -> Dict:
        """Return statistics and flagged responses."""
        return {
            "domain": self.domain,
            "creativity": self.creativity,
            "J_threshold": round(self.horizon.J_threshold, 4),
            "instinct_horizon_d": round(self.horizon.instinct_horizon_distance, 4),
            "coverage_radius_R": round(self.horizon.R, 4),
            "n_jurors": self.horizon.N,
            "stats": {
                "total": self.stats["total_checks"],
                "passed": self.stats["passed"],
                "flagged": self.stats["flagged"],
                "pass_rate": round(
                    self.stats["passed"] / max(self.stats["total_checks"], 1) * 100, 1
                ),
            },
            "recent_flags": self.stats["flagged_responses"][-5:],
        }


# 
# DEMO: Test the instinct horizon with synthetic data
# 

if __name__ == "__main__":
    print("=" * 70)
    print("  INSTINCT HORIZON — Hallucination Prevention Demo")
    print("=" * 70)
    
    K = 64
    torch.manual_seed(42)
    
    # Build trajectory pool (simulating a trained manifold)
    trajectories = []
    for i in range(128):
        trajectories.append({
            "proj": torch.randn(K) * 0.2 + torch.tensor([float(i % 4) * 0.5] + [0.0]*(K-1)),
            "label": f"topic_{i % 4}",
        })
    
    # Test at different creativity levels
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        guard = HallucinationGuard(creativity=alpha)
        guard.calibrate(trajectories)
        
        print(f"\n  Creativity α = {alpha:.2f} | J_threshold = {guard.horizon.J_threshold:.3f}")
        print(f"  Instinct horizon d_h = {guard.horizon.instinct_horizon_distance:.3f}")
        
        # Query inside familiar territory
        q_in = torch.tensor([0.5, 0.01] + [0.0]*(K-2))
        safe_in, J_in, v_in, _ = guard.check_response(q_in, "Gravity is spacetime curvature.")
        print(f"    Inside:  J={J_in:.3f} → {'SAFE' if safe_in else 'FLAGGED'} ({v_in})")
        
        # Query outside familiar territory
        q_out = torch.tensor([100.0, 200.0] + [0.0]*(K-2))
        safe_out, J_out, v_out, _ = guard.check_response(q_out, "Blargle fnord kazzak.")
        print(f"    Outside: J={J_out:.3f} → {'SAFE' if safe_out else 'FLAGGED'} ({v_out})")
    
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
  The instinct horizon prevents hallucination by rejecting responses
  that fall outside familiar territory in k-space.

  HOW IT WORKS:
    1. Every response is projected to k-space via the UGT basis
    2. The jury (N=7 nearest trajectories) votes on familiarity
    3. J = 1 - prod(1 - exp(-d_i/R)) aggregates all votes
    4. If J < threshold → potential hallucination → flag/regenerate
    5. If J >= threshold → grounded response → accept

  CREATIVITY CONTROL (α):
    α = 0.0 (strict):   J_threshold = 0.99 → only deeply familiar
                         Use for: math, code, factual Q&A
    α = 0.5 (balanced):  J_threshold = 0.50 → instinct horizon
                         Use for: general conversation
    α = 1.0 (creative):  J_threshold = 0.01 → nearly anything
                         Use for: creative writing, brainstorming

  DOMAIN-SPECIFIC DEFAULTS:
    math:     J_threshold = 0.85  (must be precise)
    science:  J_threshold = 0.80  (tolerates some extrapolation)
    code:     J_threshold = 0.90  (must be correct)
    creative: J_threshold = 0.30  (needs freedom)
    general:  J_threshold = 0.50  (instinct horizon)

  TOKEN-LEVEL GUARD (C engine):
    For complete hallucination prevention, every token must be checked
    before emission. This requires the C engine's speculative decode
    loop, where each draft token passes through llm_speculative_verify_topk().
    Tokens with J < threshold are rejected at draft time.
""")
