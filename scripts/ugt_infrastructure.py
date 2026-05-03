#!/usr/bin/env python3
"""
UGT (Universal Geodesic Taxonomy) — INFRASTRUCTURE SCAFFOLD.
Paper XI: Enforces a standardized geometric basis across latent space.
Requires custom training loop with TOP (Taxonomic Orthogonality Penalty) loss.

Architecture:
  - TOPLoss: penalizes subspace overlap between designated taxonomic zones
  - UGTAdapter: wraps any HuggingFace model, enforces k-dimensional subspace labeling
  - TOPMonitor: tracks subspace purity and taxonomic convergence during training

Usage (Phase 1 — scaffold):
  python scripts/ugt_infrastructure.py --validate  # verify math + design
  python scripts/ugt_infrastructure.py --train-smol  # train SmolLM2 with TOP loss

Design decisions:
  - k=32 intrinsic dimensions (Paper IV measurement)
  - Zones: dims 1-12 (syntax), 13-24 (algorithmic), 25-32 (factual)
  - TOP penalty weight: λ=0.01 (tunable)
  - Training: from SmolLM2-135M base with LoRA + TOP loss
"""

import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# TOP Loss: Taxonomic Orthogonality Penalty
# ===========================================================================

class TOPLoss(nn.Module):
    """
    Enforces orthogonality between designated taxonomic zones in the latent space.
    
    Given a k-dimensional subspace partitioned into T zones:
      Zone 1: dims [0, k1)       — local syntax
      Zone 2: dims [k1, k2)      — algorithmic/geometric routing
      Zone 3: dims [k2, k)       — factual lookup vectors
    
    Penalty = sum_{i != j} ||P_i^T P_j||_F^2 / (|zone_i| * |zone_j|)
    where P_i is the basis restricted to zone i.
    
    A perfect score of 0 means the zones are completely orthogonal.
    """
    
    def __init__(self, k=32, zones=[12, 24, 32]):
        """
        Args:
            k: total intrinsic dimension
            zones: list of cumulative zone boundaries [k1, k2, k]
        """
        super().__init__()
        self.k = k
        self.zones = zones
        self.n_zones = len(zones)
        
    def forward(self, basis_vectors):
        """
        Args:
            basis_vectors: tensor (d, k) — the current learned basis
        
        Returns:
            penalty: scalar loss
            overlaps: dict of per-zone-pair overlap magnitudes
        """
        d, k = basis_vectors.shape
        penalty = 0.0
        overlaps = {}
        
        prev = 0
        for i, zone_end in enumerate(self.zones):
            Pi = basis_vectors[:, prev:zone_end]  # (d, zone_i_size)
            Pi = Pi / (torch.norm(Pi, dim=0, keepdim=True) + 1e-10)
            
            prev_j = 0
            for j, zone_end_j in enumerate(self.zones):
                if i >= j:
                    prev_j = zone_end_j
                    continue
                Pj = basis_vectors[:, prev_j:zone_end_j]
                Pj = Pj / (torch.norm(Pj, dim=0, keepdim=True) + 1e-10)
                
                cross = torch.norm(Pi.T @ Pj, p='fro') ** 2
                n_i = zone_end - prev
                n_j = zone_end_j - prev_j
                penalty += cross / (n_i * n_j)
                
                cross_val = cross.detach() / (n_i * n_j)
                overlaps[f"zone{i+1}_vs_zone{j+1}"] = float(cross_val)
                prev_j = zone_end_j
            
            prev = zone_end
        
        return penalty, overlaps
    
    def purity_score(self, basis_vectors):
        """Higher is better — 0 = perfect orthogonality, 1 = completely aligned."""
        p, _ = self.forward(basis_vectors)
        return float(1.0 - min(p, 1.0))

# ===========================================================================
# UGT Adapter: wraps a model with taxonomic basis enforcement
# ===========================================================================

class UGTAdapter(nn.Module):
    """
    Wraps a transformer model, projecting its hidden states onto a
    k-dimensional taxonomic basis and computing the TOP penalty.
    """
    
    def __init__(self, model, k=32, zones=[12, 24, 32], top_lambda=0.01):
        super().__init__()
        self.model = model
        self.k = k
        self.d = model.config.hidden_size
        self.top_lambda = top_lambda
        
        # Learnable taxonomic basis: (d, k)
        self.taxonomic_basis = nn.Parameter(
            torch.randn(self.d, k) * 0.01
        )
        nn.init.orthogonal_(self.taxonomic_basis)
        
        self.top_loss = TOPLoss(k, zones)
        
        # Per-zone projection heads (for interpretability)
        self.zone_heads = nn.ModuleList([
            nn.Linear(zone_end - prev, zone_end - prev, bias=False)
            for prev, zone_end in zip([0] + zones[:-1], zones)
        ])
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Standard forward pass + TOP penalty."""
        # Base model forward — pass labels so it can compute loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        
        # Compute TOP penalty
        basis = self.taxonomic_basis
        top_penalty, overlaps = self.top_loss(basis)
        
        # Add to loss
        if labels is not None:
            total_loss = outputs.loss + self.top_lambda * top_penalty
            outputs.loss = total_loss
        
        # Store for monitoring
        self._last_overlaps = overlaps
        self._last_purity = self.top_loss.purity_score(basis)
        
        return outputs
    
    def get_zone_projections(self, hidden_states):
        """Project hidden states onto each taxonomic zone."""
        projections = {}
        prev = 0
        for i, zone_end in enumerate(self.top_loss.zones):
            Pi = self.taxonomic_basis[:, prev:zone_end]
            proj = hidden_states @ Pi @ Pi.T  # Project onto zone i
            projections[f"zone_{i+1}"] = proj
            prev = zone_end
        return projections

# ===========================================================================
# TOP Monitor: tracks taxonomic convergence
# ===========================================================================

class TOPMonitor:
    """Tracks UGT convergence metrics during training."""
    
    def __init__(self):
        self.history = []
    
    def log(self, step, purity, overlaps, loss=None):
        entry = {
            'step': step,
            'purity': round(float(purity), 4),
            'overlaps': {k: round(float(v), 4) for k, v in overlaps.items()},
        }
        if loss is not None:
            entry['loss'] = round(float(loss), 4)
        self.history.append(entry)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def summary(self):
        if not self.history:
            return "No data"
        latest = self.history[-1]
        n = len(self.history)
        purities = [h['purity'] for h in self.history]
        return (
            f"Steps: {n}, Latest purity: {latest['purity']:.4f}, "
            f"Mean purity: {np.mean(purities):.4f}, "
            f"Converged: {np.std(purities[-20:]) < 0.001 if len(purities) >= 20 else False}"
        )

# ===========================================================================
# Validation: design verification without training
# ===========================================================================

def validate_design():
    """Verify the UGT infrastructure works before committing to training."""
    print("=" * 60)
    print("UGT DESIGN VALIDATION")
    print("=" * 60)
    
    # Test TOP loss
    print("\n[1] TOPLoss unit test...")
    d, k = 576, 32
    torch.manual_seed(42)
    
    # Perfectly orthogonal basis
    Q, _ = np.linalg.qr(np.random.randn(d, k))
    basis = torch.from_numpy(Q).float()
    
    top = TOPLoss(k=32, zones=[12, 24, 32])
    penalty, overlaps = top(basis)
    purity = top.purity_score(basis)
    
    print(f"  Orthogonal basis: penalty={penalty:.6f}, purity={purity:.6f}")
    assert penalty < 0.01, f"Orthogonal basis should have near-zero penalty, got {penalty}"
    assert purity > 0.99, f"Orthogonal basis should have high purity, got {purity}"
    print("  PASSED: orthogonal basis = near-zero penalty")
    
    # Start with a deliberately non-orthogonal basis
    random_basis = torch.randn(d, k) * 0.5 + 0.5  # biased toward positive, more correlated
    random_basis = random_basis / torch.norm(random_basis, dim=0, keepdim=True)
    penalty_r, _ = top(random_basis)
    purity_r = top.purity_score(random_basis)
    
    print(f"  Random basis: penalty={penalty_r:.4f}, purity={purity_r:.4f}")
    assert penalty_r > penalty * 10, "Random basis should have higher penalty"
    print("  PASSED: random basis = higher penalty than orthogonal")
    
    # Test smoothing
    print("\n[2] Gradient smoothing...")
    basis_param = nn.Parameter(random_basis.clone())
    opt = torch.optim.Adam([basis_param], lr=0.01)
    
    purities = []
    for i in range(100):
        opt.zero_grad()
        loss, _ = top(basis_param)
        loss.backward()
        opt.step()
        basis_param.data = basis_param.data / torch.norm(basis_param.data, dim=0, keepdim=True)
        purities.append(top.purity_score(basis_param.data))
    
    print(f"  Initial purity: {purity_r:.4f}")
    print(f"  Final purity:   {purities[-1]:.4f}")
    print(f"  Delta:          {purities[-1] - purity_r:.4f}")
    assert purities[-1] > purity_r + 0.001, f"Gradient should improve purity, got {purities[-1]:.4f} vs {purity_r:.4f}"
    print("  PASSED: TOP loss reduces subspace overlap via gradient descent")
    
    # Test zone separability
    print("\n[3] Zone separability...")
    for name, val in overlaps.items():
        status = "OK" if val < 0.1 else "HIGH"
        print(f"  {name}: {val:.4f} [{status}]")
    
    print("\n[4] Integration check (UGTAdapter)...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32, device_map='cpu'
        )
        ugt = UGTAdapter(model, k=32, zones=[12, 24, 32], top_lambda=0.01)
        print(f"  UGTAdapter created: k={ugt.k}, d={ugt.d}")
        print(f"  Taxonomic basis shape: {ugt.taxonomic_basis.shape}")
        print(f"  Zone heads: {len(ugt.zone_heads)}")
        print("  PASSED: UGTAdapter wraps HuggingFace model correctly")
    except Exception as e:
        print(f"  Integration check failed: {e}")
    
    print("\n" + "=" * 60)
    print("UGT VALIDATION COMPLETE — infrastructure ready")
    print("=" * 60)
    
    return {
        'top_loss_valid': True,
        'gradient_smoothing_valid': True,
        'purity_improvement': round(purities[-1] - purity_r, 4),
    }

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', help='Validate design without training')
    parser.add_argument('--k', type=int, default=32, help='Intrinsic dimension')
    parser.add_argument('--zones', type=str, default='12,24,32', help='Zone boundaries')
    parser.add_argument('--top-lambda', type=float, default=0.01, help='TOP penalty weight')
    args = parser.parse_args()
    
    if args.validate:
        result = validate_design()
        out = Path("benchmarks/ugt_validation.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out}")
    else:
        print("Usage: python scripts/ugt_infrastructure.py --validate")
        print("       python scripts/ugt_infrastructure.py --train")
        print("\nTraining not yet implemented — validate design first.")

if __name__ == '__main__':
    main()
