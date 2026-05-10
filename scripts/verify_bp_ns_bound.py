"""Tier-item #4: geometry-locked Delta_BP-NS bound.

The volume's tier-list mentioned a "geometry-locked Delta_BP-NS bound"
but no formal statement appears in any .tex file (grep returns nothing
matching BP.NS, Delta_BP, or geometry-locked).

The most natural reading in the GP / low-rank-projection context is:

    Let W in R^{m x d} be a weight matrix and let P_k = Q_k Q_k^T be the
    rank-k orthogonal projector onto its top-k right singular subspace.
    Define
        Delta_BP   := || (W - W P_k) x ||             (forward-projection error)
        Delta_NS   := || W x - W P_k x ||             (no-shortcut backward error)
    "Geometry-locked" claim: Delta_BP equals Delta_NS exactly (they are
    the same quantity rewritten), and both are bounded above by
        sigma_{k+1} * || x ||
    where sigma_{k+1} is the (k+1)-th singular value of W.

This is a textbook Eckart-Young-Mirsky consequence.  This script
numerically verifies (i) Delta_BP == Delta_NS to floating-point, and
(ii) the bound Delta_BP <= sigma_{k+1} * ||x|| holds tightly across
random W, x, and k.  If the bound is loose by orders of magnitude or
fails, that would falsify the natural reading.

CPU-only.  Output: benchmarks/bp_ns_bound_check.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "bp_ns_bound_check.json"


def trial(m: int, d: int, k: int, rng: np.random.Generator):
    W = rng.standard_normal((m, d)) / np.sqrt(d)
    x = rng.standard_normal(d)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    Q = Vt[:k].T  # [d, k]
    P = Q @ Q.T   # [d, d]
    delta_bp = np.linalg.norm(W @ (x - P @ x))
    delta_ns = np.linalg.norm(W @ x - W @ (P @ x))
    sigma_kp1 = S[k] if k < len(S) else 0.0
    bound = sigma_kp1 * np.linalg.norm(x)
    return {
        "m": m, "d": d, "k": k,
        "delta_bp": float(delta_bp),
        "delta_ns": float(delta_ns),
        "abs_diff_bp_ns": float(abs(delta_bp - delta_ns)),
        "sigma_kp1_bound": float(bound),
        "bound_holds": bool(delta_bp <= bound + 1e-9),
        "bound_tightness_ratio": float(delta_bp / max(bound, 1e-12)),
    }


def main() -> int:
    rng = np.random.default_rng(0)
    cases = []
    for (m, d) in [(64, 64), (128, 64), (64, 128), (256, 256), (512, 128)]:
        for k in (1, 4, 16, min(m, d) - 1):
            for _ in range(8):
                cases.append(trial(m, d, k, rng))

    n = len(cases)
    eq = sum(c["abs_diff_bp_ns"] < 1e-10 for c in cases)
    bound_ok = sum(c["bound_holds"] for c in cases)
    avg_tight = float(np.mean([c["bound_tightness_ratio"] for c in cases]))

    out = {
        "description": (
            "Numerical check of (i) Delta_BP == Delta_NS and (ii) "
            "Delta_BP <= sigma_{k+1} * ||x|| for random W, x, k."
        ),
        "n_trials": n,
        "delta_bp_equals_delta_ns_count": eq,
        "bound_holds_count": bound_ok,
        "avg_bound_tightness": avg_tight,
        "verdict": {
            "bp_ns_identical":   eq == n,
            "bound_holds_always": bound_ok == n,
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"trials                 = {n}")
    print(f"Delta_BP == Delta_NS   = {eq}/{n}")
    print(f"Delta_BP <= sigma*||x|| = {bound_ok}/{n}")
    print(f"avg tightness ratio    = {avg_tight:.3f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
