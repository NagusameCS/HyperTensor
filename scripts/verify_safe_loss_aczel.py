"""Tier-item #6: SAFE-loss / Aczel functional-equation check.

The volume's tier-list mentioned a "SAFE-loss with Aczel constraint" but
no formal statement appears in any .tex file (grep on aczel/Aczel/SAFE-loss
in ARXIV_SUBMISSIONS returns nothing germane).  The Aczel functional
equation in the loss-aggregation context (Aczel & Saaty 1983, Aczel 1966)
is the additivity / associativity constraint:

    f(x + y) = f(x) + f(y)              (Cauchy)
    g(g(x, y), z) = g(x, g(y, z))       (associativity)

A "SAFE-loss" that aggregates per-sample losses L_1, ..., L_n into a
single scalar L should satisfy:

  (A1) Symmetry:        L(perm(L_1..L_n)) = L(L_1..L_n)
  (A2) Monotonicity:    L_i <= L_i' componentwise => L <= L'
  (A3) Aczel additivity: L(L_1+c, ..., L_n+c) = L(L_1,...,L_n) + c
  (A4) Homogeneity:     L(alpha L_1, ..., alpha L_n) = alpha L(...)

Under (A1)-(A4) Aczel's representation theorem forces L to be the
arithmetic mean (up to weights summing to 1).  This script checks the
four constraints numerically against three candidate aggregators:
arithmetic mean, max-loss (DRO), and log-sum-exp.

This is a closed-form sanity check, not a perf benchmark.  CPU-only.

Output: benchmarks/safe_loss_aczel_check.json
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "safe_loss_aczel_check.json"


def L_mean(x: np.ndarray) -> float:
    return float(np.mean(x))

def L_max(x: np.ndarray) -> float:
    return float(np.max(x))

def L_lse(x: np.ndarray, beta: float = 1.0) -> float:
    # log-sum-exp scaled to behave like mean at beta->0 and max at beta->inf
    return float((1.0 / beta) * (math.log(np.exp(beta * x).mean())))


def check(agg, name: str, n: int = 8, n_trials: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    sym = mon = add = hom = 0
    sym_err = mon_err = add_err = hom_err = 0.0
    for _ in range(n_trials):
        x = rng.uniform(0.0, 1.0, size=n)
        # A1 symmetry
        y = rng.permutation(x)
        e = abs(agg(x) - agg(y))
        sym_err = max(sym_err, e)
        sym += int(e < 1e-9)
        # A2 monotonicity
        d = rng.uniform(0.0, 0.1, size=n)
        e = agg(x + d) - agg(x)
        mon_err = max(mon_err, -min(e, 0))
        mon += int(e >= -1e-9)
        # A3 Aczel additivity
        c = rng.uniform(-0.5, 0.5)
        e = abs(agg(x + c) - (agg(x) + c))
        add_err = max(add_err, e)
        add += int(e < 1e-6)
        # A4 homogeneity
        a = rng.uniform(0.1, 3.0)
        e = abs(agg(a * x) - a * agg(x))
        hom_err = max(hom_err, e)
        hom += int(e < 1e-6)
    return {
        "name": name,
        "n_trials": n_trials,
        "symmetry":      {"pass": sym, "max_err": sym_err},
        "monotonicity":  {"pass": mon, "max_violation": mon_err},
        "aczel_addit.":  {"pass": add, "max_err": add_err},
        "homogeneity":   {"pass": hom, "max_err": hom_err},
        "all_aczel_axioms": (sym == n_trials and mon == n_trials
                              and add == n_trials and hom == n_trials),
    }


def main() -> int:
    out = {
        "description": (
            "Numerical check of the four Aczel-style axioms (symmetry, "
            "monotonicity, additivity, homogeneity) for three candidate "
            "loss aggregators. Aczel's representation theorem says the "
            "arithmetic mean is the unique aggregator satisfying all four."
        ),
        "aggregators": [
            check(L_mean, "arithmetic_mean"),
            check(L_max,  "max_loss_DRO"),
            check(lambda x: L_lse(x, beta=2.0), "log_sum_exp_beta2"),
        ],
    }
    # Verdict: only arithmetic mean should satisfy all four.
    out["verdict"] = {
        a["name"]: a["all_aczel_axioms"] for a in out["aggregators"]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdict"], indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
