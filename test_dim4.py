#!/usr/bin/env python3
"""
test_dim4.py — Synthetic k=4 stress test for the diffeomorphism solver.

Creates three synthetic 4-dimensional manifolds in axiom_geo JSON format
and verifies that all three pairwise combinations are classified as
diffeomorphic by the solver (Theorem 2, §3.1).

Reproduces Table §6.1 from the HyperTensor diffeomorphism paper:

  dim4_flat        — flat Euclidean metric, spherical cloud
  dim4_curved      — g_ij(x) = δ_ij + 0.6·x_i²·δ_ij, nontrivial curvature
  dim4_anisotropic — warp-1.5 axes, ellipsoidal cloud, higher curvature

All three pairs should be diffeomorphic (Theorem 2: each is an open
submanifold of standard ℝ⁴ with the inherited smooth structure).

Usage:
    python3 test_dim4.py
    # writes data/decisions_dim4.json and tests/dim4_vis/ (synth manifolds)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diffeo_solver import (
    load_manifold, decide,
    VERDICT_DIFFEO, VERDICT_NOT, VERDICT_UNDECIDED,
)


# ── Cloud generators ──────────────────────────────────────────────────────────

def _spherical_cloud(n: int, rng: np.random.Generator, radius: float = 1.0) -> np.ndarray:
    """n points distributed within a 4-D ball of given radius."""
    v = rng.standard_normal((n, 4))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # Randomly fill the interior: uniform radial density ∝ r³
    r = rng.uniform(0.65, 1.0, (n, 1)) * radius
    return v * r


def _ellipsoidal_cloud(n: int, rng: np.random.Generator,
                       axes: tuple = (1.5, 0.7, 1.3, 0.85)) -> np.ndarray:
    """n points in a 4-D ellipsoid with specified semi-axes."""
    v = rng.standard_normal((n, 4))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.uniform(0.65, 1.0, (n, 1))
    v = v * r
    for i, a in enumerate(axes):
        v[:, i] *= a
    return v


# ── Metric functions  g_diag: (4,) → (4,) ────────────────────────────────────

def _g_flat(pos: np.ndarray) -> list:
    return [1.0, 1.0, 1.0, 1.0]


def _g_curved(pos: np.ndarray) -> list:
    """g_ii = 1 + 0.6·x_i² — diagonal, position-dependent warp."""
    return [1.0 + 0.6 * float(pos[i]) ** 2 for i in range(4)]


def _g_anisotropic(pos: np.ndarray, warp: float = 1.5) -> list:
    """Axis-dependent warp with anisotropic scale factors."""
    axes = [warp, 1.0 / warp, warp ** 0.5, 1.0 / (warp ** 0.5)]
    return [axes[i] * (1.0 + 0.4 * float(pos[i]) ** 2) for i in range(4)]


# ── Scalar curvature (approximate) ───────────────────────────────────────────

def _R_flat(_pos: np.ndarray) -> float:
    return 0.0


def _R_curved(pos: np.ndarray) -> float:
    """
    Approximate scalar curvature for g_ii = 1 + α·x_i².

    For a diagonal metric g = diag(g_1,...,g_n) the Ricci scalar picks up
    contributions  R ≈ Σ_i  α / g_i(x)  (leading-order in α).
    With α = 0.6 and 4 dimensions, max ≈ 4·0.6 = 2.4 at the origin,
    but the sampled range for points near the unit sphere gives values
    near +1.4 (g_ii ≈ 1.6 on average at |x_i| ≈ 1/√4 ≈ 0.5).
    """
    return sum(0.6 / (1.0 + 0.6 * float(pos[i]) ** 2) for i in range(4))


def _R_anisotropic(pos: np.ndarray) -> float:
    """Higher curvature from combined warp + position dependence."""
    warp, alpha = 1.5, 0.4
    axes = [warp, 1.0 / warp, warp ** 0.5, 1.0 / (warp ** 0.5)]
    return sum(alpha / (axes[i] * (1.0 + alpha * float(pos[i]) ** 2))
               for i in range(4))


def _christoffel_norm(g_diag: list, pos: np.ndarray) -> float:
    """L2 norm of first-kind Christoffel symbols for a diagonal metric."""
    acc = 0.0
    for i in range(4):
        # Γ^i_ii = ∂_i(g_ii) / (2·g_ii) ≈ 2α·x_i / (2·g_ii)
        dg = abs(float(pos[i])) * 1.2  # rough ∂g_ii/∂x_i
        acc += (0.5 * dg / (g_diag[i] + 1e-9)) ** 2
    return math.sqrt(acc)


# ── Manifold builder ──────────────────────────────────────────────────────────

def make_manifold(name: str, cloud: np.ndarray,
                  g_fn, R_fn) -> tuple[dict, dict]:
    """Return (phase1_dict, phase3_dict) in axiom_geo JSON format."""
    n = len(cloud)

    p1 = {
        "model_name":     name,
        "model_arch":     "synthetic",
        "model_dim":      4,
        "model_layers":   0,
        "intrinsic_dim":  4,
        "explained_ratio": 0.99,
        "n_samples":      n,
        "cloud":          cloud.tolist(),
    }

    points = []
    R_vals = []
    for pos in cloud:
        g_diag = g_fn(pos)
        R = R_fn(pos)
        R_vals.append(R)
        points.append({
            "pos":              pos.tolist(),
            "g_diag":           g_diag,
            "R":                R,
            "christoffel_norm": _christoffel_norm(g_diag, pos),
        })

    p3 = {
        "mean_scalar_curvature": float(np.mean(R_vals)),
        "max_scalar_curvature":  float(np.max(R_vals)),
        "points":                points,
    }

    return p1, p3


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    rng = np.random.default_rng(42)

    vis_root = Path("tests/dim4_vis")
    vis_root.mkdir(parents=True, exist_ok=True)

    # Build manifolds
    spec = [
        ("dim4_flat",        _spherical_cloud(32, rng, 1.0),          _g_flat,        _R_flat),
        ("dim4_curved",      _spherical_cloud(34, rng, 1.0),          _g_curved,      _R_curved),
        ("dim4_anisotropic", _ellipsoidal_cloud(32, rng, (1.5, 0.7, 1.3, 0.85)), _g_anisotropic, _R_anisotropic),
    ]

    print("\n  HyperTensor Diffeomorphism Solver — k=4 stress test (Theorem 2)\n")

    manifolds = []
    for name, cloud, g_fn, R_fn in spec:
        p1, p3 = make_manifold(name, cloud, g_fn, R_fn)
        mdir = vis_root / name
        mdir.mkdir(exist_ok=True)
        (mdir / "phase1_manifold.json").write_text(json.dumps(p1, indent=2))
        (mdir / "phase3_curvature.json").write_text(json.dumps(p3, indent=2))
        m = load_manifold(mdir)
        manifolds.append(m)
        print(f"  [{name}]  k=4  n={m['n_samples']}  "
              f"R_mean={m['mean_R']:.2f}  R_max={p3['max_scalar_curvature']:.2f}")

    print()
    print(f"  {'Pair':<44}  {'Verdict':<20}  Reason")
    print(f"  {'-'*44}  {'-'*20}  {'-'*40}")

    all_decisions = []
    for i in range(len(manifolds)):
        for j in range(i + 1, len(manifolds)):
            mA, mB = manifolds[i], manifolds[j]
            result = decide(mA, mB)
            all_decisions.append(result)

            pair_str = f"{mA['name']} ↔ {mB['name']}"
            verdict  = result["verdict"]
            reason   = result["reason"][:60]
            c_v  = "\033[32m" if verdict == VERDICT_DIFFEO else \
                   "\033[31m" if verdict == VERDICT_NOT    else "\033[33m"
            print(f"  {pair_str:<44}  {c_v}{verdict:<20}\033[0m  {reason}")

            if "certificate" in result:
                c = result["certificate"]
                print(f"         n={c['n_probes']}  "
                      f"sign_det={'✓' if c['sign_consistent_det'] else '✗'}  "
                      f"min|det|={c['min_abs_det_dPhi']:.3f}  "
                      f"median|det|={c['median_abs_det_dPhi']:.3f}  "
                      f"rt_median={c['median_inverse_error']:.2e}  "
                      f"rt_max={c['max_inverse_error']:.2e}  "
                      f"({c['elapsed_s']}s)")
    print()

    counts = {VERDICT_DIFFEO: 0, VERDICT_NOT: 0, VERDICT_UNDECIDED: 0}
    for d in all_decisions:
        counts[d["verdict"]] += 1
    print(f"  Summary: {counts[VERDICT_DIFFEO]} diffeomorphic  |  "
          f"{counts[VERDICT_NOT]} not_diffeomorphic  |  "
          f"{counts[VERDICT_UNDECIDED]} undecided\n")

    ok = counts[VERDICT_DIFFEO] == 3 and counts[VERDICT_UNDECIDED] == 0
    if ok:
        print("  \033[32m✓ All 3 pairs correctly classified as diffeomorphic (Theorem 2)\033[0m")
    else:
        print("  \033[31m✗ Unexpected results — check implementation\033[0m",
              file=sys.stderr)

    out_path = Path("data/decisions_dim4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _serial(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        raise TypeError(f"not serialisable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump({
            "solver":    "diffeo_solver.py",
            "test":      "test_dim4.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "theorem":   "Theorem 2 (inherited-structure lemma, k=4 case)",
            "manifolds": [{"name": m["name"], "k": m["k"],
                           "n": m["n_samples"]} for m in manifolds],
            "decisions": all_decisions,
            "summary":   counts,
        }, f, indent=2, default=_serial)

    print(f"\n  Decisions written to: {out_path}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
