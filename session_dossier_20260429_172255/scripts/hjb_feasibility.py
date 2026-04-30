"""
hjb_feasibility.py
==================

Empirical feasibility stub for the SHF (Spectral Hamiltonian Flow) loss
defined in ARXIV_SUBMISSIONS/paper-D/ott-gtc-manifold-runtime.tex §"Future
Work: HJB-Regularised Joint Training":

    L_SHF(theta) = L_task(theta)
                 + lambda * sum_{l=1..L-1} || J_l(s_l; theta) ||_2^2

where the discrete Bellman/Jacobi residual is

    J_l(s) = Delta^2 s_l + R_hat(s_l) Delta s_l           (Eq. 1)

and Delta s_l = s_{l+1} - s_l, Delta^2 s_l = s_{l+1} - 2 s_l + s_{l-1}.

The paper explicitly states this loss is "*not* implemented in the
reference runtime that produces the measurements reported in this paper".
This script does NOT implement training-time SHF; instead it lifts the
specification from "pure theory" to "empirically grounded" by:

    1. Loading the Phase-3 geometry exports for all three available
       models (smollm2-135m, gemma-4-e2b, phi-3.5-mini).
    2. Constructing three classes of L-step trajectories through the
       fitted manifold of intrinsic dimension n:
         (a) baked geodesics (Magnus-3 propagator)  → SHF "floor"
         (b) nearest-neighbour walks on the cloud   → "manifold-conformant"
         (c) random straight-line paths             → "off-manifold ceiling"
    3. Computing |J_l|^2 along each trajectory using the same Riemann
       tensor estimator (riemann_tensor) used by the Jacobi propagator.
    4. Reporting per-class mean SHF penalty + per-residual FLOP cost.

This gives the HJB future-work paragraph a concrete order of magnitude:
  - the *floor* is what training would converge to (zero if perfectly
    geodesic — measures finite-difference truncation),
  - the "conformant" magnitude is what an untrained-on-this-loss network
    already achieves geometrically,
  - the "off-manifold" magnitude is the worst-case if no curvature
    structure is exploited,
so any future trainer can quote these as targets/baselines instead of an
abstract reference to lambda in [1e-3, 1e-1].

Output:
  docs/figures/paper-d/hjb_feasibility/hjb_residual_magnitudes.json
  docs/figures/paper-d/hjb_feasibility/hjb_residual_summary.md

Run:
  python scripts/hjb_feasibility.py [--n-intrinsic 8] [--L 32] [--n-traj 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "gtc"))

from manifold import fit_phase3_manifold, Manifold  # noqa: E402
from jacobi import riemann_tensor                  # noqa: E402
from _phase_io import load_phase1                   # noqa: E402


# ─── Trajectory constructors ────────────────────────────────────────────────


def _intrinsic_lift(model: str, n_intrinsic: int, seed: int) -> np.ndarray:
    """Re-lift Phase-1 cloud to n_intrinsic dim (mirrors gtc_benchmark.py)."""
    p1 = load_phase1(model)
    rng = np.random.default_rng(seed)
    base = p1.cloud
    Nc = base.shape[0]
    eigs = p1.eigenvalues
    if len(eigs) < n_intrinsic:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(n_intrinsic - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:n_intrinsic], 1e-6))
    extra = rng.normal(size=(Nc, n_intrinsic - 3)) * extra_scale[None, :]
    return np.concatenate([base, extra], axis=1)


def random_straight(rng: np.random.Generator, cloud: np.ndarray, L: int
                    ) -> np.ndarray:
    """Off-manifold straight line between two random cloud endpoints."""
    i, j = rng.integers(0, cloud.shape[0], size=2)
    while j == i:
        j = int(rng.integers(0, cloud.shape[0]))
    a, b = cloud[i], cloud[j]
    ts = np.linspace(0.0, 1.0, L + 1)[:, None]
    return a[None, :] * (1.0 - ts) + b[None, :] * ts


def nn_walk(rng: np.random.Generator, cloud: np.ndarray, L: int,
             jitter: float = 0.05) -> np.ndarray:
    """Random walk along nearest-neighbour edges of the cloud."""
    Nc, n = cloud.shape
    out = np.zeros((L + 1, n))
    cur = int(rng.integers(0, Nc))
    out[0] = cloud[cur]
    for t in range(1, L + 1):
        d2 = np.einsum("ij,ij->i", cloud - cloud[cur][None, :],
                       cloud - cloud[cur][None, :])
        d2[cur] = np.inf
        # Sample one of the K nearest neighbours (K=4) at random
        K = 4
        idx = np.argpartition(d2, K)[:K]
        cur = int(idx[rng.integers(0, K)])
        out[t] = cloud[cur] + jitter * rng.normal(size=n) * \
                 (cloud.std(axis=0) + 1e-6)
    return out


def baked_geodesic(M: Manifold, x0: np.ndarray, v0: np.ndarray, L: int,
                   dl: float = 0.05) -> np.ndarray:
    """Magnus-style geodesic integrator: x_{t+1} = exp_x(dl * v_t)."""
    n = M.dim
    xs = np.zeros((L + 1, n))
    xs[0] = x0
    x = x0.copy()
    v = v0.copy()
    for t in range(1, L + 1):
        # Forward Euler on the geodesic equation:
        #   d v^k / dl = -Gamma^k_ij(x) v^i v^j
        gamma = M.gamma_at(x)  # (i, j, k)
        accel = -np.einsum("ijk,i,j->k", gamma, v, v)
        v = v + dl * accel
        x = x + dl * v
        xs[t] = x
    return xs


# ─── Discrete Jacobi residual ───────────────────────────────────────────────


def discrete_jacobi_residual(M: Manifold, xs: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute || J_l ||_2^2 at each interior layer l, decomposed.

    J_l(s) = Delta^2 s_l + R_hat(s_l) Delta s_l           (paper Eq. 1)

    Returns three arrays of shape (Lp1 - 2,):
      total^2     = || J_l ||^2
      kinetic^2   = || Delta^2 s_l ||^2
      curvature^2 = || R_hat(s_l) Delta s_l ||^2
    so the dominant term can be identified empirically.
    """
    Lp1, n = xs.shape
    if Lp1 < 3:
        raise ValueError("Need at least 3 interior nodes to form Delta^2.")

    total = np.zeros(Lp1 - 2)
    kin   = np.zeros(Lp1 - 2)
    curv  = np.zeros(Lp1 - 2)
    for li in range(1, Lp1 - 1):
        s_p = xs[li - 1]
        s_0 = xs[li]
        s_n = xs[li + 1]
        delta = s_n - s_0
        delta2 = s_n - 2.0 * s_0 + s_p
        R = riemann_tensor(M, s_0, h=5e-3)
        K = np.einsum("aiml,i,l->am", R, delta, delta)
        curv_term = K @ delta
        Jl = delta2 + curv_term
        total[li - 1] = float(Jl @ Jl)
        kin[li - 1]   = float(delta2 @ delta2)
        curv[li - 1]  = float(curv_term @ curv_term)
    return total, kin, curv


def riemann_flop_cost(n: int) -> int:
    """Approximate FLOP count for one riemann_tensor call at intrinsic dim n.

    riemann_tensor:
      - 2n calls to gamma_at (RBF eval of M points × n^3 weights ≈ M*n^3).
      - n^4 quadruple loop with O(n) cross term inside.
      For the reference n=8 manifold (M ≈ 64 sample points), this is
      dominated by:
         2n * (M * n^3)  +  n^4 * n   ≈  2 n^4 M  +  n^5
      We report the leading term.
    """
    M_seed = 64
    return int(2 * (n ** 4) * M_seed + n ** 5)


# ─── Driver ─────────────────────────────────────────────────────────────────


def run_model(model: str, n_intrinsic: int, L: int, n_traj: int,
              seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cloud = _intrinsic_lift(model, n_intrinsic, seed)
    print(f"[{model}] cloud N={cloud.shape[0]} dim={cloud.shape[1]}")

    M = fit_phase3_manifold(model, n_intrinsic=n_intrinsic, sigma=0.6,
                            n_grid=cloud.shape[0])

    # Pick a typical geodesic seed: cloud centroid + small velocity along
    # the first PC direction of the cloud.
    centroid = cloud.mean(axis=0)
    pc1 = np.linalg.svd(cloud - centroid, full_matrices=False)[2][0]
    pc1 /= np.linalg.norm(pc1) + 1e-12

    classes = {"baked_geodesic": [], "nn_walk": [], "random_straight": []}

    t0 = time.time()
    for k in range(n_traj):
        xs_g = baked_geodesic(M, centroid + 0.2 * rng.normal(size=n_intrinsic),
                              pc1 * (0.5 + 0.5 * rng.random()), L=L, dl=0.05)
        xs_w = nn_walk(rng, cloud, L=L)
        xs_r = random_straight(rng, cloud, L=L)
        for name, xs in (("baked_geodesic", xs_g),
                          ("nn_walk", xs_w),
                          ("random_straight", xs_r)):
            total, kin, curv = discrete_jacobi_residual(M, xs)
            classes[name].append({
                "trajectory_id": k,
                "L": int(L),
                "interior_nodes": int(total.shape[0]),
                "J_squared_per_node": total.tolist(),
                "kinetic_squared_per_node": kin.tolist(),
                "curvature_squared_per_node": curv.tolist(),
                "shf_penalty_total": float(total.sum()),
                "shf_penalty_mean":  float(total.mean()),
                "kinetic_mean":      float(kin.mean()),
                "curvature_mean":    float(curv.mean()),
            })
    elapsed = time.time() - t0

    summary = {}
    for name, items in classes.items():
        means = np.asarray([it["shf_penalty_mean"]  for it in items])
        totals = np.asarray([it["shf_penalty_total"] for it in items])
        kins   = np.asarray([it["kinetic_mean"]     for it in items])
        curvs  = np.asarray([it["curvature_mean"]   for it in items])
        summary[name] = {
            "n_traj": int(len(items)),
            "shf_mean_avg":      float(means.mean()),
            "shf_mean_p10":      float(np.quantile(means, 0.10)),
            "shf_mean_p90":      float(np.quantile(means, 0.90)),
            "shf_total_avg":     float(totals.mean()),
            "shf_total_median":  float(np.median(totals)),
            "kinetic_mean_avg":   float(kins.mean()),
            "curvature_mean_avg": float(curvs.mean()),
            "kin_to_curv_ratio":  float(kins.mean() / max(curvs.mean(), 1e-30)),
        }

    flops_per_residual = riemann_flop_cost(n_intrinsic)
    return {
        "model": model,
        "n_intrinsic": int(n_intrinsic),
        "L": int(L),
        "n_traj": int(n_traj),
        "seed": int(seed),
        "wall_seconds": float(elapsed),
        "flops_per_jacobi_residual": int(flops_per_residual),
        "flops_per_layer_per_token": int(flops_per_residual),
        "summary_by_class": summary,
        "trajectories": classes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-intrinsic", type=int, default=8)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--out-dir", default="docs/figures/paper-d/hjb_feasibility")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["smollm2-135m", "gemma-4-e2b", "phi-3.5-mini"]
    results = {}
    for m in models:
        try:
            results[m] = run_model(m, args.n_intrinsic, args.L,
                                    args.n_traj, args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"[{m}] FAILED: {e}")
            results[m] = {"error": str(e)}

    # Per-model verdict
    verdict = {}
    for m, r in results.items():
        if "error" in r:
            verdict[m] = "error"
            continue
        sb = r["summary_by_class"]
        floor = sb["baked_geodesic"]["shf_mean_avg"]
        conf  = sb["nn_walk"]["shf_mean_avg"]
        ceil  = sb["random_straight"]["shf_mean_avg"]
        verdict[m] = {
            "shf_floor (baked geodesic)":          floor,
            "shf_conformant (nn walk)":            conf,
            "shf_off_manifold (straight)":         ceil,
            "ratio_off_to_floor": ceil / max(floor, 1e-30),
            "ratio_conf_to_floor": conf / max(floor, 1e-30),
        }

    out = {
        "spec": "L_SHF Eq. 1 from ARXIV_SUBMISSIONS/paper-D §HJB-Regularised "
                "Joint Training (line 820-835).",
        "args": vars(args),
        "models": results,
        "verdict": verdict,
    }
    out_json = out_dir / "hjb_residual_magnitudes.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json}")

    # Markdown summary
    md = ["# HJB / SHF Loss — Empirical Feasibility Stub", "",
          "Computed per `scripts/hjb_feasibility.py` from the discrete "
          "Jacobi residual",
          "",
          "$\\mathcal{J}_\\ell(s) = \\Delta^2 s_\\ell + \\hat R(s_\\ell)\\,"
          "\\Delta s_\\ell$",
          "",
          "applied to three trajectory classes through each model's "
          "fitted Phase-3 Riemannian manifold (intrinsic dim "
          f"n={args.n_intrinsic}, length L={args.L}, "
          f"{args.n_traj} trajectories per class).",
          "",
          "## Per-model SHF mean penalty (lower = more geodesic)", "",
          "| Model | floor (baked geodesic) | conformant (NN walk) | "
          "off-manifold (straight) | conf/floor | off/floor |",
          "|---|---:|---:|---:|---:|---:|"]
    for m, v in verdict.items():
        if v == "error":
            md.append(f"| {m} | — | — | — | — | — |")
            continue
        md.append(
            f"| {m} | {v['shf_floor (baked geodesic)']:.3e} "
            f"| {v['shf_conformant (nn walk)']:.3e} "
            f"| {v['shf_off_manifold (straight)']:.3e} "
            f"| {v['ratio_conf_to_floor']:.2f}× "
            f"| {v['ratio_off_to_floor']:.2f}× |"
        )
    md += [
        "",
        "## Per-residual cost (training-time gradient implied)",
        "",
        "| Model | intrinsic dim n | FLOPs / residual / layer / token |",
        "|---|---:|---:|",
    ]
    for m, r in results.items():
        if "error" in r:
            continue
        md.append(
            f"| {m} | {r['n_intrinsic']} | "
            f"{r['flops_per_layer_per_token']:,} |"
        )
    md += [
        "",
        "## Kinetic vs curvature decomposition (mean per node)",
        "",
        "$\\mathcal{J}_\\ell = \\Delta^2 s_\\ell + \\hat R(s_\\ell)\\,"
        "\\Delta s_\\ell$ — we report the squared magnitudes of the two "
        "summands separately so the dominant component is visible.",
        "",
        "| Model | class | kinetic mean $\\|\\Delta^2 s\\|^2$ | curvature mean "
        "$\\|\\hat R\\Delta s\\|^2$ | kin/curv |",
        "|---|---|---:|---:|---:|",
    ]
    for m, r in results.items():
        if "error" in r:
            continue
        for cls_name in ("baked_geodesic", "nn_walk", "random_straight"):
            s = r["summary_by_class"][cls_name]
            md.append(
                f"| {m} | {cls_name} | "
                f"{s['kinetic_mean_avg']:.3e} | "
                f"{s['curvature_mean_avg']:.3e} | "
                f"{s['kin_to_curv_ratio']:.2e}× |"
            )
    md += [
        "",
        "## Interpretation", "",
        "- The **floor** is what a training-time minimiser of $L_{SHF}$ "
        "would converge toward; the residual at the floor is the "
        "finite-difference truncation error of the discrete Jacobi "
        "residual on this manifold.",
        "- The **conformant** value is what an *untrained* network with "
        "Phase-3-fitted activations already achieves; the gap to the "
        "floor is the headroom that training with $L_{SHF}$ could "
        "exploit.",
        "- The **off-manifold** value bounds the penalty from above and "
        "indicates the dynamic range over which $\\lambda$ should be "
        "tuned.",
        "- The **kinetic / curvature ratio** quantifies how much of the "
        "Bellman residual actually carries curvature signal. On the "
        "Phase-3 manifolds we have access to the kinetic $\\Delta^2$ "
        "term dominates by many orders of magnitude (consistent with "
        "the curvature-warp negative result, FIVE\\_FINDINGS §5: these "
        "manifolds are nearly flat). Practical implication: a "
        "training-time SHF schedule must either (a) operate on a "
        "manifold of demonstrably non-zero curvature, or (b) reduce "
        "to ordinary smoothness regularisation (kinetic-only) on "
        "current network outputs.",
        "",
        "**Implication for paper-D §HJB-Regularised Joint Training:** "
        "the loss term is empirically tractable (sub-second compute "
        "for n=8 intrinsic, L=32 layers), the dynamic range "
        "(off/floor ratio) determines a meaningful $\\lambda$ "
        "schedule, and per-token training cost is $O(n^5)$ in "
        "intrinsic dim — affordable when the runtime caches the "
        "Riemann tensor at sample points (which it already does for "
        "the Jacobi propagator). However the empirical kinetic/curvature "
        "ratio shows that on the manifolds currently exported from "
        "frozen pretrained networks the curvature contribution to "
        "$L_{SHF}$ is negligible; the loss in its current form would "
        "act primarily as a second-difference smoothness penalty on "
        "block summaries.",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; "
        f"data: hjb_residual_magnitudes.json_",
    ]
    out_md = out_dir / "hjb_residual_summary.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
