"""
gtc/formal_hypothesis_eval.py
================================

Implements two testable hypothesis probes inspired by the formal notes:

1) Attention-logmap v0 proxy:
     v_hat(x,y) ~ -grad_x log Attn(x,y)
   where Attn is modelled as a heat-kernel-like geodesic RBF.

2) Spectral injectivity-radius proxy:
     rho_hat ~ C * pi / (sqrt(lambda_max(Cov(grad_x A))) * sigma_max(A))
   calibrated against an empirically measured local rho_true from Jacobi error
   sweeps on the fitted manifold.

This is an experimental script: it does not claim universal theorem closure.
It reports whether these proxies are useful on HyperTensor's fitted LM manifold.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _phase_io import REPO, load_phase1
from geodesic import integrate_geodesic, normalise_to_unit_speed
from jacobi import build_propagator
from manifold import fit_phase3_manifold


def _intrinsic_lift(model: str, dim: int, seed: int = 20260427) -> np.ndarray:
    p1 = load_phase1(model)
    rng = np.random.default_rng(seed)
    base = p1.cloud
    nc = base.shape[0]
    eigs = p1.eigenvalues
    if len(eigs) < dim:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(dim - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:dim], 1e-6))
    extra = rng.normal(size=(nc, dim - 3)) * extra_scale[None, :]
    return np.concatenate([base, extra], axis=1)


def _g_dist(m, x: np.ndarray, y: np.ndarray) -> float:
    g = m.g_at(0.5 * (x + y))
    v = y - x
    return float(np.sqrt(max(v @ g @ v, 0.0)))


def _attn_kernel(m, x: np.ndarray, y: np.ndarray, tau: float) -> float:
    d = _g_dist(m, x, y)
    return float(np.exp(-(d * d) / max(4.0 * tau, 1e-9)))


def _grad_neglog_attn(m, x: np.ndarray, y: np.ndarray, tau: float, h: float) -> np.ndarray:
    n = x.shape[0]
    g = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ep = x.copy(); ep[i] += h
        em = x.copy(); em[i] -= h
        ap = _attn_kernel(m, ep, y, tau)
        am = _attn_kernel(m, em, y, tau)
        fp = -math.log(max(ap, 1e-30))
        fm = -math.log(max(am, 1e-30))
        g[i] = (fp - fm) / (2.0 * h)
    return g


def _grad_neglog_attn_local_metric(m, x: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Closed-form gradient under local constant-metric approximation.

    If -log Attn(x,y) = (x-y)^T g_x (x-y) / (4 tau), then
    grad_x[-log Attn] = g_x (x-y) / (2 tau).
    """
    gx = m.g_at(x)
    return (gx @ (x - y)) / max(2.0 * tau, 1e-12)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip((a @ b) / (na * nb), -1.0, 1.0))


def _mean_rel_endpoint_err(m, x0: np.ndarray, v0: np.ndarray, phi_t: np.ndarray,
                           eps: float, n_perturb: int, dl: float, t_steps: int,
                           rng: np.random.Generator) -> float:
    xs_ref, _ = integrate_geodesic(m, x0, v0, dl=dl, T=t_steps)
    x_ref = xs_ref[-1]
    errs = []
    for _ in range(n_perturb):
        d0 = rng.normal(size=x0.shape[0])
        d0 *= eps / max(np.linalg.norm(d0), 1e-12)
        xs_true, _ = integrate_geodesic(m, x0 + d0, v0, dl=dl, T=t_steps)
        x_true = xs_true[-1]
        x_pred = x_ref + phi_t @ d0
        g_end = m.g_at(x_true)
        num = float(np.sqrt(max((x_true - x_pred) @ g_end @ (x_true - x_pred), 0.0)))
        den = max(float(np.sqrt(max((x_true - x0) @ g_end @ (x_true - x0), 0.0))), 1e-9)
        errs.append(num / den)
    return float(np.mean(errs))


def _rho_true_for_anchor(m, x0: np.ndarray, v0: np.ndarray,
                         eps_grid: list[float], tau_err: float,
                         n_perturb: int, dl: float, t_steps: int,
                         rng: np.random.Generator) -> tuple[float, list[dict]]:
    xs, vs = integrate_geodesic(m, x0, v0, dl=dl, T=t_steps)
    bank = build_propagator(m, xs, vs, dl=dl)
    phi_t = bank.Phi[-1]
    detail = []
    best = 0.0
    for eps in eps_grid:
        mean_err = _mean_rel_endpoint_err(
            m, x0, v0, phi_t, eps, n_perturb=n_perturb,
            dl=dl, t_steps=t_steps, rng=rng
        )
        detail.append({"eps": float(eps), "mean_rel_err": float(mean_err)})
        if mean_err <= tau_err:
            best = max(best, float(eps))
    return best, detail


def _rho_proxy_components(m, x: np.ndarray, nbrs: np.ndarray,
                          tau: float, h: float) -> tuple[float, float, float]:
    a = np.array([_attn_kernel(m, x, y, tau) for y in nbrs], dtype=np.float64)
    sigma_max = float(np.linalg.norm(a))

    grads = []
    n = x.shape[0]
    for y in nbrs:
        g = np.zeros(n, dtype=np.float64)
        for i in range(n):
            ep = x.copy(); ep[i] += h
            em = x.copy(); em[i] -= h
            ap = _attn_kernel(m, ep, y, tau)
            am = _attn_kernel(m, em, y, tau)
            g[i] = (ap - am) / (2.0 * h)
        grads.append(g)

    gmat = np.stack(grads, axis=0)
    cov = np.cov(gmat, rowvar=False)
    if cov.ndim == 0:
        lam = float(abs(cov))
    else:
        w = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        lam = float(max(w[-1], 1e-12))

    raw = float(math.pi / (math.sqrt(max(lam, 1e-12)) * max(sigma_max, 1e-12)))
    return raw, sigma_max, lam


def run(model: str = "smollm2-135m", dim: int = 8,
        n_queries: int = 20, n_anchors: int = 10,
        tau_attn: float = 0.5, grad_h: float = 1e-3,
        dl: float = 0.05, t_steps: int = 8,
        rho_tau_err: float = 0.05,
        rho_tau_err_strict: float = 1e-9,
    rho_local_max_g: float = 1.6,
        seed: int = 20260427) -> dict:
    rng = np.random.default_rng(seed)
    points = _intrinsic_lift(model, dim=dim, seed=seed)
    nc = points.shape[0]
    m = fit_phase3_manifold(model, n_intrinsic=dim, sigma=0.6, n_grid=nc)

    # --- A) v0 proxy test -------------------------------------------------
    q_idx = rng.choice(nc, size=min(n_queries, nc), replace=False)
    v0_rows_fd = []
    v0_rows_local = []

    for qi in q_idx:
        x = points[int(qi)]
        d = np.array([_g_dist(m, x, points[j]) for j in range(nc)], dtype=np.float64)
        order = np.argsort(d)
        # 2nd nearest avoids self-match at index 0
        yi = int(order[1]) if len(order) > 1 else int(order[0])
        y = points[yi]

        gneglog_fd = _grad_neglog_attn(m, x, y, tau=tau_attn, h=grad_h)
        gneglog_local = _grad_neglog_attn_local_metric(m, x, y, tau=tau_attn)

        # FD proxy (direct theorem-shaped form)
        v_hat_fd = normalise_to_unit_speed(m, x, -gneglog_fd)
        # Local-metric proxy: invert g_x before normalisation.
        gx = m.g_at(x)
        try:
            gx_inv = np.linalg.inv(gx + 1e-9 * np.eye(dim))
        except np.linalg.LinAlgError:
            gx_inv = np.linalg.pinv(gx)
        v_hat_local = normalise_to_unit_speed(m, x, -(gx_inv @ gneglog_local))
        v_true = normalise_to_unit_speed(m, x, y - x)

        # Compare short-horizon rollout against local target direction rollout.
        xh_fd, _ = integrate_geodesic(m, x, v_hat_fd, dl=dl, T=t_steps)
        xh_local, _ = integrate_geodesic(m, x, v_hat_local, dl=dl, T=t_steps)
        xt, _ = integrate_geodesic(m, x, v_true, dl=dl, T=t_steps)
        end_hat_fd = xh_fd[-1]
        end_hat_local = xh_local[-1]
        end_true = xt[-1]
        g_end = m.g_at(end_true)
        num_fd = float(np.sqrt(max((end_hat_fd - end_true) @ g_end @ (end_hat_fd - end_true), 0.0)))
        num_local = float(np.sqrt(max((end_hat_local - end_true) @ g_end @ (end_hat_local - end_true), 0.0)))
        den = max(float(np.sqrt(max((end_true - x) @ g_end @ (end_true - x), 0.0))), 1e-9)

        v0_rows_fd.append({
            "query_idx": int(qi),
            "target_idx": int(yi),
            "g_dist_xy": float(d[yi]),
            "cosine_vhat_vtrue": _cos(v_hat_fd, v_true),
            "rollout_rel_err": float(num_fd / den),
        })

        v0_rows_local.append({
            "query_idx": int(qi),
            "target_idx": int(yi),
            "g_dist_xy": float(d[yi]),
            "cosine_vhat_vtrue": _cos(v_hat_local, v_true),
            "rollout_rel_err": float(num_local / den),
        })

    v0_cos_fd = np.array([r["cosine_vhat_vtrue"] for r in v0_rows_fd], dtype=np.float64)
    v0_err_fd = np.array([r["rollout_rel_err"] for r in v0_rows_fd], dtype=np.float64)
    v0_cos_local = np.array([r["cosine_vhat_vtrue"] for r in v0_rows_local], dtype=np.float64)
    v0_err_local = np.array([r["rollout_rel_err"] for r in v0_rows_local], dtype=np.float64)

    # --- B) rho proxy test ------------------------------------------------
    a_idx = rng.choice(nc, size=min(n_anchors, nc), replace=False)
    eps_grid = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20, 6.40]
    rho_rows = []

    for ai in a_idx:
        x0 = points[int(ai)]
        v0 = normalise_to_unit_speed(m, x0, rng.normal(size=(dim,)))
        rho_true, sweep = _rho_true_for_anchor(
            m, x0, v0, eps_grid=eps_grid, tau_err=rho_tau_err,
            n_perturb=6, dl=dl, t_steps=t_steps, rng=rng
        )
        rho_true_strict, sweep_strict = _rho_true_for_anchor(
            m, x0, v0, eps_grid=eps_grid, tau_err=rho_tau_err_strict,
            n_perturb=6, dl=dl, t_steps=t_steps, rng=rng
        )

        # Strict-local neighbours for attention/covariance proxy.
        d = np.array([_g_dist(m, x0, points[j]) for j in range(nc)], dtype=np.float64)
        nn_local = np.where((d > 0.0) & (d <= rho_local_max_g))[0]
        if len(nn_local) < 4:
            nn = np.argsort(d)[1: min(13, nc)]
        else:
            nn = nn_local[np.argsort(d[nn_local])][: min(12, len(nn_local))]
        nbrs = points[nn]
        rho_raw, sigma_max, lam_max = _rho_proxy_components(
            m, x0, nbrs, tau=tau_attn, h=grad_h
        )

        rho_rows.append({
            "anchor_idx": int(ai),
            "rho_true": float(rho_true),
            "rho_true_strict": float(rho_true_strict),
            "rho_raw": float(rho_raw),
            "sigma_max_A": float(sigma_max),
            "lambda_max_cov_gradA": float(lam_max),
            "n_local_neighbors": int(len(nn)),
            "sweep": sweep,
            "sweep_strict": sweep_strict,
        })

    y_all = np.array([r["rho_true_strict"] for r in rho_rows], dtype=np.float64)
    x_all = np.array([r["rho_raw"] for r in rho_rows], dtype=np.float64)

    eps_lo = float(min(eps_grid))
    eps_hi = float(max(eps_grid))
    # Unsaturated strict-local anchors only: avoid floor/ceiling-clipped rho_true.
    fit_mask = (y_all > eps_lo + 1e-12) & (y_all < eps_hi - 1e-12)
    if int(np.sum(fit_mask)) >= 2:
        x_fit = x_all[fit_mask]
        y_fit = y_all[fit_mask]
    else:
        x_fit = x_all
        y_fit = y_all
        fit_mask = np.ones_like(y_all, dtype=bool)

    denom = float(np.dot(x_fit, x_fit))
    c_hat = float(np.dot(x_fit, y_fit) / denom) if denom > 1e-12 else 0.0
    y_hat = c_hat * x_all

    if len(y_fit) >= 2 and np.std(x_fit) > 1e-12 and np.std(y_fit) > 1e-12:
        corr_fit = float(np.corrcoef(x_fit, y_fit)[0, 1])
    else:
        corr_fit = 0.0

    if len(y_all) >= 2 and np.std(x_all) > 1e-12 and np.std(y_all) > 1e-12:
        corr_all = float(np.corrcoef(x_all, y_all)[0, 1])
    else:
        corr_all = 0.0

    nz = y_all > 1e-6
    if np.any(nz):
        mape = float(np.mean(np.abs(y_hat[nz] - y_all[nz]) / y_all[nz]))
    else:
        mape = float("nan")
    rmse = float(np.sqrt(np.mean((y_hat - y_all) ** 2)))

    for i, r in enumerate(rho_rows):
        r["rho_hat"] = float(y_hat[i])
        r["rho_abs_err"] = float(abs(y_hat[i] - y_all[i]))
        r["fit_unsaturated"] = bool(fit_mask[i])

    return {
        "model": model,
        "n_intrinsic": dim,
        "seed": int(seed),
        "v0_proxy": {
            "n_queries": int(len(v0_rows_fd)),
            "tau_attn": float(tau_attn),
            "grad_h": float(grad_h),
            "dl": float(dl),
            "T": int(t_steps),
            "fd_proxy": {
                "cosine_mean": float(v0_cos_fd.mean()),
                "cosine_p50": float(np.quantile(v0_cos_fd, 0.5)),
                "cosine_p95": float(np.quantile(v0_cos_fd, 0.95)),
                "rollout_rel_err_mean": float(v0_err_fd.mean()),
                "rollout_rel_err_p50": float(np.quantile(v0_err_fd, 0.5)),
                "rollout_rel_err_p95": float(np.quantile(v0_err_fd, 0.95)),
                "rows": v0_rows_fd,
            },
            "local_metric_proxy": {
                "cosine_mean": float(v0_cos_local.mean()),
                "cosine_p50": float(np.quantile(v0_cos_local, 0.5)),
                "cosine_p95": float(np.quantile(v0_cos_local, 0.95)),
                "rollout_rel_err_mean": float(v0_err_local.mean()),
                "rollout_rel_err_p50": float(np.quantile(v0_err_local, 0.5)),
                "rollout_rel_err_p95": float(np.quantile(v0_err_local, 0.95)),
                "rows": v0_rows_local,
            },
        },
        "rho_proxy": {
            "n_anchors": int(len(rho_rows)),
            "eps_grid": eps_grid,
            "rho_tau_err": float(rho_tau_err),
            "rho_tau_err_strict": float(rho_tau_err_strict),
            "rho_local_max_g": float(rho_local_max_g),
            "tau_attn": float(tau_attn),
            "grad_h": float(grad_h),
            "fit_unsaturated_count": int(np.sum(fit_mask)),
            "fit_total_count": int(len(fit_mask)),
            "calibration_C": float(c_hat),
            "pearson_corr_raw_vs_true_fit_unsaturated": float(corr_fit),
            "pearson_corr_raw_vs_true_all": float(corr_all),
            "mape_hat_vs_true": float(mape),
            "mape_nonzero_count": int(np.sum(nz)),
            "mape_total_count": int(len(y_all)),
            "rmse_hat_vs_true": float(rmse),
            "rows": rho_rows,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--anchors", type=int, default=10)
    ap.add_argument("--tau-attn", type=float, default=0.5)
    ap.add_argument("--grad-h", type=float, default=1e-3)
    ap.add_argument("--dl", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--rho-tau-err", type=float, default=0.05)
    ap.add_argument("--rho-tau-err-strict", type=float, default=1e-9)
    ap.add_argument("--rho-local-max-g", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    out = run(
        model=args.model,
        dim=args.dim,
        n_queries=args.queries,
        n_anchors=args.anchors,
        tau_attn=args.tau_attn,
        grad_h=args.grad_h,
        dl=args.dl,
        t_steps=args.T,
        rho_tau_err=args.rho_tau_err,
        rho_tau_err_strict=args.rho_tau_err_strict,
        rho_local_max_g=args.rho_local_max_g,
        seed=args.seed,
    )

    out_dir = REPO / "docs" / "figures" / "gtc"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.model}_formal_hypothesis_eval.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    v = out["v0_proxy"]
    r = out["rho_proxy"]
    v_fd = v["fd_proxy"]
    v_local = v["local_metric_proxy"]
    print(f"[formal-eval] model={args.model} dim={args.dim}")
    print(
        "  v0 fd proxy: "
        f"cos(mean/p50/p95)=({v_fd['cosine_mean']:.3f}/{v_fd['cosine_p50']:.3f}/{v_fd['cosine_p95']:.3f}) "
        f"rollout_err(mean/p95)=({v_fd['rollout_rel_err_mean']:.3%}/{v_fd['rollout_rel_err_p95']:.3%})"
    )
    print(
        "  v0 local-metric proxy: "
        f"cos(mean/p50/p95)=({v_local['cosine_mean']:.3f}/{v_local['cosine_p50']:.3f}/{v_local['cosine_p95']:.3f}) "
        f"rollout_err(mean/p95)=({v_local['rollout_rel_err_mean']:.3%}/{v_local['rollout_rel_err_p95']:.3%})"
    )
    print(
        "  rho proxy: "
        f"corr_fit={r['pearson_corr_raw_vs_true_fit_unsaturated']:.3f} "
        f"corr_all={r['pearson_corr_raw_vs_true_all']:.3f} "
        f"fit={r['fit_unsaturated_count']}/{r['fit_total_count']} "
        f"mape={r['mape_hat_vs_true']:.3%} rmse={r['rmse_hat_vs_true']:.3f} "
        f"C={r['calibration_C']:.3e}"
    )
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
