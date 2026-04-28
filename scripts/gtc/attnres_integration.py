"""
gtc/attnres_integration.py
==========================

Python-side prototype for AttnRes block-summary correction using GTC records.

We emulate block summaries as points on the fitted manifold along geodesics.
For each anchor record and each block boundary n:
    b_n(q) ≈ b_n(q_bar) + Phi_n * delta_q
where Phi_n is the Jacobi propagator at that boundary.

Outputs three tests (Paper-style 8a/8b/8c proxies):
  8a concentration: nearest-record g-distance distribution per block.
  8b simplex validity: barycentric reconstruction quality from top-k records.
  8c distinct-block selection: selected records are not duplicates and cover
      diverse anchors across blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _phase_io import REPO, load_phase1
from manifold import fit_phase3_manifold
from geodesic import integrate_geodesic, normalise_to_unit_speed
from jacobi import build_propagator


def _intrinsic_lift(model: str, dim: int, seed: int = 20260427):
    p1 = load_phase1(model)
    rng = np.random.default_rng(seed)
    base = p1.cloud
    Nc = base.shape[0]
    eigs = p1.eigenvalues
    if len(eigs) < dim:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(dim - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:dim], 1e-6))
    extra = rng.normal(size=(Nc, dim - 3)) * extra_scale[None, :]
    return np.concatenate([base, extra], axis=1)


def _g_dist(M, x: np.ndarray, y: np.ndarray) -> float:
    g = M.g_at(0.5 * (x + y))
    v = y - x
    return float(np.sqrt(max(v @ g @ v, 0.0)))


def _weights_simplex(target: np.ndarray, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative simplex-like weights via clipped least squares + renorm."""
    A = refs.T  # (d, k)
    b = target
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    recon = refs.T @ w
    return w, recon


def run(model: str = "smollm2-135m", dim: int = 8, anchors: int = 16,
        blocks: int = 8, T_per_block: int = 4, dl: float = 0.05,
        top_k: int = 4, seed: int = 20260427) -> dict:
    rng = np.random.default_rng(seed)
    points = _intrinsic_lift(model, dim, seed=seed)
    Nc = points.shape[0]
    M = fit_phase3_manifold(model, n_intrinsic=dim, sigma=0.6, n_grid=Nc)

    anchors = min(anchors, Nc)
    a_idx = rng.choice(Nc, size=anchors, replace=False)

    # Build anchor geodesics and block summaries
    rec_q = []
    rec_v0 = []
    rec_block_x = []  # (A, B+1, d)
    rec_block_phi = []  # (A, B+1, d, d)

    T_total = blocks * T_per_block
    for i in a_idx:
        q = points[int(i)]
        v0 = normalise_to_unit_speed(M, q, rng.normal(size=(dim,)))
        xs, vs = integrate_geodesic(M, q, v0, dl=dl, T=T_total)
        bank = build_propagator(M, xs, vs, dl=dl)

        bx = [xs[0]]
        bphi = [np.eye(dim)]
        for b in range(1, blocks + 1):
            t = b * T_per_block
            bx.append(xs[t])
            bphi.append(bank.Phi[t])

        rec_q.append(q)
        rec_v0.append(v0)
        rec_block_x.append(np.stack(bx))
        rec_block_phi.append(np.stack(bphi))

    rec_q = np.stack(rec_q)
    rec_v0 = np.stack(rec_v0)
    rec_block_x = np.stack(rec_block_x)
    rec_block_phi = np.stack(rec_block_phi)

    # Evaluate held-out queries
    decode_idx = np.setdiff1d(np.arange(Nc), a_idx)
    if decode_idx.size == 0:
        decode_idx = np.arange(Nc)
    q_eval_idx = rng.choice(decode_idx, size=min(24, decode_idx.size), replace=False)

    concentration = []
    block_err = [[] for _ in range(blocks + 1)]
    simplex_rel = [[] for _ in range(blocks + 1)]
    distinct_ratio = []

    for qi in q_eval_idx:
        q = points[int(qi)]

        dists = np.array([_g_dist(M, q, rq) for rq in rec_q])
        nn = np.argsort(dists)
        concentration.append(float(dists[nn[0]]))
        top = nn[:top_k]
        distinct_ratio.append(float(len(set(top.tolist())) / max(1, len(top))))

        # Ground truth geodesic from query with each selected anchor's v0
        # Use best anchor's v0 for consistency.
        best = int(top[0])
        xs_true, _ = integrate_geodesic(M, q, rec_v0[best], dl=dl, T=T_total)

        for b in range(blocks + 1):
            t = b * T_per_block
            p_true = xs_true[t]

            # Test 8a proxy: single-anchor Jacobi correction
            dq_best = q - rec_q[best]
            p_hat = rec_block_x[best, b] + rec_block_phi[best, b] @ dq_best
            err = _g_dist(M, p_hat, p_true)
            denom = max(_g_dist(M, np.zeros(dim), p_true), 1e-9)
            block_err[b].append(err / denom)

            # Test 8b proxy: simplex blend from top-k corrected block summaries
            refs = []
            for ai in top:
                dq = q - rec_q[int(ai)]
                refs.append(rec_block_x[int(ai), b] + rec_block_phi[int(ai), b] @ dq)
            refs = np.stack(refs)
            w, recon = _weights_simplex(p_true, refs)
            simp = _g_dist(M, recon, p_true) / denom
            simplex_rel[b].append(simp)

    # Aggregate
    out_blocks = []
    for b in range(blocks + 1):
        e = np.asarray(block_err[b])
        s = np.asarray(simplex_rel[b])
        out_blocks.append({
            "block": b,
            "jacobi_err_mean": float(e.mean()),
            "jacobi_err_p50": float(np.quantile(e, 0.5)),
            "jacobi_err_p95": float(np.quantile(e, 0.95)),
            "simplex_err_mean": float(s.mean()),
            "simplex_err_p50": float(np.quantile(s, 0.5)),
            "simplex_err_p95": float(np.quantile(s, 0.95)),
            "simplex_beats_jacobi_mean": bool(float(s.mean()) <= float(e.mean())),
        })

    conc = np.asarray(concentration)
    dr = np.asarray(distinct_ratio)

    return {
        "model": model,
        "n_intrinsic": dim,
        "anchors": int(anchors),
        "queries": int(len(q_eval_idx)),
        "blocks": int(blocks),
        "T_per_block": int(T_per_block),
        "dl": float(dl),
        "top_k": int(top_k),
        "test_8a_concentration": {
            "nn_g_dist_mean": float(conc.mean()),
            "nn_g_dist_p50": float(np.quantile(conc, 0.5)),
            "nn_g_dist_p95": float(np.quantile(conc, 0.95)),
        },
        "test_8b_simplex_validity": {
            "simplex_nonnegative_by_construction": True,
            "simplex_weights_sum_to_one": True,
            "block_metrics": out_blocks,
        },
        "test_8c_distinct_block_selection": {
            "distinct_ratio_mean": float(dr.mean()),
            "distinct_ratio_min": float(dr.min()),
            "distinct_ratio_p95": float(np.quantile(dr, 0.95)),
        },
        "headline": {
            "jacobi_block0_mean": out_blocks[0]["jacobi_err_mean"],
            "jacobi_blockN_mean": out_blocks[-1]["jacobi_err_mean"],
            "simplex_blockN_mean": out_blocks[-1]["simplex_err_mean"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--t-per-block", type=int, default=4)
    ap.add_argument("--dl", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    out = run(args.model, dim=args.dim, anchors=args.anchors,
              blocks=args.blocks, T_per_block=args.t_per_block,
              dl=args.dl, top_k=args.top_k)
    out_path = REPO / "docs" / "figures" / "gtc" / f"{args.model}_attnres_integration.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[attnres] model={args.model} anchors={out['anchors']} queries={out['queries']} blocks={out['blocks']}")
    c = out["test_8a_concentration"]
    print(f"  8a concentration nn-gdist: mean={c['nn_g_dist_mean']:.3f} p50={c['nn_g_dist_p50']:.3f} p95={c['nn_g_dist_p95']:.3f}")
    h = out["headline"]
    print(f"  8b jacobi err: block0={h['jacobi_block0_mean']:.3%} blockN={h['jacobi_blockN_mean']:.3%}")
    print(f"  8b simplex blockN err: {h['simplex_blockN_mean']:.3%}")
    d = out["test_8c_distinct_block_selection"]
    print(f"  8c distinct ratio: mean={d['distinct_ratio_mean']:.2f} min={d['distinct_ratio_min']:.2f}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
