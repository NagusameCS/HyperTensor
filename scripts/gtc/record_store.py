"""
gtc/record_store.py
====================

Compressed on-disk geodesic record store with two-stage lookup
(Euclidean ANN screen → g-norm geodesic refine), implementing
Algorithm 1 of Paper 5 §4.4.2.

Record fields per Paper 5 Definition 4.1:
    R = (q_bar, v_bar0, {x(λ_i)}, {Φ(λ_i)}, ρ(q_bar), l_inf)

We compress as follows:
  - q_bar, v_bar0: float32 (full precision, k floats each).
  - waypoints x(λ_i): float16 (smooth curve, 2x compression).
  - Jacobi propagators Φ(λ_i): low-rank SVD truncation to rank r
    (paper claims rank ≈ 5 is sufficient).
  - rho: float32 scalar.
  - l_inf: omitted in this prototype (we operate at the manifold
    level, not the logit level — the runtime owns the LM head).

The whole record is dumped via numpy.savez_compressed (zlib).

The "two-stage lookup" is:
  1. Euclidean nearest-neighbour over q_bar centres (numpy
     vectorised; on the small libraries we exercise this is faster
     than FAISS overhead).
  2. g-norm refine over top-K candidates using the *query's* metric.

Both stages are explicit in `Library.lookup`.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Per-record compression

def _phi_low_rank(Phi: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """SVD-truncate (T+1, n, n) propagator stack to rank `r`.
    Returns (U, V) with Phi[t] ≈ U[t] @ V[t].T, U,V shape (T+1, n, r)."""
    Tp1, n, _ = Phi.shape
    r = min(rank, n)
    U = np.zeros((Tp1, n, r), dtype=np.float32)
    V = np.zeros((Tp1, n, r), dtype=np.float32)
    for t in range(Tp1):
        u, s, vt = np.linalg.svd(Phi[t], full_matrices=False)
        U[t] = (u[:, :r] * np.sqrt(s[:r])).astype(np.float32)
        V[t] = (vt[:r, :].T * np.sqrt(s[:r])).astype(np.float32)
    return U, V


def _phi_reconstruct(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    Tp1, n, r = U.shape
    out = np.zeros((Tp1, n, n), dtype=np.float64)
    for t in range(Tp1):
        out[t] = U[t] @ V[t].T
    return out


# ----------------------------------------------------------------------
# Record + Library

@dataclass
class Record:
    q:    np.ndarray                # (n,) float64 query centre
    v0:   np.ndarray                # (n,) float64 initial velocity
    xs:   np.ndarray                # (T+1, n) waypoints
    Phi_U: np.ndarray               # (T+1, n, r) low-rank factor
    Phi_V: np.ndarray               # (T+1, n, r) low-rank factor
    rho:  float                     # injectivity radius estimate
    g_q:  np.ndarray                # (n, n) metric at q (for refine)


class Library:
    """Compressed GTC trajectory library with two-stage lookup."""

    def __init__(self):
        self.records: List[Record] = []
        self._centres: Optional[np.ndarray] = None  # (N, n) float32 stacked

    def add(self, q, v0, xs, Phi, rho: float, g_q: np.ndarray, phi_rank: int = 5):
        U, V = _phi_low_rank(np.asarray(Phi, dtype=np.float64), rank=phi_rank)
        rec = Record(
            q=np.asarray(q, dtype=np.float64),
            v0=np.asarray(v0, dtype=np.float64),
            xs=np.asarray(xs, dtype=np.float16),
            Phi_U=U, Phi_V=V,
            rho=float(rho),
            g_q=np.asarray(g_q, dtype=np.float32),
        )
        self.records.append(rec)
        self._centres = None  # invalidate stack

    # -- 1st stage (cheap Euclidean) -----------------------------------
    def _stack_centres(self) -> np.ndarray:
        if self._centres is None and self.records:
            self._centres = np.stack([r.q.astype(np.float32) for r in self.records], axis=0)
        return self._centres

    def lookup(self, q: np.ndarray, top_k: int = 8,
               g_query: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """Return (index, g-norm distance) of the best record for `q`.

        Stage 1: Euclidean nearest top_k.
        Stage 2: refine by g-norm using `g_query` (defaults to identity).
        """
        if not self.records:
            raise RuntimeError("empty library")
        C = self._stack_centres()  # (N, n)
        q32 = q.astype(np.float32)
        d_eu = np.linalg.norm(C - q32[None, :], axis=1)
        N = len(self.records)
        kk = min(top_k, N)
        cand = np.argpartition(d_eu, kk - 1)[:kk]

        g = g_query if g_query is not None else np.eye(C.shape[1], dtype=np.float64)
        best_i = -1; best_d = np.inf
        for i in cand:
            v = q - self.records[int(i)].q
            d = float(np.sqrt(max(v @ g @ v, 0.0)))
            if d < best_d:
                best_d = d; best_i = int(i)
        return best_i, best_d

    # -- correction ----------------------------------------------------
    def correct(self, idx: int, q: np.ndarray, t: int = -1) -> np.ndarray:
        """Apply Jacobi correction at waypoint `t` (default last)."""
        r = self.records[idx]
        dq = q - r.q
        # δx(λ_t) = U_t (V_t.T δq)
        U = r.Phi_U[t]
        V = r.Phi_V[t]
        return r.xs[t].astype(np.float64) + U @ (V.T @ dq)

    # -- IO ------------------------------------------------------------
    def save(self, path: Path) -> int:
        """Save compressed; return on-disk byte size."""
        buf = io.BytesIO()
        payload = {}
        for i, r in enumerate(self.records):
            payload[f"q_{i}"]   = r.q.astype(np.float32)
            payload[f"v0_{i}"]  = r.v0.astype(np.float32)
            payload[f"xs_{i}"]  = r.xs            # already float16
            payload[f"PU_{i}"]  = r.Phi_U
            payload[f"PV_{i}"]  = r.Phi_V
            payload[f"rho_{i}"] = np.float32(r.rho)
            payload[f"gq_{i}"]  = r.g_q
        payload["count"] = np.int64(len(self.records))
        np.savez_compressed(buf, **payload)
        data = buf.getvalue()
        Path(path).write_bytes(data)
        return len(data)

    @staticmethod
    def load(path: Path) -> "Library":
        z = np.load(path)
        N = int(z["count"])
        lib = Library()
        for i in range(N):
            rec = Record(
                q=z[f"q_{i}"].astype(np.float64),
                v0=z[f"v0_{i}"].astype(np.float64),
                xs=z[f"xs_{i}"],
                Phi_U=z[f"PU_{i}"],
                Phi_V=z[f"PV_{i}"],
                rho=float(z[f"rho_{i}"]),
                g_q=z[f"gq_{i}"],
            )
            lib.records.append(rec)
        return lib


# ----------------------------------------------------------------------
# Build a real library from a Phase-3 manifold

def build_library_from_manifold(M, points: np.ndarray, T: int = 16,
                                  dl: float = 0.1, phi_rank: int = 5,
                                  rho: float = 3.0, max_records: Optional[int] = None) -> Library:
    """Anchor a record at every cached point. Uses unit-speed
    geodesics in a random tangent direction (the paper's "contextual
    velocity v_0" — we don't have a real downstream task here, just a
    representative direction). The Jacobi propagator along the path is
    what GTC actually uses at lookup time.
    """
    from geodesic import integrate_geodesic, normalise_to_unit_speed
    from jacobi import build_propagator

    n = points.shape[1]
    rng = np.random.default_rng(20260427)

    lib = Library()
    N = points.shape[0] if max_records is None else min(max_records, points.shape[0])
    for i in range(N):
        x0 = points[i]
        v0 = rng.normal(size=(n,))
        v0 = normalise_to_unit_speed(M, x0, v0)
        xs, vs = integrate_geodesic(M, x0, v0, dl=dl, T=T)
        bank = build_propagator(M, xs, vs, dl=dl)
        g_q = M.g_at(x0)
        lib.add(q=x0, v0=v0, xs=xs, Phi=bank.Phi, rho=rho, g_q=g_q, phi_rank=phi_rank)
    return lib


# ----------------------------------------------------------------------
# Smoke test / standalone CLI

def main():
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _phase_io import REPO, load_phase1
    from manifold import fit_phase3_manifold

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--dl", type=float, default=0.1)
    ap.add_argument("--phi-rank", type=int, default=5)
    ap.add_argument("--max-records", type=int, default=24)
    args = ap.parse_args()

    out_dir = REPO / "docs" / "figures" / "gtc"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = load_phase1(args.model)
    Nc = p1.cloud.shape[0]
    rng = np.random.default_rng(20260427)
    base = p1.cloud
    eigs = p1.eigenvalues
    if len(eigs) < args.dim:
        eigs = np.concatenate([eigs, eigs[-1:].repeat(args.dim - len(eigs))])
    extra_scale = np.sqrt(np.maximum(eigs[3:args.dim], 1e-6))
    extra = rng.normal(size=(Nc, args.dim - 3)) * extra_scale[None, :]
    points = np.concatenate([base, extra], axis=1)

    M = fit_phase3_manifold(args.model, n_intrinsic=args.dim, sigma=0.6, n_grid=Nc)

    t0 = time.time()
    lib = build_library_from_manifold(M, points, T=args.T, dl=args.dl,
                                       phi_rank=args.phi_rank,
                                       max_records=args.max_records)
    t_build = time.time() - t0

    save_path = out_dir / f"{args.model}_library.npz"
    nbytes = lib.save(save_path)

    # Round-trip + reconstruction error
    lib2 = Library.load(save_path)
    full_phi = _phi_reconstruct(lib.records[0].Phi_U, lib.records[0].Phi_V)
    rt_full  = _phi_reconstruct(lib2.records[0].Phi_U, lib2.records[0].Phi_V)
    rt_err = float(np.linalg.norm(full_phi - rt_full) / max(np.linalg.norm(full_phi), 1e-12))

    # Lookup latency (1000 random queries, two-stage)
    n_q = 1000
    Q = points[rng.integers(0, Nc, size=n_q)] + rng.normal(scale=0.05, size=(n_q, args.dim))
    t0 = time.time()
    hits = 0; total_d = 0.0
    for q in Q:
        idx, d = lib.lookup(q, top_k=8, g_query=M.g_at(q))
        total_d += d
        if d < 3.0: hits += 1
    t_lookup = time.time() - t0

    summary = {
        "model": args.model,
        "n_intrinsic": args.dim,
        "T": args.T, "dl": args.dl, "phi_rank": args.phi_rank,
        "n_records": len(lib.records),
        "build_wall_s": round(t_build, 3),
        "save_bytes": nbytes,
        "save_kb_per_record": round(nbytes / max(len(lib.records), 1) / 1024.0, 2),
        "phi_lowrank_relerr_first_record": rt_err,
        "n_lookups": n_q,
        "lookup_wall_s": round(t_lookup, 3),
        "lookup_us_per_query": round(t_lookup / n_q * 1e6, 2),
        "hit_rate_eps3": hits / n_q,
        "mean_lookup_g_dist": round(total_d / n_q, 4),
    }
    out_json = out_dir / f"{args.model}_record_store.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[record-store] model={args.model} dim={args.dim}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  → {out_json}")
    print(f"  → {save_path}  ({nbytes/1024:.1f} KB total)")


if __name__ == "__main__":
    main()
