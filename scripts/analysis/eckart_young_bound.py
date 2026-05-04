"""
Numerical verification of Eckart--Young--Mirsky theoretical bound for GRC.

The Eckart--Young theorem states that for any matrix W ∈ ℝ^{mn} with singular
values σ_1 ≥ σ_2 ≥ ... ≥ σ_min(m,n), the best rank-k approximation in both
spectral and Frobenius norms is given by truncated SVD:

    ||W - W_k||_F^2 = Σ_{i>k} σ_i^2
    ||W - W_k||_2  = σ_{k+1}

GRC builds the projection from the *combined Gram matrix* K = Σ_i W_i^T W_i,
which is NOT a per-matrix optimal projection --- it's a shared subspace. So the
GRC reconstruction error must be ≥ the Eckart--Young lower bound.

This script measures:
  1. The Eckart--Young oracle bound for each {Q, K, V} matrix at k = 1024, 1536.
  2. The actual GRC error using the *shared* projection P_t from the combined
     Gram matrix.
  3. The ratio (GRC excess error) --- a tight gap implies the shared basis is
     near-optimal; a loose gap motivates per-matrix bases (future work item).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = Path(
    r"C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF"
    r"\blobs\7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
)
OUT = Path("docs/figures/eckart_young_bound.json")
LAYERS = [0, 15, 31]
RANKS = [512, 1024, 1536, 2048]


def deq(t):
    return np.asarray(
        dequantize(t.data, t.tensor_type),
        dtype=np.float32,
    ).reshape(tuple(reversed(t.shape.tolist())))


def eckart_young_error(W, k):
    """||W - W_k||_F^2 / ||W||_F^2 = Σ_{i>k} σ_i^2 / Σ σ_i^2."""
    sv = np.linalg.svd(W, compute_uv=False)
    e = sv ** 2
    if k >= len(sv):
        return 0.0, sv
    tail = e[k:].sum()
    total = e.sum()
    return float(tail / total), sv


def grc_error(W, P_t, k):
    """Actual GRC reconstruction error: W - W·P_t·P_t^T (rank-k projection)."""
    P = P_t[:, :k]                          # d  k
    W_hat = W @ P @ P.T                     # m  d
    err = np.linalg.norm(W - W_hat) ** 2
    total = np.linalg.norm(W) ** 2
    return float(err / total)


def build_grc_basis(W_q, W_k, W_v, n_iter=3):
    """Build P_t the way GRC does --- eigvecs of normalised combined Gram matrix."""
    K = W_q.T @ W_q + W_k.T @ W_k + W_v.T @ W_v
    K = K / np.linalg.norm(K, "fro")
    # Power iteration to stabilise top eigvecs (as in axiom_exploit.c)
    d = K.shape[0]
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    P_t = eigvecs[:, order]                  # d  d, sorted desc
    return P_t


def main():
    reader = GGUFReader(str(MODEL))
    by_name = {t.name: t for t in reader.tensors}
    results = {"layers": {}, "summary": {}}

    for L in LAYERS:
        Wq = deq(by_name[f"blk.{L}.attn_q.weight"])
        Wk = deq(by_name[f"blk.{L}.attn_k.weight"])
        Wv = deq(by_name[f"blk.{L}.attn_v.weight"])
        print(f"\n=== Layer {L} ===")
        print(f"  Wq {Wq.shape}, Wk {Wk.shape}, Wv {Wv.shape}")

        # Build GRC shared basis
        P_t = build_grc_basis(Wq, Wk, Wv)
        print(f"  P_t built: shape={P_t.shape}")

        layer_res: dict = {}
        for k in RANKS:
            row = {}
            for name, W in [("Q", Wq), ("K", Wk), ("V", Wv)]:
                ey_err, _ = eckart_young_error(W, k)
                grc_err = grc_error(W, P_t, k)
                row[name] = {
                    "eckart_young_relF2": ey_err,
                    "grc_relF2": grc_err,
                    "excess_factor": (grc_err / ey_err) if ey_err > 0 else float("inf"),
                }
                print(f"  k={k} {name}: EY={ey_err:.4f}  GRC={grc_err:.4f}  "
                      f" {row[name]['excess_factor']:.3f}")
            layer_res[k] = row
        results["layers"][L] = layer_res

    #  Aggregate 
    for k in RANKS:
        ey_vals, grc_vals, ex = [], [], []
        for L in LAYERS:
            for n in ("Q", "K", "V"):
                ey_vals.append(results["layers"][L][k][n]["eckart_young_relF2"])
                grc_vals.append(results["layers"][L][k][n]["grc_relF2"])
                ex.append(results["layers"][L][k][n]["excess_factor"])
        results["summary"][k] = {
            "ey_mean_relF2": float(np.mean(ey_vals)),
            "grc_mean_relF2": float(np.mean(grc_vals)),
            "excess_factor_mean": float(np.mean(ex)),
            "excess_factor_max": float(np.max(ex)),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Wrote {OUT}")
    print("\nSUMMARY (mean across {Q,K,V}  layers {0,15,31}):")
    for k in RANKS:
        s = results["summary"][k]
        print(f"  k={k:5d}  EY-bound rel-F²={s['ey_mean_relF2']:.4f}  "
              f"GRC rel-F²={s['grc_mean_relF2']:.4f}  "
              f"excess={s['excess_factor_mean']:.2f} (max {s['excess_factor_max']:.2f})")


if __name__ == "__main__":
    main()
