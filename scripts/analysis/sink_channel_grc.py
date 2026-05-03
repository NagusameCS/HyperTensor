"""
Sink-channel-aware GRC projection: pilot experiment.

Hypothesis (Sun et al. 2024, "Massive Activations"; Xiao et al. 2024,
"StreamingLLM"): a small number of input channels in transformer hidden
states carry disproportionately large magnitude. When GRC projects the
attention weights uniformly across all d input channels, those few
high-magnitude channels are compressed just as aggressively as the bulk
content channels, which inflates reconstruction error.

This script tests whether carving out the top-T highest-magnitude
columns of W_Q, W_K, W_V (calibration-free, weight-only proxy for
sink-channel detection) and keeping them at full rank, while applying
GRC only to the residual, reduces reconstruction error at the same or
lower effective parameter budget.

Procedure for each attention matrix W (m x d):
  1. Compute c[j] = ||W[:, j]||_2 for j = 0..d-1.
  2. Pick S = top-T indices by c[j] (treated as union across {Q,K,V} so
     all three matrices share the same sink set, matching how a runtime
     would route hidden-state channels).
  3. W_S  = W[:, S] kept exact (m x T columns, T*m params).
  4. W_R  = W with columns in S zeroed.
  5. GRC at rank k applied to W_R: P_t built from joint Gram of
     {W_Q^R, W_K^R, W_V^R}, top-k eigvecs of normalised K_R.
  6. Recon W_hat = W_S placed back + W_R @ P_R @ P_R^T.
  7. Compare to vanilla GRC at the same k (no sink exemption) AND to
     vanilla GRC at the parameter-matched rank k_eq where
       k_eq = k + T * m / (m + d)
     so the sink-exempt method gets less rank to compensate for the
     extra T*m parameters retained.

Outputs docs/figures/paper-a/sink_channel_pilot.json.

Calibration-free; runs purely off the GGUF weights. ~1-2 min on CPU.
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
OUT = Path("docs/figures/paper-a/sink_channel_pilot.json")
LAYERS = [0, 15, 31]
SINK_T = [0, 1, 2, 4, 8, 16, 32]
RANKS = [512, 1024, 1536]


def deq(t):
    return np.asarray(
        dequantize(t.data, t.tensor_type),
        dtype=np.float32,
    ).reshape(tuple(reversed(t.shape.tolist())))


def joint_col_magnitude(Wq, Wk, Wv):
    """L2 magnitude per input column, summed across Q/K/V."""
    return (
        np.linalg.norm(Wq, axis=0) ** 2
        + np.linalg.norm(Wk, axis=0) ** 2
        + np.linalg.norm(Wv, axis=0) ** 2
    ) ** 0.5


def build_grc_basis(Wq, Wk, Wv, n_iter=3):
    """Top-d eigvecs of normalised joint Gram (K_t  ranked desc)."""
    K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def grc_recon_error(W, P_t, k):
    """||W - W P P^T||_F^2 / ||W||_F^2 with first-k columns of P_t."""
    P = P_t[:, :k]
    W_hat = W @ P @ P.T
    return float((np.linalg.norm(W - W_hat) ** 2) / (np.linalg.norm(W) ** 2))


def sink_aware_recon_error(W, sink_idx, P_t_R, k_R):
    """Sink columns kept exactly; bulk projected via P_t_R at rank k_R."""
    d = W.shape[1]
    mask_R = np.ones(d, dtype=bool)
    mask_R[sink_idx] = False
    W_R = W.copy()
    W_R[:, sink_idx] = 0.0
    P = P_t_R[:, :k_R]
    W_hat_R = W_R @ P @ P.T
    W_hat = W_hat_R.copy()
    W_hat[:, sink_idx] = W[:, sink_idx]
    return float((np.linalg.norm(W - W_hat) ** 2) / (np.linalg.norm(W) ** 2))


def main():
    reader = GGUFReader(str(MODEL))
    by_name = {t.name: t for t in reader.tensors}
    results: dict = {"layers": {}, "config": {
        "model": "Llama-3.1-8B-Instruct-Q4_K_M",
        "layers_analyzed": LAYERS,
        "sink_T_values": SINK_T,
        "ranks": RANKS,
        "sink_selection": "top-T joint column L2 magnitude across {Q,K,V}",
        "metric": "rel-F-squared reconstruction error vs original W",
    }}

    for L in LAYERS:
        Wq = deq(by_name[f"blk.{L}.attn_q.weight"])
        Wk = deq(by_name[f"blk.{L}.attn_k.weight"])
        Wv = deq(by_name[f"blk.{L}.attn_v.weight"])
        m_q, d = Wq.shape
        m_k = Wk.shape[0]
        m_v = Wv.shape[0]
        print(f"\n=== Layer {L}  Wq{Wq.shape} Wk{Wk.shape} Wv{Wv.shape} ===")

        col_mag = joint_col_magnitude(Wq, Wk, Wv)
        order = np.argsort(col_mag)[::-1]

        # Vanilla GRC basis (no sink exemption).
        P_full = build_grc_basis(Wq, Wk, Wv)

        # Stats on the magnitude distribution.
        magn_total = float((col_mag ** 2).sum())
        topk_share = {
            T: float((col_mag[order[:T]] ** 2).sum() / magn_total) if T > 0 else 0.0
            for T in SINK_T
        }
        print("  col-mag^2 share captured by top-T columns:")
        for T in SINK_T:
            if T > 0:
                print(f"    T={T:3d}: {topk_share[T] * 100:.2f}%")

        layer_res: dict = {
            "shape": {"m_q": m_q, "m_k": m_k, "m_v": m_v, "d": d},
            "topT_magnitude_share": topk_share,
            "sink_indices_top32": [int(x) for x in order[:32]],
            "by_T": {},
        }

        for T in SINK_T:
            sink_idx = order[:T] if T > 0 else np.array([], dtype=np.int64)
            if T > 0:
                # Build a sink-aware basis: zero out sink cols then re-Gram.
                Wq_R = Wq.copy(); Wq_R[:, sink_idx] = 0.0
                Wk_R = Wk.copy(); Wk_R[:, sink_idx] = 0.0
                Wv_R = Wv.copy(); Wv_R[:, sink_idx] = 0.0
                P_R = build_grc_basis(Wq_R, Wk_R, Wv_R)
            else:
                P_R = P_full

            row: dict = {"by_k": {}}
            for k in RANKS:
                # --- vanilla GRC at rank k (no sink exemption) ---
                ey_err_q = grc_recon_error(Wq, P_full, k)
                ey_err_k = grc_recon_error(Wk, P_full, k)
                ey_err_v = grc_recon_error(Wv, P_full, k)
                vanilla = (ey_err_q + ey_err_k + ey_err_v) / 3.0

                # --- sink-aware at the same rank k ---
                if T > 0:
                    sa_q = sink_aware_recon_error(Wq, sink_idx, P_R, k)
                    sa_k = sink_aware_recon_error(Wk, sink_idx, P_R, k)
                    sa_v = sink_aware_recon_error(Wv, sink_idx, P_R, k)
                    sa = (sa_q + sa_k + sa_v) / 3.0
                else:
                    sa = vanilla

                # --- sink-aware at parameter-matched lower rank ---
                # extra params kept = T * (m_q + m_k + m_v); compare against
                # the cost of GRC rank-1 column = d (proj) + (m_q+m_k+m_v) (W*P
                # output rows). Effective rank-equivalent shift:
                #   k_eq = k - ceil( T * (m_q+m_k+m_v) / (d + m_q+m_k+m_v) )
                m_sum = m_q + m_k + m_v
                k_shift = int(np.ceil(T * m_sum / (d + m_sum))) if T > 0 else 0
                k_eq = max(8, k - k_shift)
                if T > 0:
                    sa_eq_q = sink_aware_recon_error(Wq, sink_idx, P_R, k_eq)
                    sa_eq_k = sink_aware_recon_error(Wk, sink_idx, P_R, k_eq)
                    sa_eq_v = sink_aware_recon_error(Wv, sink_idx, P_R, k_eq)
                    sa_eq = (sa_eq_q + sa_eq_k + sa_eq_v) / 3.0
                else:
                    sa_eq = vanilla

                row["by_k"][k] = {
                    "vanilla_relF2": vanilla,
                    "sink_aware_relF2_same_k": sa,
                    "sink_aware_relF2_param_matched": sa_eq,
                    "k_param_matched": k_eq,
                    "rel_improvement_same_k_pct": float((vanilla - sa) / vanilla * 100) if vanilla > 0 else 0.0,
                    "rel_improvement_param_matched_pct": float((vanilla - sa_eq) / vanilla * 100) if vanilla > 0 else 0.0,
                }
                print(f"  T={T:3d} k={k}  vanilla={vanilla:.5f}  "
                      f"sink-aware(k)={sa:.5f}  sink-aware(k_eq={k_eq})={sa_eq:.5f}  "
                      f"\u0394_same={(vanilla-sa)/vanilla*100:+.2f}%  "
                      f"\u0394_eq={(vanilla-sa_eq)/vanilla*100:+.2f}%")
            layer_res["by_T"][T] = row

        results["layers"][L] = layer_res

    # ----- aggregate across layers -----
    summary = {}
    for T in SINK_T:
        for k in RANKS:
            improvements_same = []
            improvements_eq = []
            for L in LAYERS:
                row = results["layers"][L]["by_T"][T]["by_k"][k]
                improvements_same.append(row["rel_improvement_same_k_pct"])
                improvements_eq.append(row["rel_improvement_param_matched_pct"])
            summary[f"T={T}_k={k}"] = {
                "mean_rel_improvement_same_k_pct": float(np.mean(improvements_same)),
                "mean_rel_improvement_param_matched_pct": float(np.mean(improvements_eq)),
            }
    results["summary"] = summary

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")
    print("\n=== Summary (mean across layers {0,15,31}) ===")
    for T in SINK_T:
        if T == 0:
            continue
        for k in RANKS:
            s = summary[f"T={T}_k={k}"]
            print(f"  T={T:3d} k={k:4d}  same-k:{s['mean_rel_improvement_same_k_pct']:+6.2f}%  "
                  f"param-matched:{s['mean_rel_improvement_param_matched_pct']:+6.2f}%")


if __name__ == "__main__":
    main()
