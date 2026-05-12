#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""Direct tests for whether the UGT basis B captures zone structure.

Three sharp experiments on the same model pair we used for universality:
SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct. 400 prompts x 4 domains.

E1 -- Concept-axis recovery.
   For each domain d, fit logistic regression d-vs-rest on raw hidden
   states; the unit-normalised weight vector w_d is the "concept axis"
   for that domain in this model.  Measure how much of w_d lies inside
   col(B) where B = top-k SVD basis: cap_in_B(d) = ||B B^T w_d||^2.
   Compare to random orthonormal B' of the same rank.  If B captures
   concept axes substantially better than random at low k, the basis IS
   directionally privileged for zones.

E2 -- Clusterability under projection.
   Project hidden states through B vs B', run KMeans(n=4), measure
   adjusted Rand index against true domain labels.  Compares whether B
   preserves the cluster structure better than random at low rank.

E3 -- Snipe-style coordinate-level ablation.
   For each domain d, find the basis coordinate j* within B whose
   removal most degrades a domain-d classifier (greedy, like Snipe).
   Project hidden states through B, zero coordinate j*, project back,
   re-classify.  Measure drop in domain-d AUROC.  Repeat with random
   B'.  If B's coordinate ablation is substantially more damaging than
   B''s coordinate ablation, the published Snipe pipeline genuinely
   uses direction structure.

Output: benchmarks/ugt_zone_recovery.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from universal_taxonomy_test import (
    build_corpus, collect_hidden, MODELS, random_orthonormal,
)

OUT = ROOT / "benchmarks" / "ugt_zone_recovery.json"
DOMAINS = ["factual", "code", "math", "creative"]


def svd_basis(H, k):
    Hc = H - H.mean(0, keepdims=True)
    _, _, Vh = np.linalg.svd(Hc, full_matrices=False)
    return Vh[:k].T  # [d, k]


# ------------- E1: concept-axis recovery -------------
def concept_axes(H, labels):
    """For each domain, fit one-vs-rest logistic regression; return
    a [n_domains, d] matrix of unit-normalised weight vectors."""
    axes = []
    for d in DOMAINS:
        y = (labels == d).astype(int)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(H, y)
        w = clf.coef_[0]
        axes.append(w / (np.linalg.norm(w) + 1e-12))
    return np.array(axes)  # [4, d]


def basis_capture(axes, B):
    """For each axis w, return ||B B^T w||^2 = squared cosine of axis to col(B).
    Returns vector of length n_axes."""
    P = B @ B.T  # [d, d] projector onto col(B)
    captured = np.einsum("ij,jk,ik->i", axes, P, axes)
    return captured  # in [0, 1]


# ------------- E2: clusterability -------------
def kmeans_ari(P, labels, n_clusters=4, n_init=10):
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=0).fit(P)
    return adjusted_rand_score(labels, km.labels_)


# ------------- E3: Snipe-style coordinate ablation -------------
def snipe_drop(H, labels, B, domain, mu=None):
    """Project H through B (so coords in R^k); for each coordinate j,
    measure logreg AUROC for {domain vs rest} after zeroing coord j and
    projecting back.  Return: baseline_auroc, best_coord_to_kill, drop."""
    if mu is None:
        mu = H.mean(0, keepdims=True)
    Hc = H - mu
    P = Hc @ B          # [N, k]
    k = P.shape[1]
    y = (labels == domain).astype(int)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    def auroc_with_coord_zero(j_kill):
        Pmod = P.copy()
        if j_kill is not None:
            Pmod[:, j_kill] = 0.0
        # rebuild ablated H by Pmod @ B^T + mu
        Habl = Pmod @ B.T + mu
        # cross-validated AUROC
        aurocs = []
        for tr, te in skf.split(Habl, y):
            clf = LogisticRegression(max_iter=2000).fit(Habl[tr], y[tr])
            scores = clf.decision_function(Habl[te])
            aurocs.append(roc_auc_score(y[te], scores))
        return float(np.mean(aurocs))

    base = auroc_with_coord_zero(None)
    drops = np.array([base - auroc_with_coord_zero(j) for j in range(k)])
    j_star = int(np.argmax(drops))
    return {"baseline_auroc": base, "best_coord": j_star, "best_drop": float(drops[j_star]),
            "mean_drop": float(np.mean(drops)), "max_drop": float(np.max(drops))}


# ------------- main -------------
def main():
    rng = np.random.default_rng(7)
    print("[corpus]")
    corpus = build_corpus(per_domain=100)
    labels = np.array([d for d, _ in corpus])

    Hs = {}
    for short, mid in MODELS:
        print(f"[hidden] {short}")
        Hs[short] = collect_hidden(mid, corpus)
        print(f"  shape {Hs[short].shape}")

    results = {"models": [m for _, m in MODELS], "n_prompts": len(corpus),
               "domains": DOMAINS, "experiments": {}}

    # ---- E1: concept-axis recovery ----
    print("\n[E1] concept-axis recovery")
    e1 = {}
    for short, mid in MODELS:
        H = Hs[short]
        print(f"  {short}: fitting concept axes...")
        axes = concept_axes(H, labels)
        per_k = {}
        for k_dim in [16, 32, 64, 128]:
            B  = svd_basis(H, k_dim)
            cap_svd = basis_capture(axes, B)
            # average over multiple random draws
            cap_rand = []
            for s in range(5):
                Br = random_orthonormal(H.shape[1], k_dim, np.random.default_rng(100 + s))
                cap_rand.append(basis_capture(axes, Br))
            cap_rand = np.array(cap_rand)  # [5, 4]

            per_k[f"k={k_dim}"] = {
                "svd_capture_per_domain": {d: float(c) for d, c in zip(DOMAINS, cap_svd)},
                "svd_capture_mean": float(cap_svd.mean()),
                "random_capture_mean": float(cap_rand.mean()),
                "random_capture_std":  float(cap_rand.mean(axis=0).std()),
                "ratio_svd_over_random": float(cap_svd.mean() / max(cap_rand.mean(), 1e-9)),
                "ratio_svd_over_kd":     float(cap_svd.mean() / (k_dim / H.shape[1])),
            }
            r = per_k[f"k={k_dim}"]
            print(f"    k={k_dim:3d}  svd={r['svd_capture_mean']:.4f}  "
                  f"random={r['random_capture_mean']:.4f}  "
                  f"k/d={k_dim/H.shape[1]:.4f}  "
                  f"svd/random={r['ratio_svd_over_random']:.2f}x  "
                  f"svd/(k/d)={r['ratio_svd_over_kd']:.2f}x")
        e1[short] = per_k
    results["experiments"]["E1_concept_axis_recovery"] = e1

    # ---- E2: clusterability ----
    print("\n[E2] clusterability (KMeans n=4, ARI vs domain labels; random=0)")
    e2 = {}
    for short, mid in MODELS:
        H = Hs[short]
        Hc = H - H.mean(0, keepdims=True)
        ari_raw = kmeans_ari(H, labels)
        per_k = {"raw_ari": float(ari_raw)}
        print(f"  {short}: raw ARI={ari_raw:.4f}")
        for k_dim in [16, 32, 64, 128]:
            B = svd_basis(H, k_dim)
            P_svd  = Hc @ B
            ari_svd = kmeans_ari(P_svd, labels)
            ari_r = []
            for s in range(5):
                Br = random_orthonormal(H.shape[1], k_dim, np.random.default_rng(200 + s))
                ari_r.append(kmeans_ari(Hc @ Br, labels))
            per_k[f"k={k_dim}"] = {
                "svd_ari": float(ari_svd),
                "random_ari_mean": float(np.mean(ari_r)),
                "random_ari_std":  float(np.std(ari_r)),
            }
            print(f"    k={k_dim:3d}  svd ARI={ari_svd:.4f}  "
                  f"random ARI={np.mean(ari_r):.4f}+/-{np.std(ari_r):.4f}")
        e2[short] = per_k
    results["experiments"]["E2_clusterability"] = e2

    # ---- E3: Snipe-style coordinate ablation ----
    print("\n[E3] Snipe-style coordinate ablation, k=64")
    e3 = {}
    K_DIM = 64
    for short, mid in MODELS:
        H = Hs[short]
        per_basis = {}
        for tag, B_fn in [
            ("svd",      lambda: svd_basis(H, K_DIM)),
            ("random_0", lambda: random_orthonormal(H.shape[1], K_DIM, np.random.default_rng(303))),
            ("random_1", lambda: random_orthonormal(H.shape[1], K_DIM, np.random.default_rng(404))),
            ("random_2", lambda: random_orthonormal(H.shape[1], K_DIM, np.random.default_rng(505))),
        ]:
            B = B_fn()
            per_dom = {}
            for d in DOMAINS:
                per_dom[d] = snipe_drop(H, labels, B, d)
            best_drops = [per_dom[d]["best_drop"] for d in DOMAINS]
            per_basis[tag] = {
                "per_domain": per_dom,
                "best_drop_mean": float(np.mean(best_drops)),
                "best_drop_max":  float(np.max(best_drops)),
            }
            print(f"  {short} {tag}: mean best_drop={np.mean(best_drops):.4f}  "
                  f"max={np.max(best_drops):.4f}")
        e3[short] = per_basis
    results["experiments"]["E3_snipe_coordinate_ablation"] = e3

    # ---- summary verdict ----
    print("\n--- summary ---")
    summary = {}
    # E1: at k=64, is svd_capture_mean > 2x random_capture_mean for both models?
    e1_pass = all(
        e1[s]["k=64"]["ratio_svd_over_random"] > 2.0 for s, _ in MODELS
    )
    summary["E1_concept_axes_in_B"] = e1_pass
    # E2: at k=32, is svd ARI > random ARI + 0.05 for both?
    e2_pass = all(
        e2[s]["k=32"]["svd_ari"] > e2[s]["k=32"]["random_ari_mean"] + 0.05
        for s, _ in MODELS
    )
    summary["E2_clusterability_better_in_B"] = e2_pass
    # E3: is svd best_drop_mean > average random best_drop_mean by 50%?
    e3_pass = []
    for s, _ in MODELS:
        svd_drop = e3[s]["svd"]["best_drop_mean"]
        rand_drops = np.mean([e3[s][f"random_{i}"]["best_drop_mean"] for i in range(3)])
        e3_pass.append(svd_drop > 1.5 * rand_drops)
        summary[f"E3_{s}_svd_drop"]    = float(svd_drop)
        summary[f"E3_{s}_random_drop"] = float(rand_drops)
        summary[f"E3_{s}_ratio"]       = float(svd_drop / max(rand_drops, 1e-9))
    summary["E3_snipe_uses_direction_structure"] = bool(all(e3_pass))
    results["summary"] = summary
    print(json.dumps(summary, indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
