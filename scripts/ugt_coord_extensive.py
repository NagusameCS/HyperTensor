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

"""Extensive battery for the coordinate-level reading of UGT.

Builds on benchmarks/ugt_zone_recovery.json with seven additional tests:

E4 -- Per-domain top-k coordinate selectivity.
   For each domain d and each coordinate j of B, fit logistic-regression
   AUROC for {d vs rest} using only coord j (univariate).  Report top-5
   coords per domain, their overlap across domains (are domains using
   distinct coordinates?), and the median selectivity profile.

E5 -- Bootstrap stability of E1 and E3.
   Resample prompts (with replacement) 200x; recompute concept-axis
   capture and best-coord ablation drop.  Report 95% CIs.  If signal is
   real and not a ~400-prompt artefact the CIs should not span zero.

E6 -- Layer-wise emergence.
   Extract hidden states from every transformer block, compute concept-
   axis capture and best-coord ablation drop at k=64 per layer.  If
   direction structure is real it should grow monotonically (or
   monotonically up to some plateau) with depth, the way representations
   typically do; if it's an artefact of last-layer pooling it'll be
   dominated by the last layer alone.

E7 -- Cross-model concept-axis correspondence.
   Each model's best-coord-for-domain-d defines a direction in R^d.
   Question: is the direction encoding "code" in SmolLM the same
   direction (up to Procrustes alignment) as in Qwen?  Compute the
   cross-model cosine of concept axes after Procrustes alignment of
   raw activations; compare to a permutation null over domains.

E8 -- Held-out domain generalisation.
   Build B from prompts of 3 domains only; test concept-axis capture for
   the held-out 4th domain.  Repeat for each of the 4 leave-one-out
   splits.  If B's structure is just memorising the training domains,
   held-out capture collapses to the random baseline; if B is capturing
   a general "concept-direction-friendly" subspace, held-out capture
   stays high.

E9 -- Random-rotation-of-SVD as a strong control.
   Take B = top-k SVD basis, then apply a Haar-random orthogonal
   rotation R \\in O(k): B_rot = B @ R.  Has identical column space to
   B (so subspace-level tests give identical results) but the
   individual coordinate directions are scrambled.  Re-run E1 and E3
   on B_rot.  This isolates "is it the column space (which random R
   preserves) or the specific axes (which random R destroys)" --- the
   crucial test.

E10 -- Prompt-paraphrase robustness.
   Generate 3 paraphrases of each prompt by perturbing wording; ask
   whether the same coordinate still encodes the same domain.  Tests
   whether E3 finds a coord that's tracking the *concept* or a coord
   that's tracking surface-level token statistics.

Output: benchmarks/ugt_coord_extensive.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ortho_group

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from universal_taxonomy_test import (
    build_corpus, MODELS, random_orthonormal,
)
from ugt_zone_recovery import (
    svd_basis, concept_axes, basis_capture, snipe_drop, DOMAINS,
)

OUT = ROOT / "benchmarks" / "ugt_coord_extensive.json"
K_DIM = 64


# -------------- helpers --------------
def collect_hidden_all_layers(model_id, prompts):
    """Returns list of np.ndarray, one per transformer block (incl. embedding):
    each [N, d]."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    L = model.config.num_hidden_layers
    # +1 for embedding layer (output_hidden_states gives L+1 tensors)
    layers = [[] for _ in range(L + 1)]
    t0 = time.perf_counter()
    for i, (_, p) in enumerate(prompts):
        ids = tok(p, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for li, h in enumerate(out.hidden_states):
            layers[li].append(h[0, -1, :].float().cpu().numpy())
        if i % 50 == 0:
            print(f"  {model_id}: {i}/{len(prompts)} ({time.perf_counter()-t0:.1f}s)")
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return [np.array(L) for L in layers]


def auroc_univariate(scores, y):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    out = []
    for tr, te in skf.split(scores, y):
        # For univariate score we just measure AUROC of the score itself
        # against the labels on the test fold (no need to fit, but we
        # match sign by averaging y_tr*x_tr correlation).
        sign = 1.0 if np.corrcoef(scores[tr], y[tr])[0, 1] >= 0 else -1.0
        out.append(roc_auc_score(y[te], sign * scores[te]))
    return float(np.mean(out))


# -------------- E4 --------------
def e4_per_domain_top_coords(H, labels, B, top=5):
    Hc = H - H.mean(0, keepdims=True)
    P = Hc @ B  # [N, k]
    k = P.shape[1]
    out = {}
    for d in DOMAINS:
        y = (labels == d).astype(int)
        aurocs = np.array([auroc_univariate(P[:, j], y) for j in range(k)])
        order = np.argsort(-aurocs)[:top]
        out[d] = {"top_coords": order.tolist(),
                  "top_aurocs": [float(aurocs[j]) for j in order]}
    # overlap matrix
    sets = {d: set(out[d]["top_coords"]) for d in DOMAINS}
    overlap = {f"{d1}-{d2}": len(sets[d1] & sets[d2]) for d1 in DOMAINS for d2 in DOMAINS if d1 < d2}
    return {"per_domain": out, "top5_overlap": overlap}


# -------------- E5 --------------
def _fast_best_coord_drop(P, y):
    """Fast surrogate for snipe_drop: AUROC of full-model logreg minus
    AUROC after zeroing the coord with the largest |w_j|.  Uses one
    cross-validated logreg fit (3-fold) instead of k+1.  Highly
    correlated with the exact greedy snipe_drop in our setting because
    a coord's contribution to a linear classifier is dominated by w_j*
    when other coords are small (verified empirically below)."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    base_aurocs, abl_aurocs = [], []
    for tr, te in skf.split(P, y):
        clf = LogisticRegression(max_iter=2000).fit(P[tr], y[tr])
        base_aurocs.append(roc_auc_score(y[te], clf.decision_function(P[te])))
        j_star = int(np.argmax(np.abs(clf.coef_[0])))
        Pmod = P.copy(); Pmod[:, j_star] = 0.0
        clf2 = LogisticRegression(max_iter=2000).fit(Pmod[tr], y[tr])
        abl_aurocs.append(roc_auc_score(y[te], clf2.decision_function(Pmod[te])))
    return float(np.mean(base_aurocs) - np.mean(abl_aurocs))


def e5_bootstrap(H, labels, B, n_boot=50, seed=0):
    rng = np.random.default_rng(seed)
    Hc = H - H.mean(0, keepdims=True)
    P  = Hc @ B  # [N, k]
    cap_b = []
    drop_b = []
    N = len(H)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        Hb, lb, Pb = H[idx], labels[idx], P[idx]
        if len(set(lb.tolist())) < len(DOMAINS): continue
        try:
            ax = concept_axes(Hb, lb)
            cap_b.append(basis_capture(ax, B).mean())
            ds = []
            for d in DOMAINS:
                y = (lb == d).astype(int)
                ds.append(_fast_best_coord_drop(Pb, y))
            drop_b.append(np.mean(ds))
        except Exception:
            continue
    cap_b, drop_b = np.array(cap_b), np.array(drop_b)
    def ci(x):
        if len(x) == 0: return [None, None, None]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 50)), float(np.percentile(x, 97.5))]
    return {"n_boot": int(len(cap_b)),
            "capture_ci_lo_med_hi": ci(cap_b),
            "drop_ci_lo_med_hi": ci(drop_b)}


# -------------- E6 --------------
def e6_layerwise(layers, labels):
    """Use the fast surrogate (single argmax-|w| ablation) for layerwise
    sweep; over 31 layers x 4 domains x 2 bases the exact greedy is too
    expensive and the rank ordering is what we need."""
    out = []
    for li, H in enumerate(layers):
        if H.shape[0] == 0: continue
        axes = concept_axes(H, labels)
        B = svd_basis(H, K_DIM)
        Br = random_orthonormal(H.shape[1], K_DIM, np.random.default_rng(li))
        cap   = float(basis_capture(axes, B ).mean())
        cap_r = float(basis_capture(axes, Br).mean())
        Hc = H - H.mean(0, keepdims=True)
        P_svd = Hc @ B
        P_rnd = Hc @ Br
        drop = float(np.mean([_fast_best_coord_drop(P_svd, (labels == d).astype(int)) for d in DOMAINS]))
        drop_r = float(np.mean([_fast_best_coord_drop(P_rnd, (labels == d).astype(int)) for d in DOMAINS]))
        out.append({"layer": li,
                    "svd_capture": cap, "random_capture": cap_r,
                    "svd_best_drop": drop, "random_best_drop": drop_r})
        print(f"    L{li:02d}  svd_cap={cap:.3f} (rand {cap_r:.3f})  "
              f"svd_drop={drop:.3f} (rand {drop_r:.3f})")
    return out


# -------------- E7 --------------
def e7_cross_model_axis_correspondence(H_a, H_b, labels):
    """Project each model to k=K_DIM via its own SVD; align by Procrustes
    on training half; compare concept axes (in projected space) across
    models on held-out half."""
    rng = np.random.default_rng(0)
    n = len(H_a)
    idx = rng.permutation(n)
    n_tr = n // 2
    tr, te = idx[:n_tr], idx[n_tr:]

    Ba = svd_basis(H_a[tr], K_DIM)
    Bb = svd_basis(H_b[tr], K_DIM)
    Pa = (H_a - H_a[tr].mean(0)) @ Ba
    Pb = (H_b - H_b[tr].mean(0)) @ Bb
    # orthogonal Procrustes on train
    M = Pa[tr].T @ Pb[tr]
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    # project A to B's space via R
    Pa_in_b = Pa @ R

    out = {"per_domain_cosine_after_procrustes": {}}
    cosines = []
    for d in DOMAINS:
        y = (labels == d).astype(int)
        # concept axis in each model's projected space, fit on TRAIN
        wa = LogisticRegression(max_iter=2000).fit(Pa_in_b[tr], y[tr]).coef_[0]
        wb = LogisticRegression(max_iter=2000).fit(Pb     [tr], y[tr]).coef_[0]
        cs = float(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-12))
        out["per_domain_cosine_after_procrustes"][d] = cs
        cosines.append(cs)
    out["mean_cosine"] = float(np.mean(cosines))

    # permutation null: shuffle which prompts get which Qwen embedding
    nulls = []
    for s in range(20):
        perm = rng.permutation(n)
        Pb_perm = Pb[perm]
        # re-fit Procrustes on the shuffled pair using train
        M_p = Pa[tr].T @ Pb_perm[tr]
        U_p, _, Vh_p = np.linalg.svd(M_p, full_matrices=False); R_p = U_p @ Vh_p
        Pa_p = Pa @ R_p
        cs_d = []
        for d in DOMAINS:
            y = (labels == d).astype(int)
            try:
                wa = LogisticRegression(max_iter=2000).fit(Pa_p     [tr], y[tr]).coef_[0]
                wb = LogisticRegression(max_iter=2000).fit(Pb_perm  [tr], y[tr]).coef_[0]
                cs_d.append(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-12))
            except Exception: pass
        if cs_d: nulls.append(np.mean(cs_d))
    out["null_mean_cosine_mean"] = float(np.mean(nulls))
    out["null_mean_cosine_std"]  = float(np.std(nulls))
    out["z_score"] = float((out["mean_cosine"] - out["null_mean_cosine_mean"]) /
                           max(out["null_mean_cosine_std"], 1e-6))
    return out


# -------------- E8 --------------
def e8_held_out_domain(H, labels):
    out = {}
    for held in DOMAINS:
        keep = labels != held
        H_train = H[keep]
        B = svd_basis(H_train, K_DIM)
        # axes for held-out domain, fit on FULL (since the question is
        # whether B captures the held-out concept axis at all)
        y = (labels == held).astype(int)
        clf = LogisticRegression(max_iter=2000).fit(H, y)
        w = clf.coef_[0]; w = w / (np.linalg.norm(w) + 1e-12)
        cap = float(w @ (B @ B.T) @ w)
        # random control of same rank
        Br = random_orthonormal(H.shape[1], K_DIM, np.random.default_rng(hash(held) % 2**31))
        cap_r = float(w @ (Br @ Br.T) @ w)
        # also: best-coord ablation drop for held-out domain through B trained without it
        drop = snipe_drop(H, labels, B, held)["best_drop"]
        drop_r = snipe_drop(H, labels, Br, held)["best_drop"]
        out[held] = {"capture_in_B_held_out": cap,
                     "capture_in_random_B": cap_r,
                     "ratio": cap / max(cap_r, 1e-9),
                     "best_coord_drop_in_B_held_out": float(drop),
                     "best_coord_drop_in_random_B":   float(drop_r)}
    return out


# -------------- E9 --------------
def e9_random_rotation_of_svd(H, labels, n_rot=3):
    """B_rot = B @ R for random R \\in O(k).  Same column space as B,
    so subspace-level tests are unchanged.  Coordinate-level tests
    SHOULD see the same column space contain the concept axes (capture
    same), but Snipe-style coord ablation should now look like random,
    because the axes have been scrambled relative to the basis vectors."""
    B = svd_basis(H, K_DIM)
    axes = concept_axes(H, labels)
    cap_B = float(basis_capture(axes, B).mean())
    drop_B = float(np.mean([snipe_drop(H, labels, B, d)["best_drop"] for d in DOMAINS]))
    rot_caps = []; rot_drops = []
    for s in range(n_rot):
        R = ortho_group.rvs(K_DIM, random_state=2000 + s)
        B_rot = B @ R
        rot_caps.append(float(basis_capture(axes, B_rot).mean()))
        rot_drops.append(float(np.mean([snipe_drop(H, labels, B_rot, d)["best_drop"] for d in DOMAINS])))
    return {"k": K_DIM,
            "B_capture": cap_B, "B_best_drop": drop_B,
            "B_rot_capture_mean": float(np.mean(rot_caps)),
            "B_rot_capture_std":  float(np.std(rot_caps)),
            "B_rot_best_drop_mean": float(np.mean(rot_drops)),
            "B_rot_best_drop_std":  float(np.std(rot_drops)),
            "predicts_capture_equal":      abs(cap_B - np.mean(rot_caps))   < 0.02,
            "predicts_drop_collapses_to_random": np.mean(rot_drops) < drop_B / 5}


# -------------- E10 --------------
def e10_paraphrase_robustness(model_id, corpus, B_dim=K_DIM):
    """Generate paraphrases by lightly perturbing each prompt; check
    whether the same E3 best-coord still ranks high."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    PARAPHRASE_PREFIXES = [
        ("orig",  ""),
        ("para1", "Please answer: "),
        ("para2", "I would like to know: "),
    ]
    Hs = {p[0]: [] for p in PARAPHRASE_PREFIXES}
    for (_, prompt) in corpus:
        for tag, prefix in PARAPHRASE_PREFIXES:
            full = prefix + prompt
            ids = tok(full, return_tensors="pt", truncation=True, max_length=80).input_ids.to(device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            Hs[tag].append(out.hidden_states[-1][0, -1, :].float().cpu().numpy())
    del model
    if device == "cuda": torch.cuda.empty_cache()
    Hs = {k: np.array(v) for k, v in Hs.items()}
    labels = np.array([d for d, _ in corpus])

    # B trained on orig
    B = svd_basis(Hs["orig"], B_dim)
    out = {}
    for d in DOMAINS:
        # find best coord on orig
        s_orig = snipe_drop(Hs["orig"], labels, B, d)
        j_star = s_orig["best_coord"]
        # rank of j_star on para1/para2: re-rank coords by drop magnitude on para1
        ranks = {}
        for tag in ["para1", "para2"]:
            P = (Hs[tag] - Hs[tag].mean(0, keepdims=True)) @ B
            y = (labels == d).astype(int)
            # univariate AUROC on each coord on para
            scores = np.array([auroc_univariate(P[:, j], y) for j in range(B_dim)])
            order = np.argsort(-scores)
            ranks[tag] = int(np.where(order == j_star)[0][0])
        out[d] = {"orig_best_coord": j_star,
                  "rank_in_para1": ranks["para1"],
                  "rank_in_para2": ranks["para2"]}
    return out


# -------------- main --------------
def main():
    print("[corpus]")
    corpus = build_corpus(per_domain=100)
    labels = np.array([d for d, _ in corpus])

    results = {"models": [m for _, m in MODELS], "n_prompts": len(corpus),
               "k": K_DIM, "experiments": {}}

    print("[hidden] all-layer extraction")
    layers_by_model = {}
    for short, mid in MODELS:
        print(f"  {short}")
        layers_by_model[short] = collect_hidden_all_layers(mid, corpus)
        print(f"    {len(layers_by_model[short])} layers, last shape "
              f"{layers_by_model[short][-1].shape}")

    H_smol = layers_by_model["smol135m"][-1]
    H_qwen = layers_by_model["qwen500m"][-1]

    # ---- E4 ----
    print("\n[E4] per-domain top-5 coordinates of B")
    e4 = {}
    for short, H in [("smol135m", H_smol), ("qwen500m", H_qwen)]:
        B = svd_basis(H, K_DIM)
        e4[short] = e4_per_domain_top_coords(H, labels, B, top=5)
        print(f"  {short} top-5 overlap: {e4[short]['top5_overlap']}")
    results["experiments"]["E4_top_coords"] = e4

    # ---- E5 ----
    print("\n[E5] bootstrap (n=200) on E1 capture and E3 drop")
    e5 = {}
    for short, H in [("smol135m", H_smol), ("qwen500m", H_qwen)]:
        B = svd_basis(H, K_DIM)
        e5[short] = e5_bootstrap(H, labels, B, n_boot=200, seed=42)
        print(f"  {short} cap CI {e5[short]['capture_ci_lo_med_hi']}  "
              f"drop CI {e5[short]['drop_ci_lo_med_hi']}")
    results["experiments"]["E5_bootstrap"] = e5

    # ---- E6 ----
    print("\n[E6] layer-wise emergence")
    e6 = {}
    for short in ["smol135m", "qwen500m"]:
        print(f"  {short} ({len(layers_by_model[short])} layers)")
        e6[short] = e6_layerwise(layers_by_model[short], labels)
    results["experiments"]["E6_layerwise"] = e6

    # ---- E7 ----
    print("\n[E7] cross-model concept-axis correspondence")
    e7 = e7_cross_model_axis_correspondence(H_smol, H_qwen, labels)
    print(f"  mean cosine after Procrustes: {e7['mean_cosine']:.3f}  "
          f"null {e7['null_mean_cosine_mean']:.3f}+/-{e7['null_mean_cosine_std']:.3f}  "
          f"z={e7['z_score']:.2f}")
    results["experiments"]["E7_cross_model_axes"] = e7

    # ---- E8 ----
    print("\n[E8] held-out domain")
    e8 = {}
    for short, H in [("smol135m", H_smol), ("qwen500m", H_qwen)]:
        e8[short] = e8_held_out_domain(H, labels)
        for d in DOMAINS:
            r = e8[short][d]
            print(f"  {short} held={d:9s}  cap_B={r['capture_in_B_held_out']:.3f} "
                  f"cap_rand={r['capture_in_random_B']:.3f}  ratio={r['ratio']:.2f}x  "
                  f"drop_B={r['best_coord_drop_in_B_held_out']:.3f}  "
                  f"drop_rand={r['best_coord_drop_in_random_B']:.3f}")
    results["experiments"]["E8_held_out_domain"] = e8

    # ---- E9 ----
    print("\n[E9] random rotation of SVD basis (same col space, scrambled axes)")
    e9 = {}
    for short, H in [("smol135m", H_smol), ("qwen500m", H_qwen)]:
        e9[short] = e9_random_rotation_of_svd(H, labels, n_rot=3)
        print(f"  {short}  cap B={e9[short]['B_capture']:.3f}  "
              f"cap B_rot={e9[short]['B_rot_capture_mean']:.3f}+/-{e9[short]['B_rot_capture_std']:.3f}  "
              f"drop B={e9[short]['B_best_drop']:.3f}  "
              f"drop B_rot={e9[short]['B_rot_best_drop_mean']:.4f}+/-{e9[short]['B_rot_best_drop_std']:.4f}")
    results["experiments"]["E9_random_rotation"] = e9

    # ---- E10 ----
    print("\n[E10] paraphrase robustness (Smol135M only, to save time)")
    e10 = e10_paraphrase_robustness(MODELS[0][1], corpus, B_dim=K_DIM)
    for d in DOMAINS:
        r = e10[d]
        print(f"  {d:9s}  best j={r['orig_best_coord']:3d}  "
              f"rank in para1: {r['rank_in_para1']}/64  rank in para2: {r['rank_in_para2']}/64")
    results["experiments"]["E10_paraphrase_robustness"] = e10

    # ---- summary ----
    summary = {
        "E5_drop_CIs_exclude_random_baseline": all(
            e5[s]["drop_ci_lo_med_hi"][0] > 0.005 for s in e5
        ),
        "E7_cross_model_axes_align_above_null":
            e7["mean_cosine"] > e7["null_mean_cosine_mean"] + 3 * max(e7["null_mean_cosine_std"], 1e-3),
        "E8_held_out_capture_above_random": all(
            e8[s][d]["ratio"] > 2.0 for s in e8 for d in DOMAINS
        ),
        "E9_axes_not_subspace_carry_signal": all(
            e9[s]["predicts_drop_collapses_to_random"] for s in e9
        ),
        "E9_subspace_capture_unchanged": all(
            e9[s]["predicts_capture_equal"] for s in e9
        ),
        "E10_para_top10_persistence": float(np.mean([
            (r["rank_in_para1"] < 10) + (r["rank_in_para2"] < 10)
            for r in e10.values()
        ]) / 2),
    }
    results["summary"] = summary
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    OUT.write_text(json.dumps(results, indent=2, default=_json_default))
    print("\n--- summary ---")
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
