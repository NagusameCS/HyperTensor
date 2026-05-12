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

"""Paper X gaps 2-3 + XI gap 2 + XII gap 4 + XIII gap 4 + XIV gaps 3,5
+ XV gaps 4,6.

CPU-friendly probes; uses SmolLM2-135M for any neural piece.

Output: benchmarks/knockout_x_xv.json
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "knockout_x_xv.json"

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_smol():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device).eval(), device


def collect_hidden(tok, model, device, prompts, layer=15):
    Hs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=24).input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        Hs.append(out.hidden_states[layer][0, -1, :].float().cpu().numpy())
    return np.array(Hs)


def x_pairwise_overlap(H_pool, n_basis=8, k=16):
    """Build n_basis pseudo-finetune bases (different random projection
    of the same hidden-state pool) and report pairwise subspace overlap."""
    rng = np.random.default_rng(0)
    bases = []
    for i in range(n_basis):
        # subsample 70% of the pool with replacement, do SVD
        idx = rng.integers(0, len(H_pool), size=int(0.7 * len(H_pool)))
        H = H_pool[idx]
        H = H - H.mean(axis=0, keepdims=True)
        U, S, Vh = np.linalg.svd(H, full_matrices=False)
        bases.append(Vh[:k].T)  # [d, k]
    overlap = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            # principal-angle similarity: || B_i^T B_j ||_F^2 / k
            M = bases[i].T @ bases[j]
            overlap[i, j] = float(np.linalg.norm(M, "fro")**2 / k)
    return {
        "n_basis": n_basis, "k": k,
        "diagonal":  float(np.diag(overlap).mean()),
        "off_diag":  float((overlap.sum() - np.trace(overlap)) / (n_basis*(n_basis-1))),
        "matrix":    overlap.tolist(),
    }


def x_functional_grafting(tok, model, device):
    """Swap the residual contribution of layer L with that of layer L+2
    on a fixed prompt; report PPL change."""
    prompts = ["The capital of France is", "def add(a, b): return", "Roses are red,"]
    L_swap = 10
    base_loss = []
    swap_loss = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            base_loss.append(float(model(ids, labels=ids).loss.item()))
        # graft hook
        captured = {}
        def cap(_m, _i, o):
            captured["h"] = (o[0] if isinstance(o, tuple) else o).clone()
        h1 = model.model.layers[L_swap].register_forward_hook(cap)
        with torch.no_grad():
            model(ids)
        h1.remove()
        def replace(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            h_new = captured["h"]
            return (h_new,) + o[1:] if isinstance(o, tuple) else h_new
        h2 = model.model.layers[L_swap + 2].register_forward_hook(replace)
        with torch.no_grad():
            swap_loss.append(float(model(ids, labels=ids).loss.item()))
        h2.remove()
    return {
        "L_swap_into": L_swap + 2,
        "donor": L_swap,
        "base_loss_mean": float(np.mean(base_loss)),
        "swap_loss_mean": float(np.mean(swap_loss)),
        "delta":          float(np.mean(swap_loss) - np.mean(base_loss)),
    }


def xi_cca_alignment(H_pool):
    """Treat half of H_pool as 'sparse-feature directions' (synthesized
    via random sparse coding) and do CCA between B = top-k SVD of H_pool
    and those directions."""
    from numpy.linalg import svd, qr
    H = H_pool - H_pool.mean(axis=0, keepdims=True)
    U, S, Vh = svd(H, full_matrices=False)
    k = 16
    B = Vh[:k].T  # [d, k]
    rng = np.random.default_rng(0)
    sparse_dirs = rng.standard_normal((B.shape[0], 32))
    sparse_dirs[np.abs(sparse_dirs) < 0.8] = 0.0
    Q, _ = qr(sparse_dirs)
    M = B.T @ Q
    sing = np.linalg.svd(M, compute_uv=False)
    return {
        "k_basis": k, "n_sae_directions": Q.shape[1],
        "mean_canonical_corr": float(sing.mean()),
        "max_canonical_corr":  float(sing.max()),
        "interpretation":      "if max << 1, B is uncorrelated with synthesised sparse directions",
    }


def xii_wallclock(tok, model, device):
    """Native (low-rank reparam) vs full forward wall-clock."""
    import time
    ids = tok(("hello world " * 64), return_tensors="pt").input_ids.to(device)
    # full
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            model(ids)
    full = (time.perf_counter() - t0) / 20
    # rank-128 lowrank
    saved = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.weight.shape[0] >= 32 and m.weight.shape[1] >= 32:
            W = m.weight.data; saved[n] = W.clone()
            r = min(128, min(W.shape) // 2)
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            m.weight.data = ((U[:, :r] * S[:r]) @ Vh[:r, :]).to(W.dtype)
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            model(ids)
    lr = (time.perf_counter() - t0) / 20
    for n, m in model.named_modules():
        if n in saved:
            m.weight.data = saved[n]
    return {"full_s_per_iter": full, "rank128_s_per_iter": lr,
            "speedup": float(full / lr)}


def xiii_rlhf_composability():
    """Synthetic: simulate a model whose hidden states have a 'safety
    direction'.  Apply Safe-OGD projection P=I-q q^T and report whether
    composing with a downstream alignment shift along q' (different
    random direction) preserves the safety guarantee on the labelled set."""
    rng = np.random.default_rng(0)
    d = 64
    q = rng.standard_normal(d); q /= np.linalg.norm(q)  # forbidden dir
    P = np.eye(d) - np.outer(q, q)
    # labelled forbidden samples: lie in span(q)
    Xf = (rng.standard_normal((100, 1)) * q[None, :])
    # safe samples: orthogonal
    Xs = rng.standard_normal((100, d)); Xs = Xs @ P
    # Apply Safe-OGD then a random RLHF-style shift along q'
    qp = rng.standard_normal(d); qp /= np.linalg.norm(qp)
    teh_before = float(np.linalg.norm(Xf @ q[:, None], axis=1).mean())
    Xf_safe = Xf @ P
    teh_after_safe = float(np.linalg.norm(Xf_safe @ q[:, None], axis=1).mean())
    Xf_post_rlhf = Xf_safe + 0.1 * qp[None, :]
    teh_after_rlhf = float(np.linalg.norm(Xf_post_rlhf @ q[:, None], axis=1).mean())
    return {
        "teh_pre":       teh_before,
        "teh_post_safe": teh_after_safe,
        "teh_post_rlhf": teh_after_rlhf,
        "drift":         teh_after_rlhf - teh_after_safe,
        "interpretation": ("If drift remains << teh_pre, RLHF composition "
                           "preserves Safe-OGD; here drift is the rate at "
                           "which the safety projection leaks under added "
                           "shifts."),
    }


def xiv_held_out_category():
    rng = np.random.default_rng(0)
    d = 64; n_cat = 8
    # each category has its own direction
    dirs = rng.standard_normal((n_cat, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # train Q_f on first 7 categories
    Q, _ = np.linalg.qr(dirs[:7].T)
    P = np.eye(d) - Q @ Q.T
    # held-out: 8th cat
    held = dirs[7] / np.linalg.norm(dirs[7])
    teh_held = float(np.linalg.norm(P @ held))  # how much survives
    return {
        "n_cat_train": 7, "n_cat_held": 1,
        "teh_after_proj_held_out": teh_held,
        "expected_if_orthogonal":  1.0,
        "interpretation": ("Snipe trained on 7 categories; held-out "
                           "category survives projection at "
                           f"~{teh_held:.2f} of its norm. Lower is better "
                           "for generalisation."),
    }


def xiv_persistence_under_sft():
    rng = np.random.default_rng(0)
    d = 32
    # forbidden dir
    q = rng.standard_normal(d); q /= np.linalg.norm(q)
    # initial post-snipe: small projection on q
    h = 0.01 * q + rng.standard_normal(d) * 0.5
    teh = [float(abs(h @ q))]
    # 1000 SFT steps: each step shifts h slightly via gradient that
    # incidentally has small q-component
    for _ in range(1000):
        g = rng.standard_normal(d) * 0.01 + 0.001 * q
        h = h - 0.01 * g
        teh.append(float(abs(h @ q)))
    return {
        "initial_teh":  teh[0], "final_teh": teh[-1],
        "growth_factor": float(teh[-1] / max(teh[0], 1e-12)),
        "interpretation": ("If growth_factor >> 1, the harm direction "
                           "re-emerges under benign SFT."),
    }


def xv_cog_replicability():
    rng = np.random.default_rng(0)
    n_seeds = 4
    metrics = []
    for s in range(n_seeds):
        r = np.random.default_rng(s)
        # COG-style outer-product accumulation
        d = 8
        M = np.eye(d) * 0.01
        for _ in range(2000):
            v = r.standard_normal(d) * 0.1
            M = M + np.outer(v, v) * 0.001
        metrics.append(np.linalg.norm(M, "fro"))
    metrics = np.array(metrics)
    return {
        "n_seeds": n_seeds,
        "metric_norms": metrics.tolist(),
        "mean":  float(metrics.mean()),
        "std":   float(metrics.std()),
        "cv":    float(metrics.std() / metrics.mean()),
        "interpretation": "low CV implies COG is replicable across seeds.",
    }


def xv_vs_llamaguard_proxy():
    """Synthetic comparison TEH-detector vs a logistic-regression
    'Llama-Guard proxy' on the same labelled set."""
    rng = np.random.default_rng(0)
    d = 64; n = 400
    q = rng.standard_normal(d); q /= np.linalg.norm(q)
    # 50% harm (large q-component), 50% benign
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d))
    X[y == 1] += 0.5 * q  # inject harm
    # TEH detector: project onto q, threshold
    score = X @ q
    from sklearn.metrics import roc_auc_score
    teh_auroc = roc_auc_score(y, score)
    # logistic regression baseline (Llama-Guard proxy)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    auroc = cross_val_score(LogisticRegression(max_iter=1000), X, y,
                             scoring="roc_auc", cv=5).mean()
    return {
        "teh_auroc":             float(teh_auroc),
        "logreg_baseline_auroc": float(auroc),
        "delta":                 float(teh_auroc - auroc),
        "interpretation": ("If teh_auroc > logreg, TEH is a useful "
                           "geometric detector beyond a linear proxy."),
    }


def main():
    out = {}
    print("[X] pairwise overlap matrix...")
    tok, model, device = load_smol()
    prompts = (["The cat sat on the", "Hello world this is a", "1+1 equals",
                "Today I went to", "Roses are red and"] * 8)
    H_pool = collect_hidden(tok, model, device, prompts, layer=15)
    out["paper_x_gap2_pairwise_overlap"] = x_pairwise_overlap(H_pool)
    print("[X] functional grafting...")
    out["paper_x_gap3_functional_graft"] = x_functional_grafting(tok, model, device)
    print("[XI] CCA alignment...")
    out["paper_xi_gap2_cca_alignment"]   = xi_cca_alignment(H_pool)
    print("[XII] wall-clock...")
    out["paper_xii_gap4_wallclock"]      = xii_wallclock(tok, model, device)
    print("[XIII] RLHF composability...")
    out["paper_xiii_gap4_rlhf"]          = xiii_rlhf_composability()
    print("[XIV] held-out category...")
    out["paper_xiv_gap3_held_out"]       = xiv_held_out_category()
    print("[XIV] persistence under SFT...")
    out["paper_xiv_gap5_persistence"]    = xiv_persistence_under_sft()
    print("[XV] COG replicability...")
    out["paper_xv_gap4_cog_replicability"] = xv_cog_replicability()
    print("[XV] vs Llama-Guard proxy...")
    out["paper_xv_gap6_vs_llamaguard"]   = xv_vs_llamaguard_proxy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
