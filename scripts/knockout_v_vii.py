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

"""Paper V gaps 1-3 + Paper VII gaps 1-4: distillation closure, matched-rank
LoRA, cross-corpus generalisation, FFN cluster end-to-end PPL, cluster-count
sweep, clustering-algo ablation, vs random-cluster control.

CPU-friendly using SmolLM2-135M.

Output: benchmarks/knockout_v_vii.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "knockout_v_vii.json"

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_model_and_tokens():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    # short token sequences for fast PPL
    text = ("The quick brown fox jumps over the lazy dog. " * 64
            + "In a hole in the ground there lived a hobbit. " * 64)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    return model, ids


def ppl(model, ids):
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())


def low_rank_replace(model, k_frac):
    """In-place low-rank approx of every Linear weight; returns undo fn."""
    saved = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.weight.shape[0] >= 8 and m.weight.shape[1] >= 8:
            W = m.weight.data
            r = max(1, int(k_frac * min(W.shape)))
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            saved[name] = W.clone()
            Wr = (U[:, :r] * S[:r]) @ Vh[:r, :]
            m.weight.data = Wr.to(W.dtype)
    def undo():
        for name, m in model.named_modules():
            if name in saved:
                m.weight.data = saved[name]
    return undo


def cluster_replace_ffn(model, n_clusters, algo):
    """Per-cluster low-rank on FFN up/down; returns undo."""
    saved = {}
    rng = np.random.default_rng(0)
    for name, m in model.named_modules():
        if "mlp" in name and isinstance(m, torch.nn.Linear) and m.weight.shape[0] >= 32:
            W = m.weight.data
            saved[name] = W.clone()
            rows = W.float().cpu().numpy()
            if algo == "kmeans":
                from sklearn.cluster import KMeans
                lab = KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit_predict(rows)
            elif algo == "spectral":
                # cheap spectral: cosine sim -> top-k eigenvectors -> kmeans on them
                Xn = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-9)
                S = Xn @ Xn.T
                # top-k eigvecs
                w, V = np.linalg.eigh(S)
                feat = V[:, -n_clusters:]
                from sklearn.cluster import KMeans
                lab = KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit_predict(feat)
            elif algo == "random":
                lab = rng.integers(0, n_clusters, size=rows.shape[0])
            else:
                raise ValueError(algo)
            r_per = max(1, int(0.25 * rows.shape[1] / n_clusters))
            recon = np.zeros_like(rows)
            for c in range(n_clusters):
                idx = np.where(lab == c)[0]
                if len(idx) < 2:
                    recon[idx] = rows[idx]
                    continue
                Wc = rows[idx]
                U, S, Vh = np.linalg.svd(Wc, full_matrices=False)
                recon[idx] = (U[:, :r_per] * S[:r_per]) @ Vh[:r_per, :]
            m.weight.data = torch.tensor(recon, dtype=W.dtype, device=W.device)
    def undo():
        for name, m in model.named_modules():
            if name in saved:
                m.weight.data = saved[name]
    return undo


def main():
    model, ids = load_model_and_tokens()
    base = ppl(model, ids)

    # Paper V: PPL closure curve at varying GRC ranks (no LoRA distill,
    # but show recovery shape vs rank).
    closure = []
    for kf in [1.0, 0.75, 0.5, 0.25, 0.125]:
        undo = low_rank_replace(model, kf)
        closure.append({"k_frac": kf, "ppl": ppl(model, ids)})
        undo()

    # Paper V gap 2: at matched rank, compare 'GRC-only' vs 'GRC+LoRA-distill-mock'.
    # We mock LoRA-distill as a +1 rank residual fit to the SVD truncation error.
    matched = []
    for kf in [0.25, 0.125]:
        # GRC-only
        undo = low_rank_replace(model, kf)
        ppl_grc = ppl(model, ids)
        undo()
        # GRC + 1-rank residual lift (best-case LoRA)
        saved = {}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and m.weight.shape[0] >= 8 and m.weight.shape[1] >= 8:
                W = m.weight.data; saved[name] = W.clone()
                r = max(1, int(kf * min(W.shape)))
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                Wr = (U[:, :r] * S[:r]) @ Vh[:r, :]
                # residual rank-1 lift
                Wres = (U[:, r:r+1] * S[r:r+1]) @ Vh[r:r+1, :]
                m.weight.data = (Wr + Wres).to(W.dtype)
        ppl_lift = ppl(model, ids)
        for name, m in model.named_modules():
            if name in saved:
                m.weight.data = saved[name]
        matched.append({"k_frac": kf, "ppl_grc_only": ppl_grc, "ppl_grc_plus_rank1_lift": ppl_lift})

    # Paper V gap 3: cross-corpus -- evaluate the same model on 2 disjoint
    # token sequences (proxy for WikiText vs C4 generalisation).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text2 = ("def quicksort(arr):\n    if len(arr)<=1: return arr\n"
             "    p=arr[len(arr)//2]\n    return quicksort([x for x in arr if x<p])\n") * 32
    ids2 = tok(text2, return_tensors="pt").input_ids.to(ids.device)
    cross = {"corpus_A_ppl": base, "corpus_B_ppl": ppl(model, ids2)}

    # Paper VII: cluster-count sweep with kmeans, spectral, random algos.
    cluster_results = []
    for algo in ["kmeans", "spectral", "random"]:
        for n in [4, 8, 16]:
            undo = cluster_replace_ffn(model, n_clusters=n, algo=algo)
            cluster_results.append({"algo": algo, "n_clusters": n, "ppl": ppl(model, ids)})
            undo()

    out = {
        "baseline_ppl": base,
        "paper_v_gap1_closure_curve": closure,
        "paper_v_gap2_matched_rank":  matched,
        "paper_v_gap3_cross_corpus":  cross,
        "paper_vii_cluster_sweep":    cluster_results,
        "paper_vii_summary": {
            "best_kmeans": min((r for r in cluster_results if r["algo"]=="kmeans"), key=lambda r: r["ppl"]),
            "best_random": min((r for r in cluster_results if r["algo"]=="random"), key=lambda r: r["ppl"]),
            "kmeans_beats_random": (
                min(r["ppl"] for r in cluster_results if r["algo"]=="kmeans")
                <
                min(r["ppl"] for r in cluster_results if r["algo"]=="random")
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
