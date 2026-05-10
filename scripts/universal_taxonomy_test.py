"""Universal taxonomy of knowledge: empirical test.

This script runs the Platonic-Representation-Hypothesis-style test on the
specific model pair we use elsewhere (SmolLM2-135M-Instruct, Qwen2.5-0.5B-
Instruct), and then asks whether a UGT-style SVD basis preserves the
cross-model agreement at low rank, vs. a random orthonormal control.

Three independent measurements:

H1. Cross-model structural agreement on raw hidden states.
    For 400 prompts spanning 4 domains, take the last-token last-layer
    hidden state in each model.  Build a mutual-k-NN graph (k=10) within
    each model; compute the Jaccard overlap of neighbour-sets across
    models prompt-by-prompt, plus linear CKA.  Compare to a label-shuffled
    null.

H2. Does a UGT-style SVD basis preserve that structure at low rank?
    Compute B_M = top-k singular vectors of centred hidden-states for each
    model M.  Project hidden states onto B_M (so the prompt is now in
    R^k), redo the mutual-k-NN agreement.  Compare to a random orthonormal
    B'_M of the same rank.  Sweep k in {16, 32, 64, 128}.

H3. Does UGT's cross-model alignment do better than per-model SVD?
    Solve the orthogonal Procrustes problem to find a rotation
    R: R^k -> R^k aligning B_smol-coordinates with B_qwen-coordinates on
    a held-out 50% of prompts; measure the residual on the other 50%.
    Compare to the residual when starting from random B'.

Output: benchmarks/universal_taxonomy.json
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "universal_taxonomy.json"

MODELS = [
    ("smol135m", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("qwen500m", "Qwen/Qwen2.5-0.5B-Instruct"),
]

# --------------- prompt corpus ---------------
def build_corpus(per_domain: int = 100) -> list[tuple[str, str]]:
    """Return (domain, prompt) pairs.  Domains: factual, code, math, creative."""
    rng = np.random.default_rng(0)

    factual_seeds = [
        "The capital of {} is", "The currency of {} is", "The official language of {} is",
        "The largest city in {} is", "The president of {} in 2020 was", "The longest river in {} is",
        "The chemical symbol for {} is", "The atomic number of {} is", "The boiling point of {} in Celsius is",
        "The author of {} is", "The composer of {} was", "The director of {} was",
    ]
    factual_fillers = [
        "France", "Germany", "Japan", "Brazil", "Canada", "Egypt", "India", "Italy",
        "iron", "gold", "helium", "carbon", "oxygen", "silver", "copper", "lead",
        "Hamlet", "1984", "Macbeth", "Dune", "Frankenstein", "Beethoven's 9th symphony",
        "The Godfather", "Pulp Fiction", "Inception",
    ]

    code_seeds = [
        "def factorial(n):\n    if n <= 1:\n        return", "def is_prime(n):\n    if n < 2:\n        return",
        "for i in range(10):\n    print", "x = [1, 2, 3]\nx.append", "import numpy as np\nnp.zeros",
        "class Stack:\n    def __init__(self):\n        self.items =", "while True:\n    ",
        "try:\n    open('f.txt')\nexcept", "lambda x: x **", "def quicksort(arr):\n    if len(arr) <= 1:\n        return",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b =",
        "[x for x in range(10) if x %",
    ]

    math_seeds = [
        "Solve for x: 2x + 5 =", "What is 17 times 23?", "The derivative of x^3 is",
        "The integral of cos(x) is", "The square root of 144 is", "log_2(8) =",
        "sin(pi/2) =", "cos(0) =", "The sum 1+2+3+...+100 =",
        "If f(x) = x^2 then f'(x) =", "The limit of (1 + 1/n)^n as n -> infinity is",
        "The number of permutations of 5 distinct objects is",
    ]

    creative_seeds = [
        "Once upon a time in a small village", "She opened the box and found",
        "The wind whispered through the", "On a stormy night, the lighthouse keeper",
        "The dragon awoke after a thousand years to discover",
        "He looked into the mirror and saw a stranger who",
        "The last human on Earth sat alone in a room",
        "Music drifted from the abandoned theatre, and",
        "In the garden, the roses had begun to",
        "The detective lit her cigarette and stared at",
        "Beneath the floorboards lay a secret",
        "The robot dreamed for the first time, and the dream was",
    ]

    out = []
    for seed in factual_seeds[:per_domain // 2]:
        for filler in factual_fillers:
            if "{}" in seed:
                out.append(("factual", seed.format(filler)))
                if len(out) >= per_domain * 1: break
        if len([p for p in out if p[0] == "factual"]) >= per_domain: break

    for s in code_seeds:
        out.append(("code", s))
    for s in math_seeds:
        out.append(("math", s))
    for s in creative_seeds:
        out.append(("creative", s))

    # pad each domain up to per_domain by re-using seeds with small perturbations
    for dom, seeds in (("code", code_seeds), ("math", math_seeds), ("creative", creative_seeds)):
        cur = [p for p in out if p[0] == dom]
        i = 0
        while len(cur) < per_domain:
            extra = seeds[i % len(seeds)] + " " + str(i)
            out.append((dom, extra))
            cur.append((dom, extra))
            i += 1

    rng.shuffle(out)
    return out


# --------------- hidden-state extraction ---------------
def get_last_hidden(model, tok, prompt, device):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].float().cpu().numpy()


def collect_hidden(model_id, prompts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    H = []
    t0 = time.perf_counter()
    for i, (_, p) in enumerate(prompts):
        H.append(get_last_hidden(model, tok, p, device))
        if i % 50 == 0:
            print(f"  {model_id}: {i}/{len(prompts)} ({time.perf_counter()-t0:.1f}s)")
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return np.array(H)


# --------------- metrics ---------------
def mutual_knn_agreement(H_a, H_b, k=10):
    """Build k-NN graph in each space, return mean Jaccard overlap of
    neighbour-sets prompt-by-prompt."""
    def normed(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Sa = normed(H_a) @ normed(H_a).T
    Sb = normed(H_b) @ normed(H_b).T
    np.fill_diagonal(Sa, -np.inf); np.fill_diagonal(Sb, -np.inf)
    Na = np.argsort(-Sa, axis=1)[:, :k]
    Nb = np.argsort(-Sb, axis=1)[:, :k]
    overlaps = []
    for i in range(len(H_a)):
        a, b = set(Na[i].tolist()), set(Nb[i].tolist())
        overlaps.append(len(a & b) / len(a | b))
    return float(np.mean(overlaps)), overlaps


def linear_cka(X, Y):
    """Centered Kernel Alignment, linear kernel."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / max(den, 1e-30))


def shuffle_null(H_a, H_b, k=10, n_shuffles=20, rng=None):
    rng = rng or np.random.default_rng(1)
    out = []
    for _ in range(n_shuffles):
        perm = rng.permutation(len(H_b))
        m, _ = mutual_knn_agreement(H_a, H_b[perm], k=k)
        out.append(m)
    return float(np.mean(out)), float(np.std(out))


def random_orthonormal(d, k, rng):
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def project_and_score(H_a, H_b, k_dim, mode, rng):
    """mode in {'svd', 'random'}.  Project each model's H onto either its
    top-k SVD basis or a random orthonormal k-frame, then recompute mutual
    kNN."""
    def basis(H):
        Hc = H - H.mean(0, keepdims=True)
        if mode == "svd":
            U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
            return Vh[:k_dim].T  # [d, k]
        elif mode == "random":
            return random_orthonormal(H.shape[1], k_dim, rng)
        raise ValueError(mode)
    Ba, Bb = basis(H_a), basis(H_b)
    Pa = (H_a - H_a.mean(0)) @ Ba
    Pb = (H_b - H_b.mean(0)) @ Bb
    knn, _ = mutual_knn_agreement(Pa, Pb, k=10)
    cka  = linear_cka(Pa, Pb)
    return {"mutual_knn": knn, "cka": cka}


def procrustes_residual(H_a, H_b, k_dim, mode, rng, train_frac=0.5):
    """Project both models to k_dim via mode in {svd, random}, fit
    orthogonal Procrustes alignment R minimising ||P_a R - P_b||_F on
    train half, report residual on test half (lower = better cross-model
    universal structure)."""
    n = len(H_a)
    idx = rng.permutation(n)
    n_tr = int(train_frac * n)
    tr, te = idx[:n_tr], idx[n_tr:]

    def basis(H, tr):
        Hc = H - H[tr].mean(0, keepdims=True)
        if mode == "svd":
            U, S, Vh = np.linalg.svd(Hc[tr], full_matrices=False)
            return Vh[:k_dim].T, H[tr].mean(0)
        elif mode == "random":
            return random_orthonormal(H.shape[1], k_dim, rng), H[tr].mean(0)
        raise ValueError(mode)

    Ba, mu_a = basis(H_a, tr); Bb, mu_b = basis(H_b, tr)
    Pa = (H_a - mu_a) @ Ba
    Pb = (H_b - mu_b) @ Bb
    # orthogonal Procrustes: min || Pa[tr] R - Pb[tr] ||_F over R orthogonal
    M = Pa[tr].T @ Pb[tr]
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    aligned_te = Pa[te] @ R
    res = float(np.linalg.norm(aligned_te - Pb[te], ord="fro") / np.linalg.norm(Pb[te], ord="fro"))
    return res


# --------------- main ---------------
def main():
    rng = np.random.default_rng(42)
    print("[corpus] building...")
    corpus = build_corpus(per_domain=100)  # 400 prompts
    print(f"  {len(corpus)} prompts: " + ", ".join(f"{d}={sum(1 for x in corpus if x[0]==d)}" for d in {"factual","code","math","creative"}))

    Hs = {}
    for short, mid in MODELS:
        print(f"[hidden] {short} ({mid}) ...")
        Hs[short] = collect_hidden(mid, corpus)
        print(f"  shape {Hs[short].shape}")

    H_a, H_b = Hs["smol135m"], Hs["qwen500m"]

    print("[H1] cross-model agreement on raw hidden states...")
    knn_raw, _ = mutual_knn_agreement(H_a, H_b, k=10)
    cka_raw = linear_cka(H_a, H_b)
    null_mean, null_std = shuffle_null(H_a, H_b, k=10, n_shuffles=20, rng=rng)

    print(f"  mutual-kNN(10) = {knn_raw:.4f}   null = {null_mean:.4f} +/- {null_std:.4f}")
    print(f"  linear CKA     = {cka_raw:.4f}")

    h1 = {
        "n_prompts": len(corpus),
        "k_neighbors": 10,
        "mutual_knn_raw": knn_raw,
        "linear_cka_raw": cka_raw,
        "shuffle_null_knn_mean": null_mean,
        "shuffle_null_knn_std":  null_std,
        "z_score_vs_null":       (knn_raw - null_mean) / max(null_std, 1e-6),
    }

    print("[H2] same agreement after projecting to k-dim subspace...")
    h2 = {}
    for k_dim in [16, 32, 64, 128]:
        h2[f"k={k_dim}"] = {
            "svd":    project_and_score(H_a, H_b, k_dim, "svd",    rng),
            "random": project_and_score(H_a, H_b, k_dim, "random", rng),
        }
        print(f"  k={k_dim}  svd: knn={h2[f'k={k_dim}']['svd']['mutual_knn']:.4f}  "
              f"cka={h2[f'k={k_dim}']['svd']['cka']:.4f}   "
              f"random: knn={h2[f'k={k_dim}']['random']['mutual_knn']:.4f}  "
              f"cka={h2[f'k={k_dim}']['random']['cka']:.4f}")

    print("[H3] Procrustes alignment residual (lower = more universal)...")
    h3 = {}
    for k_dim in [32, 64, 128]:
        # average a few random-control draws
        rs_svd, rs_rand = [], []
        for s in range(3):
            r = np.random.default_rng(100 + s)
            rs_svd .append(procrustes_residual(H_a, H_b, k_dim, "svd",    r))
            rs_rand.append(procrustes_residual(H_a, H_b, k_dim, "random", r))
        h3[f"k={k_dim}"] = {
            "svd_mean":    float(np.mean(rs_svd)),
            "svd_std":     float(np.std(rs_svd)),
            "random_mean": float(np.mean(rs_rand)),
            "random_std":  float(np.std(rs_rand)),
            "ratio_random_over_svd": float(np.mean(rs_rand) / np.mean(rs_svd)),
        }
        print(f"  k={k_dim}  svd residual = {np.mean(rs_svd):.4f}+/-{np.std(rs_svd):.4f}   "
              f"random = {np.mean(rs_rand):.4f}+/-{np.std(rs_rand):.4f}   "
              f"ratio = {np.mean(rs_rand)/np.mean(rs_svd):.2f}x")

    # --------------- summary ---------------
    summary = {
        "interpretation": (
            "H1: if mutual_knn_raw >> shuffle_null, the two independently "
            "trained models share structural agreement on the same prompts "
            "(the Platonic-representation reading); H2: if svd preserves "
            "that agreement at low k while random does not, an SVD-derived "
            "basis B captures real structure rather than just rank; "
            "H3: lower Procrustes residual under SVD vs random means UGT's "
            "alignment construction does work at the level of the basis "
            "directions."
        ),
        "verdict_h1_universal_structure": h1["mutual_knn_raw"] > h1["shuffle_null_knn_mean"] + 5 * max(h1["shuffle_null_knn_std"], 1e-3),
        "verdict_h2_svd_preserves_structure": all(
            h2[f"k={k}"]["svd"]["mutual_knn"] > h2[f"k={k}"]["random"]["mutual_knn"] + 0.02
            for k in [32, 64, 128]
        ),
        "verdict_h3_alignment_uses_directions": all(
            h3[f"k={k}"]["ratio_random_over_svd"] > 1.05 for k in [32, 64, 128]
        ),
    }

    out = {
        "models": [m for _, m in MODELS],
        "H1_raw": h1, "H2_projected": h2, "H3_procrustes": h3,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n--- summary ---")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
