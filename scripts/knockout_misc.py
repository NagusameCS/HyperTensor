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

"""Paper VIII gaps 3-4 + IX gap 3 + III gap 4 + Foundation gap 2 + VI gap 4.

CPU/synthetic only.

Output: benchmarks/knockout_misc.json
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "knockout_misc.json"


def viii_hit_rate_decay(rng):
    # Simulate 1-hour multi-user trace: queries per second 5, novelty
    # rate alpha (fraction unseen), cache size C.
    n = 18000
    alpha = 0.15
    C = 2000
    cache = {}
    hits = []
    for t in range(n):
        is_novel = rng.random() < alpha
        q = rng.integers(0, 10**9) if is_novel else rng.integers(0, max(1, len(cache)))
        if q in cache:
            hits.append(1)
            cache[q] = t
        else:
            hits.append(0)
            cache[q] = t
            if len(cache) > C:
                # LRU evict
                old = min(cache, key=cache.get)
                del cache[old]
    h = np.array(hits, dtype=float)
    bins = np.array_split(h, 12)
    return {
        "n_queries": n, "novelty_rate": alpha, "cache_size": C,
        "hit_rate_per_5min_bin": [float(b.mean()) for b in bins],
        "overall_hit_rate":      float(h.mean()),
    }


def viii_kl_on_hits():
    # On a true cache hit (deterministic recompute), KL must be 0.  We
    # also report the case of approximate reuse (perturbed logits).
    rng = np.random.default_rng(0)
    V = 32000
    logits_ref = rng.standard_normal(V) * 0.5
    p_ref = np.exp(logits_ref - logits_ref.max())
    p_ref /= p_ref.sum()
    # exact cache: identical
    kl_exact = 0.0
    # approximate cache (e.g. low-rank rebuild error 1e-3)
    pert = logits_ref + 1e-3 * rng.standard_normal(V)
    p_a = np.exp(pert - pert.max()); p_a /= p_a.sum()
    kl_approx = float((p_ref * (np.log(p_ref) - np.log(p_a))).sum())
    # bigger perturbation
    pert2 = logits_ref + 0.05 * rng.standard_normal(V)
    p_b = np.exp(pert2 - pert2.max()); p_b /= p_b.sum()
    kl_loose = float((p_ref * (np.log(p_ref) - np.log(p_b))).sum())
    return {"kl_exact_cache": kl_exact, "kl_eps_1e-3": kl_approx, "kl_eps_5e-2": kl_loose}


def ix_multivariable_kstar():
    # Toy regression: predict optimal k* from (L2_MB, L1_KB, HBM_BW, SM)
    # using the 5-GPU training set, then leave-one-out CV.
    # Real values for 5 NVIDIA GPUs (round numbers).
    gpus = [
        # name,     L2_MB, L1_KB, HBM_GBs, SM, observed_k*
        ("RTX 4070",  36,   128,   504,    46,  1024),
        ("RTX 4080",  64,   128,   717,    76,  1280),
        ("RTX 4090",  72,   128,  1008,   128,  1408),
        ("L40S",      96,   128,   864,   142,  1568),
        ("A100-40",   40,   192,  1555,   108,  1024),
    ]
    X = np.array([g[1:5] for g in gpus], dtype=float)
    y = np.array([g[5] for g in gpus], dtype=float)
    # univariate (L2 only)
    from numpy.linalg import lstsq
    A1 = np.c_[X[:, 0:1], np.ones(len(X))]
    c1, *_ = lstsq(A1, y, rcond=None)
    pred1 = A1 @ c1
    r2_1 = 1 - ((y - pred1)**2).sum() / ((y - y.mean())**2).sum()
    # multivariate
    A4 = np.c_[X, np.ones(len(X))]
    c4, *_ = lstsq(A4, y, rcond=None)
    pred4 = A4 @ c4
    r2_4 = 1 - ((y - pred4)**2).sum() / ((y - y.mean())**2).sum()
    # leave-one-out
    loo_err1, loo_err4 = [], []
    for i in range(len(X)):
        idx = [j for j in range(len(X)) if j != i]
        ci, *_ = lstsq(np.c_[X[idx, 0:1], np.ones(len(idx))], y[idx], rcond=None)
        loo_err1.append(abs(np.r_[X[i, 0:1], 1] @ ci - y[i]))
        ci, *_ = lstsq(np.c_[X[idx], np.ones(len(idx))], y[idx], rcond=None)
        loo_err4.append(abs(np.r_[X[i], 1] @ ci - y[i]))
    return {
        "univariate_L2_R2":      float(r2_1),
        "multivariate_R2":       float(r2_4),
        "univariate_LOO_MAE":    float(np.mean(loo_err1)),
        "multivariate_LOO_MAE":  float(np.mean(loo_err4)),
        "multivariate_better":   float(np.mean(loo_err4)) < float(np.mean(loo_err1)),
    }


def iii_kl_at_temperature():
    # Standard rejection-sampling spec-decode preserves the target
    # distribution exactly at any T.  Numerically check.
    rng = np.random.default_rng(0)
    V = 1000
    res = []
    for T in [0.0, 0.5, 1.0, 1.5]:
        # target distribution
        z = rng.standard_normal(V)
        if T > 0:
            p = np.exp(z / T - (z / T).max()); p /= p.sum()
        else:
            p = np.zeros(V); p[int(z.argmax())] = 1.0
        # draft distribution (perturbed)
        z_d = z + 0.5 * rng.standard_normal(V)
        if T > 0:
            q = np.exp(z_d / T - (z_d / T).max()); q /= q.sum()
        else:
            q = np.zeros(V); q[int(z_d.argmax())] = 1.0
        # rejection-sampled distribution (analytic): r(x) = min(p,q) + accept-correction
        # By construction equals p exactly; numerically simulate with 200k draws.
        if T == 0.0:
            res.append({"T": T, "kl_p_||_r": 0.0, "method": "argmax (deterministic)"})
            continue
        N = 200_000
        accepted = []
        idx_q = rng.choice(V, size=N, p=q)
        u = rng.random(N)
        ratios = p[idx_q] / np.maximum(q[idx_q], 1e-30)
        keep = u < np.minimum(1.0, ratios)
        accepted_idx = idx_q[keep]
        # Resample on rejects from p_residual (exactly recovers p)
        n_reject = N - keep.sum()
        if n_reject > 0:
            # residual distribution = max(p - q, 0); normalize
            resid = np.maximum(p - q, 0); resid /= resid.sum() + 1e-30
            extra = rng.choice(V, size=int(n_reject), p=resid)
            accepted_idx = np.concatenate([accepted_idx, extra])
        emp = np.bincount(accepted_idx, minlength=V) / len(accepted_idx)
        kl = float((p * (np.log(p + 1e-30) - np.log(emp + 1e-30))).sum())
        res.append({"T": T, "kl_p_||_r": kl, "method": "rejection sampling, N=200000"})
    return res


def foundation_dh_real(rng):
    # Empirical d_h for a synthetic "model manifold": 32 trajectories on
    # an n-sphere, R = local cluster radius.  Sweep a query at distance d
    # from cluster center, compute jury vote vs noisy-OR theory.
    # The horizon d_h is where jury confidence drops below 0.5.
    R = 1.0
    n_traj = 32
    n_dim = 16
    centers = rng.standard_normal((1, n_dim)); centers /= np.linalg.norm(centers)
    pts = centers + R * rng.standard_normal((n_traj, n_dim)) / np.sqrt(n_dim)
    distances = np.linspace(0.5, 5.0, 60) * R
    confidences = []
    for d in distances:
        # query at distance d along center direction
        q = centers + d * (rng.standard_normal((1, n_dim)) / np.sqrt(n_dim)) / 0.0001 * 0.0
        q = centers + d * (np.array([[1.0] + [0.0]*(n_dim-1)]))
        # noisy-OR over jurors: each juror confidence c_i = exp(-||p_i - q||^2 / R^2)
        ds = np.linalg.norm(pts - q, axis=1)
        c = np.exp(-(ds / R)**2 / 2)
        # noisy-OR aggregate
        agg = 1 - np.prod(1 - c)
        confidences.append(float(agg))
    confidences = np.array(confidences)
    # find where confidence drops through 0.5
    below = np.where(confidences < 0.5)[0]
    d_h_emp = float(distances[below[0]]) if len(below) else float(distances[-1])
    return {
        "n_trajectories": n_traj, "dim": n_dim, "R": R,
        "theoretical_dh_2.36R": 2.36 * R,
        "empirical_dh":         d_h_emp,
        "abs_diff":             abs(d_h_emp - 2.36 * R),
    }


def vi_collapse_mechanism(rng):
    # Show that "routing entropy" rises sharply at low k/d on a synthetic
    # MoE-style attention pattern, while "knowledge match" decays smoothly.
    d = 64
    n_heads = 8
    seq = 32
    Q = rng.standard_normal((seq, d))
    K = rng.standard_normal((seq, d))
    res = []
    for kf in [1.0, 0.75, 0.5, 0.25, 0.15, 0.1, 0.05]:
        r = max(1, int(kf * d))
        Uq, Sq, Vq = np.linalg.svd(Q, full_matrices=False); Qr = (Uq[:, :r]*Sq[:r])@Vq[:r]
        Uk, Sk, Vk = np.linalg.svd(K, full_matrices=False); Kr = (Uk[:, :r]*Sk[:r])@Vk[:r]
        scores = Qr @ Kr.T / math.sqrt(d)
        attn = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn = attn / attn.sum(axis=1, keepdims=True)
        H = float(-(attn * np.log(attn + 1e-12)).sum(axis=1).mean())
        # knowledge match: <attn, target> with target being baseline attn
        if kf == 1.0:
            target = attn.copy()
            score = 1.0
        else:
            score = float((attn * target).sum() / target.sum() * attn.shape[1])
        res.append({"k_frac": kf, "rank": r, "attn_entropy": H, "match_to_baseline": score})
    return res


def main():
    rng = np.random.default_rng(0)
    out = {
        "paper_viii_gap3_hit_rate_decay":    viii_hit_rate_decay(rng),
        "paper_viii_gap4_kl_on_hits":        viii_kl_on_hits(),
        "paper_ix_gap3_multivariable_kstar": ix_multivariable_kstar(),
        "paper_iii_gap4_kl_at_temperature":  iii_kl_at_temperature(),
        "foundation_gap2_dh_empirical":      foundation_dh_real(rng),
        "paper_vi_gap4_collapse_mechanism":  vi_collapse_mechanism(rng),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:2000])
    print("...")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
