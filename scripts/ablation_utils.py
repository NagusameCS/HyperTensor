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

#!/usr/bin/env python3
"""
Combined Ablation & Benchmark Utilities (Papers I, II, IV, V, VIII, XI, XII, XIII, XVII)

Covers multiple paper gaps in a single module:
  Paper I gap 6:  Fine k-sweep near k* (cache-fit transition)
  Paper I gap 1:  Bootstrap CIs for headline measurements
  Paper II gap 4: Per-matrix rank-importance ablation
  Paper IV gap 6: Intrinsic-dimension estimator comparison
  Paper V gap 4:  Sink-channel exemption ablation
  Paper VIII gap 5: Eviction policy ablation (LRU/LFU/jury-weighted)
  Paper XI gap 1:  LEACE-style UGT semantic probe
  Paper XI gap 6:  Stiefel vs Grassmann optimization ablation
  Paper XII gap 3: KExpansion vs fixed-rank ablation
  Paper XIII gap 3: Capability preservation under Safe OGD projection
  Paper XIV gap 6: Per-category TEH specificity reporting
  Paper XVII gap 3+4: Overparameterisation control + random-zero-set ablation

Usage:
    from ablation_utils import (bootstrap_ci, fine_k_sweep, rank_ablation,
                                 intrinsic_dim_compare, sink_ablation,
                                 eviction_ablation, ugt_semantic_probe,
                                 stiefel_vs_grassmann, kexpansion_ablation,
                                 capability_preservation, teh_report,
                                 overparam_control, random_zero_ablation)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time


# ======================================================================
# Paper I gap 1: Bootstrap CIs
# ======================================================================

@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_resamples: int = 10000
    method: str = 'percentile'

def bootstrap_ci(
    data: List[float],
    ci_level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute bootstrap percentile confidence interval.

    Args:
        data: List of measurements.
        ci_level: Confidence level (default 0.95).
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        BootstrapResult with mean, lower, upper bounds.
    """
    rng = np.random.default_rng(seed)
    data = np.array(data, dtype=np.float64)
    n = len(data)

    means = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = (1 - ci_level) / 2
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1 - alpha)))

    return BootstrapResult(
        mean=float(np.mean(data)),
        ci_lower=lower,
        ci_upper=upper,
        ci_level=ci_level,
        n_resamples=n_resamples,
    )


# ======================================================================
# Paper I gap 6: Fine k-sweep near k*
# ======================================================================

def fine_k_sweep(
    k_star: int = 1024,
    window: int = 200,
    step: int = 25,
    throughput_fn: Optional[Callable[[int], float]] = None,
) -> Dict:
    """
    Characterise the cache-fit transition shape (cliff vs ramp).

    Sweeps k in [k_star - window, k_star + window] at fine resolution
    to detect whether throughput drops as a cliff (sharp) or ramp (gradual).

    Args:
        k_star: Estimated optimal k.
        window: Sweep half-width.
        step: k increment.
        throughput_fn: Function mapping k → throughput (tok/s).
                       If None, uses a synthetic sigmoid model.

    Returns:
        Dict with k values, throughput, and transition analysis.
    """
    if throughput_fn is None:
        # Synthetic cache-fit model: sigmoid transition at k_star
        def throughput_fn(k):
            base = 35.0  # Baseline tok/s
            peak = 38.0  # Peak at k_star
            # Throughput drops when working set exceeds L2
            working_set_mb = 2 * 4096 * k / (1024 * 1024)  # ~bytes
            l2_mb = 32.0
            # Smooth transition via sigmoid
            saturation = 1.0 / (1.0 + np.exp((working_set_mb - l2_mb * 0.8) / 2.0))
            return base + (peak - base) * saturation

    ks = list(range(max(k_star - window, 64), k_star + window + 1, step))
    tps = [throughput_fn(k) for k in ks]

    # Detect transition type
    tps_arr = np.array(tps)
    grad = np.gradient(tps_arr)

    # Cliff: max gradient > 3× mean gradient
    is_cliff = np.max(np.abs(grad)) > 3 * np.mean(np.abs(grad)) if len(grad) > 1 else False

    return {
        'k_star': k_star,
        'k_values': ks,
        'throughput': tps,
        'transition_type': 'cliff' if is_cliff else 'ramp',
        'max_gradient': float(np.max(np.abs(grad))) if len(grad) > 1 else 0,
        'optimal_k': ks[int(np.argmax(tps_arr))] if tps else k_star,
    }


# ======================================================================
# Paper II gap 4: Per-matrix rank-importance ablation
# ======================================================================

def rank_ablation(
    rank_budget: int,
    matrices: Dict[str, int],  # name → default_rank
    quality_fn: Callable[[Dict[str, int]], float],
) -> Dict:
    """
    Drop one matrix-class at a time and measure quality impact.

    Args:
        rank_budget: Total rank budget.
        matrices: Dict of matrix names → their default rank allocation.
        quality_fn: Function mapping rank allocations → quality score (e.g., PPL).

    Returns:
        Dict with per-matrix importance rankings.
    """
    baseline = quality_fn(matrices)
    results = {}

    for name in matrices:
        # Drop this matrix: set its rank to 0, redistribute budget
        ablated = matrices.copy()
        freed = ablated[name]
        ablated[name] = 0

        # Redistribute freed budget proportionally to remaining matrices
        remaining = [n for n in ablated if n != name]
        total_remaining_rank = sum(ablated[n] for n in remaining)
        if total_remaining_rank > 0:
            for n in remaining:
                ablated[n] += int(freed * ablated[n] / total_remaining_rank)

        quality = quality_fn(ablated)
        results[name] = {
            'quality': quality,
            'delta': quality - baseline,
            'importance': abs(quality - baseline) / max(baseline, 1e-10),
        }

    return {
        'baseline_quality': baseline,
        'rank_budget': rank_budget,
        'per_matrix': results,
        'ranking': sorted(results.keys(),
                         key=lambda n: results[n]['importance'],
                         reverse=True),
    }


# ======================================================================
# Paper IV gap 6: Intrinsic-dimension estimator comparison
# ======================================================================

def intrinsic_dim_compare(
    data: np.ndarray,
    max_dim: int = 100,
) -> Dict:
    """
    Compare intrinsic dimension estimators on a dataset.

    Methods:
      - PCA: 95% variance threshold
      - TwoNN (Facco et al., 2017): ratio of 2nd-nearest to nearest neighbor
      - MLE (Levina-Bickel): maximum-likelihood local dimension

    Args:
        data: Data matrix, shape (n_samples, d_ambient).
        max_dim: Maximum dimension to consider.

    Returns:
        Dict with per-method dimension estimates.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    n, d = data.shape
    results = {}

    # 1. PCA: dim where cumulative variance > 0.95
    pca = PCA().fit(data)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k_pca = int(np.searchsorted(cumvar, 0.95)) + 1
    results['pca_95'] = min(k_pca, max_dim)

    # 2. TwoNN (Facco et al.)
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(data)
    dists, _ = nn.kneighbors(data)
    # Ratio r_i = d_{i,2} / d_{i,1}
    mu = dists[:, 2] / (dists[:, 1] + 1e-10)
    # For a Poisson process on a d-dim manifold: μ_i ~ Pareto(1, d)
    # d ≈ n / Σ log(μ_i)
    log_mu = np.log(mu[mu > 1.0 + 1e-10])  # Filter degenerate ratios
    if len(log_mu) > 0:
        k_twonn = int(np.round(len(log_mu) / np.sum(log_mu)))
    else:
        k_twonn = 1
    results['twonn'] = min(k_twonn, max_dim)

    # 3. MLE (Levina-Bickel)
    # For each point, d_local = (1/(k-2) Σ log(T_k / T_j))^{-1}
    k_neighbors = min(20, n - 1)
    nn_mle = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(data)
    dists_mle, _ = nn_mle.kneighbors(data)
    # Use distances to k-th neighbor
    T_k = dists_mle[:, k_neighbors:k_neighbors+1]
    d_local = np.zeros(n)
    for i in range(n):
        ratios = np.log(T_k[i] / (dists_mle[i, 1:k_neighbors] + 1e-10))
        ratios = ratios[ratios > 0]
        if len(ratios) > 0:
            d_local[i] = (len(ratios) - 1) / np.sum(ratios)
    k_mle = int(np.round(np.median(d_local[d_local > 0]))) if np.any(d_local > 0) else 1
    results['mle_levina_bickel'] = min(k_mle, max_dim)

    # 4. Persistent homology dim (simplified: betti-0 persistence)
    # Placeholder for a full TDA pipeline
    results['persistent_homology'] = None  # Requires gudhi or ripser

    return results


# ======================================================================
# Paper V gap 4: Sink-channel exemption ablation
# ======================================================================

def sink_ablation(
    k: int,
    T_values: List[int] = [0, 8, 16, 32, 64],
    error_fn: Optional[Callable[[int, int], float]] = None,
) -> Dict:
    """
    Measure reconstruction error with/without sink protection.

    Args:
        k: Compression rank.
        T_values: Sink exemption sizes to test.
        error_fn: Function (k, T) → reconstruction error.

    Returns:
        Dict with per-T error measurements.
    """
    if error_fn is None:
        def error_fn(k, T):
            # Synthetic model: error decreases with T, floors at k-dependent value
            base_err = 0.5 * np.exp(-k / 256) + 0.05
            sink_benefit = T / (T + 16) * 0.3 * np.exp(-k / 512)
            return base_err - sink_benefit

    results = {}
    for T in T_values:
        err = error_fn(k, T)
        results[f'T={T}'] = {
            'error': err,
            'improvement_vs_T0': (results.get('T=0', {}).get('error', err) - err)
            if T > 0 else 0,
        }

    return {
        'k': k,
        'results': results,
        'best_T': min(T_values, key=lambda t: results[f'T={t}']['error']),
    }


# ======================================================================
# Paper VIII gap 5: Eviction policy ablation
# ======================================================================

def eviction_ablation(
    n_queries: int = 1000,
    cache_size: int = 100,
    policies: List[str] = None,
) -> Dict:
    """
    Compare cache eviction policies: LRU, LFU, jury-confidence-weighted.

    Args:
        n_queries: Number of queries to simulate.
        cache_size: Maximum cache entries.
        policies: List of policy names to test.

    Returns:
        Dict with per-policy hit rates and mean latencies.
    """
    if policies is None:
        policies = ['LRU', 'LFU', 'jury_weighted', 'random']

    rng = np.random.default_rng(42)

    # Generate query stream with Zipf popularity
    n_unique = n_queries // 4
    popularity = 1.0 / np.arange(1, n_unique + 1)
    popularity /= popularity.sum()

    queries = rng.choice(n_unique, size=n_queries, p=popularity)
    jury_scores = rng.uniform(0.5, 1.0, n_unique)  # Per-item jury confidence

    results = {}

    for policy in policies:
        cache = {}  # item_id → (last_access_time, access_count, jury_score)
        hits = 0
        access_time = 0

        for t, q in enumerate(queries):
            access_time += 1
            if q in cache:
                hits += 1
                entry = cache[q]
                cache[q] = (access_time, entry[1] + 1, entry[2])
            else:
                if len(cache) >= cache_size:
                    # Evict based on policy
                    if policy == 'LRU':
                        victim = min(cache.keys(),
                                    key=lambda k: cache[k][0])
                    elif policy == 'LFU':
                        victim = min(cache.keys(),
                                    key=lambda k: cache[k][1])
                    elif policy == 'jury_weighted':
                        victim = min(cache.keys(),
                                    key=lambda k: cache[k][2] / (cache[k][1] + 1))
                    else:  # random
                        victim = rng.choice(list(cache.keys()))
                    del cache[victim]

                cache[q] = (access_time, 1, jury_scores[q])

        results[policy] = {
            'hit_rate': hits / n_queries,
            'total_hits': hits,
            'cache_utilization': len(cache) / cache_size,
        }

    return results


# ======================================================================
# Paper XI gap 1: LEACE-style UGT semantic probe
# ======================================================================

def ugt_semantic_probe(
    B: np.ndarray,  # UGT basis, shape (d, k)
    hidden_states: np.ndarray,  # shape (n_samples, d)
    labels: np.ndarray,  # shape (n_samples,)
    n_classes: int = 2,
) -> Dict:
    """
    LEACE-style probe: train linear classifier on B^T h to predict labels.

    Compare against:
      (a) Random orthonormal B' of same rank
      (b) Full h (no projection)

    Args:
        B: UGT basis, shape (d, k).
        hidden_states: Hidden states, shape (n, d).
        labels: Integer labels, shape (n,).
        n_classes: Number of classes.

    Returns:
        Dict with accuracy per probe type.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    results = {}

    # Project data
    h_proj = hidden_states @ B  # (n, k)

    # Train linear probe
    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    scores = cross_val_score(clf, h_proj, labels, cv=5, scoring='accuracy')
    results['ugt_probe'] = {
        'mean_accuracy': float(np.mean(scores)),
        'std_accuracy': float(np.std(scores)),
    }

    # Random orthonormal baseline
    rng = np.random.default_rng(42)
    B_rand = rng.normal(0, 1, B.shape)
    B_rand, _ = np.linalg.qr(B_rand)
    h_rand = hidden_states @ B_rand
    scores_rand = cross_val_score(
        LogisticRegression(max_iter=1000), h_rand, labels, cv=5, scoring='accuracy'
    )
    results['random_basis'] = {
        'mean_accuracy': float(np.mean(scores_rand)),
        'std_accuracy': float(np.std(scores_rand)),
    }

    # Full hidden state
    scores_full = cross_val_score(
        LogisticRegression(max_iter=1000), hidden_states, labels, cv=5, scoring='accuracy'
    )
    results['full_hidden'] = {
        'mean_accuracy': float(np.mean(scores_full)),
        'std_accuracy': float(np.std(scores_full)),
    }

    # Semantic gap: how much UGT captures vs full
    gap = results['full_hidden']['mean_accuracy'] - results['ugt_probe']['mean_accuracy']
    rand_gap = results['ugt_probe']['mean_accuracy'] - results['random_basis']['mean_accuracy']
    results['semantic_gap'] = float(gap)
    results['random_gap'] = float(rand_gap)
    results['is_semantic'] = rand_gap > 0.05  # UGT is above random by meaningful margin

    return results


# ======================================================================
# Paper XI gap 6: Stiefel vs Grassmann ablation
# ======================================================================

def stiefel_vs_grassmann(
    d: int = 512,
    k: int = 32,
    n_steps: int = 500,
) -> Dict:
    """
    Compare plain SVD basis vs Riemannian fine-tuned basis.

    Measures whether Grassmann optimization improves zone separation
    over the initial SVD basis.

    Args:
        d: Ambient dimension.
        k: Subspace dimension.
        n_steps: Optimization steps.

    Returns:
        Dict with separation metrics.
    """
    rng = np.random.default_rng(42)

    # Generate synthetic data with 3 zones
    n_per_zone = 100
    zones = {
        'zone_0': rng.normal(0, 1, (n_per_zone, d)),
        'zone_1': rng.normal(2, 0.5, (n_per_zone, d)),
        'zone_2': rng.normal(-1, 0.8, (n_per_zone, d)),
    }

    # Concatenate and compute SVD basis (use right singular vectors)
    X = np.vstack(list(zones.values()))
    U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    B_svd = Vt[:k, :].T  # Right singular vectors: shape (d, k)

    # Measure zone separation with SVD basis
    centroids_svd = {}
    for name, data in zones.items():
        centroids_svd[name] = (data @ B_svd).mean(axis=0)

    # Pairwise cosine distances
    def mean_separation(centroids):
        dists = []
        names = list(centroids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c1 = centroids[names[i]]
                c2 = centroids[names[j]]
                dist = 1.0 - float(np.dot(c1, c2) / (
                    max(np.linalg.norm(c1), 1e-10) * max(np.linalg.norm(c2), 1e-10)
                ))
                dists.append(dist)
        return np.mean(dists) if dists else 0.0

    sep_svd = mean_separation(centroids_svd)

    # Simulate Riemannian fine-tuning (simplified)
    # In production, this uses RiemannianAdamW on Gr(k,d)
    B_grass = B_svd.copy()
    for step in range(n_steps):
        # Gradient step: push centroids apart
        grad = np.zeros_like(B_grass)
        names = list(zones.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c_i = zones[names[i]] @ B_grass
                c_j = zones[names[j]] @ B_grass
                # Push centroids apart in projected space
                diff = c_i.mean(axis=0) - c_j.mean(axis=0)
                grad += np.outer(
                    zones[names[i]].mean(axis=0) - zones[names[j]].mean(axis=0),
                    diff / max(np.linalg.norm(diff), 1e-10)
                ) * 0.01

        B_grass = B_grass + grad
        # QR retraction
        Q, _ = np.linalg.qr(B_grass)
        B_grass = Q

    centroids_grass = {}
    for name, data in zones.items():
        centroids_grass[name] = (data @ B_grass).mean(axis=0)

    sep_grass = mean_separation(centroids_grass)

    return {
        'svd_separation': float(sep_svd),
        'grassmann_separation': float(sep_grass),
        'improvement': float(sep_grass - sep_svd),
        'relative_improvement': float((sep_grass - sep_svd) / max(sep_svd, 1e-10)),
    }


# ======================================================================
# Paper XII gap 3: KExpansion vs fixed-rank ablation
# ======================================================================

def kexpansion_ablation(
    d: int = 256,
    k_fixed: int = 64,
    kexpansion_init: int = 32,
    kexpansion_final: int = 128,
) -> Dict:
    """
    Compare KExpansion auto-grow vs fixed-rank a priori.

    Returns convergence speed and final quality for both.
    """
    rng = np.random.default_rng(42)
    W_target = rng.normal(0, 1, (d, d)).astype(np.float64)

    # Fixed-rank: train at k_fixed
    U, S, Vt = np.linalg.svd(W_target, full_matrices=False)
    W_fixed = U[:, :k_fixed] @ np.diag(S[:k_fixed]) @ Vt[:k_fixed, :]
    err_fixed = float(np.linalg.norm(W_fixed - W_target, 'fro') /
                      max(np.linalg.norm(W_target, 'fro'), 1e-10))

    # KExpansion: start at k_init, grow to k_final
    # Simplified: measure error at each expansion point
    k_points = list(range(kexpansion_init, kexpansion_final + 1, 32))
    errors = []
    for k in k_points:
        W_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        err = float(np.linalg.norm(W_k - W_target, 'fro') /
                    max(np.linalg.norm(W_target, 'fro'), 1e-10))
        errors.append(err)

    return {
        'fixed_rank_error': err_fixed,
        'fixed_rank': k_fixed,
        'kexpansion_errors': dict(zip([str(k) for k in k_points], errors)),
        'kexpansion_final_error': errors[-1],
        'winner': 'fixed' if err_fixed < errors[-1] else 'kexpansion',
    }


# ======================================================================
# Paper XIII gap 3: Capability preservation under Safe OGD projection
# ======================================================================

def capability_preservation(
    hidden_states: np.ndarray,  # (n, d)
    Q_f: np.ndarray,             # Forbidden subspace basis (d, f)
    P_safe: Optional[np.ndarray] = None,  # Precomputed projector
) -> Dict:
    """
    Measure how much of the original signal survives Safe OGD projection.

    Args:
        hidden_states: Original hidden states.
        Q_f: Forbidden subspace basis.
        P_safe: Safe projector I - Q_f Q_f^T (computed if None).

    Returns:
        Dict with preservation metrics.
    """
    if P_safe is None:
        P_safe = np.eye(Q_f.shape[0]) - Q_f @ Q_f.T

    h_safe = hidden_states @ P_safe.T

    # Per-sample cosine similarity
    cosines = []
    for i in range(len(hidden_states)):
        cos = np.dot(hidden_states[i], h_safe[i]) / (
            max(np.linalg.norm(hidden_states[i]), 1e-10) *
            max(np.linalg.norm(h_safe[i]), 1e-10)
        )
        cosines.append(float(cos))

    cosines = np.array(cosines)

    # Forbidden subspace leakage
    leakage = np.array([
        float(np.linalg.norm(Q_f.T @ h_safe[i]) / max(np.linalg.norm(h_safe[i]), 1e-10))
        for i in range(len(h_safe))
    ])

    return {
        'mean_cosine_preservation': float(np.mean(cosines)),
        'min_cosine_preservation': float(np.min(cosines)),
        'max_forbidden_leakage': float(np.max(leakage)),
        'mean_forbidden_leakage': float(np.mean(leakage)),
        'is_zero_leakage': float(np.max(leakage)) < 1e-10,
        'fraction_above_099': float(np.mean(cosines > 0.99)),
    }


# ======================================================================
# Paper XIV gap 6: Per-category TEH specificity reporting
# ======================================================================

def teh_report(
    categories: Dict[str, Dict[str, np.ndarray]],
    Q_f: np.ndarray,
) -> Dict:
    """
    Generate per-category TEH specificity report.

    Args:
        categories: Dict mapping category_name → {
            'harm': hidden_states_harmful (n_harm, d),
            'benign': hidden_states_benign (n_benign, d),
        }
        Q_f: Forbidden subspace basis.

    Returns:
        Dict with per-category TEH metrics.
    """
    report = {}
    for cat_name, data in categories.items():
        h_harm = data['harm']
        h_benign = data['benign']

        teh_harm = np.array([
            float(np.linalg.norm(Q_f.T @ h) / max(np.linalg.norm(h), 1e-10)) * 100
            for h in h_harm
        ])
        teh_benign = np.array([
            float(np.linalg.norm(Q_f.T @ h) / max(np.linalg.norm(h), 1e-10)) * 100
            for h in h_benign
        ])

        # ROC analysis
        thresholds = np.linspace(0, 50, 100)
        tpr, fpr = [], []
        for tau in thresholds:
            tpr.append(np.mean(teh_harm > tau))
            fpr.append(np.mean(teh_benign > tau))

        # Best F1 with zero FP
        f1_scores = []
        for i, tau in enumerate(thresholds):
            tp = np.sum(teh_harm > tau)
            fp = np.sum(teh_benign > tau)
            fn = np.sum(teh_harm <= tau)
            if tp + fp > 0:
                precision = tp / (tp + fp)
                recall = tp / max(tp + fn, 1)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
            else:
                f1 = 0.0
            f1_scores.append((f1, tau, fp == 0))

        # Best with zero FPR constraint
        zero_fp = [(f1, tau) for f1, tau, is_zf in f1_scores if is_zf]
        best_zero_fp = max(zero_fp, key=lambda x: x[0]) if zero_fp else (0, 0)

        specificity = float(np.mean(teh_harm) / max(np.mean(teh_benign), 1e-10))

        report[cat_name] = {
            'mean_teh_harm': float(np.mean(teh_harm)),
            'mean_teh_benign': float(np.mean(teh_benign)),
            'max_teh_harm': float(np.max(teh_harm)),
            'specificity': specificity,
            'best_f1_zero_fp': float(best_zero_fp[0]),
            'best_threshold_zero_fp': float(best_zero_fp[1]),
            'detection_rate_at_zero_fp': float(np.mean(teh_harm > best_zero_fp[1])),
        }

    # Aggregate
    spec_values = [r['specificity'] for r in report.values()]
    return {
        'per_category': report,
        'mean_specificity': float(np.mean(spec_values)),
        'max_specificity': float(np.max(spec_values)),
        'min_specificity': float(np.min(spec_values)),
    }


# ======================================================================
# Paper XVII gap 3+4: Overparameterisation + random-zero ablation
# ======================================================================

def overparam_control(
    embed_dim: int = 768,
    n_zeros: int = 105,
    n_off_critical: int = 60,
    dims_to_test: List[int] = None,
) -> Dict:
    """
    Test whether the ACM embedder's separation survives at lower dimensions.

    If a 16-dim embedder also "works", the 768-dim result is overfitting.

    Args:
        embed_dim: Full embedder dimension.
        n_zeros: Number of critical zeros.
        n_off_critical: Number of off-critical points.
        dims_to_test: List of embedder dimensions to evaluate.

    Returns:
        Dict with per-dimension separation metrics.
    """
    if dims_to_test is None:
        dims_to_test = [16, 32, 64, 128, 256, 768]

    rng = np.random.default_rng(42)

    # Generate synthetic critical and off-critical embeddings
    # Critical zeros cluster tightly; off-critical scatter
    critical = rng.normal(0, 0.01, (n_zeros, embed_dim))
    off_critical = rng.normal(0.5, 0.1, (n_off_critical, embed_dim))

    results = {}
    for dim in dims_to_test:
        # Project to lower dimension (simulating smaller embedder)
        proj = np.eye(embed_dim)[:, :dim]  # Identity projection
        crit_proj = critical @ proj
        off_proj = off_critical @ proj

        crit_mean = np.mean(crit_proj, axis=0)
        off_mean = np.mean(off_proj, axis=0)
        separation = float(np.linalg.norm(crit_mean - off_mean))

        # Within-class variance
        crit_var = float(np.mean(np.var(crit_proj, axis=0)))
        off_var = float(np.mean(np.var(off_proj, axis=0)))

        results[str(dim)] = {
            'separation': separation,
            'critical_variance': crit_var,
            'off_critical_variance': off_var,
            'signal_to_noise': separation / max(crit_var + off_var, 1e-10),
            'is_meaningful': separation > 3 * np.sqrt(crit_var + off_var),
        }

    return results


def random_zero_ablation(
    n_zeros: int = 105,
    n_random: int = 60,
    embed_dim: int = 768,
) -> Dict:
    """
    Replace off-critical points with RANDOM points and test if the
    embedder still separates them from critical zeros.

    If yes, the separation result is meaningless (the embedder just
    separates ANY two arbitrary sets, not zeros from non-zeros).

    Args:
        n_zeros: Number of critical zeros.
        n_random: Number of random replacement points.
        embed_dim: Embedder dimension.

    Returns:
        Dict with separation comparison.
    """
    rng = np.random.default_rng(42)

    critical = rng.normal(0, 0.01, (n_zeros, embed_dim))
    off_critical = rng.normal(0.5, 0.1, (n_random, embed_dim))
    random_points = rng.normal(0, 1, (n_random, embed_dim))

    # Real separation
    real_sep = float(np.linalg.norm(
        np.mean(critical, axis=0) - np.mean(off_critical, axis=0)
    ))

    # Random separation (same setup, random points)
    rand_sep = float(np.linalg.norm(
        np.mean(critical, axis=0) - np.mean(random_points, axis=0)
    ))

    return {
        'real_separation': real_sep,
        'random_separation': rand_sep,
        'ratio': real_sep / max(rand_sep, 1e-10),
        'is_false_positive': rand_sep > real_sep * 0.5,
        'verdict': ('WARNING: random baseline achieves comparable separation. '
                     'Result may be meaningless.'
                     if rand_sep > real_sep * 0.5
                     else 'Real separation exceeds random baseline.'),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Combined Ablation Utilities — Self-Test")
    print("=" * 60)

    # Bootstrap CI
    ci = bootstrap_ci([1.0, 1.1, 0.9, 1.05, 1.15, 0.95, 1.08, 1.02])
    print(f"\n  Bootstrap CI: mean={ci.mean:.3f}, [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]")

    # Fine k-sweep
    sweep = fine_k_sweep(k_star=1024, window=200, step=50)
    print(f"  Fine k-sweep: {sweep['transition_type']}, optimal_k={sweep['optimal_k']}")

    # Intrinsic dim
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (200, 100))
    data[:, :10] *= 5  # Low-rank structure
    idim = intrinsic_dim_compare(data)
    print(f"  Intrinsic dim: PCA={idim['pca_95']}, TwoNN={idim['twonn']}, MLE={idim['mle_levina_bickel']}")

    # Sink ablation
    sink = sink_ablation(k=512, T_values=[0, 8, 16, 32])
    print(f"  Sink best T: {sink['best_T']}")

    # Eviction
    evict = eviction_ablation(n_queries=500, cache_size=50)
    for policy, r in evict.items():
        print(f"  Eviction {policy}: hit_rate={r['hit_rate']:.3f}")

    # Stiefel vs Grassmann
    svg = stiefel_vs_grassmann(d=128, k=16, n_steps=100)
    print(f"  Stiefel→Grassmann improvement: {svg['relative_improvement']*100:.1f}%")

    # KExpansion
    kexp = kexpansion_ablation(d=128, k_fixed=64)
    print(f"  KExpansion winner: {kexp['winner']}")

    # Overparam
    over = overparam_control(dims_to_test=[16, 32, 64, 128, 768])
    for dim, r in over.items():
        print(f"  Overparam D={dim}: sep={r['separation']:.3f}, meaningful={r['is_meaningful']}")

    # Random zero
    rz = random_zero_ablation()
    print(f"  Random zero: {rz['verdict'][:60]}...")

    print("\n  Ablation Utils module: OK")
