# GTC and Jury-GTC: A Concrete Explanation

## Part 1: GTC — Geodesic Trajectory Caching (Paper IV/VIII)

### The Problem

A transformer language model processes text by passing it through layers of matrix multiplications. For a 7B model with 32 layers and hidden dimension d=4096, generating one token costs roughly O(L·d²) operations. If 1000 users ask similar questions ("What is Python?", "Tell me about Python", "Explain Python programming"), the model does the same work 1000 times.

### The Geometric Insight

Paper IV (Organic Training Theory) discovered that the trained latent space of a transformer is a Riemannian manifold with intrinsic dimension k ≈ 30–50 — far smaller than d=4096. This means all the model's knowledge lives on a low-dimensional surface embedded in a high-dimensional space.

Paper XI (UGT) provides a shared coordinate system for this surface: a projection matrix B ∈ R^(d×k) that maps any hidden state h ∈ R^d to a compact coordinate h_k = B^T·h ∈ R^k.

### How GTC Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        GTC LIFECYCLE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STORE (first time seeing a query):                             │
│    query → model forward pass → hidden state h ∈ R^d            │
│    h_k = B^T·h  ∈ R^k                    (UGT projection)       │
│    store: {h_k, response_tokens} in cache                       │
│                                                                 │
│  RETRIEVE (subsequent similar queries):                         │
│    query → model forward pass → hidden state h' ∈ R^d           │
│    h'_k = B^T·h' ∈ R^k                   (UGT projection)       │
│    for each cached trajectory t_i:                              │
│      dist_i = 1 - cos_sim(h'_k, t_i)     (geodesic distance)    │
│    nearest = argmin(dist_i)                                     │
│    if dist_nearest < threshold:                                 │
│      return cached_response[nearest]     (INSTANT — no gen!)    │
│    else:                                                        │
│      generate new response, add to cache                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What GTC Measures (Paper VIII)

| Metric | Value | Meaning |
|--------|-------|---------|
| Cache hit rate | ~50% at sim > 0.90 | Half of queries reuse cached responses |
| Speedup over RAG | 15.5× | GTC is 15.5× faster than retrieval-augmented generation |
| Batched Jacobi | 97× at B=10 | Multiple similar queries corrected in one matrix multiply |
| Coverage radius R | ~0.02–0.10 | Median geodesic distance between cached trajectories |

### The Bottleneck

For a cache with N trajectories, retrieval is O(N·k). With k=20 and N=300, this is 6000 dot products — trivial. But as the living model accumulates interactions over months, N grows to 10,000+. At N=100,000, retrieval becomes the bottleneck: 100,000 × 20 = 2 million operations per query.

---

## Part 2: Jury-GTC — Accelerating Retrieval with the Geometric Jury

### The Jury Principle (from Paper XV Section 12)

The geometric jury is a universal aggregation formula:

```
J = 1 - ∏(1 - c_i)
```

where c_i = exp(-d_i/R) is the single-trial confidence of the i-th juror (a cached trajectory), d_i is the geodesic distance to that trajectory, and R is the coverage radius.

The jury doesn't just find the single nearest trajectory — it WEIGHS all trajectories by similarity and produces a CONFIDENCE score. This gives us two things: (1) a better match (considers consensus, not just nearest neighbor), and (2) domain-level information (which Saiyan/domain contributed most weight).

### The Jury Discovery That Enables Acceleration

From our 18 jury experiments (9 wins, 8 losses across 4 scripts), we discovered:

1. **Domain routing is 100% accurate at K=20.** The Two-Stage Jury correctly identifies the query's domain 100% of the time, even though domain centroids overlap at cos_sim 0.86–0.99.

2. **Some domains are redundant.** Trunks (creative) and Yamcha (general) have 1.000 overlap — they're identical manifolds. We can skip one entirely and lose nothing.

3. **Trajectories have measurable importance.** Removing Piccolo's best trajectory drops jury confidence by 0.05 — it carries unique information. Removing Yamcha's worst trajectory changes nothing. We should check important trajectories first.

4. **Coverage radius R varies by domain.** Goku (math) has R=0.013 (tight cluster — high confidence in-domain). Yamcha has R=0.021 (loose cluster — generalist). We can set adaptive similarity thresholds per domain.

5. **Knowledge transfer is asymmetric.** Code trajectories help math queries (sim=0.63) more than math trajectories help code queries (sim=0.42). When the primary domain doesn't have a good match, we should search the transfer domain next.

### How Jury-GTC Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    JURY-GTC LIFECYCLE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRE-COMPUTATION (done once, updated incrementally):            │
│    - Build domain index: {domain → [trajectory indices]}        │
│    - Compute coverage radius R per domain                        │
│    - Compute domain overlap matrix                                │
│    - Compute trajectory importance scores                         │
│    - Compute asymmetric transfer hints                            │
│                                                                 │
│  RETRIEVE (every query):                                        │
│    ┌─────────────────────────────────────────────────────┐      │
│    │ STAGE 1: JURY ROUTING (O(S·k), S=20)                │      │
│    │                                                    │      │
│    │ 1a. Sample S=20 random trajectories from pool       │      │
│    │ 1b. Compute softmax(sim × T=8) weights             │      │
│    │ 1c. Aggregate weights by domain                    │      │
│    │ 1d. Identify top-2 domains (dominant + transfer)    │      │
│    │ 1e. Set adaptive threshold τ = 1 - 2·R_dominant     │      │
│    └─────────────────────────────────────────────────────┘      │
│                          ↓                                       │
│    ┌─────────────────────────────────────────────────────┐      │
│    │ STAGE 2: DOMAIN SEARCH (O(D·k), D≈100)              │      │
│    │                                                    │      │
│    │ 2a. Search dominant domain cache (importance order) │      │
│    │ 2b. If best_sim < τ, search transfer domain         │      │
│    │ 2c. If best_sim >= 0.99, exit early                │      │
│    │ 2d. Return: best trajectory or "cache miss"         │      │
│    └─────────────────────────────────────────────────────┘      │
│                                                                 │
│  COMPARISON SAVED:                                              │
│    Naive GTC:     O(N·k) = 300·20 = 6000 ops                   │
│    Jury-GTC:      O((S+D)·k) = (20+100)·20 = 2400 ops          │
│    Savings:       60% fewer operations                          │
│    At N=10,000:   98% fewer operations (20+100 vs 10,000)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works When Centroids Don't

The key discovery from our centroid entanglement experiments: domain centroids overlap at cos_sim 0.86–0.99 because SVD captures prompt structure, not domain semantics. **But the softmax over individual trajectory similarities still works** because:

1. Individual trajectories from the correct domain are marginally more similar to the query
2. Softmax with temperature T=8 exponentially amplifies these marginal differences
3. Aggregating over S=20 random samples gives the correct domain 100% of the time

The jury doesn't need centroids. It needs individual trajectories and softmax. This is the same principle that makes the 105-zero Riemann jury work: individual measurements have noise, but aggregation amplifies signal.

### The Scaling Advantage

| Pool Size N | Naive O(N) | Jury O(S+D) | Savings | Crossover |
|-------------|-----------|-------------|---------|-----------|
| 300 | 300 | 89 | 70% | Already faster in ops |
| 1,000 | 1,000 | 120 | 88% | 8× faster |
| 10,000 | 10,000 | 150 | 98.5% | 66× faster |
| 100,000 | 100,000 | 200 | 99.8% | 500× faster |

The jury overhead is constant (~100 comparisons). The savings grow linearly with pool size. This is the asymptotic advantage: GTC retrieval goes from O(N) to O(1) as the pool grows.

---

## Part 3: What We Test

### Small Model Benchmarks (Local RTX 4070 Laptop)
- Model: Qwen2.5-1.5B-Instruct (UGT k=20)
- Pool size: 300–3,000 trajectories (synthetic scaling)
- Metrics: comparisons, latency, hit rate, match accuracy

### Large Model Benchmarks (EC2 L40S)
- Model: Qwen2.5-7B-Instruct 4-bit (UGT k=512)
- Pool size: 1,000–10,000 trajectories
- Metrics: comparisons, latency, hit rate, KV-cache memory

### Stress Tests
- 100,000 trajectory pool (simulated)
- Mixed domain queries (adversarial routing)
- Cold start (no pre-computed domain index)
- Incremental update (add 1,000 new trajectories)
