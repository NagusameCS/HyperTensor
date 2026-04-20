# Autonomous Axiomatic Subsystem — Beta-3

Real-geometry implementation of the 5-phase manifold analysis pipeline for neural network models. Beta-3 extends Beta-2 with decode-aligned oracle targets, persistent warp-state storage, knowledge injection plumbing, warm-manifold caching, and a tuned active-learning budget for fast-mode operation.

## 1. What Changed from Beta-2 to Beta-3

| Aspect | Beta-2 | Beta-3 |
|--------|--------|--------|
| Phase 4 oracle budget | Fixed cap (16) | Adaptive (2–4 fast, 12 full) |
| Phase 4 candidate selection | EMA scoring | Uncertainty-driven, early stop |
| Phase 5 target | Random vocab token | **Decode-aligned oracle token** |
| Phase 5 velocity prior | Zero | Curvature-informed (Christoffel acceleration) |
| Phase 5 retry | None | Adaptive step/velocity damping on divergence |
| Knowledge injection | Not implemented | Prototype: local Christoffel warp + Gaussian decay |
| Warp state persistence | None | `axiom_warp_state.dat` survives restarts |
| Warp recompute trigger | None | Threshold-based: full Phase 3+4 refresh |
| Warm-cache (Phase 3) | None | LRU token_id → hidden state; 197 s → 0.17 s |
| MRR (fast mode) | ~0.032 | **~0.067** |
| Total fast-mode time | ~1218 ms | **~977 ms** |

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    axiom_beta.c                          │
│  Phase 1 ─→ Phase 2 ─→ Phase 3 ─→ Phase 4 ─→ Phase 5  │
└────┬───────────┬───────────┬───────────┬──────────┬──────┘
     │           │           │           │          │
     ▼           ▼           ▼           ▼          ▼
  axiom_linalg  llm.h     axiom_geo  axiom_geo  axiom_geo
  (PCA, TwoNN)  (model    (metric    (oracle    (geodesic
                 weights)   field,    calls)     integrator)
                           Christoffel,
                           curvature)
```

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `runtime/nn/axiom_linalg.h` | ~130 | Dense matrix, PCA, TwoNN, dequantization API |
| `runtime/nn/axiom_linalg.c` | ~490 | Jacobi eigendecomp, economy-mode PCA, TwoNN ID estimator, Q4_0/Q8_0/Q6_K/F16/BF16/F32 dequant |
| `runtime/nn/axiom_geo.h` | ~165 | Metric field, Christoffel, curvature, geodesic, Fisher types |
| `runtime/nn/axiom_geo.c` | ~540 | IDW metric interpolation, Christoffel via finite differences, Ricci contraction, RK4 geodesic integrator |
| `runtime/nn/axiom_beta.h` | ~165 | Per-phase result structs, expanded config |
| `runtime/nn/axiom_beta.c` | ~730 | Complete 5-phase pipeline with real model probes |

### Modified Files

- `runtime/nn/llm.h` — added `llm_get_model()`, `llm_get_embedding_vec()`
- `runtime/nn/llm.c` — accessor implementations
- `host/main.c` — updated CLI for Beta-2 config, new `--axiom-skip-geodesic` flag
- `build_host.ps1` — added axiom_linalg.c, axiom_geo.c to SOURCES

## 3. Five-Phase Pipeline

### Phase 1: Manifold Identification

**Method**: Sample N random token embeddings from the model's embedding matrix (dequantized via `llm_get_embedding_vec()`), compute PCA on the N×dim cloud, then estimate intrinsic dimensionality via the Facco et al. TwoNN estimator.

**Key algorithms**:
- Jacobi iterative eigenvalue decomposition for symmetric matrices
- Economy-mode PCA: uses Gram matrix X·X^T when n < d (avoids d×d covariance)
- TwoNN: sorts pairwise distances, computes μ = r₂/r₁ ratios, estimates ID via MLE

**Outputs**: `axiom_phase1_t`
- `intrinsic_dim` — TwoNN integer estimate
- `twonn_raw` — raw TwoNN estimator value
- `pca_components_kept` — components retained at variance threshold
- `explained_ratio` — fraction of total variance explained
- `embedding_dim`, `samples_used`, `total_variance`

**Example result** (Gemma 4 E2B, 256 samples):
- Intrinsic dim: 41 (TwoNN raw=40.95)
- PCA: 221 components at 95.1% variance

### Phase 2: Symmetry Extraction

**Method**: For each sampled layer, extract attention head Q-weight statistics. In Beta-3, Phase 2 uses **real dequantized Q-weight rows** (not block statistics) for per-head energy fingerprints, then measures pairwise similarity between heads.

**Key insight**: Heads with similar weight distributions are "permutation invariant" — swapping them preserves model behavior. Each such pair corresponds to a generator of the model's symmetry group.

**Outputs**: `axiom_phase2_t`
- `symmetry_score` — mean pairwise head similarity
- `generators_found` — estimated Lie algebra dimension
- `permutation_invariant_heads` — count of near-identical head pairs
- `head_similarity_mean`, `head_similarity_max`, `total_heads_tested`

**Example result**: score=0.81, 80 invariant pairs, 64 generators

### Phase 3: Nonlinearity Absorption (Curvature)

**Method**: Build a metric tensor field in PCA subspace (capped at max 64 dims for O(d³) tractability). At each of N sample points, compute local covariance of nearby embedding projections as the metric tensor. Then:
1. Numerical Christoffel symbols Γ^k_{ij} via finite-difference metric derivatives
2. Ricci tensor R_{ij} via algebraic Γ·Γ contraction
3. Scalar curvature R = g^{ij} R_{ij}

**Key algorithms**:
- IDW metric interpolation (k=8 nearest neighbors, Shepard p=2)
- Gauss-Jordan matrix inversion with 1e-10 regularization
- Christoffel: Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

**Outputs**: `axiom_phase3_t`
- `mean_scalar_curvature`, `max_scalar_curvature`, `min_scalar_curvature`
- `curvature_std` — standard deviation of curvature across sample points
- `high_curvature_loci` — points with |R| > mean + 2σ
- `metric_field_points`, `christoffel_computed`

### Phase 4: Axiom Formalization

**Method**: Generate axiom candidates from geometric features discovered in Phases 1–3. Four axiom types:
- **METRIC** — distance structure from PCA variance ratio
- **SYMMETRY** — head-permutation invariance from similarity score
- **GEODESIC** — curvature constraints on token trajectories
- **BOUNDARY** — embedding space boundary behavior

Active learning loop selects least-confident candidates for oracle testing. Oracle validation: compare embedding pairs via distance/cosine metrics. Bayesian confidence update: posterior = 0.7·prior + 0.3·evidence.

**Outputs**: `axiom_phase4_t`
- `axiom_count` — unique axioms after deduplication
- `consistency_score` — mean confidence of accepted axioms
- `candidates_tested`, `candidates_accepted`, `oracle_calls_used`
- `information_gain` — total information gained from oracle calls

### Phase 5: Geodesic Pilot

**Method**: Proof-of-concept for geodesic inference. For N test token pairs:
1. Project start/end embeddings into PCA subspace (capped at 64 dims)
2. Build local metric field at 32 sample points
3. Compute Christoffel symbols
4. Integrate geodesic equation with RK4: ẍ^k + Γ^k_{ij} ẋ^i ẋ^j = 0
5. Compare geodesic endpoint with target embedding

**Outputs**: `axiom_phase5_t`
- `geodesic_cosine_similarity` — endpoint vs target cosine sim
- `geodesic_reconstruction_error` — L2 distance to target
- `geodesic_path_length` — integrated arc length
- `projected_speedup` — O(n²dL) / O(n·ID²) complexity ratio
- `geodesic_converged` — whether all test geodesics converged

**Example result**: 8/8 converged, speedup=8187x (projected)

## 4. Linear Algebra Library (axiom_linalg)

Zero-dependency C11 linear algebra for the axiomatic pipeline:

- **Dense matrix**: row-major `axmat_t`, create/destroy/multiply/transpose
- **Eigendecomposition**: Jacobi iterative method for symmetric matrices (max 100 sweeps, tol=1e-12)
- **PCA**: handles n≥d (standard covariance) and n<d (economy Gram matrix). Variance thresholding for component selection
- **TwoNN**: Facco et al. intrinsic dimensionality estimator via nearest-neighbor distance ratios
- **Dequantization**: Q4_0, Q8_0, Q6_K, F16, BF16, F32 → f32/f64

## 5. Differential Geometry Engine (axiom_geo)

Custom Riemannian geometry for neural manifold analysis:

- **Metric field**: N sample points with d×d symmetric positive-definite metric tensors. IDW interpolation for querying arbitrary points
- **Christoffel symbols**: Γ^k_{ij} from metric finite differences with off-axis displacement filtering (threshold=0.3)
- **Curvature**: Ricci tensor via **full Riemann computation** (∂Γ finite-difference derivative + Γ·Γ algebraic contraction). Scalar curvature via trace with inverse metric. AVX2 dot product helper (`ott_dot_kk`) for PCA tensor scans.
- **Geodesic integrator**: 4th-order Runge-Kutta with divergence detection (velocity norm > 1e10 or NaN). Supports trajectory recording for path analysis
- **Geodesic length**: numerical integration via midpoint metric evaluation

## 6. CLI

```
--axiom-beta-run          Run 5-phase survey after model load
--axiom-beta-only         Run survey then exit
--axiom-fast              Fast-mode clamps (64 samples, 12 oracle calls max)
--axiom-gpu               Use CUDA for Phase 3/5 matrix operations
--axiom-report <path>     JSON report path (default: axiom_beta_report.json)
--axiom-samples <n>       Embedding samples (default: 256)
--axiom-probe <n>         Phase 5 vocab probe size (default: 1024)
--axiom-seed <n>          Deterministic RNG seed
--axiom-skip-geodesic     Skip Phase 5 geodesic pilot
-v                        Verbose per-phase logging
```

## 7. Example Run

```powershell
.\build_host\geodessical.exe model.gguf --axiom-beta-only --axiom-fast --axiom-gpu -v
```

Output (Gemma 4 E2B, fast mode, April 2026):
```
[AXIOM-BETA-3] Phase 1: ID=14, PCA=... components, 128 ms
[AXIOM-BETA-3] Phase 2: score=0.8149, generators=64, 1 ms
[AXIOM-BETA-3] Phase 3: mean_R=..., high-curv=7, 43 ms (warm cache)
[AXIOM-BETA-3] Phase 4: 49 axioms, consistency=0.8530, oracle_calls=8, 669 ms
[AXIOM-BETA-3] Phase 5: cos_sim=..., L2_err=..., top1=0.000, mrr=0.067, 43 ms
[AXIOM-BETA-3] Complete: 977 ms total
```

## 8. Known Limitations (Beta-3)

1. **Curvature magnitudes**: Scalar curvature values can be very large due to sparse metric field sampling. Need denser sampling or regularized Christoffel computation.
2. **Geodesic reconstruction**: cosine similarity remains low (MRR ≈ 0.067 fast mode) — geodesic in embedding PCA subspace does not yet model the full transformer computation. Layer-wise hidden-state trajectory matching is the next quality gate.
3. **Symmetry probing**: Beta-3 upgraded to real dequantized Q-weight rows. Accurate for all supported quantization types (Q4_0, Q8_0, Q6_K, F16, BF16, F32).
4. **Memory**: Phase 1 PCA with N>1024 samples at dim=1536 requires ~20 MB. Phase 3 Christoffel with d=64 requires ~2 MB. All allocations bounded.
5. **Knowledge injection**: Warp plumbing complete, but training-time policy and expanded manifold recomputation coupling still needed for production use.
6. **Geodesic forward pass**: Phase 5 is a pilot evaluator, not the default generation engine. Geodesic proposals are not yet the runtime decode path.

## 9. Roadmap

### Beta-4: Geodesic Inference Prototype
- First working geodesic forward pass replacing transformer for candidate token generation
- Layer-wise geodesic paths through layer-specific metric fields
- Accuracy parity gates against standard forward pass (cosine sim + top-1 match)
- Promote Phase 5 from evaluator to candidate token proposer in decode loop

### Beta-5: Production Pilot
- Accuracy parity gates against standard forward pass
- Hybrid mode: standard inference with geodesic verification
- O(n·k²) complexity profiling vs transformer O(n²·d·L)
- LRU hidden-state cache (items 21–26 in GEODESSICAL_PLAN.md)

### v1.0: Geodesic Default
- Geodesic inference as the primary Geodessical decode path
- Knowledge injection production API
- Formal diffeomorphism for softmax/LayerNorm/GeLU → curvature absorption
