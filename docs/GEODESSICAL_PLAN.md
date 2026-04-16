# Geodessical Development Plan — Organic Training Theory

## Foundation

This plan is derived from **Organic Training Theory (OTT)** by NagusameCS (2026).
The existing Geodessical inference engine (v0.5 "Synapse") remains the primary runtime.
OTT-based geodesic inference is built alongside it as the next-generation system.

---

## Core Thesis

A trained model's weight space is a Riemannian manifold whose intrinsic curvature
encodes all learned knowledge. Inference collapses from O(n²·d·L) to O(n·k²) by
solving the geodesic equation through the manifold in intrinsic coordinates (k ≈ 40),
eliminating explicit attention, nonlinearities, and layered forward passes.

---

## Current State vs OTT Requirements

| OTT Concept | Phase | Current State | Gap |
|---|---|---|---|
| Geometric Survey | 1 | ✅ PCA + TwoNN on embeddings | Need Fisher Information Matrix from output Jacobians |
| Intrinsic Dimensionality | 1 | ✅ TwoNN estimator (k≈41 measured) | Done — refine with denser sampling |
| Symmetry Mining | 2 | 🟡 Block statistics (crude) | Need real weight permutation probing + Lie algebra |
| Christoffel Symbols | 3 | 🟡 Finite differences on covariance metric | Need FIM-based metric, full derivative terms |
| Riemann Tensor | 3 | 🟡 Algebraic Γ·Γ only | Missing ∂Γ derivative terms for full curvature |
| Diffeomorphism φ | 3 | ❌ Not implemented | **Open problem**: absorb nonlinearities into curvature |
| Axiom Formalization | 4 | 🟡 Random generation + EMA | Need active learning, formal grammar, real oracle |
| Geodesic Validation | 5 | 🟡 Synthetic metric | Need hidden-state trajectory comparison |
| Geodesic Forward Pass | — | ❌ Not implemented | Core OTT inference: RK4 solve of geodesic equation |
| Knowledge Injection | — | ❌ Not implemented | Local Christoffel perturbation with Gaussian decay |
| O(n·k²) Runtime | — | ❌ Not implemented | Full replacement of transformer forward pass |

---

## Phased Roadmap

### Milestone 1: Real Geometry Foundation (Beta-3)
**Goal**: Replace all surrogate math with real geometric computation.

1. **Fisher Information Matrix**
   - Wire `tensor_bridge` into forward pass for per-layer hidden state capture
   - Compute FIM: g_ij(x) = E[∂log p(y|x)/∂x^i · ∂log p(y|x)/∂x^j]
   - Use FIM as the Riemannian metric instead of covariance approximation
   - Storage: k×k metric at each sampled point (k ≈ 40, manageable)

2. **Full Riemann Curvature**
   - Add derivative terms ∂_ρ Γ^μ_νσ to curvature computation
   - Implement full Riemann tensor R^μ_νρσ (currently only algebraic part)
   - Compute Ricci tensor R_μν by contraction
   - Compute scalar curvature R = g^μν R_μν

3. **Real Symmetry Mining**
   - Dequantize attention weight heads (not just block byte stats)
   - Apply random weight permutations, measure output deltas
   - Fit Lie algebra generators from near-zero-delta transformations
   - Identify symmetry group G of the manifold

4. **Active Axiom Discovery**
   - Replace random axiom generation with active learning
   - Define formal axiom grammar (metric relations, curvature bounds, symmetries)
   - Use the model itself as oracle (forward pass to validate axioms)
   - Prune inconsistent candidates; derive minimal axiom set A_θ

   Progress (Apr 2026):
   - Phase 4 now mixes geometry evidence with real model-oracle evidence from deterministic token-generation checks.
   - Added bounded oracle budget to keep runtime practical (fast mode currently uses 8 model-oracle calls).
   - JSON now exposes `phase4_axioms.model_oracle_calls` for observability.

### Milestone 2: Geodesic Inference Prototype (Beta-4)
**Goal**: First working geodesic forward pass, validating OTT's core claim.

5. **Geodesic Forward Pass**
   - Implement the geodesic equation solver:
     d²x^μ/dλ² + Γ^μ_νρ(x) · dx^ν/dλ · dx^ρ/dλ = 0
   - Use RK4 integration (already implemented in axiom_geo.c)
   - Initial conditions: x^μ(0) = Embed(input tokens)
   - Map geodesic endpoint back to logit space via linear projection

6. **Initial Velocity Computation**
   - Derive v₀ from attention structure without full attention
   - Christoffel symbols encode how context bends the space
   - This encodes "contextual direction" for the geodesic

   Progress (Apr 2026):
   - Phase-5 now uses a curvature-informed initial velocity prior in `runtime/nn/axiom_beta.c`.
   - It starts from endpoint direction and applies a bounded local acceleration term from interpolated Christoffel symbols.
   - This is a practical bridge toward full context-derived v0 while preserving numerical stability.
   - Added adaptive geodesic retry integration (step/velocity damping) to improve convergence robustness in high-curvature regions.
   - Quick Gemma4 fast run (`--axiom-fast --axiom-probe 128 --axiom-gpu`) remains convergent and improved token MRR from 0.000 to 0.005.

7. **Validation Against Transformer**
   - Compare geodesic trajectory with actual hidden-state trajectory
   - Measure cosine similarity and reconstruction error per layer
   - This is the critical test: if geodesics match hidden states, OTT works

8. **Diffeomorphism Research** (Open Problem)
   - Study how softmax, LayerNorm, GeLU map to curvature regions
   - Find coordinate transform φ: M_θ → N where nonlinearities become curvature
   - This is the hardest unsolved problem — may require iterative approximation

### Beta-4+ Additions (from arXiv:2603.15031 AttnRes)
**Goal**: Improve depth-wise information flow and system efficiency in the geodesic stack.

15. **Depth-wise Softmax Aggregation for Geodesic States**
   - Replace fixed/equal depth accumulation with learned softmax weights over prior manifold states.
   - Use one pseudo-query vector per geodesic layer/state transition to keep parameter overhead minimal.
   - Enforce competitive normalization (softmax, not sigmoid) to avoid depth dilution.

16. **Block Geodesic Aggregation (Scalable Variant)**
   - Partition geodesic depth into blocks and aggregate block representations instead of all prior states.
   - Target complexity/memory reduction from O(Ld) history access to O(Nd) block-history access.
   - Keep a tunable block count (N) to trade quality vs latency.

17. **Two-Phase Runtime Execution**
   - Phase 1 (parallel): batch inter-block attention/aggregation against cached block summaries.
   - Phase 2 (sequential): intra-block updates with online-softmax merge.
   - Integrate online softmax merge to preserve numerical equivalence while reducing I/O.

18. **Key Normalization + Stable Initialization**
   - Apply RMSNorm on depth keys to prevent magnitude-dominated layer selection.
   - Initialize pseudo-query vectors to zero so early training starts as uniform depth averaging.
   - Add guardrails for smooth warm-up and reduced training volatility.

19. **Pipeline and Prefill Infrastructure**
   - Add cross-stage cache reuse for block summaries under pipeline parallelism to remove redundant transfers.
   - Add sequence-sharded block-summary caching for long context prefill.
   - Track per-device memory overhead and communication reduction vs naive transfer.

20. **Depth Dynamics Observability (Required Diagnostics)**
   - Add dashboards/metrics for hidden-state magnitude vs depth and gradient norm vs depth.
   - Verify that depth aggregation mitigates PreNorm-style dilution and flattens gradient imbalance.
   - Record learned depth-attention heatmaps to validate locality + selective long-range retrieval.

### Acceptance Criteria for AttnRes-derived Additions
- Inference overhead target: <= 2% latency increase for block-depth aggregation path.
- Communication target: eliminate redundant cross-stage block transfers via cache reuse.
- Memory target: bounded block-history footprint proportional to O(Nd), not O(Ld).
- Quality target: measurable gain in Phase-5 token metrics (top1/MRR) at equal compute budget.

### Beta-4++ Memory Efficiency (from arXiv:2603.15031 §4 & Appendix)
**Goal**: Eliminate redundant ott_get_hidden_state forward passes and reduce probe pool peak memory,
using concrete infrastructure techniques from the AttnRes paper.

21. **ott_get_hidden_state Result Cache (LRU token_id → hidden state)**
   - Problem: Phase 1 = 31.5s (256 calls ~130ms each), Phase 3 = 248s (~1900 calls), Phase 5 = 65.6s (~1032 calls). Same token IDs are resampled across phases.
   - Fix: In-process LRU cache keyed by (token_id, layer): `float[dim]`. On hit, skip llm_generate_tokens entirely.
   - Size: 2048 entries x 576 x 4B = 4.7MB (negligible relative to model).
   - Expected: 70-90% cache hit rate on Phase 3's ~1900 random vocab samples; Phase 3: 248s -> ~25-75s.
   - Paper basis (SS4): Block AttnRes caches N block representations and reuses them across all L layers. Cross-stage cache eliminates V* redundant inter-stage transfers.

22. **Block-Level Manifold Sampling -- Reduce ott_get_hidden_state Call Count**
   - Problem: Phase 1 samples 256 tokens individually; Phase 3 samples ~1900 individually.
   - Fix: Partition vocab into N=8 blocks by token_id range. Sample S tokens per block; aggregate b_n = mean(h_i for i in block_n). Use N block vectors for PCA / metric field fitting.
   - Reduction: Phase 1: 256 -> N*S = 8*4 = 32 calls (8x). Phase 3: ~1900 -> N*S = 8*32 = 256 calls (7x). Phase 3: 248s -> ~33s at 130ms/call.
   - Memory: O(N*d) = 8*576*4B = 18KB vs O(n_samples*d) = 590KB.
   - Paper basis (SS3): 'N=~8 recovers most of the benefit across model scales.' Block AttnRes reduces memory from O(Ld) to O(Nd).

23. **Online Softmax Merge for Phase 5 Probe Pool Scoring**
   - Problem: cand_mat_f32[n_probe * dim] materializes all 1024 probe hidden states: 1024*576*4B = 2.3MB. GPU scores entire matrix in one shot.
   - Fix: Process probe tokens in chunks of S=32. Maintain running (m_max, l_logsumexp, o_weighted_sum). Merge chunks via Milakov 2018 online softmax:
     m = max(m1, m2); l = exp(m1-m)*l1 + exp(m2-m)*l2; o = (exp(m1-m)*o1 + exp(m2-m)*o2) / l
   - Peak active memory: 32*576*4B = 72KB (32x reduction). GPU allocation drops proportionally.
   - Paper basis (SS4 Algorithm 1 Phase 2): online softmax merge naturally admits kernel fusion with RMSNorm; used for intra-block sequential + inter-block parallel merge.

24. **RMSNorm Normalization of Captured Hidden State Keys**
   - Problem: Last-layer hidden states have widely variable magnitudes across token types; high-magnitude tokens bias Phase 3 covariance structure.
   - Fix: After ott_get_hidden_state captures h, compute k = h / sqrt(mean(h^2) + 1e-6). Use normalized k as the metric field sample point.
   - Expected: Phase 3 max_R reduced organically; Christoffel normalization scale factor approaches 1.0 naturally.
   - Paper basis (SS3 eq. phi(q,k) = exp(q^T * RMSNorm(k))): 'The RMSNorm inside phi prevents layers with large-magnitude outputs from dominating the attention weights.'

25. **Depth-Sink Layer Detection for Optimal Hidden State Capture Layer**
   - Problem: ott_get_hidden_state uses layer=-1 uniformly. Certain layers are depth sinks -- they attract consistently high attention weight regardless of input, giving more stable representations.
   - Fix: On axiom_beta_run init, probe 16 diverse tokens with tensor_bridge across all L layers. For each layer, compute variance of activation L2-norm across the 16 tokens. Layer with minimum variance = depth-sink candidate. Cache as ott_sink_layer; use for all subsequent captures.
   - Paper basis (SS6.1 Discussions): 'Input-dependent M of AttnRes reveals depth-wise attention sinks, where certain layers consistently attract high weight regardless of input -- mirroring sequence-wise attention sinks (Xiao 2023).'

26. **Two-Phase Batch I/O for Hidden State Collection (10x I/O reduction)**
   - Problem: Each ott_get_hidden_state runs a full sequential forward pass. For Phase 3 metric sampling, all target tokens are known upfront -- ideal for batching.
   - Fix (Appendix formula): Partition n_total_samples tokens into N=8 blocks of S tokens each.
     Phase 1 (parallel): batch all S queries against N cached block-KV pairs via single matmul.
     Phase 2 (sequential): walk intra-block sums; merge with Phase 1 via online softmax.
   - I/O per token: (S+N)*d = (32+8)*576 = 23,040 floats vs naive full-scan per call.
   - Paper formula (Appendix A): Read per layer = (S+N-2)*d, Write = 2*d. Total I/O = (S+N)*d.
     Typical (L=54, N=9, S=6): 15*d = 8,640 floats vs naive 55*d = 31,680 floats = 3.7x I/O reduction.
   - Requires: new llm_batch_hidden_states(token_ids[], n, layer, out[][dim]) API in llm.c.

### Axiom Beta Benchmark Snapshot (Apr 14, 2026)

Gemma 4 E2B Q4_0, host-mode geodessical runtime, `--axiom-beta-only`:

- samples=64, probe=256: total=543.2ms, p5=59.3ms, id=14, cos=-0.0289, l2=7.9265, top1=0.0000, mrr=0.0153, supports=1
- samples=128, probe=512: total=1012.8ms, p5=69.4ms, id=16, cos=0.1870, l2=8.0595, top1=0.0000, mrr=0.0000, supports=1
- samples=256, probe=1024: total=3208.9ms, p5=15.4ms, id=41, top1=0.0000, mrr=0.0000, supports=0 (extreme curvature regime)

Notes:
- 64-128 samples currently provide stable geodesic pilot convergence.
- 256 samples improves manifold detail but currently destabilizes geodesic convergence; treat as a robustness target for Beta-4+.

Latest fast/full matrix (Gemma 4 E2B Q4_0, auto model discovery):
- fast + GPU phase5 (`probe=128`): total=2130.3ms, p4=1818.8ms, p5=44.9ms, mrr=0.030, model_oracle_calls=8
- fast + CPU phase5 (`probe=128`, `--axiom-no-gpu`): total=2114.3ms, p4=1807.6ms, p5=43.0ms, mrr=0.030, model_oracle_calls=8
- full + GPU phase5 (`probe=256`): total=6709.4ms, p4=3656.6ms, p5=163.2ms, mrr=0.000, model_oracle_calls=16

Post-tuning update (Apr 14, 2026):
- fast + GPU phase5 (`probe=128`, uncertainty-gated model-oracle budget=4): total=1210.7ms, p4=902.2ms, p5=42.3ms, mrr=0.020, model_oracle_calls=4
- Delta vs previous fast+GPU: total -43.2%, phase4 -50.4% (quality tradeoff: MRR -0.010)
- fast + GPU phase5 (`probe=128`, fast-mode oracle cap=16 + uncertainty early-stop): total=1218.2ms (rerun 1255.8ms), p4=909.3ms (rerun 923.9ms), p5=42.5ms, mrr=0.032, model_oracle_calls=4, oracle_calls=16
- Delta vs uncertainty-gated budget-only run: total +0.6% (best run), phase4 +0.8%, MRR +0.012, oracle_calls_used -75%
- fast + GPU phase5 (`probe=128`, adaptive model-oracle budget 2..4 + fast cap oracle=12, iter=96): total=996.6ms (rerun 977.0ms), p4=687.5ms (rerun 668.7ms), p5=44.8ms, mrr=0.068/0.067, model_oracle_calls=3, oracle_calls=12
- Delta vs previous fast cap=16 run: total -18.6% (best), phase4 -26.5% (best), MRR +0.035 (best)
- Iteration 4: implemented OTT-style local curvature knowledge injection API (`axgeo_apply_local_warp`, `axgeo_apply_local_warp_many`) and Phase-5 controls (`enable_knowledge_injection`, `injection_alpha`, `injection_sigma`, `injection_points`) with JSON observability.
- Injection-enabled sample (`inj=1`, full path): knowledge injection telemetry appears in report; fast path keeps injection off by default (`inj=0`) to protect throughput.
- Iteration 5: added warp accumulation + recalculation trigger plumbing (`enable_recalc_trigger`, `recalc_cross_term_threshold`, `recalc_warp_budget`) and Phase-5 telemetry (`warp_points_accumulated`, `warp_cross_term_estimate`, `recalc_triggered`).
- Iteration 5.1: added persistent warp-state storage (`axiom_warp_state.dat`) and validated full-path runtime trigger firing (`recalc=1`) with persisted accumulation support across runs.
- Iteration 6: recalc trigger now orchestrates in-run axiom refresh (Phase-4 rerun upon trigger). Reduced full-path validation (`samples=64`, `probe=64`) reached `recalc=1` with improved pilot token metrics: top1=0.250, mrr=0.265.
- Iteration 6.1: moved recalc orchestration out of Phase-5 internals into post-Phase-5 full recompute flow (Phase-3 + Phase-4 refresh), and fixed persistent warp-state reset behavior to preserve accumulation across normal runs.
- Iteration 6.2: Phase-5 target selection is now decode-aligned by default (`geodesic_use_oracle_target=1`), using deterministic model next-token oracle targets instead of random token pairs. Added telemetry (`oracle_target_count`, `random_target_count`) and CLI override (`--axiom-random-targets`) for legacy random-target behavior.
- Iteration 6.2 reduced fast validation (`samples=64`, `probe=128`, `--axiom-fast`): total=3841.1ms, p4=1657.7ms, p5=1866.9ms, cos=0.8138, l2=5.2694, top1=0.000, mrr=0.088, targets(o/r)=8/0.

Interpretation:
- Phase 4 oracle validation remains the dominant cost center, but this iteration drops it below 700ms in fast mode.
- GPU acceleration still helps Phase 5 scoring; end-to-end wins now come primarily from tighter Phase 4 policy.
- Knowledge injection + recalculation trigger + persistence are now implemented at prototype level; remaining gap is automatic full manifold recomputation orchestration and geodesic-first generation path.
- Knowledge injection + recalculation trigger + persistence + in-run axiom-refresh orchestration are now implemented; remaining gap is broader manifold recomputation policy and geodesic-first generation path.
- Knowledge injection + recalculation trigger + persistence + post-Phase-5 recompute orchestration are now implemented; remaining primary gap is geodesic-first generation path and broader diffeomorphism/nonlinearity-absorption integration.
- Decode-aligned geodesic pilot is now in place (oracle next-token targets), reducing gap-to-production by validating proposal quality on generation-relevant targets; remaining primary gap is wiring geodesic proposal into default decode execution path.

### smollm2-135m Benchmark Snapshot (Apr 15, 2026)

smollm2-135m-instruct-q8_0.gguf, host-mode geodessical runtime, `--ott-full`, GPU forward pass with CPU fallback for bridge capture:

**First run (cold cache, `ott_hs_disk.dat` absent):**
- Phase 1: 22.5s (256 samples × ~88ms), Phase 3: 197s (2288 block-sampled calls × ~86ms), Total: 306s
- OTT-HS: hits=8269 misses=3516 (70.2%), disk: saved 2532 entries (4.7MB)
- Phase 5: cos_sim=0.2737, L2_err=9.2067, top1=0.125, mrr=0.131, probe=1024

**Second+ run (warm disk cache):**
- Phase 1: 0.78s (-97%), Phase 3: 0.17s (-99.9%), Total: **86s** (-73%)
- OTT-HS: hits=10801 misses=984 (91.7% hit rate)
- Phase 5: cos_sim=0.2737, L2_err=9.2067, top1=0.125, mrr=0.127
- Geometry: layer=9 (depth-sink, mean_cos=0.8207), ID=22, PCA=122, ch_n=88, max_R=648

Key improvements implemented (Apr 15, 2026):
- Todo 21/24: LRU hidden-state cache + RMSNorm key normalization
- Todo 22: Adaptive block-partitioned Phase 3 sampling (k_local=sub_dim+4, min_pts=4×sub_dim=88)
- Todo 23: Streaming S=32 probe pool chunks (peak 73KB vs 2.36MB prior)
- Todo 25: Depth-sink layer detection — selects layer 9 (most diverse, mean_cos=0.8207)
- Todo 26: Disk-persistent HS cache (ott_hs_disk.dat) — Phase 3 cold→warm: 197s→0.17s (-99.9%)
- GPU bridge capture fix (llm.c): CPU fallback when BRIDGE_MODE_CAPTURE active (gpu_fwd bypass)

### Milestone 3: Knowledge Injection (Beta-5)
**Goal**: Implement OTT's training replacement — local curvature warping.

9. **Knowledge Point Injection**
   - Implement the perturbation formula:
     Γ̃^μ_νρ(x) = Γ^μ_νρ(x) + α·Φ^μ_νρ(p)·exp(-d_g(x,p)²/2σ²)
   - Compute geodesic distance d_g(x, p) from RK4 integrator
   - Φ encodes directional structure in tangent space T_p M_θ
   - σ controls warp radius, α controls injection strength

10. **Multi-Point Superposition**
    - Implement additive superposition: Γ̃ = Γ + Σ ΔΓ_i
    - Track accumulated perturbation count N
    - Detect cross-term threshold where axiom recalculation is needed

11. **Axiom Recalculation**
    - Trigger when accumulated warps cause significant cross-terms
    - Recompute Riemann tensor from accumulated perturbations
    - Derive new minimal axiom set
    - Cost: O(k⁴) ≈ 2.56 × 10⁶ at k=40

### Milestone 4: Production Geodesic Inference (v1.0)
**Goal**: O(n·k²) inference as the default mode.

12. **Full Pipeline Integration**
    ```
    Human text in
    → Token IDs (boundary only)
    → Embed → manifold coordinates
    → Geodesic solve O(n·k²)
    → Project → logit tensor
    → Human text out (boundary only)
    ```

13. **Performance Targets**
    - Speedup = n·L/d (linear with context length)
    - At 32k tokens, 70B model: theoretical 320× over standard transformer
    - At 4k tokens, 4B model: theoretical ~40× over current Geodessical

14. **Topological Boundary Handling**
    - Token embedding spaces violate manifold hypothesis (discrete singularities)
    - Need robust boundary map at input/output discontinuities
    - Text appears ONLY at the two human-facing boundaries

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Geodessical                     │
│                                                  │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Standard Engine  │  │  Geodesic Engine     │  │
│  │  (v0.5 Synapse)  │  │  (OTT-based)         │  │
│  │                  │  │                      │  │
│  │  GGUF → Layers   │  │  GGUF → Manifold     │  │
│  │  → Attention     │  │  → Christoffel       │  │
│  │  → FFN           │  │  → Geodesic solve    │  │
│  │  → Logits        │  │  → Logits            │  │
│  │                  │  │                      │  │
│  │  O(n²·d·L)      │  │  O(n·k²)             │  │
│  └──────────────────┘  └──────────────────────┘  │
│           ↑                      ↑               │
│           │    --mode standard   │ --mode geodesic│
│           └──────────┬───────────┘               │
│                      │                           │
│              ┌───────┴────────┐                  │
│              │  Shared Core   │                  │
│              │  GGUF parser   │                  │
│              │  Tokenizer     │                  │
│              │  HAL / SMP     │                  │
│              │  Backend vtable│                  │
│              └────────────────┘                  │
└─────────────────────────────────────────────────┘
```

- `--mode standard` (default): Current transformer inference engine
- `--mode geodesic`: OTT-based geodesic inference (when ready)
- Both share GGUF loading, tokenizer, HAL, threading

---

## Key Mathematical Objects (from OTT)

| Object | Formula | Role |
|---|---|---|
| Metric tensor | g_ij(x) ≈ FIM of p_θ(y\|x) | Measures distance at point x |
| Christoffel symbols | Γ^μ_νρ = ½g^μσ(∂_ν g_σρ + ∂_ρ g_σν − ∂_σ g_νρ) | Encodes curvature |
| Geodesic equation | d²x^μ/dλ² + Γ^μ_νρ dx^ν/dλ dx^ρ/dλ = 0 | Forward pass |
| Riemann tensor | R^μ_νρσ = ∂_ρ Γ^μ_νσ − ∂_σ Γ^μ_νρ + Γ^μ_ρλ Γ^λ_νσ − Γ^μ_σλ Γ^λ_νρ | Full curvature |
| Knowledge warp | ΔΓ^μ_νρ(x,p) = α·Φ^μ_νρ(p)·exp(−d_g(x,p)²/2σ²) | Training |
| Intrinsic dim | k ≈ 30–50 (measured k≈41 on Gemma 4B) | Computational tractability |

---

## Open Problems (from theory)

1. **Diffeomorphism construction**: The map φ absorbing softmax/LayerNorm/GeLU into curvature. Hardest unsolved problem.
2. **Geodesic initial velocity**: Computing v₀ without full attention. Theoretically motivated but not algorithmically specified.
3. **Multi-point interaction threshold**: The N at which cross-terms require full axiom recalculation. Needs empirical calibration.
4. **Topological boundary**: Token embedding spaces are discrete; manifold hypothesis breaks at input/output boundaries.

---

## Files Structure

```
runtime/nn/
  axiom_linalg.h/c    — Linear algebra (PCA, eigendecomp, TwoNN)
  axiom_geo.h/c        — Differential geometry (metric, Christoffel, curvature, geodesic)
  axiom_beta.h/c       — 5-phase discovery pipeline
  geodesic_engine.h/c  — [NEW] Geodesic inference engine
  fisher_metric.h/c    — [NEW] Fisher Information Matrix computation
  knowledge_warp.h/c   — [NEW] Knowledge point injection
```

---

## Design Principles

1. **Theory-first**: Every implementation must trace back to OTT equations
2. **Validate against transformer**: Geodesic trajectories must match hidden states
3. **Incremental**: Standard engine stays default until geodesic is validated
4. **k-dimensional**: All geometry operates in intrinsic subspace (k ≈ 40)
5. **Zero external deps**: Pure C, builds with zig cc
