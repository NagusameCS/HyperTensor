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

### Axiom Beta Benchmark Snapshot (Apr 14, 2026)

Gemma 4 E2B Q4_0, host-mode geodessical runtime, `--axiom-beta-only`:

- samples=64, probe=256: total=543.2ms, p5=59.3ms, id=14, cos=-0.0289, l2=7.9265, top1=0.0000, mrr=0.0153, supports=1
- samples=128, probe=512: total=1012.8ms, p5=69.4ms, id=16, cos=0.1870, l2=8.0595, top1=0.0000, mrr=0.0000, supports=1
- samples=256, probe=1024: total=3208.9ms, p5=15.4ms, id=41, top1=0.0000, mrr=0.0000, supports=0 (extreme curvature regime)

Notes:
- 64-128 samples currently provide stable geodesic pilot convergence.
- 256 samples improves manifold detail but currently destabilizes geodesic convergence; treat as a robustness target for Beta-4+.

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
