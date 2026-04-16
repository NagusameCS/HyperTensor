# Axiom Beta Internal Design

## Purpose

This document explains the current Axiom Beta implementation in geodessical, the main performance/quality tradeoffs, and the distance to the ideal Organic Training Theory (OTT) model target.

## Scope

- Runtime implementation: [runtime/nn/axiom_beta.c](runtime/nn/axiom_beta.c)
- Supporting math/geometry: [runtime/nn/axiom_geo.c](runtime/nn/axiom_geo.c), [runtime/nn/axiom_linalg.c](runtime/nn/axiom_linalg.c)
- Plan and benchmark log: [docs/GEODESSICAL_PLAN.md](docs/GEODESSICAL_PLAN.md)

## Pipeline Overview

Axiom Beta is a five-phase autonomous survey pipeline:

1. Phase 1 (manifold identification)
- Samples real embeddings, runs PCA, estimates intrinsic dimension with TwoNN.
- Outputs ID and PCA reduction used by later phases.

2. Phase 2 (symmetry extraction)
- Dequantizes head weight structure and estimates permutation-style invariance.
- Produces symmetry score and generator count proxy.

3. Phase 3 (curvature field)
- Builds Fisher-blended local metric field and Christoffel symbols.
- Computes curvature statistics and caches geometry for reuse.

4. Phase 4 (axiom formalization)
- Generates geometry-conditioned candidates.
- Uses uncertainty-driven active learning with bounded model-oracle calls.
- Dominant runtime cost in fast mode.

5. Phase 5 (geodesic pilot)
- Integrates geodesics in intrinsic subspace.
- Scores geodesic endpoint against token embedding probe set.
- Uses decode-aligned oracle targets by default (`geodesic_use_oracle_target=1`),
	where target token comes from deterministic model next-token generation.
- Reports cosine/L2/top1/MRR and projected complexity speedup.
- Can now apply OTT-style local curvature warps (knowledge injection prototype)
	to Christoffel symbols before geodesic evaluation.
- Includes warp accumulation and threshold-based recalculation trigger telemetry
	(`warp_points_accumulated`, `warp_cross_term_estimate`, `recalc_triggered`).
- Includes persistent warp-state storage (`axiom_warp_state.dat`) so accumulation
	and recalc triggers survive process restarts.
- Recalc orchestration now runs in post-Phase-5 control flow with full
	Phase-3 + Phase-4 refresh, avoiding Phase-5 internal coupling.
- Emits target-source telemetry (`oracle_target_count`, `random_target_count`)
	so decode-aligned vs legacy random-target pilot behavior is measurable.

## Fast-Mode Design (Current)

Current fast-mode policy in [runtime/nn/axiom_beta.c](runtime/nn/axiom_beta.c):

- Workload clamps:
- `embedding_samples <= 64`
- `symmetry_trials <= 128`
- `metric_sample_points <= 64`
- `active_iterations <= 96`
- `oracle_calls_max <= 12`
- `geodesic_test_tokens <= 8`
- `geodesic_vocab_probe <= 512`
- `enable_knowledge_injection = 0` (kept off on fast path by default)

- Phase 4 policy:
- uncertainty-based candidate selection
- early stop after sustained low uncertainty
- adaptive model-oracle budget in fast mode (2..4)
- stricter fast-mode uncertainty floor for model-oracle trigger

- Knowledge injection controls (full path):
- `enable_knowledge_injection`
- `injection_alpha`
- `injection_sigma`
- `injection_points`

Rationale:
- Keep geometric signal quality while aggressively controlling model-oracle overhead.
- Preserve or improve MRR while reducing phase4 wall time.

## Latest Measured Behavior

From latest reports and benchmark history in [docs/GEODESSICAL_PLAN.md](docs/GEODESSICAL_PLAN.md):

- Best recent fast run:
- total about 977 ms
- phase4 about 669 ms
- phase5 about 43 ms
- MRR about 0.067

- Previous fast policy baseline (cap=16):
- total about 1218 ms
- phase4 about 909 ms
- MRR about 0.032

Net effect of latest iteration:
- meaningful runtime drop
- meaningful MRR increase
- phase4 still dominant but significantly reduced

## How Close We Are To Ideal OTT Model

Reference target is the OTT theory manuscript (Downloads path provided by user), especially these goals:

1. Replace transformer forward with geodesic forward pass (not just pilot)
2. Encode nonlinearities via diffeomorphism into curvature
3. Inject new knowledge via local curvature warp with geodesic distance decay
4. Reach true runtime scaling near O(n*k^2) in production path

Current readiness estimate (engineering, not theorem-proof):

- Geometry foundation (metric/christoffel/curvature): 70%
- Axiom discovery loop (active + model oracle): 65%
- Geodesic inference replacement path: 30%
- Geodesic inference replacement path: 35%
- Knowledge injection by local curvature warp: 55%
- End-to-end OTT production replacement: 25%

Overall closeness to ideal OTT model: about 70%

Why not higher yet:
- Phase 5 is still a pilot evaluator, not the default generation engine.
- Decode-aligned targets are now used in pilot mode, but geodesic proposals
	are still not the runtime default in the main generation loop.
- Diffeomorphism that absorbs softmax/layernorm/activation into curvature is not implemented.
- Knowledge injection now has accumulation + trigger plumbing, but still needs
	expanded full manifold recomputation workflow coupling and training-time policy.
- Current complexity wins are still largely projected, not production decode-path replacement.

## Next Engineering Moves

1. Promote Phase 5 from evaluator to candidate token proposer in decode loop.
2. Add local curvature-warp injection API in `axiom_geo` and integrate with phase3 cache.
3. Add geodesic-vs-transformer trajectory matching metrics layer-by-layer.
4. Add persistent multi-point warp superposition with threshold-triggered axiom recalculation.
5. Add benchmark harness for median/p95 phase timings across 5-10 runs.

## Risks

- Over-aggressive phase4 pruning can inflate apparent speed at the cost of brittle axioms.
- Geodesic scoring quality is sensitive to probe-set composition and endpoint reconstruction.
- Curvature estimation noise can destabilize policy decisions without robust confidence bounds.

## Operational Note

Axiom Beta is currently strongest as a high-fidelity geometric survey and planning subsystem.
The major milestone to align with OTT is making geodesic inference the primary token-generation path, then layering local curvature knowledge injection on top.
