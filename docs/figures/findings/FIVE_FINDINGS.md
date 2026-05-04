# Five Findings: Source Notes for Paper Revisions

This document is the canonical source-of-record for five measured but un-published
findings as of 2026-04-22. Each is intended to be lifted into a specific paper
revision; the target paper is named for each.

---

## Finding 1 --- Phase 3 Warm-Cache: 197 s -> 0.17 s

Target paper: `04-organic-training-theory.html` (Axiom Beta-3 / Phase 3 section).

Claim. A token-keyed LRU cache over hidden states reduces SmolLM2-135M Phase 3
manifold recomputation from 197 s cold to 0.17 s warm, a 99.9 % wall-time
reduction. This makes full Phase 3 + Phase 4 refresh routinely affordable inside
a training step.

How. Cache keyed on `(token_id, layer_index)`; invalidated on weight update
that touches the corresponding layer. Cold path is unchanged.

Why it matters. Removes the dominant cost in the OTT loop. Previously Phase 3
was the bottleneck preventing per-step manifold refresh; now it is below the cost
of the Phase 4 active-learning loop.

Reproduce. SmolLM2-135M, Phase 3 cold vs warm; cf.
`docs/figures/figure_03_phase3_speedup.svg`.

---

## Finding 2 --- Phase 4 Oracle-Budget Squeeze: 909 ms -> 669 ms (−26 %)

Target paper: `04-organic-training-theory.html` (Phase 4 section).

Claim. Adaptive oracle budget in `--axiom-fast` mode (2--4 model--oracle calls,
down from 16) plus an uncertainty-based early-stop reduces Phase 4 wall-clock by
26 % without measurable MRR loss.

Mechanism. Run-length-encoded sustained-low-uncertainty heuristic gates the
oracle trigger; once the active-learner uncertainty stays below threshold for `n`
consecutive candidates, Phase 4 exits.

Reproduce. Gemma 4 E2B, axiom-fast, before/after configs in
`scripts/benchmark_whitepaper_finalize.ps1`; values in `CHANGELOG.md` v0.6.0.

---

## Finding 3 --- Decode-Aligned MRR: 0.032 -> 0.067 (×2.1)

Target paper: `02-geodesic-projection.html` (Phase 5 evaluation) and
`04-organic-training-theory.html` (Beta-3 results).

Claim. Replacing Phase 5's random next-token oracle with the model's actual
deterministic next-token (greedy decode) doubles MRR from 0.032 to 0.067. This
is not a method change to the geodesic itself --- it is a measurement-protocol
correction. Random targets undercount the manifold's true alignment with decode.

Why it matters. It establishes that geodesic-projection retrieval is closer to
decode-relevant than the random-target evaluation suggested. The headline bound
on geodesic-pilot quality should be the decode-aligned number.

Caveat. MRR 0.067 is still small in absolute terms; the result is "the metric
was wrong by ×2", not "the method is now production-quality".

---

## Finding 4 --- Persistent Warp State (`axiom_warp_state.dat`)

Target paper: `04-organic-training-theory.html` (new sub-section on persistent
knowledge injection).

Claim. Knowledge-injection warp accumulations are now persisted to
`axiom_warp_state.dat` between sessions. Threshold-triggered manifold recomputation
runs as a post-Phase-5 control-flow step (no Phase 5 timing coupling). Warp points
accumulate across runs.

Mechanism. Per-session warp tuples `(centre, direction, strength, sigma)` are
appended to a binary store. On startup, the manifold is reconstructed by replaying
warps in order. A counter triggers full Christoffel recomputation every `N` warps.

Status. Plumbing complete; controlled by `enable_knowledge_injection`,
`injection_alpha`, `injection_sigma`, `injection_points`. Training-time coupling
pending --- for now warps are externally seeded.

Cross-reference. Beta-3 controlled experiment is the curvature-warp prototype
under `scripts/curvature_warp/`; see `Finding 5` and the GTC results note.

---

## Finding 5 --- Archived `WHITEPAPER_DIFFEOMORPHISM.md` (Stallings + Inherited-Structure Lemma)

Target paper: Sidebar / Appendix in `04-organic-training-theory.html`.

Claim. A short note (Stallings' theorem on PL-homeomorphism between high-dim
manifolds + an "Inherited-Structure Lemma") was drafted and archived. It is not
a solution to Paper 4's open ϕ-construction problem and should not be cited as
one. The lemma supplies a transfer principle for inherited structure across
diffeomorphic manifolds, but the actual learnable diffeomorphism ϕ used in OTT
remains an open construction.

Why archive instead of publish. The framing risked being read as "OTT solves
the diffeomorphism problem", which it does not. The honest position is:
ϕ is hand-engineered per-architecture, OTT defines the target of ϕ but not its
construction.

Future direction. Re-introduce the lemma alongside an explicit statement of
the open problem and a concrete proposal for ϕ on transformers.

---

## Companion Quantitative Findings (this turn)

These are not part of the original five but are documented here so the next paper
revision has all numbers in one place.

### GTC Coverage on SmolLM2-135M
- k=16 cached samples (25 % of cloud) -> 91.0 % hit rate at ε=3.0 (g-norm).
- k=6 -> 58.6 %, k=32 -> 99.8 %, k=48 -> 100 %.
- Validity radius: errors < 0.1 % out to ε=5.0.
- Sphere sanity (n=4): ε⋆(τ=5 %)=0.05, quadratic scaling matches Jacobi bound.
- Source: `docs/figures/gtc/GTC_RESULTS.md`.

### Curvature-Warp Prototype: Falsifiable Negative
- 32-config sweep over (strength, sigma, dl).
- 0/32 configurations meet the success criterion (≥ 50 % reduction in target
  geodesic endpoint error AND ≤ 5 % spillover at non-target points).
- Best improvement 16 % at (strength=0.99, sigma=1.20, dl=0.10), but spillover
  diverges (NaN at high strength).
- Interpretation: simple Gaussian metric-warp `g'(x) = g(x) − α(1−exp(−|x−c|²/2σ²))wwᵀ`
  is too blunt for SmolLM2's near-flat manifold. Either too weak to redirect the
  geodesic, or strong enough to redirect but with global metric perturbation.
- Source: `docs/figures/curvature_warp/smollm2-135m_sweep.json`.

### Llama-3.1-8B PPL Sweep (this turn)
- baseline 6.7902, k=1024 -> 10.9585 (+61.4 %), k=1536 -> 7.6936 (+13.30 %),
  k=2048 -> 7.6936 (identical to k=1536).
- Mechanistic explanation for k=1536 ≡ k=2048: GQA K/V dim is 1024, so once
  k ≥ 1024, K/V are full-rank; Q's PCA energy saturates by k=1536. k=1536 is
  the Pareto rank for `--axex-attn-only` on Llama-3.1-8B.
- Source: `docs/figures/ppl_sweep/README.md`.
