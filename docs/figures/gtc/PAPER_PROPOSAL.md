# Should GTC Be a Standalone Paper?

Date: 2026-04-27
Verdict: Yes.

## Why this is publishable now

1. Clear technical contribution
- Geodesic trajectory caching is distinct from parameter compression.
- It introduces a new decode-time substitution mechanism with a formal
  Riemannian/Jacobi basis and measurable systems implications.

2. Strong positive empirical anchors
- Three-model scaling invariance for cache coverage near 91% at 25% cache.
- Batch resonance measured up to 60x at B=10000 with numerical-equivalence error.
- Record store efficiency (5.96 KB/record, ~31 us query).
- In-regime correction accuracy at numerical floor with 160-184x vs fresh
  geodesic solves.

3. Honest falsifiable negatives already documented
- Curvature-warp v1 and v2 do not meet redirection success criteria yet.
- Live runtime cloud density is insufficient for high correctable-rate at
  rho=0.4 despite high lookup hit rate.
- Simplex blending in AttnRes prototype currently underperforms single-anchor
  Jacobi correction.

This mix of positives and negatives strengthens scientific credibility.

## Paper framing recommendation

Title direction:
"Geodesic Trajectory Caching for Decode-Time Substitution: Validity Regimes,
Coverage-Density Limits, and Batched Resonance"

Core message:
- The mechanism works very well where geometry says it should.
- Deployment benefit is gated by manifold-sampling density, not by Jacobi
  approximation quality.

## Minimum additions before submission-quality

1. Runtime hook and real trace density study
- Add per-step hidden-state export in geodessical runtime.
- Measure correctable-rate vs sample density and decoding task family.

2. AttnRes integration refinement
- Keep single-anchor correction as baseline.
- Redesign blend policy (confidence gating / local convex hull check) rather
  than naive simplex averaging.

3. Curvature-warp section as negative result
- Keep v2 as "spillover fixed, redirection unresolved".
- Include why this is still useful (constraints for future warp design).

## Novelty beyond existing GRC paper

- GRC: compresses parameters.
- GTC: compresses trajectories and step executions.
- Joint story is compositional but each can stand alone as a contribution.

## Recommendation

Proceed with a dedicated GTC paper, with explicit regime statements and density
scaling as the central deployment theorem.
