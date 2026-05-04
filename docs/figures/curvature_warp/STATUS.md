# Curvature-Warp Cross-Model --- STATUS

Date: 2026-04-29
Driver: `scripts/curvature_warp/cross_model.py`
Inputs: `legacy/axiom_vis/<model>/{phase1_manifold,phase3_curvature}.json` for
`gemma-4-e2b`, `phi-3.5-mini`, `smollm2-135m` (all cached locally).
Output: `docs/figures/curvature_warp/cross_model_summary.json`,
`docs/figures/curvature_warp/cross_model_log.txt`
Wall: 146.3 s on local CPU (numpy single-thread).

## Setup

Twelve configurations: 3 models × 2 intrinsic-dim values (n_intrinsic ∈ {6, 8}) ×
2 protocol variants:

- v1 Christoffel-warp: Gaussian "pull" along the gradient of an intrinsic
  coordinate (strength=0.7, sigma=1.2) over T=16 integration steps with dl=0.1.
- v2 Covariant edit: Riemannian-exponential push of magnitude alpha=0.35 in
  a radius-1.2 ball, parallel-transported by the cached Christoffel symbols.

Success criteria: post-warp error on the targeted fact reduced by ≥ 50 % AND
mean spill-over to non-targeted facts < 5 %.

## Result --- negative

0 / 12 configurations satisfy both criteria. v1 best case (gemma, n=6) gets
39.4 % error reduction with 60 % p95 spill-over --- neither criterion met. v2
mostly produces negative improvements (i.e. error grows after edit) on phi
and gemma; only one configuration meets the spill-over criterion alone
(gemma n=8 v2: 1.8 % error growth, 14 % p95 spill).

phi-3.5-mini v1 n=6 produces NaN post-error: the warp drives the geodesic
solver out of its convergence basin. The negative result is robust.

| model | variant | n_intrinsic | improvement | spill mean | passes |
|---|---|---:|---:|---:|---|
| gemma-4-e2b | v1 | 6 | +39.4 % | 0.20 | no |
| gemma-4-e2b | v2 | 6 | −57.4 % | 0.05 | no |
| gemma-4-e2b | v1 | 8 | + 5.0 % | 0.08 | no |
| gemma-4-e2b | v2 | 8 |  −1.8 % | 0.03 | spill only |
| phi-3.5-mini | v1 | 6 | NaN     | NaN  | no |
| phi-3.5-mini | v2 | 6 |  −9.4 % | 0.16 | no |
| phi-3.5-mini | v1 | 8 | −23.9 % | NaN  | no |
| phi-3.5-mini | v2 | 8 | −23.7 % | 0.42 | no |
| smollm2-135m | v1 | 6 | +21.6 % | NaN  | no |
| smollm2-135m | v2 | 6 |  ...    | ...  | no |
| smollm2-135m | v1 | 8 |  ...    | ...  | no |
| smollm2-135m | v2 | 8 |  ...    | ...  | no |

(See `cross_model_summary.json` for the full numeric table.)

## Implications for Paper-D §X (curvature-warp)

The previous draft cited a single negative result on smollm2-135m. This
cross-model run strengthens the negative conclusion: neither protocol
generalises, and v2 is consistently worse than the trivial baseline on at
least one model (phi). The "next revision" placeholder in §X can be replaced
with a definite negative.

## Notes

- This experiment was originally tagged "EC2" but is pure Python+numpy on
  cached JSON manifolds --- no GPU required. Ran locally to save EC2 cost.
- Manifolds were captured by previous `axiom-beta` runs and stored under
  `legacy/axiom_vis/`.
