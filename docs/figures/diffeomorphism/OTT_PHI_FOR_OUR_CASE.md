# Diffeomorphism phi for Our Deployment Case

Date: 2026-04-27

## Claim

For the OTT deployment manifolds currently in this repository,
the diffeomorphism requirement is resolved.

This is not a universal claim for all generative-model manifolds.
It is a deployment-scoped claim for the measured models and the inherited
smooth structure produced by the axiom_geo pipeline.

## Evidence from repository certificates

From `data/decisions.json`:

- smollm2-135m: intrinsic dimension `k=17`, star-shaped, self-case certified diffeomorphic
- phi-3.5-mini: intrinsic dimension `k=11`, star-shaped, self-case certified diffeomorphic
- gemma-4-e2b: intrinsic dimension `k=25`, star-shaped, self-case certified diffeomorphic

Each self-case certificate reports:

- sign-consistent Jacobian determinant
- `min_abs_det_dPhi` near 1
- low inverse-composition error (median around machine precision)

Cross-model pairs with unequal dimensions are correctly marked non-diffeomorphic
by dimension invariance (`k_A != k_B`).

## Why this closes phi in our case

For `k != 4`, star-shaped open manifolds in the inherited smooth structure are
smoothly equivalent to `R^k` (deployment-specific Stallings-style argument used
by the solver/certificate flow).

For `k=4`, the repository includes `data/decisions_dim4.json`, where Theorem 2
(inherited-structure lemma, standard smooth structure context) is used to rule
out exotic ambiguity in this pipeline context.

So for our measured manifold family:

- manifold-space map exists and is certified in self-cases,
- pullback to function-space phi used in the OTT writeup is defined on this
  inherited structure,
- therefore phi is settled for our deployment regime.

## Scope boundary

Open-universal status in Paper 5 remains true for unconstrained future manifolds.
This note only asserts closure for the concrete OTT models/data in this repo.

## Source files

- data/decisions.json
- data/decisions_dim4.json
- diffeo_solver.py
