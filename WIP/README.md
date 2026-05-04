# WIP — Millennium Problems (Volume 2)

Status: Work in progress. Computational architectures under development.

## Papers

| Paper | Problem | Method | Status |
|-------|---------|--------|--------|
| XIX | BSD Conjecture | Z_2 + SVD: iota(s)=2-s, fixed point s=1 | Architecture validated (5/5 tests) |
| XX | Generalized RH | Z_2 + SVD (identical to RH structure) | Architecture validated |
| XXI | Yang-Mills Mass Gap | Spectral gap via gauge correlation | Architecture validated (2/2 tests) |
| XXII | Navier-Stokes Regularity | Spectral divergence tracking | Architecture validated (2/2 tests) |

## Directory

- scripts/ — Computational architecture scripts
- benchmarks/ — Verification outputs
- docs/ — Paper drafts

## Note

These are computational proof architectures, NOT peer-reviewed mathematical proofs.
The Riemann Hypothesis (Papers XVI-XVIII) is the most complete architecture and is
documented in the main repository (docs/RIEMANN_PROOF.md, docs/HANDOFF_TO_PHD.md).

The BSD, YM, and NS architectures demonstrate the transferability of the HyperTensor
geometric framework but require substantial further development before they reach
the level of completeness achieved for the Riemann Hypothesis.
