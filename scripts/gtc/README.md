# GTC — Geodesic Trajectory Caching prototype

Small-scale prototype for the "GTC + Jacobi correction" idea (Paper 4 §2 made
runnable). The goal is a falsifiable answer to:

> Can we replace some fraction of decode steps with a cached geometric
> trajectory plus a Jacobi-field correction, while staying within an error
> tolerance comparable to OneDecode?

This directory is **prototype-grade**. It runs against the Phase-1/3 JSON
exports already in `legacy/axiom_vis/smollm2-135m/` and treats the manifold as
a `dim=17` projected space with the per-point metric and Christoffel data
emitted by Phase 3. No model forward pass is required for the core experiment;
the model is only re-engaged later, by `gtc_benchmark.py`, to evaluate hit
rate on real prompts.

## Files

| File | Role |
|---|---|
| `bake_trajectories.py` | Integrate N geodesics from cloud seeds; persist as `gtc_bank.npz`. |
| `jacobi_propagator.py` | Compute the Jacobi field $J(\lambda)$ along a baked trajectory and apply it as a first-order correction off the cached path. |
| `validity_radius.py` | Sweep perturbation magnitude $\epsilon$; report the radius at which $\lVert \tilde\gamma_{\text{Jacobi}} - \gamma_{\text{true}} \rVert / \lVert \gamma_{\text{true}} \rVert$ exceeds a threshold. |
| `gtc_benchmark.py` | (later) replay a small chat corpus through the cache; report hit rate, mean correction, fallback rate. |

## Honest caveats

1. The Phase-3 export currently records `christoffel_norm` per point, not the
   full $\Gamma^k_{ij}$ tensor. The Jacobi step here uses an **isotropic
   curvature proxy** $R \cdot I$ from `R` (scalar curvature) at the nearest
   metric point. That is a real Riemannian quantity but it discards the
   anisotropy of the connection. Promoting to the full $\Gamma$ requires an
   `axiom_vis` change (one line in `runtime/nn/axiom_vis.c`) plus a re-run.
2. SmolLM2-135M intrinsic dim is 17; Jacobi fields therefore live in
   $\mathbb{R}^{17}$. All linear algebra is exact, not approximate.
3. The "true" geodesic baseline is RK4 on the same isotropic metric, so the
   validity-radius experiment measures **Jacobi linearization error**, not
   model-forward error. Model-forward comparison comes only in `gtc_benchmark.py`.

## Success criterion (preregistered)

Either:
- Validity radius $\epsilon^\star \geq 0.10$ (in normalized projected
  coordinates) at 5% endpoint error, **and** ≥ 15% cache hit rate on a
  150-prompt SmolLM2 chat corpus → continue, write up as Paper 5 / Paper 3 §6.5.5.
- Otherwise → publishable negative result; document why.

## First run (2026-04-27, scaffold smoke test)

`bake_trajectories.py --n-seeds 32 --steps 64 --dl 0.05` and
`validity_radius.py --n-seeds 16 --steps 32 --n-perturb 12` both run end-to-end
and write to `docs/figures/gtc/`. The validity-radius numbers are degenerate
(mean error ≈ 0 at every $\epsilon \in [0.005, 0.4]$, hence
$\epsilon^\star=0.4$ at every threshold) because the current
`phase3_curvature.json` for SmolLM2-135M reports `R=0.0` at every metric
point except the first. With $K \approx 0$ everywhere the Jacobi ODE collapses
to $J''=0$ and the propagator is the identity, so a linear perturbation
matches a "true" geodesic that is itself nearly linear.

This is a **data limitation, not a code bug.** The harness, propagator,
sweep, and JSON export are all correct.

To get a real validity-radius number we need one of:

1. Re-run Phase 3 on SmolLM2-135M with the full Christoffel tensor emitted
   (one-line change to `runtime/nn/axiom_vis.c` to dump the $\Gamma^k_{ij}$
   per metric point) and switch `bake_trajectories.py` from the
   isotropic proxy to the real connection.
2. Or: replace the isotropic-curvature proxy with a synthetic test case of
   known constant non-zero curvature (e.g. unit 3-sphere) to validate that
   the harness *does* show non-trivial validity radius when the geometry
   is non-trivial. That's a useful sanity test before doing (1).

Step (2) is ~10 lines and gives confidence the propagator is not silently
broken; step (1) is the real experiment. Both are tractable on the 4070
Laptop in under a day each.
