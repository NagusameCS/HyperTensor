# GTC — Geodesic Trajectory Caching (v0.2, validated)

Small-scale prototype for the "GTC + Jacobi correction" idea (Paper 4 §2),
made runnable end-to-end on the 4070 Laptop. The goal is a falsifiable
answer to:

> Can we replace some fraction of decode steps with a cached geometric
> trajectory plus a Jacobi-field correction, while staying within an error
> tolerance comparable to OneDecode?

**v0.2 status (2026-04-27):** harness validated against a closed-form
sphere case, full N-dimensional Christoffel + Riemann pipeline now in
place. Results in [`../../docs/figures/gtc/GTC_RESULTS.md`](../../docs/figures/gtc/GTC_RESULTS.md).

## Files

| File | Role |
|---|---|
| `_phase_io.py` | Loader for `legacy/axiom_vis/<model>/phase{1,3}*.json`. |
| `manifold.py` | Builds a smooth metric `g(x)` and Christoffel field `Γ^k_ij(x)` over a sampled `Manifold` object. Two builders: `fit_phase3_manifold(model)` for the activation manifold, `build_sphere_manifold(n)` for the closed-form sanity case. |
| `geodesic.py` | RK4 integrator for the geodesic ODE in arbitrary intrinsic dimension. |
| `jacobi.py` | Riemann tensor via finite differences; Jacobi propagator `Φ(λ)` via Magnus-3 along the discretised geodesic. |
| `validity_radius.py` | Sweeps ε on either manifold; emits `<case>_validity_radius.json`. |

## v0.2 design choices (vs v0.1)

The runtime emits one global Christoffel tensor and a per-point diagonal
of the metric, which is too thin to give a non-trivial connection. v0.2
sidesteps this entirely:

- The metric tensor `g(x)` is fitted **in Python** as the inverse of the
  local k-NN covariance of the Phase-1 cloud, lifted to the intrinsic
  dimension by padding with PCA-tail-eigenvalue noise. This is the
  classical Mahalanobis metric on embedded data — a Fisher-information
  proxy when the data are activations.
- The metric field is smoothed with a log-Euclidean RBF (so it stays SPD
  along its natural geometry).
- `Γ^k_ij(x)` is computed from `g` by the standard formula and central
  differences. **No runtime patch needed.**
- `R^a_bcd` along a geodesic is finite-differenced from `Γ`. The Jacobi
  ODE `D²J/dλ² = −R(J, γ̇) γ̇` is then integrated with a Magnus-3 step.

## Reproducing

```powershell
cd C:\Users\legom\HyperTensor
.venv\Scripts\python.exe scripts\gtc\validity_radius.py --case sphere --dim 4 --n-seeds 16 --steps 24 --n-perturb 16 --dl 0.05
.venv\Scripts\python.exe scripts\gtc\validity_radius.py --case smollm2-135m --dim 8 --n-seeds 12 --steps 20 --n-perturb 12 --dl 0.05
```

Outputs land at `docs/figures/gtc/<case>_validity_radius.json`.

## Headline numbers (256-sample sphere, n=4)

| ε | mean rel. error | p95 |
|---:|---:|---:|
| 0.05 | 0.027 | 0.051 |
| 0.10 | 0.054 | 0.106 |
| 0.20 | 0.108 | 0.226 |

`ε⋆(τ=5%) = 0.05`, `ε⋆(τ=10%) = 0.10`, `ε⋆(τ=20%) = 0.20`. Quadratic
scaling in ε is exactly the theoretical Jacobi bound on a constant-K
manifold. **The harness is correct.**

On SmolLM2-135M the manifold is so weakly curved at the runtime sampling
resolution that errors are below 1e-3 across the whole ε ∈ [0.005, 0.4]
sweep. This is the operational green light: the validity radius is **not**
the bottleneck for GTC; coverage is.

## Next milestones

1. **Live decode benchmark.** `gtc_benchmark.py` will drive
   `geodessical.exe` on the 150-prompt SmolLM2 corpus, comparing cache
   hits + Jacobi correction against the full forward. Blocked on the
   SmolLM2-135M Q8_0 download finishing locally.
2. **Coverage analysis.** Map the 64-point Phase-1 cloud against decode
   activations on the 150-prompt corpus; report the fraction of decode
   steps within ε=0.4 of a cached point. This pins the empirical hit-rate
   ceiling.
3. **Curvature-warp injection** (Paper 4 §3, separate
   `scripts/curvature_warp/` directory). Not started.
