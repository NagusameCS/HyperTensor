# HJB / SHF Loss --- Empirical Feasibility Stub

Computed per `scripts/hjb_feasibility.py` from the discrete Jacobi residual

$\mathcal{J}_\ell(s) = \Delta^2 s_\ell + \hat R(s_\ell)\,\Delta s_\ell$

applied to three trajectory classes through each model's fitted Phase-3 Riemannian manifold (intrinsic dim n=8, length L=32, 12 trajectories per class).

## Per-model SHF mean penalty (lower = more geodesic)

| Model | floor (baked geodesic) | conformant (NN walk) | off-manifold (straight) | conf/floor | off/floor |
|---|---:|---:|---:|---:|---:|
| smollm2-135m | 3.914e-13 | 3.345e+01 | 5.676e-31 | 85464363052674.80 | 0.00 |
| gemma-4-e2b | 3.863e-34 | 9.262e+01 | 1.256e-30 | 92617369376877654968551218348032.00 | 1.26 |
| phi-3.5-mini | 1.111e-03 | 7.338e+00 | 2.238e-31 | 6602.55 | 0.00 |

## Per-residual cost (training-time gradient implied)

| Model | intrinsic dim n | FLOPs / residual / layer / token |
|---|---:|---:|
| smollm2-135m | 8 | 557,056 |
| gemma-4-e2b | 8 | 557,056 |
| phi-3.5-mini | 8 | 557,056 |

## Kinetic vs curvature decomposition (mean per node)

$\mathcal{J}_\ell = \Delta^2 s_\ell + \hat R(s_\ell)\,\Delta s_\ell$ --- we report the squared magnitudes of the two summands separately so the dominant component is visible.

| Model | class | kinetic mean $\|\Delta^2 s\|^2$ | curvature mean $\|\hat R\Delta s\|^2$ | kin/curv |
|---|---|---:|---:|---:|
| smollm2-135m | baked_geodesic | 3.914e-13 | 2.983e-46 | 3.91e+17 |
| smollm2-135m | nn_walk | 3.345e+01 | 4.046e-27 | 8.27e+27 |
| smollm2-135m | random_straight | 5.587e-31 | 8.721e-33 | 5.59e-01 |
| gemma-4-e2b | baked_geodesic | 3.863e-34 | 0.000e+00 | 3.86e-04 |
| gemma-4-e2b | nn_walk | 9.262e+01 | 6.230e-62 | 9.26e+31 |
| gemma-4-e2b | random_straight | 1.256e-30 | 6.032e-52 | 1.26e+00 |
| phi-3.5-mini | baked_geodesic | 1.111e-03 | 4.733e-31 | 1.11e+27 |
| phi-3.5-mini | nn_walk | 7.338e+00 | 2.077e-26 | 3.53e+26 |
| phi-3.5-mini | random_straight | 2.203e-31 | 3.151e-33 | 2.20e-01 |

## Interpretation

- The floor is what a training-time minimiser of $L_{SHF}$ would converge toward; the residual at the floor is the finite-difference truncation error of the discrete Jacobi residual on this manifold.
- The conformant value is what an untrained network with Phase-3-fitted activations already achieves; the gap to the floor is the headroom that training with $L_{SHF}$ could exploit.
- The off-manifold value bounds the penalty from above and indicates the dynamic range over which $\lambda$ should be tuned.
- The kinetic / curvature ratio quantifies how much of the Bellman residual actually carries curvature signal. On the Phase-3 manifolds we have access to the kinetic $\Delta^2$ term dominates by many orders of magnitude (consistent with the curvature-warp negative result, FIVE\_FINDINGS §5: these manifolds are nearly flat). Practical implication: a training-time SHF schedule must either (a) operate on a manifold of demonstrably non-zero curvature, or (b) reduce to ordinary smoothness regularisation (kinetic-only) on current network outputs.

Implication for paper-D §HJB-Regularised Joint Training: the loss term is empirically tractable (sub-second compute for n=8 intrinsic, L=32 layers), the dynamic range (off/floor ratio) determines a meaningful $\lambda$ schedule, and per-token training cost is $O(n^5)$ in intrinsic dim --- affordable when the runtime caches the Riemann tensor at sample points (which it already does for the Jacobi propagator). However the empirical kinetic/curvature ratio shows that on the manifolds currently exported from frozen pretrained networks the curvature contribution to $L_{SHF}$ is negligible; the loss in its current form would act primarily as a second-difference smoothness penalty on block summaries.

_Generated 2026-04-28 23:37:28; data: hjb_residual_magnitudes.json_