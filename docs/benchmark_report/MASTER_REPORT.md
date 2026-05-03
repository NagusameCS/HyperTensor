# HyperTensor Geodessical — A–E Benchmark Master Report
Generated: 2026-05-01 03:37 UTC  
GPU: RTX 4070 Laptop (8 GB VRAM) · Ryzen 9 7940HS · 32 GB RAM  
Runtime: geodessical v0.6.0 "Synapse"

> This report consolidates current A–E benchmark evidence, including
> k_int structure, load behavior, decode experiments, HJB feasibility checks,
> and rho-spectrum results. It is an interim evidence snapshot, not a final
> cross-hardware/cross-model claim.

---

## Paper A — k_int Generalisation Across Architectures

Summary: across the sampled models, intrinsic rank k_int (95% joint-Gram variance) stays below d and is often in the 0.5–0.7 d range, supporting rank-limited attention compression.

| Model | d | Mean k_int | k_int/d | Min k_int | Max k_int | Layers sampled |
|-------|---|-----------|---------|-----------|-----------|----------------|
| SmolLM2-135M | 576 | 299.4 | 0.5198 | 122 | 378 | 5 |
| Gemma4-2B | 1536 | 947.6 | 0.6169 | 867 | 1121 | 7 |
| Qwen3.5-35B | 2048 | 1385.0 | 0.6763 | 1325 | 1475 | 5 |
| Qwen3.5-MoE-30B-A3.5B | 2048 | 1385.0 | 0.6763 | 1325 | 1475 | 5 |
| Gemma3-4B | 2560 | 1460.5 | 0.5705 | 1261 | 1635 | 6 |
| Gemma3-12B | 3840 | 2499.6 | 0.6509 | 1719 | 2885 | 7 |
| Gemma4-27B | 5376 | 3803.2 | 0.7074 | 3165 | 4342 | 6 |
| Gemma4-31B | 5376 | 3803.2 | 0.7074 | 3165 | 4342 | 6 |

### Per-layer k_int detail

SmolLM2-135M (d=576)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 122 | 0.2118 |
| 5 | 364 | 0.6319 |
| 10 | 378 | 0.6562 |
| 15 | 315 | 0.5469 |
| 20 | 318 | 0.5521 |

Gemma4-2B (d=1536)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1002 | 0.6523 |
| 4 | 1121 | 0.7298 |
| 8 | 918 | 0.5977 |
| 12 | 867 | 0.5645 |
| 16 | 921 | 0.5996 |
| 20 | 917 | 0.5970 |
| 25 | 887 | 0.5775 |

Qwen3.5-35B (d=2048)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1382 | 0.6748 |
| 10 | 1391 | 0.6792 |
| 20 | 1352 | 0.6602 |
| 30 | 1325 | 0.6470 |
| 39 | 1475 | 0.7202 |

Qwen3.5-MoE-30B-A3.5B (d=2048)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1382 | 0.6748 |
| 10 | 1391 | 0.6792 |
| 20 | 1352 | 0.6602 |
| 30 | 1325 | 0.6470 |
| 39 | 1475 | 0.7202 |

Gemma3-4B (d=2560)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1564 | 0.6109 |
| 6 | 1450 | 0.5664 |
| 12 | 1367 | 0.5340 |
| 18 | 1486 | 0.5805 |
| 24 | 1635 | 0.6387 |
| 30 | 1261 | 0.4926 |

Gemma3-12B (d=3840)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1719 | 0.4477 |
| 8 | 2559 | 0.6664 |
| 16 | 2433 | 0.6336 |
| 24 | 2263 | 0.5893 |
| 32 | 2821 | 0.7346 |
| 40 | 2885 | 0.7513 |
| 45 | 2817 | 0.7336 |

Gemma4-27B (d=5376)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 3416 | 0.6354 |
| 12 | 3165 | 0.5887 |
| 24 | 3896 | 0.7247 |
| 36 | 3998 | 0.7437 |
| 48 | 4342 | 0.8077 |
| 58 | 4002 | 0.7444 |

Gemma4-31B (d=5376)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 3416 | 0.6354 |
| 12 | 3165 | 0.5887 |
| 24 | 3896 | 0.7247 |
| 36 | 3998 | 0.7437 |
| 48 | 4342 | 0.8077 |
| 58 | 4002 | 0.7444 |

### Data-Quality Notes

Some model pairs have identical per-layer k_int vectors. This may reflect shared checkpoints, aliasing in model selection, or a pipeline mapping issue and should be verified before publication.

- Identical k_int profile: Gemma4-27B, Gemma4-31B
- Identical k_int profile: Qwen3.5-35B, Qwen3.5-MoE-30B-A3.5B

### LaTeX Generalisation Table (copy-paste)

```latex
\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Model & $d$ & $\bar{k}_\mathrm{int}$ & $\bar{k}_\mathrm{int}/d$ \\
\midrule
  SmolLM2-135M & 576 & 299.4 & 0.5198 \\
  Gemma4-2B & 1536 & 947.6 & 0.6169 \\
  Qwen3.5-35B & 2048 & 1385.0 & 0.6763 \\
  Qwen3.5-MoE-30B-A3.5B & 2048 & 1385.0 & 0.6763 \\
  Gemma3-4B & 2560 & 1460.5 & 0.5705 \\
  Gemma3-12B & 3840 & 2499.6 & 0.6509 \\
  Gemma4-27B & 5376 & 3803.2 & 0.7074 \\
  Gemma4-31B & 5376 & 3803.2 & 0.7074 \\
\bottomrule
\end{tabular}
\caption{Intrinsic rank $k_\mathrm{int}$ at 95\% joint Gram variance}
\end{table}
```

---

## Paper B — Load & VRAM Efficiency

Summary: load logs show several compressed models fitting within an 8 GB-class GPU budget, with model-dependent offload behavior.

| Model | Load (ms) | GPU tensors | GPU VRAM (MB) | Offload from | tok/s |
|-------|----------|-------------|---------------|--------------|-------|
| GLM-4.7-Flash | 817 | 5 | 457 | — | — |
| Gemma3-12B | 5400 | 236 | 6019 | 47 | 3.8 |
| Gemma3-4B | 2280 | 171 | 2174 | — | 58.3 |
| Gemma4-31B | 6155 | 103 | 6444 | 21 | — |
| Qwen3.5-35B | 1220 | 82 | 1051 | — | — |

---

## Paper C — Decode Throughput Under GRC Compression

Summary: preliminary decode measurements indicate usable throughput with GRC and provide early OTT/AttnRes interaction evidence.

No farm decode log rows yet; showing OTT/AttnRes empirical sections below.


---

# OTT Speculative Decode — Empirical Speedup Table

Paper C empirical anchor data  
Rows show mean tok/s ± 95% CI and acceptance rate α across 10 locked prompts.
Speedup = mean_tok_s / baseline_tok_s for the same model.

| Model | Mode | thresh | batch | tok/s | ±CI | α (%) | ±CI | geo_frac | Speedup |
|-------|------|--------|-------|-------|-----|-------|-----|----------|---------|
| SmolLM2-135M | baseline | 0.0 | 0 | 71.92 | ±21.96 | — | ±— | — | 1.0 |
| SmolLM2-135M | spec | 0.45 | 4 | 33.66 | ±19.22 | — | ±— | — | 0.468 |
| SmolLM2-135M | spec_grc | 0.45 | 4 | 81.31 | ±12.57 | 46.9 | ±— | 46.9% | 1.131 |

## Key observations

- Geodesic hit rate (`geo_frac`) shows what fraction of accepted tokens came
  from the Riemannian geodesic draft vs. the transformer verifier correction path.
- Speedup > 1.0 confirms the speculative path outperforms autoregressive decode
  on this hardware. Speedup < 1.0 means the verifier overhead dominates.
- spec_grc rows test whether GRC compression at k=1024 affects α.
  Significant α drop would indicate the compressed attention manifold diverges
  from the uncompressed verifier's predicted distribution.

---

## Paper D — HJB Feasibility Spectrum

# HJB / SHF Loss — Empirical Feasibility Stub

Computed per `scripts/hjb_feasibility.py` from the discrete Jacobi residual

$\mathcal{J}_\ell(s) = \Delta^2 s_\ell + \hat R(s_\ell)\,\Delta s_\ell$

applied to three trajectory classes through each model's fitted Phase-3 Riemannian manifold (intrinsic dim n=8, length L=32, 8 trajectories per class).

## Per-model SHF mean penalty (lower = more geodesic)

| Model | floor (baked geodesic) | conformant (NN walk) | off-manifold (straight) | conf/floor | off/floor |
|---|---:|---:|---:|---:|---:|
| smollm2-135m | 3.448e-13 | 3.460e+01 | 4.013e-31 | 100348941481125.25 | 0.00 |
| gemma-4-e2b | 4.028e-34 | 9.452e+01 | 1.136e-30 | 94515620943131423831110002409472.00 | 1.14 |
| phi-3.5-mini | 1.571e-03 | 7.929e+00 | 2.557e-31 | 5047.47 | 0.00 |

## Per-residual cost (training-time gradient implied)

| Model | intrinsic dim n | FLOPs / residual / layer / token |
|---|---:|---:|
| smollm2-135m | 8 | 557,056 |
| gemma-4-e2b | 8 | 557,056 |
| phi-3.5-mini | 8 | 557,056 |

## Kinetic vs curvature decomposition (mean per node)

$\mathcal{J}_\ell = \Delta^2 s_\ell + \hat R(s_\ell)\,\Delta s_\ell$ — we report the squared magnitudes of the two summands separately so the dominant component is visible.

| Model | class | kinetic mean $\|\Delta^2 s\|^2$ | curvature mean $\|\hat R\Delta s\|^2$ | kin/curv |
|---|---|---:|---:|---:|
| smollm2-135m | baked_geodesic | 3.448e-13 | 2.876e-46 | 3.45e+17 |
| smollm2-135m | nn_walk | 3.460e+01 | 5.084e-27 | 6.81e+27 |
| smollm2-135m | random_straight | 3.879e-31 | 1.308e-32 | 3.88e-01 |
| gemma-4-e2b | baked_geodesic | 4.028e-34 | 0.000e+00 | 4.03e-04 |
| gemma-4-e2b | nn_walk | 9.452e+01 | 6.761e-62 | 9.45e+31 |
| gemma-4-e2b | random_straight | 1.136e-30 | 1.186e-68 | 1.14e+00 |
| phi-3.5-mini | baked_geodesic | 1.571e-03 | 7.100e-31 | 1.57e+27 |
| phi-3.5-mini | nn_walk | 7.929e+00 | 2.975e-26 | 2.66e+26 |
| phi-3.5-mini | random_straight | 2.507e-31 | 4.611e-33 | 2.51e-01 |

## Interpretation

- The floor is what a training-time minimiser of $L_{SHF}$ would converge toward; the residual at the floor is the finite-difference truncation error of the discrete Jacobi residual on this manifold.
- The conformant value is what an untrained network with Phase-3-fitted activations already achieves; the gap to the floor is the headroom that training with $L_{SHF}$ could exploit.
- The off-manifold value bounds the penalty from above and indicates the dynamic range over which $\lambda$ should be tuned.
- The kinetic / curvature ratio quantifies how much of the Bellman residual actually carries curvature signal. On the Phase-3 manifolds we have access to the kinetic $\Delta^2$ term dominates by many orders of magnitude (consistent with the curvature-warp negative result, FIVE\_FINDINGS §5: these manifolds are nearly flat). Practical implication: a training-time SHF schedule must either (a) operate on a manifold of demonstrably non-zero curvature, or (b) reduce to ordinary smoothness regularisation (kinetic-only) on current network outputs.

Implication for paper-D §HJB-Regularised Joint Training: the loss term is empirically tractable (sub-second compute for n=8 intrinsic, L=32 layers), the dynamic range (off/floor ratio) determines a meaningful $\lambda$ schedule, and per-token training cost is $O(n^5)$ in intrinsic dim — affordable when the runtime caches the Riemann tensor at sample points (which it already does for the Jacobi propagator). However the empirical kinetic/curvature ratio shows that on the manifolds currently exported from frozen pretrained networks the curvature contribution to $L_{SHF}$ is negligible; the loss in its current form would act primarily as a second-difference smoothness penalty on block summaries.

Generated 2026-04-30 12:13:45; data: hjb_residual_magnitudes.json

---

## Paper E — Rho / Distillation Spectrum

| Model | Rank | LoRA rank | Layers | Mean ρ | Source |
|-------|------|-----------|--------|--------|--------|
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 1024 | 8 | 32 | 0.1340 | docs/figures/paper-e/rho_sweep/rho_summary.json |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 1536 | 8 | 32 | 0.1355 | docs/figures/paper-e/rho_sweep_1536/rho_summary.json |
| models/smollm2-135m-instruct-q8_0.gguf | 256 | 8 | 30 | 0.3443 | docs/figures/paper-e/rho_sweep_spectrum/smollm2_135m/rho_summary.json |

### Per-layer ρ (highest mean: models/smollm2-135m-instruct-q8_0.gguf)

| Layer | ρ |
|-------|---|
| 0 | 0.4111 |
| 1 | 0.3243 |
| 2 | 0.3307 |
| 3 | 0.3314 |
| 4 | 0.3212 |
| 5 | 0.3325 |
| 6 | 0.3246 |
| 7 | 0.3356 |
| 8 | 0.3375 |
| 9 | 0.3397 |
| 10 | 0.3255 |
| 11 | 0.3248 |
| 12 | 0.3339 |
| 13 | 0.3274 |
| 14 | 0.3419 |
| 15 | 0.3409 |
| 16 | 0.3324 |
| 17 | 0.3494 |
| 18 | 0.3565 |
| 19 | 0.3468 |
| 20 | 0.3578 |
| 21 | 0.3486 |
| 22 | 0.3487 |
| 23 | 0.3902 |
| 24 | 0.3645 |
| 25 | 0.3402 |
| 26 | 0.3622 |
| 27 | 0.3422 |
| 28 | 0.3571 |
| 29 | 0.3492 |
