## Paper E --- Rho / Distillation Spectrum

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
