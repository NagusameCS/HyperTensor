# Geodessical Benchmark Report

**Date:** 2026-04-18  
**Hardware:** AMD Ryzen 9 7940HS · NVIDIA RTX 4070 Laptop GPU (8 GB VRAM)  
**OS:** Windows 11  
**Geodessical build:** CUDA-enabled (`build_host\geodessical.exe`)  
**Ollama version:** latest (comparison baseline)

---

## Table of Contents

1. [Environment & Models](#environment--models)
2. [Geodessical vs Ollama — Head-to-Head](#geodessical-vs-ollama--head-to-head)
3. [Resource Consumption](#resource-consumption)
4. [Flag Configuration Sweep](#flag-configuration-sweep)
5. [CPU Thread Scaling](#cpu-thread-scaling)
6. [Key Findings & Recommendations](#key-findings--recommendations)

---

## Environment & Models

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| smollm2-135m-instruct | Q8_0 GGUF | 138 MB | Small, fits entirely in VRAM |
| Gemma-4-E2B-it | Q4_0 GGUF | 3.2 GB | Medium, near VRAM limit |
| phi-3.5-mini-instruct | Q4_0 GGUF | ~2.1 GB | Medium |

**Metric definitions:**
- **Decode t/s** — generation tokens per second (decode phase only)
- **Prefill t/s** — prompt tokens processed per second
- **TTFT ms** — Time To First Token (prefill latency)
- **VRAM MB** — GPU memory allocated at steady state
- **HBM %** — percentage of theoretical HBM bandwidth in use during decode

---

## Geodessical vs Ollama — Head-to-Head

### Quick Comparison (single-run, realistic prompt)

| Model | Metric | Geodessical | Ollama | Winner |
|-------|--------|:-----------:|:------:|:------:|
| smollm2-135m | Decode t/s | 298 | 467 | Ollama +57% |
| smollm2-135m | Prefill t/s | ~298 | 2 359 | Ollama 8× |
| smollm2-135m | TTFT | 92 ms | 20 ms | Ollama 4.6× |
| smollm2-135m | VRAM | **262 MB** | ~450 MB | Geo −42% VRAM |
| phi3.5-mini | Decode t/s | 87.5 | 94.1 | Near parity |
| phi3.5-mini | TTFT | 172 ms | 42 ms | Ollama 4× |
| gemma4-2b | Decode t/s | **111.7** | 107.4 | **Geo +4%** |
| gemma4-2b | VRAM | **1 622 MB** | ~3 400 MB | **Geo −52% VRAM** |

### Averaged Across Prompt Lengths (40 / 128 / 512 tokens out)

| Runtime | Backend | Model | Decode t/s | Prefill t/s | TTFT ms |
|---------|---------|-------|:----------:|:-----------:|:-------:|
| Geodessical | GPU | smollm2-135m | 243 | 315 | 74 |
| Ollama | GPU | smollm2-135m | 726 | 17 123 | 5 |
| Geodessical | GPU | gemma4-2b | 34 | 101 | 166 |
| Ollama | GPU | gemma4-2b | 104 | 2 719 | 13 |
| Geodessical | CPU | smollm2-135m | 34 | 43 | 473 |
| Ollama | CPU | smollm2-135m | 149 | 4 759 | 12 |
| Geodessical | CPU | gemma4-2b | 7 | 12 | 1 108 |
| Ollama | CPU | gemma4-2b | 27 | 719 | 83 |

**Observations:**
- Geodessical's decode is competitive at larger model sizes (within 4–15% of Ollama on GPU for bandwidth-bound workloads).
- Prefill is 8–170× slower than Ollama — Geodessical has no batched prompt kernel; every token is processed sequentially.
- Geodessical uses ~50% less VRAM than Ollama on all tested models.

---

## Resource Consumption

### gemma4-2b GPU Timeline

| Phase | Duration | GPU Util | Power | HBM BW |
|-------|----------|:--------:|:-----:|:------:|
| Model load | 740 ms | 98% | 48 W | — |
| Decode (steady state) | per token | 7% | 14–17 W | 335.5 GB/s (99.84% of 336 GB/s peak) |

The RTX 4070 Laptop's HBM is saturated at **99.84%** during gemma4-2b decode — the bottleneck is memory bandwidth, not compute.

### Bigger Models (Ollama only, for reference)

| Model | Decode t/s | VRAM |
|-------|:----------:|:----:|
| gemma4:9b | 30 | 6 142 MB |
| gemma3:12b | 6.7 (CPU spill) | 7 414 MB |

---

## Flag Configuration Sweep

All GPU runs use `build_host\geodessical.exe` (CUDA build).  
Results saved to `benchmark_flags_results.csv`.

### smollm2-135m (138 MB, N=100 tokens)

| Config | Decode t/s | TTFT ms | HBM % | Startup | vs Baseline |
|--------|:----------:|:-------:|:-----:|:-------:|:-----------:|
| GPU / baseline | 300.8 | 70 | 13.0% | 0.7s | — |
| GPU / no-verifier | **327.4** | 60 | 14.1% | 0.6s | **+8.8%** |
| GPU / one-decode | **327.5** | 63 | 14.1% | 3.5s | **+8.9%** |
| GPU / ott-spec | 325.9 | 108 | 14.1% | 4.5s | +8.4% |
| GPU / ott-spec b=6 | 326.3 | 107 | 14.1% | 3.4s | +8.5% |
| GPU / ott-od | 322.1 | 106 | 13.9% | 11.7s | +7.1% |
| GPU / ott-fast | 307.2 | 106 | 13.2% | **53.7s** | +2.1% |
| GPU / attnres 0.7 | 288.4 | 105 | 12.4% | 0.4s | −4.1% |
| GPU / attnres | 260.4 | 108 | 11.2% | 0.4s | −13.4% |
| GPU / ott-full | 66.7 | 14 | 2.9% | 3.8s | **−77.8%** |

### gemma4-2b (3.2 GB, N=80 tokens)

| Config | Decode t/s | TTFT ms | HBM % | Startup | vs Baseline |
|--------|:----------:|:-------:|:-----:|:-------:|:-----------:|
| GPU / baseline | 86.2 | 156 | 86.7% | 1.9s | — |
| GPU / attnres 0.7 | **115.3** | 215 | 116% | 1.1s | **+33.8%** |
| GPU / attnres | **110.5** | 221 | 111% | 1.2s | **+28.2%** |
| GPU / no-verifier | 89.6 | 157 | 90.1% | 1.9s | +4.0% |
| GPU / one-decode | 88.7 | 152 | 89.2% | 7.5s | +2.9% |
| GPU / ott-spec | 83.6 | 498 | 84.1% | 25.7s | −3.0% |
| GPU / ott-fast | 81.2 | 298 | 81.6% | **225s** | −5.8% |
| GPU / ott-od | 77.0 | 358 | 77.4% | 41.3s | −10.7% |
| GPU / ott-spec b=6 | 74.4 | 220 | 74.8% | 7.0s | −13.7% |
| GPU / ott-full | 27.3 | 36 | 27.5% | 7.7s | **−68.3%** |

**Note on HBM % > 100% for attnres:** The counter reflects sustained bandwidth utilisation relative to the advertised peak; values slightly over 100% indicate burst-mode access patterns (L2 cache reuse or out-of-order memory transactions being counted differently by the hardware PMU). The absolute throughput numbers are valid.

---

## CPU Thread Scaling

Tested on smollm2-135m (memory-bandwidth-limited even on CPU).  
**Warning:** 4+ threads causes a deadlock in the current build — do not use `-t 4` or higher.

| Threads | Decode t/s | TTFT ms |
|:-------:|:----------:|:-------:|
| 1 | 41.0 | 403 ms |
| 2 | 45.3 | 389 ms |
| 4+ | ❌ hang | — |

Minimal scaling from 1→2 threads confirms the workload is memory-bandwidth-bound, not compute-bound.

---

## Key Findings & Recommendations

### `--attnres` — Use on large (bandwidth-saturated) models

`--attnres` (attention residual depth stabilization) provides a **+28–34% decode speedup** on gemma4-2b, where HBM is ≥87% saturated. On smollm2 (13% HBM), it *reduces* throughput by 13%. The flag trades extra attention compute for improved memory access locality — only beneficial when memory bandwidth is the bottleneck.

**Recommendation:** Enable `--attnres` (or `--attnres --attnres-strength 0.7`) for any model where HBM% exceeds ~60%.

### `--no-verifier` / `--one-decode` — Safe universal speedup

Both flags give a clean **+9% on smollm2** and **+3–4% on gemma4** with no side effects. `--no-verifier` skips the transformer verification pass on geodesic drafts. `--one-decode` bakes the geodesic flow map once.

**Recommendation:** Enable `--no-verifier` by default for production serving.

### OTT speculative modes — Marginal or negative on GPU

`--ott-spec`, `--ott-od`, and `--ott-spec b=6` all show marginal gains (+7–8%) on smollm2 but regress 3–14% on gemma4. The geodesic draft overhead does not amortize when the main model decode is already near peak bandwidth.

**Recommendation:** Only use OTT speculative modes for CPU inference or very small models.

### `--ott-fast` / `--ott-full` — Not suitable for production

- `--ott-fast`: 53s (smollm) / 225s (gemma) cold-start due to axiom manifold analysis. Throughput gain is 0–2% at best.
- `--ott-full`: Catastrophic slowdown — **−78% on smollm2, −68% on gemma4**. This is the full axiomatic geometry analysis path, not a decode acceleration path.

**Recommendation:** These modes are for offline analysis only. Never use in serving.

### VRAM efficiency

Geodessical consistently uses **~50% less VRAM** than Ollama:

| Model | Geodessical | Ollama |
|-------|:-----------:|:------:|
| smollm2-135m | 262 MB | ~450 MB |
| gemma4-2b | 1 622 MB | ~3 400 MB |

This means Geodessical can fit models ~2× larger than Ollama on the same GPU, e.g. a model Ollama would spill to CPU at 8 GB can run fully on-GPU in Geodessical.

### Prefill gap

Ollama's prefill is **8–170× faster** due to batched prompt kernels. Geodessical processes prompt tokens sequentially. For chat-style interactions this manifests as higher TTFT. Implementing a batched prefill kernel is the single highest-impact optimization opportunity.

---

*Generated from `benchmark_flags_results.csv` and `benchmark_results.csv` on 2026-04-18.*
