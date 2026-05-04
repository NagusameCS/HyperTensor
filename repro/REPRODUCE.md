# Geodessical GRC --- Reproduction Guide

Version: 0.6.0
Primary validated pack: `benchmarks/whitepaper_pack_20260427_121815`
Reference hardware: RTX 4070 Laptop GPU (8 GB VRAM), Ryzen 9 7940HS, Windows

This document gives the exact commands needed to reproduce the core benchmark tables
from the technical report (`docs/WHITEPAPER.md`). Everything runs from the repository root.

---

## Prerequisites

| Requirement | Version / Notes |
|------------|-----------------|
| CUDA-capable GPU | Minimum 8 GB VRAM for 8B Q4_K_M model |
| CUDA driver | 12.x or later (tested: 595.79) |
| Build tools | Zig cc (see `build_host.ps1` for version) |
| PowerShell | 7+ recommended |
| Model file | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` from [bartowski on HuggingFace](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| Disk space | ~6 GB (model 4.6 GB + W_proj cache 1.1 GB) |

---

## Step 1 --- Build the runtime

```powershell
cd <repo_root>
.\build_host.ps1
```

Expected: `build_host\geodessical.exe` created (~1.1 MB binary).

---

## Step 2 --- Set model path

```powershell
$MODEL = "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

---

## Step 3 --- Verify baseline runs

```powershell
.\build_host\geodessical.exe $MODEL -p "Hello, world" -n 32
```

Expected output includes a token-generation line like:
```
[GD] 32 tokens in 891.4 ms (35.9 tok/s)
```

Absolute tok/s will vary by GPU. The important number is this forms the baseline
that all GRC comparisons are relative to.

---

## Step 4 --- Run the full benchmark suite

This runs the rank sweep (k=1024/1536/2048), 12-rep CI pack, and 5-rep PPL pack
with the 30-second cooldown protocol. Expected duration: 40--70 minutes.

```powershell
.\scripts\benchmark_whitepaper_finalize.ps1 -Model $MODEL -CooldownSec 30
```

This creates a timestamped pack directory under `benchmarks/`.

---

## Step 5 --- Validate gates

```powershell
$PACK = "benchmarks\<pack_dir_from_step_4>"
.\scripts\validation_cycle.ps1 -PackDir $PACK
```

Expected terminal output:
```
Gate k1024_decode  (>=95.0%):  PASS  [106.27%]
Gate k1536_decode  (>=75.0%):  PASS  [97.55%]
Gate k2048_decode  (>=75.0%):  PASS  [101.04%]
Gate k2048_prefill (<=225.0%): PASS  [108.48%]
Gate CI_lower95_coding   (>=67.0%): PASS  [86.60%]
Gate CI_lower95_reasoning(>=67.0%): PASS  [85.64%]
Gate PPL_delta     (<=15.0%):  PASS  [+13.30%]

STRONG_CLAIM_READY: True
```

---

## Step 6 --- Run single-shot quality check (PPL only, faster)

To reproduce just the perplexity numbers without the full throughput suite (~15 min):

```powershell
# Baseline
.\build_host\geodessical.exe $MODEL --ppl-eval

# GRC k=1536 (k=2048 request is capped to k=1536)
.\build_host\geodessical.exe $MODEL `
    --axex-compress --axex-attn-only --axex-skip-o --axex-weight-pca `
    --axex-compress-rank 2048 --ppl-eval
```

Expected values (deterministic --- same across all runs):

| Run | PPL |
|-----|-----|
| Baseline | 6.7902 |
| GRC k=1536 | 7.6936 |
| Delta | +13.30% |

---

## Expected Benchmark Outputs

The following tables are from the primary validated pack
(`benchmarks/whitepaper_pack_20260427_121815`). Your numbers will differ in absolute
tok/s depending on GPU, but the relative percentages should match within ±5%
on hardware with similar VRAM bandwidth characteristics.

### Rank Sweep (`rank_sweep_aggregate.csv`)

| Rank | Decode % of baseline | Overall % of baseline | Prefill % of baseline |
|------|---------------------|--------------------|----------------------|
| 1024 | 106.27% | 105.72% | 102.67% |
| 1536 | 97.55%  | 95.80%  | 114.61% |
| 2048 | 101.04% | 99.34%  | 108.48% |

The k=1024 above-baseline result is a GPU L2-cache fit effect and may not reproduce
on GPUs with substantially different cache sizes (e.g., A100 80GB).

### CI Pack (`ci_pack_summary.csv`)

| Prompt | Baseline decode | GRC decode | Retention | Lower-95 |
|--------|----------------|-----------|-----------|---------|
| coding/256 | 35.68 ± 0.35 tok/s | 34.86 ± 2.02 tok/s | 97.70% | 86.60% |
| reasoning/256 | 35.58 ± 0.31 tok/s | 35.22 ± 2.42 tok/s | 98.99% | 85.64% |

### PPL (`ci_ppl_5run.csv`)

| Rep | Baseline PPL | GRC PPL | Delta |
|-----|-------------|---------|-------|
| 1--5 (all) | 6.7902 | 7.6936 | +13.30% |

PPL is fully deterministic --- identical values across all 5 repetitions.

---

## Reference Artifacts

Exact reference CSVs from the primary pack are in `repro/expected_outputs/`:

```
repro/expected_outputs/
  rank_sweep_aggregate.csv
  ci_pack_summary.csv
  ci_ppl_5run.csv
  validation_cycle.json
```

---

## Phase 3 Transfer (Cross-Model)

To run the same protocol on a different model:

```powershell
.\scripts\phase3_transfer.ps1 `
    -Model "C:\path\to\other-model-Q4_K_M.gguf" `
    -ModelTag "model_shortname"
```

This creates a new pack, runs the validator, and prints a comparison prompt.
Results are saved to `benchmarks/phase3_<ModelTag>_<timestamp>/`.

Transfer pass criteria (qualitative):
- Same rank ordering (k=1024 fastest or within 10% of k=1536)
- No gate-critical metric contradicting the primary pack
- PPL delta within 5% of the +13.30% measured on Llama 3.1 8B (±5 percentage points)

---

## Known Reproduction Caveats

1. Thermal protocol is required. Without the 30-second cooldown (`-CooldownSec 30`),
   GPU thermal throttling can reduce GRC throughput to 50--60% of baseline on sustained runs,
   making the results look like a regression. Always use the cooldown.

2. W_proj cache must be warm. First-run calibration (60--120 s) adds latency that is not
   included in throughput numbers. The benchmark harness handles this automatically ---
   it discards the first run if the cache was just built.

3. AXEX_MANIFOLD_K_MAX = 1536. Requesting k=2048 is silently capped. k=2048 and k=1536
   rows share the same W_proj cache.

4. Batch-prefill is disabled on GRC path. Prefill runs token-by-token when GRC is active,
   causing the 108--115% prefill overhead. This is an implementation constraint, not a
   fundamental property of the compression method.

5. PPL is deterministic. If your PPL values differ from 6.7902/7.6936, suspect a
   different model file (wrong quantisation variant) or a different WikiText-2 dataset slice.
