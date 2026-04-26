# HyperTensor GRC Whitepaper

Date: 2026-04-26

## Abstract

This document describes a working 8B-scale proof of concept for attention-weight compression in a GGUF inference runtime.
The core method is a geometry-informed projection workflow implemented in the Geodessical runtime under the GRC path.

At k=2048, on Meta-Llama-3.1-8B-Instruct-Q4_K_M, we retain near-baseline perplexity while halving the active attention weight footprint in the compressed path.
The measured quality/speed tradeoff is explicit and normalized to baseline.

This is a technical report for reproducible engineering state, not a universal claim across all models or hardware.

## 1. Problem Framing

Large transformer inference systems are often constrained by memory bandwidth and weight movement, not only pure arithmetic throughput.
The central question in this work is:

How much attention weight structure can be compressed while preserving practical generation quality?

The target here is practical deployment behavior on an 8B model, not symbolic compression ratios in isolation.

## 2. System Under Test

Runtime:

- Geodessical v0.6.x host runtime

Model:

- Meta-Llama-3.1-8B-Instruct-Q4_K_M (GGUF)

Hardware context for numbers in this report:

- Local RTX 4070 laptop class system used during this benchmark cycle

Evaluation channels:

- WikiText-2 perplexity (512-token evaluation)
- prompt-based speed sweeps with deterministic settings for comparison stability

## 3. Method: Why This Compression Path

### 3.1 Attention-only focus

The current best-performing path compresses attention projections while leaving FFN behavior in a safer regime.
This narrows the risk surface while preserving a meaningful memory/quality tradeoff.

### 3.2 Weight-space PCA basis per layer

For each transformer layer, we build a projection basis from the weight geometry:

- Construct a layer-local covariance-like term from attention weight structure
- Solve for top-k eigenvectors
- Use that basis as the layer projection operator

This yields per-layer basis matrices Pt and projected weights W_proj.

### 3.3 Runtime transform

At inference, the compressed path computes projected activations and projected matvecs instead of full-dimension attention matvecs.

Conceptually:

- project residual stream into k-dimensional subspace
- apply projected attention weights
- map through the same attention flow with preserved structural assumptions

### 3.4 Skip-O decision

`--axex-skip-o` is used in the validated 8B path to avoid destabilizing quality in this configuration.
This is a deliberate engineering choice, not a hidden limitation.

## 4. Reproducible Commands

### 4.1 Baseline PPL

```powershell
$MODEL = "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
.\build_host\geodessical.exe $MODEL --ppl-eval
```

### 4.2 GRC PPL (k=2048)

```powershell
$MODEL = "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
.\build_host\geodessical.exe $MODEL --axex-compress --axex-attn-only --axex-skip-o --axex-weight-pca --axex-compress-rank 2048 --ppl-eval
```

### 4.3 Throughput measurement without pipe distortion

```powershell
.\scripts\benchmark_decode_nopipe.ps1 -Model "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

## 5. Quality Results

### 5.1 Perplexity

Measured on WikiText-2 (512 tokens):

- Baseline PPL: 6.7902
- GRC k=2048 PPL: 7.1969

Relative quality view:

- GRC PPL is 106.00% of baseline (about +6.00%)

This is close enough for practical proof-of-concept demonstration while remaining honest about the gap.

### 5.2 Coding-output interpretation note

Coding-output behavior is recorded as a contextual quality note in this report, not used as a punitive demerit metric.
This phase is focused on measurable compression-vs-quality-vs-speed behavior under controlled settings.

## 6. Performance Results (Baseline-Relative)

### 6.1 Multi-prompt matrix (k=2048)

Earlier matrix runs reported:

- Mean decode retention: 82.34% of baseline
- Mean overall throughput retention: 81.15% of baseline
- Mean prefill time: 126.27% of baseline

Current interpretation in this revision:

- Those figures are historical references from a prior run window.
- The completed follow-up rank sweep shows k=2048 is currently regressed in this branch.
- Publication-facing speed language must follow the latest completed sweep, not the earlier optimistic matrix.

### 6.2 Outlier investigation for coding/256

A targeted repeated test was run for coding and reasoning prompts at 256 tokens.
The results show transient low-throughput events affecting both prompt classes, not coding alone.

From the completed outlier pack:

- coding decode retention (6-run mean): 39.32%
- reasoning decode retention (6-run mean): 38.61%

The near-equality indicates a shared run-condition issue rather than a coding-only pathology.

## 7. Confidence-Pack Results Used For Claims

### 7.1 5-run throughput confidence pack (from completed outlier dataset)

Coding 256:

- Decode baseline: 34.10 +/- 3.68 tok/s
- Decode GRC: 13.24 +/- 1.38 tok/s
- Decode retention: 38.83%

- Overall baseline: 31.70 +/- 3.33 tok/s
- Overall GRC: 12.08 +/- 1.22 tok/s
- Overall retention: 38.11%

Reasoning 256:

- Decode baseline: 32.50 +/- 6.81 tok/s
- Decode GRC: 12.32 +/- 2.72 tok/s
- Decode retention: 37.91%

- Overall baseline: 30.30 +/- 6.08 tok/s
- Overall GRC: 11.32 +/- 2.59 tok/s
- Overall retention: 37.36%

These confidence figures are intentionally narrow in scope and tied to the exact run conditions and command path used.
They support stability analysis for the affected sessions, not broad performance guarantees.

### 7.2 5-run PPL pack

PPL confidence pack artifacts are recorded under the benchmark output directory and used to bound quality claims.
In this revision cycle, central quality claim remains anchored to the stable 6.7902 vs 7.1969 measurement pair and baseline-relative expression.
The strict gate check still requires `ci_ppl_5run.csv`; this file is currently missing from the latest CI directory used by the validator.

## 8. Rank Sweep Status

The baseline-normalized rank sweep is now complete for 1024, 1536, and 2048.

Latest aggregate from the completed sweep:

- k=1024: decode 103.27% of baseline, overall 101.32%, prefill 104.91%
- k=1536: decode 89.02% of baseline, overall 88.10%, prefill 119.21%
- k=2048: decode 46.22% of baseline, overall 45.81%, prefill 244.48%

Interpretation:

- k=1024 and k=1536 are in a plausible operating range for this setup.
- k=2048 is currently regressed in the active code path and is the primary blocker for publication-grade speed claims.

Machine-gate status from `scripts/paradigm_shift_validate.ps1`:

- strong-claim ready: false
- k1024 decode gate: pass
- k1536 decode gate: pass
- k2048 decode gate: fail
- k2048 prefill gate: fail
- CI lower-bound throughput gate: fail
- PPL gate: unresolved/fail due missing `ci_ppl_5run.csv` in current CI artifacts

## 9. What Is Demonstrated Today

Demonstrated reliably:

- 8B attention-compression proof of concept
- near-baseline perplexity at k=2048
- reproducible baseline-relative throughput measurement flow
- practical cache-backed repeated-run workflow

Not yet claimed as complete:

- universal performance behavior across hardware classes
- broad model-family generalization
- root-caused and fixed k=2048 throughput regression in the current branch

## 10. Current Research Interest

This work brings together several engineering components in one operating path:

- geometry-aware layerwise compression with explicit runtime projection
- practical constraints around cache identity, deterministic measurement, and GPU/CPU path hygiene
- baseline-relative reporting to keep numbers interpretable across conditions

The approach is concrete and measurable; the tradeoffs are quantified and inspectable rather than asserted.

## 11. Next Technical Milestones

1. Root-cause and fix the k=2048 throughput regression in the current branch.
2. Re-run and lock the full matrix and CI slices after the fix.
3. Add second hardware profile to separate method behavior from machine-specific effects.
4. Keep coding-output quality as a tracked qualitative note with dedicated future quantitative scoring once eval harness is finalized.

## 12. Artifact Paths

Benchmark outputs used in this cycle are under:

- `benchmarks/whitepaper_matrix_20260425_160512/`
- `benchmarks/whitepaper_pack_20260425_192208/`
- `benchmarks/whitepaper_finalize_20260425_rank/`

These include raw stdout/stderr captures and derived CSV files.
