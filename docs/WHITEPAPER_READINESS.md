# White Paper Readiness Assessment

Date: 2026-04-26

## Scope

This assessment evaluates whether HyperTensor GRC (attention-only, weight-PCA, skip-O, k=2048) is ready for a proper white paper release.
All performance figures are reported relative to the original baseline model path.

## Data Sources

- Benchmark matrix (4 prompt types x 3 generation lengths x 2 modes = 24 runs):
  - benchmarks/whitepaper_matrix_20260425_160512/matrix_results.csv
  - benchmarks/whitepaper_matrix_20260425_160512/matrix_relative_to_baseline.csv
- Repeatability check (3 reps, coding prompt, 256 tokens):
  - benchmarks/whitepaper_matrix_20260425_160512/repeatability_256.csv
- Perplexity checks (WikiText-2, 512 tokens):
  - baseline PPL = 6.7902
  - GRC k=2048 PPL = 7.1969

## Relative-to-Baseline Performance Summary

Across the 12 prompt/length cases:

- Mean decode speed: 82.34% of baseline
- Mean overall throughput: 81.15% of baseline
- Mean prefill time: 126.27% of baseline

Best and worst cases:

- Best decode retention: 84.93% (coding, 128 tokens)
- Worst decode retention: 64.69% (coding, 256 tokens, matrix run)
- Best overall retention: 84.18% (reasoning, 256 tokens)
- Worst overall retention: 65.49% (coding, 256 tokens)

Interpretation:

- Typical decode/throughput retention clusters around ~82-85% of baseline.
- There is one clear outlier case (coding/256) that needs investigation before publication claims about stable speed retention.
- Prefill is consistently slower under GRC by about 26% on average.

## Relative-to-Baseline Quality Summary

Perplexity (lower is better):

- Baseline: 6.7902
- GRC k=2048: 7.1969
- Relative PPL: 106.00% of baseline (approximately +6.00%)

Interpretation:

- Quality is near-baseline for this setup, but not parity.
- The quality/speed tradeoff is now quantifiable and coherent.

## Repeatability (3-run check)

Coding prompt, 256 tokens:

- Baseline decode: 36.30 +/- 0.00 tok/s
- GRC decode: 30.50 +/- 0.22 tok/s
- GRC decode retention: 84.02% of baseline

- Baseline overall: 33.62 +/- 0.04 tok/s
- GRC overall: 27.64 +/- 1.03 tok/s
- GRC overall retention: 82.21% of baseline

Interpretation:

- Baseline is very stable.
- GRC has modest variance but remains in the expected low-80% retention band in this repeated test.

## White Paper Readiness Verdict

Current state: Not ready for a proper white paper submission.

What is complete:

- Relative quality degradation is measured and moderate (+6% PPL).
- Relative speed tradeoff is measured across multiple prompts and lengths.
- Rank sweep table (1024/1536/2048) is now completed and normalized to baseline.

Why this is not yet ready:

- The completed rank sweep shows a severe k=2048 slowdown in current runs.
- Evidence is single-model and single-hardware; publication-grade claims usually require at least multi-model or multi-hardware confirmation.
- No confidence intervals or significance testing are included yet.

Latest completed rank sweep aggregate (current run state):

- k=1024: decode 103.27% of baseline, overall 101.32%, prefill 104.91%
- k=1536: decode 89.02% of baseline, overall 88.10%, prefill 119.21%
- k=2048: decode 46.22% of baseline, overall 45.81%, prefill 244.48%

Interpretation:

- The k=2048 path is currently regressed and does not match earlier retained-throughput claims.
- This is now a primary engineering blocker for publication-facing claims at k=2048.

## Outlier Investigation Update (coding/256)

Targeted repeated runs (6 reps each for coding and reasoning at 256 tokens) show similar retention collapse in both prompt classes under affected runs:

- coding decode retention (6-run mean): 39.32%
- reasoning decode retention (6-run mean): 38.61%

Interpretation:

- The gap between coding and reasoning retention is small, so the observed collapse is not coding-specific.
- The stronger explanation is run-condition instability in those sessions (process-state / warmup / long-run environment effects), not a coding-quality penalty.

5-run confidence slices from the same dataset (reps 1-5) support the same picture:

- Coding decode: baseline 34.10 +/- 3.68 tok/s, GRC 13.24 +/- 1.38 tok/s (38.83% of baseline)
- Reasoning decode: baseline 32.50 +/- 6.81 tok/s, GRC 12.32 +/- 2.72 tok/s (37.91% of baseline)

## Coding Quality Note

Coding output quality remains a contextual note, not a demerit score, in this readiness pass.
The readiness decision is grounded in baseline-relative quantitative metrics (PPL, decode retention, overall retention, and variance bounds).

## Minimum Work to Reach Proper White Paper State

1. Root-cause and fix the k=2048 regression so retention is stable and reproducible.
2. Re-run the full matrix and confidence slices after the fix using the same no-pipe harness.
3. Add at least one additional model and one additional hardware profile.
4. Expand repeatability to >= 5 runs per key scenario and report confidence intervals.
5. Freeze a benchmark protocol appendix so all future results are reproducible and comparable.

## Claim Language Safe to Use Now

- "At k=1024 and k=1536 on Llama-3.1-8B-Instruct-Q4_K_M, current runs show strong baseline-relative throughput retention, while k=2048 is currently regressed and under active investigation."
- "Quality remains near baseline at the reported PPL pair (6.7902 vs 7.1969), and speed claims are reported strictly as baseline-relative percentages under the current harness."
