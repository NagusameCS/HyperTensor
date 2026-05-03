# Benchmark Protocol (Locked)

Date: 2026-04-26

This protocol defines the minimum standard for making strong performance or quality claims in this repository.

## Goal

Convert benchmark evidence from ad-hoc runs into reproducible pass/fail validation.

## Required Conditions

- Runtime build: current `build_host\geodessical.exe` built from working tree under test.
- Model class for this milestone: 8B and below.
- Primary model for continuity: Meta-Llama-3.1-8B-Instruct-Q4_K_M.
- Temperature: `--temp 0` for all claim-critical runs.
- Output capture: no live tee/pipeline during measurement. Redirect to files.
- Warmup: one warmup pass before timed runs when evaluating throughput retention.
- Cache policy: explicitly record whether projection cache was reused or rebuilt.

## Required Artifacts

A claim package must include all of:

1. Outlier investigation
- File: `outlier_investigation.csv`
- Scope: coding/256 and reasoning/256, minimum 6 reps each mode.

2. Baseline-normalized rank sweep
- Files:
  - `rank_sweep_raw.csv`
  - `rank_sweep_relative_to_baseline.csv`
  - `rank_sweep_aggregate.csv`
- Ranks required: 1024, 1536, 2048.

3. 5-run confidence pack
- Files:
  - `ci_pack_raw.csv`
  - `ci_pack_summary.csv`
  - `ci_ppl_5run.csv`
- Cases required: coding_256 and reasoning_256 for throughput; baseline vs k=2048 for PPL.

## Interpretation Rules

- Coding quality is a contextual note, not a demerit axis for this stage.
- Throughput claims must be baseline-relative percentages, not isolated absolute tok/s values.
- If rank-2048 is regressed, publication language must explicitly say so and avoid optimistic wording.

## Pass/Fail Readiness Gates

Use `scripts/validation_cycle.ps1` to generate a machine-readable readiness report.

A package is "strong-claim ready" only if all gates pass:

- k1024 mean decode retention >= 95%
- k1536 mean decode retention >= 85%
- k2048 mean decode retention >= 75%
- k2048 mean prefill <= 150% of baseline
- 5-run coding and reasoning decode lower 95% bound >= 75%
- PPL delta <= +8% over baseline

If any gate fails, claims must be framed as proof-of-concept and blocker-focused.
