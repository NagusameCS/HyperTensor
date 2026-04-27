# White Paper Readiness Assessment

Date: 2026-04-26

## Scope

This assessment evaluates whether HyperTensor GRC (attention-only, weight-PCA, skip-O, k=2048) is ready for a proper white paper release.
All performance figures are reported relative to the original baseline model path.

## Data Sources

- Finalized benchmark pack:
  - benchmarks/whitepaper_pack_20260426_212526/rank_sweep_relative_to_baseline.csv
  - benchmarks/whitepaper_pack_20260426_212526/rank_sweep_aggregate.csv
  - benchmarks/whitepaper_pack_20260426_212526/ci_pack_summary.csv
  - benchmarks/whitepaper_pack_20260426_212526/ci_ppl_5run.csv

## Relative-to-Baseline Performance Summary

Across the completed rank sweep aggregate:

- k=1024 mean decode speed: 106.51% of baseline
- k=1536 mean decode speed: 79.21% of baseline
- k=2048 mean decode speed: 82.04% of baseline
- k=2048 mean prefill time: 163.14% of baseline

Interpretation:

- k=1024 is healthy and above baseline throughput in this pack.
- k=1536 and k=2048 decode retention remain below target gates for strong-claim publication.
- k=2048 prefill remains above the <=150% gate.

## Relative-to-Baseline Quality Summary

Perplexity (lower is better):

- Baseline: 6.7902
- GRC k=2048: 7.6936
- Relative PPL: 113.30% of baseline (approximately +13.30%)

Interpretation:

- Quality is currently outside the readiness target (+8% max delta).
- Throughput and quality can both be measured reproducibly, but the gate set is not yet cleared.

## Repeatability (5-run CI pack)

Coding prompt, 256 tokens:

- Baseline decode: 34.14 +/- 1.34 tok/s
- GRC decode: 31.68 +/- 3.33 tok/s
- GRC decode retention: 92.79% of baseline

Reasoning prompt, 256 tokens:

- Baseline decode: 34.84 +/- 0.25 tok/s
- GRC decode: 31.54 +/- 3.54 tok/s
- GRC decode retention: 90.53% of baseline

Lower-95 decode retention from validator:

- coding lower-95: 73.69%
- reasoning lower-95: 70.59%

Interpretation:

- Means are improved, but lower-bound gates still fail the >=75% criterion.

## White Paper Readiness Verdict

Current state: Not ready for a proper white paper submission.

Latest machine-validated gate status (paradigm shift validator):

- Validation artifact: benchmarks/whitepaper_pack_20260426_212526/paradigm_shift_validation.json
- Strong-claim ready: False
- Gate pass/fail:
  - k1024 decode >=95%: pass (106.51%)
  - k1536 decode >=85%: fail (79.21%)
  - k2048 decode >=75%: pass (82.04%)
  - k2048 prefill <=150%: fail (163.14%)
  - CI lower-bound decode >=75%: fail (coding 73.69%, reasoning 70.59%)
  - PPL delta <= +8%: fail (+13.30%)

Interpretation:

- The blocker is now a concrete 4-gate miss under a reproducible validator.
- Remaining blockers are k1536 decode retention, k2048 prefill inflation, CI lower-bound retention, and PPL delta.

What is complete:

- Relative quality degradation is measured and moderate (+7.56% PPL).
- Relative speed tradeoff is measured across multiple prompts and lengths.
- Rank sweep table (1024/1536/2048) is now completed and normalized to baseline.

Why this is not yet ready:

- The completed rank sweep shows a severe k=2048 slowdown in current runs.
- Evidence is single-model and single-hardware; publication-grade claims usually require at least multi-model or multi-hardware confirmation.
- No confidence intervals or significance testing are included yet.

Latest completed rank sweep aggregate (current run state):

- k=1024: decode 106.51% of baseline, overall 106.14%, prefill 101.24%
- k=1536: decode 79.21% of baseline, overall 76.11%, prefill 166.66%
- k=2048: decode 82.04% of baseline, overall 78.82%, prefill 163.14%

Interpretation:

- The catastrophic k=2048 collapse is no longer present, but the high-rank path still fails prefill and confidence-lower-bound gates.
- Publication-facing claims remain blocked until the remaining gate set is cleared.

## Outlier Investigation Update (coding/256)

Targeted repeated runs (6 reps each for coding and reasoning at 256 tokens) in this pack:

- coding decode retention (6-run mean): 91.03%
- reasoning decode retention (6-run mean): 89.04%

Interpretation:

- Prior catastrophic outlier behavior did not reproduce under the fixed harness.
- Remaining blockers are now gate-threshold margins and quality delta, not prompt-specific collapse.

## Coding Quality Note

Coding output quality remains a contextual note, not a demerit score, in this readiness pass.
The readiness decision is grounded in baseline-relative quantitative metrics (PPL, decode retention, overall retention, and variance bounds).

## Minimum Work to Reach Proper White Paper State

1. Raise k1536 decode retention from 79.21% to >=85% without worsening quality.
2. Reduce k2048 prefill from 163.14% to <=150%.
3. Lift CI lower-95 decode bounds from 73.69%/70.59% to >=75%.
4. Reduce PPL delta from +13.30% to <=+8%.
5. Re-run validator on a fresh pack and only then claim strong readiness.

## Claim Language Safe to Use Now

- "On Llama-3.1-8B-Instruct-Q4_K_M under the current harness, k=1024 and k=2048 decode gates pass while k1536 decode, k2048 prefill, CI lower-bound retention, and PPL delta still fail strong-claim thresholds."
- "Speed and quality claims are reported strictly as baseline-relative percentages from the latest validated pack (whitepaper_pack_20260426_212526)."
