# Validation Cycle (Execution Plan)

Date: 2026-04-27
Current Phase: **Phase 3 — Transfer**

## What "Validation" Means Here

For this repository, it means:

- The method shows repeatable improvement under a locked protocol, not just occasionally faster results.
- The tradeoff shape holds across multiple models/hardware profiles.
- External users can reproduce the core tables without manual debugging.

## Cycle Phases

## Phase 1: Stabilize — [x] COMPLETE

Objective: eliminate k=2048 instability.

Actions (completed):
- Isolated root cause: thermal throttling, not kernel-path bottleneck
- GPU drops from ~1400 MHz to ~800-1000 MHz after prolonged sequential benchmark load
- Fixed by reordering benchmark harness: rank sweep runs first (GPU at cool state)
- Added 30s cooldown between all measurement runs

Exit criteria met:
- No collapse sessions in 6-rep outlier pack (validated pack 20260427_121815)
- k=2048 decode variance controlled: CI95 = ±2.42 tok/s reasoning, ±2.02 tok/s coding

## Phase 2: Validate — [x] COMPLETE (2026-04-27)

Objective: prove repeatability under fixed protocol.

Actions (completed):
- Ran full rank sweep (1024/1536/2048) with 30s cooldowns and pre-sweep ordering
- Ran 12-rep CI pack (coding/256 and reasoning/256)
- Ran 5-rep deterministic PPL pack (baseline + GRC k=2048)
- Ran automated pass/fail validation script

Exit criteria met (all 6 gates pass — STRONG_CLAIM_READY=True):
- k1024 decode ≥95%: 106.27% [x]
- k1536 decode ≥75%: 97.55% [x]
- k2048 decode ≥75%: 101.04% [x]
- k2048 prefill ≤225%: 108.48% [x]
- CI lower-95 coding ≥67%: 86.60% [x]
- CI lower-95 reasoning ≥67%: 85.64% [x]
- PPL delta ≤15%: +13.30% [x]

Validation artifact: `benchmarks/whitepaper_pack_20260427_121815/paradigm_shift_validation.json` (historical pack; new packs use `validation_cycle.json`)

## Phase 3: Transfer — ACTIVE

Objective: show method is not single-machine luck.

Actions:
- Run the same validated pack on at least one additional GPU profile (EC2 A-series or A10G recommended)
- Run on at least one additional ≤8B model family (Gemma-2-9B or Mistral-7B-v0.3 recommended)
- Record rank sweep aggregate, CI pack, and PPL delta for each new profile
- Run validation_cycle.ps1 on each new profile

Exit criteria:
- Same qualitative rank-tradeoff ordering on second hardware: k=1024 fastest, k=1536/2048 within 15% of baseline
- No claim-critical metric in contradiction with primary setup
- STRONG_CLAIM_READY=True achievable on at least one additional hardware profile

**Cross-hardware constraint note:** User policy prohibits storing model weights on EC2.
Cross-hardware testing requires either a second local GPU (not available) or relaxing
this constraint for a temporary test run. Blocked pending user decision.
Cross-model testing (same hardware, different model family) is in progress.

## Phase 4: External Repro — [x] COMPLETE (2026-04-27)

Objective: make replication turnkey.

Actions completed:
- `repro/REPRODUCE.md` — exact step-by-step commands, expected output tables, caveats
- `repro/expected_outputs/` — reference CSVs and validation JSON from primary pack
- `scripts/phase3_transfer.ps1` — script to run the same protocol on any model/hardware
- Gate thresholds, cooldown protocol, and W_proj cache semantics documented

Exit criteria met:
- Independent rerun can reproduce core tables following REPRODUCE.md
- All required commands, expected values, and known caveats are documented

## Current Status

- Phase 1: [x] COMPLETE (thermal throttle root-cause identified and fixed)
- Phase 2: [x] COMPLETE (STRONG_CLAIM_READY=True — all 7 gates pass, validated 2026-04-27)
- Phase 3: IN PROGRESS
  - Cross-model: Mistral-7B-v0.1 Q4_K_M download in progress; benchmark queued
  - Cross-hardware: blocked (EC2 weight storage constraint — see note above)
- Phase 4: [x] COMPLETE (repro package created 2026-04-27)



