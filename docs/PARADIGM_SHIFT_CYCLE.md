# Paradigm-Shift Cycle (Execution Plan)

Date: 2026-04-27
Current Phase: **Phase 3 — Transfer**

## What "Paradigm Shift" Means Here

For this repository, it means:

- The method shows repeatable improvement under a locked protocol, not just occasionally faster results.
- The tradeoff shape holds across multiple models/hardware profiles.
- External users can reproduce the core tables without manual debugging.

## Cycle Phases

## Phase 1: Stabilize — ✓ COMPLETE

Objective: eliminate k=2048 instability.

Actions (completed):
- Isolated root cause: thermal throttling, not kernel-path bottleneck
- GPU drops from ~1400 MHz to ~800-1000 MHz after prolonged sequential benchmark load
- Fixed by reordering benchmark harness: rank sweep runs first (GPU at cool state)
- Added 30s cooldown between all measurement runs

Exit criteria met:
- No collapse sessions in 6-rep outlier pack (validated pack 20260427_121815)
- k=2048 decode variance controlled: CI95 = ±2.42 tok/s reasoning, ±2.02 tok/s coding

## Phase 2: Validate — ✓ COMPLETE (2026-04-27)

Objective: prove repeatability under fixed protocol.

Actions (completed):
- Ran full rank sweep (1024/1536/2048) with 30s cooldowns and pre-sweep ordering
- Ran 12-rep CI pack (coding/256 and reasoning/256)
- Ran 5-rep deterministic PPL pack (baseline + GRC k=2048)
- Ran automated pass/fail validation script

Exit criteria met (all 6 gates pass — STRONG_CLAIM_READY=True):
- k1024 decode ≥95%: 106.27% ✓
- k1536 decode ≥75%: 97.55% ✓
- k2048 decode ≥75%: 101.04% ✓
- k2048 prefill ≤225%: 108.48% ✓
- CI lower-95 coding ≥67%: 86.60% ✓
- CI lower-95 reasoning ≥67%: 85.64% ✓
- PPL delta ≤15%: +13.30% ✓

Validation artifact: `benchmarks/whitepaper_pack_20260427_121815/paradigm_shift_validation.json`

## Phase 3: Transfer — ACTIVE

Objective: show method is not single-machine luck.

Actions:
- Run the same validated pack on at least one additional GPU profile (EC2 A-series or A10G recommended)
- Run on at least one additional ≤8B model family (Gemma-2-9B or Mistral-7B-v0.3 recommended)
- Record rank sweep aggregate, CI pack, and PPL delta for each new profile
- Run paradigm_shift_validate.ps1 on each new profile

Exit criteria:
- Same qualitative rank-tradeoff ordering on second hardware: k=1024 fastest, k=1536/2048 within 15% of baseline
- No claim-critical metric in contradiction with primary setup
- STRONG_CLAIM_READY=True achievable on at least one additional hardware profile

## Phase 4: External Repro

Objective: make replication turnkey.

Actions:
- publish command list and expected outputs
- publish benchmark artifact paths and validation report

Exit criteria:
- independent rerun can reproduce core tables with minimal friction

## Current Status

- Phase 1: ✓ COMPLETE (thermal throttle root-cause identified and fixed)
- Phase 2: ✓ COMPLETE (STRONG_CLAIM_READY=True — all 6 gates pass, validated 2026-04-27)
- Phase 3: ACTIVE — cross-hardware and cross-model transfer
- Phase 4: pending

## Near-Term Priorities

1. Set up EC2 transfer benchmark run (A10G or similar)
2. Run validated pack on Gemma-2-9B or Mistral-7B-v0.3
3. Update whitepaper with Phase 3 transfer results
4. Produce external reproduction package (Phase 4)
