# Paradigm-Shift Cycle (Execution Plan)

Date: 2026-04-26

This cycle defines the minimum work needed to move from strong proof-of-concept to field-shifting claim strength.

## What "Paradigm Shift" Means Here

For this repository, it means:

- The method is reproducibly better under a locked protocol, not just occasionally faster.
- The tradeoff shape holds across multiple models/hardware profiles.
- External users can reproduce the core tables without manual debugging.

## Cycle Phases

## Phase 1: Stabilize

Objective: eliminate k=2048 instability.

Actions:
- isolate kernel-path bottlenecks and launch-overhead pathologies
- keep quality fixed while recovering decode retention
- rerun targeted coding/256 + reasoning/256 checks after each patch

Exit criteria:
- no collapse sessions in 6-rep outlier pack
- k=2048 decode retention variance controlled and interpretable

## Phase 2: Validate

Objective: prove repeatability under fixed protocol.

Actions:
- run full rank sweep (1024/1536/2048)
- run 5-run CI pack (throughput + PPL)
- run automated pass/fail validation script

Exit criteria:
- all `BENCHMARK_PROTOCOL.md` gates pass

## Phase 3: Transfer

Objective: show method is not single-machine luck.

Actions:
- run the same package on at least one additional hardware profile
- run on at least one additional <=8B model family

Exit criteria:
- same qualitative rank-tradeoff ordering
- no claim-critical metric in contradiction with primary setup

## Phase 4: External Repro

Objective: make replication turnkey.

Actions:
- publish command list and expected outputs
- publish benchmark artifact paths and validation report

Exit criteria:
- independent rerun can reproduce core tables with minimal friction

## Current Status

- Phase 1: in progress (k=2048 still regressed in latest completed sweep)
- Phase 2: mostly complete for artifacts, not complete for gates due k=2048
- Phase 3: pending
- Phase 4: pending

## Near-Term Priorities

1. finish k=2048 stabilization
2. rerun full validation package
3. produce gate report and update whitepaper claims strictly from latest gate status
