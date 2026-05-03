# GTC Serious Draft (Problem-First, Paper-Grade)

Date: 2026-04-27
Scope: Decode-time OTT speculative path in HyperTensor, with GTC/GRC integration claims.
Intent: This is a complete problem inventory for publication-readiness decisions.

## Executive State (No Spin)

The system is implementation-complete for a working end-to-end path on SmolLM2-135M,
but not publication-complete for strong deployment claims.

What is already true:
- Build is clean and reproducible.
- OTT speculative decode executes end-to-end in host runtime.
- Runtime telemetry export works (`test_live.tsv` is populated).
- Core geometry/Jacobi static results are strong (coverage/resonance/record-store).

What is not yet true:
- Speculative acceptance is high enough for robust deployment claims.
- GRC online feedback is demonstrably improving decode behavior in measured runs.
- Evaluation breadth is sufficient for publication-grade generalization claims.

## Hard Problems You Asked To Solve

### P1. Acceptance rate is low in live OTT decode

Observed evidence:
- Verified runtime run shows `acceptance_rate=8.3%` with
  `geo_accepted=2`, `xfmr=22`, `od_drafts=3` for 24 tokens.

Why this is a publication blocker:
- A low acceptance regime weakens the practical speedup argument for live substitution.
- It is currently easy for reviewers to classify the result as "works, but mostly falls back."

Root-cause candidates already visible in repo behavior:
- Draft proposal quality (OD draft token quality mismatch to verifier distribution).
- Dynamic batch schedule and threshold policy under-tuned (`--ott-spec-batch`, `--ott-spec-thresh`).
- Early-turn context mismatch effects (prompt/template/tokenization interactions).
- Sparse local manifold support at runtime (nearest anchors too far for confident acceptance).

What must be shown to remove blocker:
- Acceptance curve vs thresholds/batch policy with confidence intervals.
- Net decode TPS gain attributable to accepted drafts, not just optimistic projection.
- Stability across prompt families, not one smoke prompt.

### P2. GRC online feedback not materially contributing yet

Observed evidence:
- Run output includes `GRC online feedback: ok=0 fail=0 skip=22 (grc_count=0)`.

Why this is a publication blocker:
- Current narrative suggests compositional GRC+GTC behavior, but measured online
  contribution is negligible in the referenced run.

Root-cause candidates already visible in repo behavior:
- Feedback path gates too strict for early sessions.
- `step_fast` / curvature accumulation needs longer warmup before nonzero contribution.
- Runtime conditions for feedback eligibility are rarely met in short decode windows.

What must be shown to remove blocker:
- Measured increase in `grc_count`/`ok` over turns and sessions.
- Ablation: with-feedback vs without-feedback on same prompts/seeds.
- Demonstrable impact on acceptance/TPS/quality, not just counter increments.

### P3. Evaluation breadth is not yet paper-grade

Observed evidence:
- Live runtime success is currently anchored to a narrow set of smoke runs.
- Existing strong numbers are concentrated in static/offline geometry evaluations.

Why this is a publication blocker:
- Reviewers require robustness across prompts, seeds, models, and hardware envelopes.
- Single-path success can be interpreted as overfitting to one configuration.

What must be shown to remove blocker:
- Multi-prompt, multi-seed distributions for acceptance/TPS/quality.
- Cross-model runtime runs (at least small + medium families).
- Reproducible protocol with locked command lines and artifact pack.

## Full Problem Inventory (Publication Readiness Matrix)

### A. Runtime Performance Claims

A1. Live acceptance currently too low for strong substitution claim.
- Severity: High
- Current state: Partially working, not yet performant enough.
- Needed artifact: acceptance/tok-s gain distributions with variance bounds.

A2. Verifier dominates too many steps in observed runs.
- Severity: High
- Current state: `xfmr` path still carries majority of tokens.
- Needed artifact: verifier-call reduction curve and net speedup decomposition.

A3. OD draft frequency/quality not yet calibrated per regime.
- Severity: High
- Current state: low draft count (`od_drafts=3` in representative run).
- Needed artifact: OD proposal precision/recall style diagnostics by step index.

### B. GRC Composition Claims

B1. Online feedback path is not yet measurably active in short-run telemetry.
- Severity: High
- Current state: skip-heavy, zero effective contributions in run cited.
- Needed artifact: longitudinal run showing nonzero contributions and benefit.

B2. Warmup and gating behavior are insufficiently characterized.
- Severity: Medium
- Current state: comments indicate expected delayed activation, but no hard curve.
- Needed artifact: warmup-length vs contribution curve with confidence intervals.

### C. Geometry-to-Runtime Bridging

C1. Offline geometry strength does not yet fully transfer to live decode substitution rates.
- Severity: High
- Current state: strong static validity/coverage, weak live acceptance.
- Needed artifact: bridge analysis mapping offline distances to live acceptance outcomes.

C2. Runtime cloud density mismatch remains a likely bottleneck.
- Severity: High
- Current state: prior docs acknowledge sparse runtime cloud effects.
- Needed artifact: density sweep showing acceptance sensitivity to cloud enrichment.

C3. Calibration parameters (`rho`, thresholds, batch policy) lack globally tuned defaults.
- Severity: Medium
- Current state: ad-hoc values work functionally, not yet optimized.
- Needed artifact: systematic grid/BO search with held-out validation prompts.

### D. Quality/Correctness Evidence

D1. Need stronger output quality analysis under speculation.
- Severity: High
- Current state: throughput and counters dominate current reporting.
- Needed artifact: perplexity/task-quality deltas for speculative vs baseline decode.

D2. Need error taxonomy for rejected drafts and correction paths.
- Severity: Medium
- Current state: aggregate counts exist; semantic failure classes not formalized.
- Needed artifact: per-step labeled rejection reasons and representative cases.

### E. Generalization and Reproducibility

E1. Prompt diversity currently insufficient for publication-level confidence.
- Severity: High
- Current state: smoke prompts used; broad families not yet locked.
- Needed artifact: stratified prompt suite (coding/reasoning/chat/structured output).

E2. Seed variance not yet fully reported for live runtime metrics.
- Severity: High
- Current state: point estimates dominate narrative.
- Needed artifact: multi-seed CIs for acceptance/TPS/quality.

E3. Cross-model runtime evidence incomplete for strong universality language.
- Severity: High
- Current state: strongest live proof is SmolLM2-focused.
- Needed artifact: at least 2 additional model families with same pipeline.

E4. External reproduction package for runtime OTT claims not finalized.
- Severity: Medium
- Current state: commands exist, but paper-ready bundle is incomplete.
- Needed artifact: frozen scripts + exact artifacts + one-command replay docs.

### F. Framing and Claim Hygiene

F1. Some docs still mix "theory-complete" and "deployment-complete" language.
- Severity: Medium
- Current state: mixed messaging across files can invite reviewer pushback.
- Needed artifact: one canonical claim table with scope tags.

F2. Strong positive sections need paired limitations in same narrative unit.
- Severity: Medium
- Current state: caveats exist but are not always colocated with headline claims.
- Needed artifact: claim-by-claim limitation and threat-to-validity table.

## Problem Dependencies (What Unlocks What)

- Solving P1 (acceptance) is prerequisite for strong runtime speedup claims.
- Solving P2 (GRC contribution) is prerequisite for compositional GRC+GTC claims.
- Solving P3 (breadth) is prerequisite for publication-grade generalization claims.

Dependency map:
- C2/C3 -> P1
- B2 -> P2
- E1/E2/E3 -> P3
- D1/D2 + F1/F2 are required to defend all three in peer review.

## Minimum Publishable Evidence Bundle (No New Theory Required)

To move from "working implementation" to "paper-grade":

1. Runtime acceptance package
- Acceptance/tok-s/quality over a stratified prompt suite and >=5 seeds.
- Parameter sweeps for threshold/batch policy.

2. GRC contribution package
- Longitudinal online-feedback runs showing nonzero measurable gains.
- With/without feedback ablation with matched conditions.

3. Generalization package
- Repeat runtime package on >=2 additional model families.
- Include one constrained hardware transfer run if feasible.

4. Reproducibility package
- Frozen command scripts, raw logs, summary tables, and exact env metadata.

## What Is Ready Right Now vs Not Ready

Ready now:
- Functional OTT runtime path and telemetry.
- Strong offline geometry/Jacobi/record-store results.
- Honest baseline for "works end-to-end" statement.

Not ready now:
- High-confidence live substitution efficiency claims.
- Strong compositional GRC+GTC online claims.
- Broad publication-level generalization claims.

## Decision Surface For Your Next Inputs

You said you will choose solutions and I will implement them. Use this as the menu:

- If you choose to prioritize runtime acceptance first:
  focus on P1 + C2/C3 + D2.

- If you choose to prioritize compositional story first:
  focus on P2 + B2 + D1.

- If you choose to prioritize publication confidence first:
  focus on P3 + E1/E2/E3 + F1/F2.

This draft is intentionally problem-first and complete enough to drive an implementation queue without additional discovery work.

---

## Implementation Status (P1 Track — D2 / C3 / C2)

**Implemented:** Session ending 2026-04-27

### D2 — Rejection Taxonomy 
- Runtime: `--ott-rejection-log <path>` flag added to `host/main.c`
  - Writes `geodessical_rejection_log_v1` TSV on every speculative verification
  - Columns: `step`, `draft_pos`, `draft_tok`, `draft_piece`, `verifier_tok`,
    `verifier_piece`, `n_drafts`, `warmup`, `source` (od / geo)
- Analysis script: `scripts/ott/rejection_taxonomy.py --log rejection_log.tsv`
  - Classifies each rejection into Type I (vocab mismatch), Type II (manifold
    divergence), Type III (early-turn context collapse)
  - Outputs console table + optional JSON report
- Smoke-run result on "What is the capital of France?" (24 tokens):
  - 1 rejection: Type II (draft "its" → verifier "working", pos=0, source=od)
  - Primary driver: **Type II** → Manifold Divergence
  - Action: GRC correction budget / C3 calibration sweep

### C3 — Calibration Sweep 
- Script: `scripts/ott/calibration_sweep.ps1`
  - Grid: `--ott-spec-thresh` ∈ {0.25, 0.35, 0.45, 0.55, 0.65, 0.75}
    × `--ott-spec-batch` ∈ {1, 2, 3, 4}
  - 5 locked validation prompts, configurable reps
  - Outputs `calibration_sweep_results.csv` + `calibration_sweep_report.txt`
    with Pareto-optimal operating point
- Run with:
  ```powershell
  .\scripts\ott\calibration_sweep.ps1 -MaxTokens 32 -Reps 2
  ```

### C2 — Cloud Density Bridge Analysis 
- Script: `scripts/ott/cloud_density.py`
  - Reads `test_live.tsv` (online cloud) + `axiom_beta_report.json` (offline)
  - Computes consecutive distances, nearest-neighbour distribution, offline-to-online
    density ratio, and enrichment recommendation
  - Optional matplotlib chart: `cloud_density_chart.png`
- Smoke-run result (24 online states, dim=576):
  - Consecutive cosine distance: mean=0.275, p90=0.477, max=0.801
  - NN distance: mean=0.132, p90=0.348 → moderate spread, manageable divergence
  - Offline axiom report had no `coverage_radius` field → bridge gap skipped
    (run `--ott-full` on a richer prompt set to populate axiom report)
- Run with:
  ```powershell
  .\.venv\Scripts\python.exe scripts\ott\cloud_density.py --telemetry test_live.tsv
  ```

### Next Steps
1. Run C3 calibration sweep to find the Pareto-optimal `(thresh, batch)` pair
2. Re-run with the new parameters and capture `rejection_log.tsv` at scale
3. Use D2 on the larger rejection log to confirm Type II dominance decreases
4. Consult C2 bridge gap once `axiom_beta_report.json` is enriched with `coverage_radius`

