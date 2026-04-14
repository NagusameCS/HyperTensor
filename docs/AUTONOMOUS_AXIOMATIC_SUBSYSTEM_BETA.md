# Autonomous Axiomatic Subsystem (Beta)

This document defines the HyperTensor-hosted beta implementation of an autonomous axiomatic manifold subsystem.

Scope for this phase:
- HyperTensor-only integration (no TensorOS runtime changes yet)
- measurable, deterministic 5-phase pipeline
- model-specific report artifact for iterative R&D
- explicit separation between implemented behavior and future geodesic-native inference goals

## 1. Objective

Treat a trained model as a constrained mathematical world and derive a compact, testable axiom set describing that world.

In this beta, the subsystem does not replace standard transformer inference. Instead, it provides:
- manifold and symmetry survey primitives
- curvature/nonlinearity surrogate analysis
- axiom-set candidate scoring
- projected complexity comparison between standard transformer cost and geodesic-style cost

The intended outcome is reproducible diagnostics and a hard engineering base for future single-step geodesic inference work.

## 2. Design Principles

1. Deterministic first
- Every run is reproducible from a seed.
- Outputs are suitable for regression tracking.

2. Honest signal labeling
- Surrogate metrics are marked as surrogates.
- No claim of replacing forward pass until objective parity thresholds are met.

3. Incremental integration
- Additive beta module in runtime/nn.
- Optional CLI path via host main.
- No disruption to baseline inference paths.

4. Production constraints
- bounded runtime
- bounded memory
- graceful failure and explicit status codes

## 3. Five-Phase Pipeline (Implemented Beta Mapping)

### Phase 1: Manifold Identification

Implemented now:
- model context capture: dim, layers, vocab, parameter count
- intrinsic dimensionality estimate (architecture-scale heuristic)
- local metric rank estimate
- Fisher-trace proxy via deterministic perturbation energy sampling

Output fields:
- intrinsic_dim_estimate
- metric_rank_estimate
- fisher_trace_proxy
- uses_surrogate_metric

### Phase 2: Symmetry Extraction

Implemented now:
- deterministic random invariance probes
- invariance score accumulation
- symmetry generator count estimate

Output fields:
- symmetry_invariance_score
- symmetry_generators_estimate

### Phase 3: Nonlinearity Absorption Proxy

Implemented now:
- local nonlinearity gap proxy (SiLU vs linear local approximation)
- curvature_proxy score as nonlinearity concentration indicator

Output fields:
- curvature_proxy
- uses_surrogate_curvature

### Phase 4: Axiom Formalization

Implemented now:
- active-iteration candidate scoring loop
- acceptance thresholding
- minimal axiom count estimate
- consistency score estimate

Output fields:
- axiom_count_estimate
- axiom_consistency_score

### Phase 5: Native Inference Projection

Implemented now:
- projected baseline transformer complexity: O(n^2 * d * L)
- projected geodesic-style complexity: O(n * ID^2)
- projected_speedup ratio

Output fields:
- projected_transformer_cost
- projected_geodesic_cost
- projected_speedup
- supports_single_step_native_infer = 0 (explicitly disabled in beta)

## 4. CLI and Runtime Integration

New host CLI options:
- --axiom-beta-run
- --axiom-beta-only
- --axiom-report <path>
- --axiom-samples <n>
- --axiom-seed <n>

Execution flow:
1. Load model as normal.
2. If --axiom-beta-run is enabled, execute the 5-phase pipeline.
3. Emit summary to stdout.
4. Write JSON report to --axiom-report path (default: axiom_beta_report.json).
5. Continue normal inference, unless --axiom-beta-only is specified.

## 5. Report Schema

Top-level sections:
- subsystem
- model
- config
- phases
- timings_us

The schema is designed for machine ingestion in future regression dashboards.

## 6. Current Limitations

1. Hidden-state probes are not yet fully exposed through public runtime API.
2. Fisher metric and Christoffel symbols are approximated by surrogate measurements.
3. No geodesic solver is in the inference hot path.
4. Report is a diagnostics artifact, not a formal proof certificate.

These are intentional for Beta-1 and tracked as next milestones.

## 7. Validation Strategy

Near-term validation targets:
1. Stability
- same model + same seed => identical report values

2. Sensitivity
- altered model checkpoints should produce measurable report drift

3. Correlation
- report trends should correlate with measured decode throughput and perplexity changes

4. Safety
- subsystem failure must not affect baseline generation path

## 8. Roadmap to Geodesic-Native Inference

### Beta-2: Observability Upgrade
- expose internal hidden-state trajectory taps
- add local Jacobian probes on controlled token traces
- replace Fisher proxy with measured local Fisher blocks

### Beta-3: Geometry Core
- estimate local metric tensor field g_ij(x)
- compute approximate Christoffel symbols Gamma^u_vr from fitted metric patches
- add manifold charts and transition consistency checks

### Beta-4: Axiom Compiler
- formal candidate grammar for primitive/metric/symmetry/geodesic axioms
- active-learning oracle loop against model behavior
- automated contradiction pruning

### Beta-5: Geodesic Pilot
- limited-scope geodesic integrator for selected subspaces
- parity checks against baseline forward pass on bounded tasks
- acceptance criteria for moving from projection to partial native path

### Beta-6: Production Candidate
- optional hybrid runtime mode with guarded geodesic path
- strict accuracy gates, fallback guarantees, telemetry, and rollback hooks

## 9. Engineering Acceptance Gates

Before any claim of native single-step inference:
1. quality parity within agreed tolerance on benchmark suites
2. deterministic behavior across repeated runs
3. robust fallback to standard inference path
4. no regression in reliability or serving stability

## 10. Files Added in HyperTensor (Beta-1)

- runtime/nn/axiom_beta.h
- runtime/nn/axiom_beta.c
- host/main.c (CLI and runtime hook)
- build_host.ps1 (source wiring)
- docs/AUTONOMOUS_AXIOMATIC_SUBSYSTEM_BETA.md

## 11. Practical Use Right Now

Example command:

```powershell
.\build_host\hypertensor.exe <model.gguf> --axiom-beta-run --axiom-report axiom_beta_report.json --axiom-samples 4096 --axiom-seed 1337 -n 256
```

Survey only:

```powershell
.\build_host\hypertensor.exe <model.gguf> --axiom-beta-only --axiom-report axiom_beta_report.json
```

This gives a deterministic geometry/axiom diagnostics report while leaving baseline inference behavior intact.
