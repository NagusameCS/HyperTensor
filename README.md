# HyperTensor

HyperTensor is a C11 inference runtime and compression research codebase centered on one practical question:

How far can we compress attention weights while keeping usable quality and predictable throughput on real hardware?

This repository currently has a working proof of concept for the 8B regime.

## What Is In This Repo

- Geodessical runtime: GGUF inference engine (Windows/Linux) with CUDA and CPU execution paths
- GRC pipeline: geometry-informed attention compression via weight-space PCA bases and projected weights
- Evaluation tools: perplexity and benchmark scripts used to measure quality/speed tradeoffs

## Current Proof-Of-Concept Status (8B)

Reference model:

- Meta-Llama-3.1-8B-Instruct-Q4_K_M

Quality (WikiText-2, 512-token eval):

- Baseline PPL: 6.7902
- GRC PPL (k=2048): 7.1969
- Relative PPL: 106.00% of baseline (about +6.00%)

Compression (attention path used in this setup):

- Q/K/V attention weights: 3072 MB -> 1536 MB (50% reduction)

Speed note:

- Current completed rank sweep shows stable behavior at k=1024 and k=1536, while k=2048 is currently regressed in this branch.
- Throughput must be measured without live terminal piping. Redirect to files for reliable numbers.

## Important Interpretation Notes

- Coding quality is tracked as an observation, not scored as a demerit in this phase.
- In this repo state, coding outputs are useful for prototyping and demonstration workloads.
- Whitepaper conclusions are benchmark-driven and normalized against baseline throughput and PPL.

## Recommended Demo Command (k=2048)

```powershell
$MODEL = "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
.\build_host\geodessical.exe $MODEL --axex-compress --axex-attn-only --axex-skip-o --axex-weight-pca --axex-compress-rank 2048 -i
```

## Build

```powershell
.\build_host.ps1
```

## Benchmark Commands

No-pipe decode benchmark (avoid measurement artifacts):

```powershell
.\scripts\benchmark_decode_nopipe.ps1 -Model "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

## Documentation

- [docs/WHITEPAPER.md](docs/WHITEPAPER.md): full technical writeup, design reasoning, and measured tradeoffs
- [docs/WHITEPAPER_READINESS.md](docs/WHITEPAPER_READINESS.md): readiness criteria and publication gap checklist tied to current benchmark evidence
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): runtime architecture and dataflow
- [CHANGELOG.md](CHANGELOG.md): version and milestone record

## Scope Boundaries

What is working today:

- 8B attention-compression proof of concept with near-baseline quality
- reproducible local eval path
- practical caching workflow for repeated runs

What is still research:

- broad cross-model generalization claims
- full FFN compression with baseline-level quality retention
- publication-grade multi-hardware validation

## License

See [LICENSE](LICENSE).
