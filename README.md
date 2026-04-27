# HyperTensor

> **📄 Read the full research paper:** [nagusamecs.github.io/HyperTensor](https://nagusamecs.github.io/HyperTensor)
> &nbsp;&nbsp;·&nbsp;&nbsp; **📘 Whitepaper:** [docs/WHITEPAPER.md](docs/WHITEPAPER.md)
> &nbsp;&nbsp;·&nbsp;&nbsp; **🔁 Reproduce:** [repro/REPRODUCE.md](repro/REPRODUCE.md)

HyperTensor is a C11 inference runtime and compression research codebase centered on one practical question:

How far can we compress attention weights while keeping usable quality and predictable throughput on real hardware?

This repository currently has a working proof of concept for the 8B regime.

## What Is In This Repo

- Geodessical runtime: GGUF inference engine (Windows/Linux) with CUDA and CPU execution paths
- GRC pipeline: geometry-informed attention compression via weight-space PCA bases and projected weights
- Evaluation tools: perplexity and benchmark scripts used to measure quality/speed tradeoffs

## Current Validated Status (8B, April 2026)

Reference model: **Meta-Llama-3.1-8B-Instruct-Q4_K_M**

**Throughput** (vs uncompressed baseline, locked 30-second cooldown protocol):

- k=1024: **106.27%** decode — *above baseline* (GPU L2 cache-fit effect)
- k=1536: **97.55%** decode — near-lossless throughput
- k=2048† decode: 101.04% (capped to k=1536 by `AXEX_MANIFOLD_K_MAX`)

**Quality** (WikiText-2, 512-token eval, deterministic across 5 runs):

- Baseline PPL: 6.7902
- GRC PPL k=1536: 7.6936  →  **+13.30%**

**Validation:** all 7 automated gates pass under the locked protocol.
Pack: `benchmarks/whitepaper_pack_20260427_121815/`.

**Compression** (attention Q/K/V only):

- Disk W_proj cache: 1,093 MB (k=1536)
- VRAM delta during decode: +36 MiB peak

See [docs/WHITEPAPER.md](docs/WHITEPAPER.md) §6 for full results, §7 for the cache-fit analysis,
§9 for the limitations table.

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
