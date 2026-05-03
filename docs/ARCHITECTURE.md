# HyperTensor Architecture

This document explains how the runtime, compression path, and supporting infrastructure fit together in practical terms.

## 1. Top-Level Components

HyperTensor has two primary tracks in one repository:

- Geodessical host runtime
- Geometry/compression research path (GRC / OTT family)

For the current proof-of-concept work, Geodessical + GRC is the active execution path.

## 2. Runtime Layers

### 2.1 Host entry and orchestration

Primary host entry point:

- `host/main.c`

Responsibilities:

- CLI flag parsing
- model loading orchestration
- compression mode setup
- benchmark and eval routing

### 2.2 Inference core

Primary core files:

- `runtime/nn/llm.c`
- `runtime/nn/gguf.c`
- `runtime/nn/flash_attn.c`
- `runtime/nn/backend*.c`

Responsibilities:

- tokenization and decode loop
- KV cache management
- quantized weight execution
- backend dispatch (CPU/CUDA)

### 2.3 Compression and geometry modules

Primary files:

- `runtime/nn/axiom_exploit.c`
- `runtime/nn/axiom_linalg.c`
- `runtime/nn/axiom_beta.c`

Responsibilities:

- weight-space PCA basis generation
- projected weight construction and caching
- runtime projected-matvec path
- geometry survey and diagnostics

## 3. Dataflow Through The Compressed Path

For attention-only GRC mode:

1. Load model and map tensors.
2. Resolve or build compressed projection artifacts (Pt, W_proj).
3. Upload projected artifacts to GPU when GPU path is active.
4. During decode, project hidden state into the k-subspace and use projected attention weights.
5. Continue through normal decode/sampling flow.

Why this matters:

- It pushes heavy attention computation into a smaller working subspace.
- It introduces explicit quality/speed tradeoffs controlled by rank `k`.

## 4. Cache and Identity Controls

Compression artifacts are cached to avoid expensive recomputation.
A critical correctness guard is that cache identity includes rank and mode-relevant settings, so different ranks do not alias the same artifact.

## 5. CPU and GPU Execution Model

### CPU path

- Quantized matvec kernels
- AVX2/FMA optimized hot loops where available
- Threaded execution for larger workloads

### CUDA path

- Model weights and selected runtime buffers uploaded to VRAM
- GPU-resident decode path with fallback behavior retained
- Compression artifacts (Pt/W_proj) uploaded and reused

## 6. Benchmarking Philosophy In This Repo

Benchmark outputs are normalized to baseline and reported in percentages.
This keeps claims interpretable across conditions and avoids overemphasis on absolute tok/s values from a single run.

Practical rule used throughout:

- no live `Tee-Object` pipelines when measuring speed
- redirect stdout/stderr to files, then parse

## 7. Why Attention-Only First

The repo currently prioritizes attention compression because it gives a manageable engineering surface for:

- preserving quality
- keeping runtime behavior inspectable
- iterating quickly on rank and projection choices

FFN compression remains present in research paths but is not the main quality-retention claim in this proof-of-concept stage.

## 8. Current Risk Areas

Known areas that still need careful handling:

- first-run warmup cost for high-rank projected artifacts
- throughput variance under heavy or interrupted benchmark sessions
- rank-sweep completion robustness under long-running process conditions

## 9. Files To Start Reading

If you are new to the codebase and want the current active path quickly, start here:

1. `host/main.c`
2. `runtime/nn/llm.c`
3. `runtime/nn/axiom_exploit.c`
4. `runtime/nn/axiom_linalg.c`
5. `scripts/benchmark_decode_nopipe.ps1`

## 10. Design Intent

The design intent is pragmatic:

- keep the runtime operational and measurable
- expose compression controls explicitly
- make tradeoffs auditable with baseline-relative metrics

That is the architecture standard used for the present 8B proof-of-concept milestone.
