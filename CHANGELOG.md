# Changelog

All notable changes to HyperTensor are documented in this file.
The focus here is code and measured behavior, not release-note marketing.

---

## [unreleased] — 2026-04-27 — "GTC v0.3: Scaling + Resonance + Record Store"

### Verified Results (this turn)

#### Three-model scaling — Paper 5 "flag flip" claim verified
| Model        | Params | Coverage @ k=16 (25%), ε=3.0 |
|--------------|-------:|-----------------------------:|
| SmolLM2-135M | 135M   | 91.0 %                       |
| Phi-3.5-mini | 3.8B   | 90.4 %                       |
| Gemma-4-E2B  | 4.5B   | 91.5 %                       |
Scale-invariant within ±0.5 %.

#### Batch Jacobi resonance (Paper 5 Tests 4a–4c on real LM data)
| B     | Speedup | Paper-5 target | rel.err |
|------:|--------:|---------------:|--------:|
| 10    | 97.9×   | 2.7×           | 1.1e-16 |
| 100   | 27.4×   | 12.5×          | 1.2e-16 |
| 1 000 | 44.5×   | 7.0×           | 1.2e-16 |
| 10 000| 60.0×   | (new)          | 1.2e-16 |
Source: `docs/figures/gtc/smollm2-135m_batch_jacobi.json`.

#### Compressed record store + two-stage lookup (Paper 5 Algorithm 1)
- 5.96 KB/record at k=8 (paper target: 50–80 KB at k=40 — well under budget).
- Rank-5 Φ SVD truncation: reconstruction error 0.0 (paper: "rank ≈ 5 sufficient" — verified).
- Two-stage Euclidean→g-norm lookup: 30.9 µs/query (paper target <5 ms — 160× faster).
- Source: `scripts/gtc/record_store.py`, `docs/figures/gtc/smollm2-135m_library.npz`.

### Code added
- `scripts/gtc/record_store.py` — compressed Library with two-stage lookup.
- `scripts/gtc/batch_jacobi.py` — Paper 5 §4.5 resonance benchmark.

### Paper 5 status
12 of 17 testable Paper 5 claims now have measured replicable results.
Remaining gaps: live decode replacement, AttnRes integration, and the
two open problems flagged by the paper itself (ϕ, v₀). See
`docs/figures/gtc/GTC_RESULTS.md` for the full gap analysis table.

---

## [unreleased] — 2026-04-22 — "GTC + PPL Sweep"

### Verified Results (this turn)

#### Llama-3.1-8B PPL sweep
| Case        | Rank | PPL      | % of baseline |
|-------------|-----:|---------:|--------------:|
| baseline    |   —  |  6.7902  | 100.00 %      |
| GRC k=1024  | 1024 | 10.9585  | 161.39 %      |
| GRC k=1536  | 1536 |  7.6936  | **113.30 %**  |
| GRC k=2048  | 2048 |  7.6936  | 113.30 %      |

k=1536 ≡ k=2048 (bit-identical PPL) because Llama-3.1 GQA K/V dim is 1024:
once k ≥ 1024 the K/V matrices are full-rank and only Q is being truncated;
Q's PCA energy saturates by k=1536. **k=1536 is the Pareto rank** for
`--axex-attn-only` on Llama-3.1-8B. Source: `docs/figures/ppl_sweep/`.

#### GTC v0.2 coverage on SmolLM2-135M
- k=16 cached samples (25 % of cloud) → **91.0 %** hit rate at ε=3.0.
- Validity radius: errors < 0.1 % out to ε=5.0.
- Sphere sanity (n=4) matches theoretical Jacobi quadratic scaling.
- Source: `docs/figures/gtc/GTC_RESULTS.md`.

#### Curvature-warp prototype (negative result)
- 32-config sweep over (strength, sigma, dl). 0/32 pass success criterion.
- Best improvement 16 %; spillover diverges at high strength.
- Mechanism: SmolLM2 manifold too flat for local Gaussian metric warp to
  redirect a geodesic without global side effects.
- Source: `docs/figures/curvature_warp/`, `scripts/curvature_warp/`.

#### Model inventory
SmolLM2-135M Q8_0 (138 MB), Mistral-7B Q4_K_M (4166 MB), Llama-3.1-8B
Q4_K_M (4693 MB) — `docs/figures/model_inventory.{json,md}`.

### Code added
- `scripts/run_ppl_sweep.ps1` — wrapper for the four-case PPL sweep.
- `scripts/gtc/gtc_benchmark.py` — coverage sweep harness (NEW).
- `scripts/curvature_warp/{inject.py,sweep.py}` — Paper 4 §3 prototype.
- `scripts/inventory_models.py` — GGUF inventory generator.
- `docs/figures/findings/FIVE_FINDINGS.md` — source-of-record for the five
  un-published findings (Phase 3 warm-cache, Phase 4 oracle-budget, Phase 5
  decode-aligned MRR, `axiom_warp_state.dat`, archived
  `WHITEPAPER_DIFFEOMORPHISM.md`).

---

## [0.6.1] — 2026-04-25 — "GRC 8B Milestone"

### Summary

This release records the first single-digit perplexity result for GRC on Llama-3.1-8B-Instruct-Q4_K_M while compressing attention weights by 50%.

### Verified Results

| Metric | Value |
|---|---:|
| Baseline PPL (WikiText-2, 512 tok) | 6.79 |
| GRC PPL (k=1024) | 10.6869 |
| GRC PPL (k=2048) | 7.1969 |
| Attention compression | 3072 MB -> 1536 MB |

### Key Technical Changes

- Added Frobenius-normalized weight-PCA accumulation for Q/K/V to avoid Q-dominance.
- Kept power iteration count at 3 (5 iterations was measurably worse on PPL).
- Enforced skip-O in practical runs for this path due to structural GQA mismatch.
- Added `axex_compress_rank` to W_proj cache hash to prevent stale cache reuse between k values.

### Notes

- Quality is now demo-ready for the 8B path.
- Throughput at k=2048 remains an optimization target.
- This is still a research milestone, not a production guarantee across all model families.

---

## [0.6.0] — 2026-04-18 — "Synapse"

### Summary

**Production host runtime with geometric inference research integration.** Geodessical
v0.6.0 "Synapse" ships as a fully featured host-mode inference engine while running the
Axiom Beta-3 OTT survey pipeline in parallel. Peak decode reaches **107.7 tok/s** on
Gemma 4 E2B (RTX 4070 Laptop), **22.7% ahead of Ollama gemma3:4b** on the same hardware.

### OTT / Axiom Beta-3

#### Decode-Aligned Oracle Targets (Phase 5)
Phase 5 geodesic pilot now uses deterministic model next-token generation as the target
embedding instead of random vocabulary selection. This aligns the pilot with actual
decode behavior and makes top1/MRR metrics directly meaningful.

Telemetry: `oracle_target_count` vs `random_target_count` in JSON report.

#### Persistent Warp State
Knowledge-injection warp accumulations now survive process restarts via
`axiom_warp_state.dat`. Warp points accumulated across sessions; threshold-triggered
manifold recomputation runs in post-Phase-5 control flow (no Phase-5 coupling).

#### Improved Phase 4 Active Learning
- Uncertainty-based candidate selection with early stop after sustained low uncertainty
- Adaptive model-oracle budget in fast mode: 2–4 calls (down from 16)
- Stricter fast-mode uncertainty floor for oracle trigger
- Result: Phase 4 wall time: 909 ms → **669 ms** (−26%)

#### Phase 5 MRR Improvement
Curvature-informed initial velocity prior in Phase 5 (bounded local acceleration from
interpolated Christoffel symbols). Adaptive geodesic retry with step/velocity damping.

| Metric | Previous (cap=16) | Current |
|--------|------------------|---------|
| Total time | ~1218 ms | ~977 ms |
| Phase 4 | ~909 ms | ~669 ms |
| Phase 5 | — | ~43 ms |
| MRR | ~0.032 | **~0.067** |

#### Phase 3 Warm-Cache
LRU hidden-state cache (keyed by token_id × layer) reduces Phase 3 manifold recomputation
from **197 s cold → 0.17 s warm** (−99.9%) on SmolLM2. Full Phase 3 + Phase 4 refresh
now triggerable without prohibitive cost.

#### Knowledge Injection Prototype
`enable_knowledge_injection`, `injection_alpha`, `injection_sigma`, `injection_points`
controls added. Applies OTT-style local Christoffel warp with Gaussian distance decay.
Warp accumulation + recalc trigger plumbing fully implemented; training-time coupling
pending.

#### Fast-Mode Clamp Policy
`--axiom-fast` activates:
- `embedding_samples ≤ 64`
- `metric_sample_points ≤ 64`
- `oracle_calls_max ≤ 12`
- `geodesic_test_tokens ≤ 8`
- `geodesic_vocab_probe ≤ 512`

#### Axiom Beta Benchmark Snapshot (Gemma 4 E2B, April 14, 2026)
| Config | Total | Phase5 | ID | top1 | MRR |
|--------|-------|--------|----|------|-----|
| samples=64, probe=256 | 543 ms | 59 ms | 14 | 0.000 | 0.0153 |
| samples=128, probe=512 | 1013 ms | 69 ms | 16 | 0.000 | 0.0000 |
| samples=256, probe=1024 | 3209 ms | 15 ms | 41 | 0.000 | 0.0000 |

### Performance (Geodessical Inference Engine)

| Metric | Value | Context |
|--------|-------|---------|
| Decode (Gemma4 E2B, GPU, long/512) | **107.7 tok/s** | RTX 4070 Laptop, decode-only |
| End-to-end (Gemma4 E2B, GPU) | **92.5 tok/s** | Includes prefill, 256 tokens |
| vs Ollama gemma3:4b | **+22.7%** | Same prompt, same hardware |
| vs Ollama gemma4:latest | **+206.2%** | Same prompt, same hardware |
| SmolLM2-135M GPU (long) | 174–271 tok/s | Q8_0, variable prompt length |

### New Features (Inference Engine)
- Gemma4 architecture: interleaved sliding-window attention (ISWA), dual RoPE bases, doubled FFN layers 15+
- `--ott-fast`: speed-first OTT (spec-decode batch=16, AttnRes, fast axiom, max TPS)
- `--ott-speculative`: geodesic spec-decode (batch=2, geodesic drafts + transformer verify)
- `--ott-perfect`: exact greedy rollout upper bound (100% draft acceptance rate)
- `--ott-full`: full OTT pipeline (axiom + geodesic-first + AttnRes + OneDecode prep)
- `--ott-theorem`: adds depth-attn to ott-full for maximum reasoning quality
- `--one-decode`: bake geodesic flow map once → `ott_one_decode.bin` for instant decode
- `--ott-od`: OTT-OD protocol — OneDecode map as speculative draft source
- `--ott-swarm <K>`: OD-SWARM fan-out (K candidates per draft slot)
- `--attnres` / `--attnres-strength`: attention residual depth stabilization
- `--depth-attn` / `--depth-attn-strength` / `--depth-attn-window`: depth-wise residual cross-layer attention
- `--no-think`, `--force-think`, `--show-think`: thinking token control for reasoning models
- CUDA: dynamic DLL dispatch (`cuda_kernels.dll`), ~50 GPU operations, CUDA Graph capture
- CUDA: fused QKV (triple_q4_0), batch prefill, add_rmsnorm, iswa_combine, async transfers
- CUDA: uploads Q4_0, Q4_1, Q8_0, Q6_K, F16, BF16, F32 (expanded from Q4_0/Q8_0 only)
- 13 JIT SSE2 kernels (added: gelu, layernorm, q8_0_gemv, q4_0_q8_0_gemv)
- `--axiom-gpu` flag: runs Phase 3/5 matrix ops on CUDA device
- `--ctx-size`: user override for context window size
- `--log-level`: verbosity control (0=quiet to 3=trace)
- OTT readiness report (`ott_readiness_report.json`) with subsection flags
- JSON axiom report: `phase5_geodesic.oracle_target_count`, `warp_points_accumulated`

---

## [0.4.0] — 2026-03-30

### Summary

**Host-mode runtime + CUDA GPU offload + speculative execution.** First release that
runs on Windows/Linux as a native host application without a bootable kernel image.
Introduced CUDA GPU dispatch (RTX 4070: 29% decode speedup over CPU), five speculative
execution techniques, and an HTTP API server for programmatic access.

### New Features

#### Host-Mode Runtime (HAL)
Full hardware abstraction layer for running as a host process:
- Memory-mapped model loading (GGUF mmap, no copy into heap)
- Native POSIX/Win32 threads replacing bare-metal SMP dispatch
- `host/main.c` CLI: flags for prompt, token count, temperature, GPU mode
- Cross-platform build: `build_host.ps1` / `build_host.sh`

#### CUDA GPU Offload
Selective dispatch to RTX/Quadro/A-series GPUs via CUDA runtime:
- Threshold: `out_dim ≥ 8192` (captures all large projection layers)
- Gemma-4 E2B: GPU decode ~14.5 tok/s at launch; improves to 92.5 tok/s by v0.5
- `cudaMemcpy` weight staging on first call; cached for subsequent tokens
- Fallback: CPU AVX2 path for sub-threshold layers

#### Speculative Neural Execution (SNE) — 5 Techniques
| Technique | Principle |
|-----------|-----------|
| **Adaptive Precision Cascade (APC)** | Low entropy inputs fast-path at INT16; high entropy re-runs at FP32 |
| **Speculative Layer Fusion (SLF)** | Skip matmul when layer input signature matches cached activation |
| **Entropy-Aware Neuron Pruning (EANP)** | Zero-entropy neurons pruned at runtime (no retraining) |
| **Compute DAG Scheduling** | Tomasulo-inspired tensor dependency DAG with resource ordering |
| **Confidence-Gated Early Exit** | Execution depth proportional to input difficulty |

These techniques operate within the SNE engine (`runtime/nn/speculative.c`) as
microarchitecture-level acceleration of neural inference, independent of the
OTT speculative decode path.

#### HTTP API Server
REST API on `localhost:8080`:
- `POST /v1/generate` — single-turn text completion
- `POST /v1/chat` — multi-turn conversation (OpenAI-compatible)
- `GET /v1/models` — list loaded models
- `GET /v1/version` — runtime version string

#### Additional Architectures
Added GGUF loaders for: **Qwen2.5**, **LLaMA 3**, **Gemma 2**, **SmolLM 2**, **Mistral**

#### Axiom Beta-1 (Initial Research Build)
Placeholder geometry survey with architecture-heuristic manifold ID and surrogate
curvature metrics. Not yet using real model weights. Serves as integration scaffold for
Beta-2 real-geometry implementation.

### Performance

| Metric | v0.3.0 | v0.4.0 | Improvement |
|--------|--------|--------|-------------|
| Decode speed (CPU) | 162 ms/tok | 138 ms/tok | 1.2× |
| Decode speed (GPU) | N/A | 69 ms/tok | GPU enabled |
| Host binary size | N/A | ~1.1 MB | Host target |
| Supported models | 2 | 7 | +5 architectures |

---



### Summary

**2.8× performance improvement.** Decode speed improved from 454 ms/tok to 162 ms/tok
through SMP parallel GEMV, JIT-compiled forward kernels, and critical bug fixes in
both the JIT loop counters and the SMP trampoline.

### Critical Fixes

#### SMP Page Table Relocation
Page tables relocated from `0x1000` to `0x10000` (18 pages). The original address
collided with the BIOS Data Area, causing AP bootstrap failures. All 4 CPUs now
come online reliably.

#### JIT Loop Counter Bugs
All JIT forward kernels (`vadd`, `dot`, `vmul`, `rmsnorm`) emitted `vecs/2` as the
loop count instead of `vecs`. This halved the computation, producing incorrect results
for every JIT-accelerated operation.

#### SMP Trampoline Stack Alignment
Changed AP entry from `jmp rax` to `call rax` to ensure 16-byte stack alignment
required by the System V ABI. Misalignment caused SSE2 `movaps` faults on APs.

### New Features

#### JIT Forward Kernels
Six native x86_64 kernels lazy-compiled on first LLM inference:
- `vadd` (dim=3072) — residual connections
- `dot` (head_dim=96) — attention score computation
- `axpy` (head_dim=96) — attention value accumulation
- `fused_silu_mul` (ff_dim=8192) — FFN gate ⊙ up projection
- `rope` (head_dim=96) — rotary position encoding
- `rmsnorm` (dim=3072) — RMS normalization

Emitted into a 2 MB W^X code pool (max 64 concurrent buffers).

#### SMP Parallel GEMV
- `smp_dispatch()` partitions GEMV rows across all online CPUs
- Supports Q4_0 and Q8_0 fused AVX2 GEMV paths
- Dispatches when `ncpu > 1 && out_dim >= 64`
- BSP + APs synchronized via `smp_wait_all()`

#### AVX2 Integer SIMD Emitters
7 new AVX2 integer SIMD instruction emitters added to the JIT engine.
Integer Q4×Q8 GEMV compiler implemented (disabled pending correctness verification).

### Performance

| Metric | v0.2.0 | v0.3.0 | Improvement |
|--------|--------|--------|-------------|
| Decode speed | 454 ms/tok | 162 ms/tok | 2.8× faster |
| CPUs used | 1 | 4 | SMP dispatch |
| JIT kernels | 0 | 6 | Forward pass JIT |
| JIT pool | 1 MB | 2 MB | Doubled capacity |

---

## [0.2.0] — 2026-03-23

### Summary

**First coherent LLM inference achieved.** Phi-3.5 Mini Instruct (3.8B params, Q4_0)
now generates correct English text on bare-metal x86_64, running under QEMU WHPX at
~800 ms/tok (454 ms/tok decode, 5.5s prefill for 12 tokens).

Prompt: `"What is an operating system?"`
Output: `"An operating system (OS) is a complex piece of software that man..."`

This release fixes critical numerical bugs in quantized inference that produced
garbage output since the initial LLM integration.

### Critical Fixes

#### Q4_0 Dequantization Layout (Root Cause of Garbage Output)
All Q4_0 (4-bit quantized) dequantization code used an **interleaved nibble layout**
instead of the GGML standard layout. For each 32-element block packed into 16 bytes:

- **Wrong (interleaved):** `out[2*j] = lo_nibble, out[2*j+1] = hi_nibble`
- **Correct (GGML standard):** `out[j] = lo_nibble, out[j+16] = hi_nibble`

This caused element-order corruption at every F32 boundary (RMSNorm multiply,
residual connections), compounding through all 32 transformer layers. Aggregate
statistics (min/max/sum) were identical since values were just permuted within
blocks, making the bug invisible to earlier verification.

**Files fixed:**
- `runtime/nn/llm.c` — `llm_embed()` Q4_0 case
- `runtime/nn/llm.c` — `q4_0_dot32()` (SSE2 + aarch64 paths)
- `runtime/nn/llm.c` — `q4_1_dot32()` (SSE2 + aarch64 paths)
- `runtime/nn/llm.c` — `q4_0_dot32_avx2()` (AVX2 8-wide path)
- `runtime/nn/llm.c` — `llm_gemv_q4_fused_avx2()` (4-row batched GEMV)
- `runtime/nn/llm.c` — `llm_gemv_q4_fused_range_avx2()` (parallel worker GEMV)
- `runtime/nn/llm.c` — AVX2 helper replaced: `q4_unpack_v8f` → `q4_unpack_lo_v8f` + `q4_unpack_hi_v8f`

#### RMSNorm Epsilon Mismatch
Hardcoded `1e-6f` replaced with model-specific epsilon loaded from GGUF metadata
(`general.rms_norm_eps`). Phi-3.5 uses `1e-5`.

#### Math Library Precision (Prior Session)
Complete rewrites of `sinf`, `cosf`, `expf`, `logf`, `sqrtf` — custom bare-metal
implementations had catastrophic precision errors affecting RoPE frequency computation
and softmax normalization.

### New Features

#### LongRoPE Scaling Factors
- Loaded `rope_factors_short` and `rope_factors_long` tensors from GGUF
- Applied frequency scaling in `llm_rope_precompute()` based on position vs
  original context length (4096 for Phi-3.5)
- Enables correct positional encoding for extended-context models

#### New Subsystems (Scaffolding)
- `runtime/nn/flash_attn.c` — Flash Attention kernel interface
- `runtime/nn/paged_attn.c` — PagedAttention (vLLM-style) interface
- `runtime/nn/safetensors.c` — Safetensors format loader
- `runtime/nn/onnx.c` — ONNX Runtime integration interface
- `runtime/compute/vulkan_compute.c` — Vulkan/WebGPU compute backend interface
- `runtime/pseudocode/pseudo_stdlib.c` — Pseudocode standard library
- `kernel/drivers/dma/pcie_dma.c` — PCIe DMA engine
- `kernel/net/distributed.c` — Distributed inference networking
- `boot/uefi_stub.c` — UEFI boot support

### Performance

| Metric | Before | After |
|--------|--------|-------|
| Decode speed | N/A (garbage) | 454 ms/tok |
| Prefill (12 tok) | N/A | 5,475 ms |
| End-to-end (16 tok) | N/A | 12,793 ms |
| Model load | 5.5s | 5.5s |
| Binary size | ~808 KB | ~808 KB |

### Numerical Verification

All values now match a Python/NumPy reference implementation exactly:

| Checkpoint | Python | TensorOS | Match |
|------------|--------|----------|-------|
| Embedding abssum | 72.35 | 72.35 | ✅ |
| Embed[0] | -0.009949 | -0.009949 | ✅ |
| Q[0] after GEMV | -0.419630 | -0.419630 | ✅ |
| L0 output min | -3.5961 | -3.5961 | ✅ |
| L0 output max | 2.5063 | 2.5063 | ✅ |
| L0 output abssum | 135.21 | 135.21 | ✅ |

Previously, Q[0] was -0.014042 (30× too small) and L0 output range was
[-0.37, 0.37] instead of [-3.60, 2.51] — a 10× dynamic range loss.

---

## [0.1.0] — 2025 (Initial Release)

### Features
- Multiboot1 bootloader with x86_64 long mode + SSE2
- SMP bootstrap (INIT-SIPI-SIPI)
- Tensor-aware memory manager (heap + arena + slab)
- GGUF model format parser
- Complete transformer forward pass
- Q4_0, Q4_1, Q6_K, Q8_0 quantization support
- AVX2+FMA SIMD acceleration with 4-row batched GEMV
- x86_64 JIT code generator for Q8_0 GEMV kernels
- SentencePiece and BPE tokenizers
- Temperature sampling with top-k and nucleus filtering
- Virtio-blk and virtio-net drivers
- ARP/IPv4/UDP/ICMP network stack
- AI shell with 20+ commands
- Pseudocode JIT compiler (lexer, parser, IR, 4-tier optimization)
- Model package manager
- ARM64 / Raspberry Pi 4 boot stub
