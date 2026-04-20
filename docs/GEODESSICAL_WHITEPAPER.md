# Geodessical — System Whitepaper
## High-Performance GGUF Inference Runtime with Bare-Metal OS Integration
### Version 0.6.0 "Synapse" — April 2026
**Author**: NagusameCS | **Project**: HyperTensor

---

## Abstract

Geodessical is a C11 inference runtime for GGUF-format large language models that operates both as a host application on Windows/Linux and as the inference core of TensorOS — a bare-metal x86_64 AI operating system. The system achieves **107.7 tok/s decode** (92.5 tok/s end-to-end) on Gemma 4 E2B Q4_0 using an RTX 4070 Laptop GPU, outperforming Ollama on the same hardware by 22.7% on a 256-token generation task.

Geodessical integrates JIT-compiled SIMD kernels, SMP-parallel GEMV, selective CUDA offload, and five speculative execution techniques into a ~1.1 MB binary with zero external runtime dependencies. The same inference core additionally powers the Axiom Beta-3 research pipeline — an autonomous geometric survey system implementing early components of Organic Training Theory (OTT).

---

## 1. System Overview

### 1.1 Three-Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HyperTensor                                │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   Geodessical   │  │    TensorOS     │  │        OTT          │ │
│  │ Host Inference  │  │  Bare-Metal OS  │  │  Research Runtime   │ │
│  │    Runtime      │  │  Kernel x86_64  │  │  Axiom Beta-3       │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
│           │                   │                       │             │
│           └───────────────────┴───────────────────────┘             │
│                     Shared Inference Core                           │
│         (GGUF parser, BPE tokenizer, JIT, SMP GEMV, CUDA)          │
└─────────────────────────────────────────────────────────────────────┘
```

All three systems share the same `runtime/nn/` inference core. Geodessical adds a HAL for host-mode execution; TensorOS boots the same core on bare metal; OTT research runs on top of both.

### 1.2 Supported Model Architectures

| Architecture | Representative Models | Quantization Formats |
|---|---|---|
| LLaMA 3 | Meta-Llama-3-8B-Instruct | Q4_0, Q8_0, F16 |
| Qwen2.5 | Qwen2.5-7B-Instruct | Q4_0, Q8_0, F16 |
| Gemma 2 | gemma-2-2b-it | Q4_0, Q8_0 |
| Gemma 4 | gemma-4-E2B-it | Q4_0, Q8_0 |
| Phi-2 / Phi-3 / Phi-3.5 | Phi-3.5-mini-instruct | Q4_0, Q8_0 |
| SmolLM 2 | SmolLM2-135M-Instruct | Q8_0 |
| Mistral | Mistral-7B-Instruct | Q4_0, Q8_0 |

---

## 2. Architecture Deep Dive

### 2.1 GGUF Model Loading

GGUF is the binary model format from llama.cpp. Geodessical's loader (`runtime/nn/gguf.c`):

1. **Header parse**: magic `GGUF`, version, tensor count, KV metadata count
2. **Metadata read**: architecture string, context length, rope parameters, RMSNorm epsilon, GQA groups, vocab size — stored in `gguf_kv_t` hash table
3. **Tensor index**: name → {type, shape, byte_offset} hash table (O(1) lookup)
4. **Memory mapping**: `mmap` (Linux) / `MapViewOfFile` (Windows) for zero-copy weight access
5. **Tokenizer**: BPE vocabulary loaded from `tokenizer.ggml.tokens` with O(1) hash table lookup

### 2.2 Transformer Forward Pass

For each decode token:

```
x = embed(token)                              # Embedding lookup (Q4_0 dequant)
for layer in 0..L:
    x_norm = rmsnorm(x, w_attn_norm)          # JIT kernel: rmsnorm (dim=3072)
    q,k,v = gemv(x_norm, Wq,Wk,Wv)           # SMP AVX2 GEMV / CUDA (large layers)
    q,k = rope(q, k, pos)                     # JIT kernel: rope (head_dim=96)
    kv_cache[pos] = (k, v)                    # KV cache write
    scores = dot(q, kv_cache.k) / sqrt(head_dim)  # JIT kernel: dot
    weights = softmax(scores)                  # Scalar (variable length)
    attn = dot(weights, kv_cache.v)           # JIT kernel: axpy
    x = x + gemv(attn, Wo)                    # Residual + projection
    x_norm = rmsnorm(x, w_ff_norm)            # JIT kernel: rmsnorm
    gate = gemv(x_norm, Wgate)                # GEMV (CUDA or CPU AVX2)
    up = gemv(x_norm, Wup)                    # GEMV
    x = x + gemv(fused_silu_mul(gate, up), Wdown)  # JIT: fused_silu_mul
x_norm = rmsnorm(x, w_out_norm)
logits = gemv(x_norm, Wembed.T)              # Tied embedding or output proj
token = sample(logits, temp, top_k, top_p)
```

### 2.3 JIT-Compiled Kernels

The JIT engine (`runtime/jit/llm_jit.c`) emits SSE2 native code (4-wide `v4f` vectors) at first inference. AVX2 is used in separate GEMV helper paths (`llm_gemv_q4_fused_range_avx2`).

| Kernel | Operation |
|--------|-----------|
| `fast_exp` | Fast scalar exp (building block for silu/gelu/softmax) |
| `silu` | SiLU activation: x / (1 + e^{−x}) |
| `rmsnorm` | RMSNorm: x / √(mean(x²) + ε) · w |
| `rope` | Rotary position encoding (non-Gemma4 only; SWA head_dim guard) |
| `elmul` | Element-wise multiply a[i] × b[i] |
| `eladd` | Element-wise add a[i] + b[i] |
| `dot` | Σ a[i]·b[i] → scalar |
| `axpy` | a[i] += α·b[i] |
| `fused_silu_mul` | silu(a[i]) · b[i] in one pass |
| `gelu` | GELU activation |
| `layernorm` | LayerNorm: (x − μ) / σ · w + b |
| `q8_0_gemv` | Q8_0 quantized GEMV |
| `q4_0_q8_0_gemv` | Q4_0 × Q8_0 integer GEMV |

JIT pool: 2 MB W^X executable memory, up to 64 concurrent buffers. Buffers cached by signature (op type × dimension).

Known planned extensions (JIT-TODOs in source): softmax kernel, Gemma4-safe RoPE (blocked on lhd≠hd guard), LRU GEMV kernel cache, fused RMSNorm+scale, batched GEMV, AVX-512 variant.

### 2.4 SMP Parallel GEMV

```
smp_dispatch(gemv_worker, row_range):
    CPU 0 (BSP) → rows [0, R/N)
    CPU 1 (AP)  → rows [R/N, 2R/N)    via IPI 0xFE
    CPU 2 (AP)  → rows [2R/N, 3R/N)   via IPI 0xFE
    CPU 3 (AP)  → rows [3R/N, R)       via IPI 0xFE
smp_wait_all()   ← barrier
```

Activates when: `ncpu > 1 && out_dim >= 64`

Each worker calls `llm_gemv_q4_fused_range_avx2()` — a 4-row batched AVX2 GEMV that processes 4 output rows per inner loop, accumulating with FMA instructions.

### 2.5 CUDA GPU Offload

CUDA acceleration is implemented as a **dynamically loaded dispatch table** (`cuda_kernels.dll` / `cuda_kernels.so`), compiled separately by nvcc and loaded at startup via `LoadLibraryA`/`dlopen`. The main binary has zero compile-time CUDA dependencies and falls back to CPU if the DLL is absent.

At model load, all weight matrices for supported quantization types are uploaded to GPU VRAM and registered in a 512-entry host→device pointer map:

```
Supported types: Q4_0, Q4_1, Q8_0, F32, F16, Q6_K, BF16
GPU weight map: up to 512 entries (host_ptr → dev_ptr lookup)
```

The dispatch table exposes ~50 GPU operations:

| Category | Operations |
|----------|-----------|
| Arithmetic | gemv, gemm, add, mul, scale, dot, dequantize |
| Activations | silu, gelu, fused_swiglu, fused_geglu, gelu_mul, softcap |
| Normalization | rmsnorm, layernorm, batched_rmsnorm, add_rmsnorm, rmsnorm_add |
| Attention | fused_qk_norm_rope, v_norm, attention, kv_update, embed |
| Fused | gemv_dual_q4_0, gemv_triple_q4_0, fused_rmsnorm_triple_q4_0 |
| Async transfer | gemv_async, upload_async, download_async, stream_sync |
| Batched prefill | batch_fused_qk_norm_rope, batch_v_norm, batch_kv_update, prefill_attn_batched |
| CUDA Graph | graph_begin_capture, graph_end_capture, graph_launch, graph_destroy |
| Utilities | set_decode_pos, argmax |

**CUDA Graph**: After the first prefill, the decode-step kernel sequence is captured as a CUDA Graph and replayed on each subsequent token, eliminating per-token kernel-launch overhead.

**GPU-resident forward pass**: Entire transformer loop on GPU with no CPU round-trips for supported models. Gemma4 uses CUDA GEMV offload mode (GPU-resident pass disabled pending ISWA quality validation).

**Batched prefill**: Multiple prompt tokens processed in a single batched GPU pass. Replaces the per-token attention loop with `cuda_prefill_attn_batched` for significant prefill speedup.

**ISWA (Gemma4)**: Interleaved sliding-window attention with GPU-resident scratch buffers (`d_iswa_per_layer`, `d_iswa_proj`), eliminating per-layer CPU round-trips for ISWA gating.

### 2.6 Speculative Execution (SNE — Speculative Neural Execution)

The speculative engine in `runtime/nn/speculative.c` implements five techniques from computer architecture applied to neural inference:

| Technique | Principle | Implementation |
|-----------|-----------|----------------|
| **Adaptive Precision Cascade (APC)** | Easy inputs confident at INT16; ambiguous inputs escalate to FP32 | Shannon entropy gate: H < threshold → accept INT16, else re-run FP32 |
| **Speculative Layer Fusion (SLF)** | Temporal coherence in activations enables speculative reuse | If layer input signature matches cached value, skip the matmul |
| **Entropy-Aware Neuron Pruning (EANP)** | Shannon entropy quantifies neuron usefulness at runtime | Dead neurons (zero entropy) pruned without retraining |
| **Compute DAG Scheduling** | Tomasulo's algorithm applied to tensor ops | Dependency DAG of tensor operations, monotonic resource ordering for deadlock-free execution |
| **Confidence-Gated Early Exit** | Execution depth proportional to input difficulty | Easy inputs exit after 1–2 layers; hard inputs run full depth |

Together these form the **Speculative Neural Execution (SNE)** engine — the first implementation of CPU microarchitecture principles (speculation, out-of-order execution, branch prediction, dynamic scheduling) applied to neural inference at the OS level.

Additionally, the OTT inference modes (see Section 5) implement a separate transformer-level speculative decode path: geodesic drafts verified by the transformer in a batch.

---

## 3. Performance Results

### 3.1 Test Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 7940HS (8 cores / 16 threads, 5.4 GHz boost) |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6) |
| RAM | 32 GB DDR5 |
| OS | Windows 11 (host mode) |
| Build | MSVC + zig cc, AVX2+FMA, CUDA 12.x |

### 3.2 Inference Benchmarks (April 13–14, 2026)

**Gemma 4 E2B Q4_0 (2B params, 3.2 GB)**:

| Engine | Mode | Decode tok/s | E2E tok/s | TTFT ms |
|--------|------|-------------|-----------|---------|
| Geodessical | GPU (256 tok) | **107.7** | **92.5** | ~350 |
| Geodessical | GPU (512 tok) | **88.4** | ~80 | ~700 |
| Geodessical | CPU only | ~11 | ~9 | ~800 |
| Ollama gemma3:4b | GPU | — | 75.36 | — |
| Ollama gemma4:latest | GPU | — | 30.21 | — |

**SmolLM2-135M Q8_0 (138 MB)**:

| Prompt Length | Decode tok/s | Prefill tok/s |
|---|---|---|
| Short (32 tok) | 228–266 | >500 |
| Long (512 tok) | 174–271 | 180 |

### 3.3 Comparative Analysis

| Comparison | Geodessical | Competitor | Delta |
|---|---|---|---|
| vs Ollama gemma3:4b | 92.5 tok/s | 75.36 tok/s | **+22.7%** |
| vs Ollama gemma4:latest | 92.5 tok/s | 30.21 tok/s | **+206.2%** |

The performance advantage comes primarily from:
1. Tighter CUDA dispatch (no Python overhead, no GGML compute graph)
2. AVX2 FMA SIMD on CPU layers (custom kernels vs llama.cpp defaults)
3. SMP parallel GEMV on small layers that CUDA skips

### 3.4 Efficiency Metrics

| Engine | Mode | tok/s per Watt |
|--------|------|----------------|
| Geodessical | GPU (long/512) | 1.408 |
| Ollama | GPU (long/512) | 1.825 |

Geodessical's lower efficiency at long prompts reflects lower GPU utilization (0–84% vs Ollama's 71–88%) — primarily because Geodessical dispatches fewer layers to GPU. Closing this gap is a target for v0.6.

---

## 4. TensorOS Integration

### 4.1 Bare-Metal Boot

```
BIOS → multiboot_stub.asm (Multiboot1) → boot.asm (long mode) → entry64.asm → kernel_main()
```

The same inference core that runs as a host application is linked into `kernel64.bin` and booted from a Multiboot1-compatible loader. Page tables at `0x10000` (18 pages, 16 GB identity map with 2 MB huge pages). Kernel loaded at `0x200000`.

### 4.2 Memory Layout (Bare Metal)

```
0x10000    Page tables (18 pages, PML4+PDPT+8×PD)
0x200000   Kernel code + data
0x8000     SMP trampoline
           AP stacks (65 KB each, up to 64 CPUs)
[dynamic]  Tensor heap (bump + free-list)
[dynamic]  Model weight LRU cache (64 slots)
[dynamic]  JIT code pool (2 MB, W^X)
```

### 4.3 SMP on Bare Metal

BSP sends INIT-SIPI-SIPI to up to 64 application processors. Each AP:
1. Enters real mode at `0x8000`
2. Transitions to protected → long mode
3. Gets 65 KB stack
4. Increments `smp.ap_started` counter
5. Enters idle loop, waiting for IPI 0xFE work dispatch

Under bare metal, `smp_dispatch()` calls the same `gemv_worker` functions used in host mode. The only difference is the synchronization primitive (spinlock on AP mailbox vs pthread mutex).

### 4.4 AI Shell

TensorOS boots to an interactive AI shell (`aishell_main()`) with commands:
- `load <model.gguf>` — load model into LRU cache
- `generate <prompt>` — run inference
- `axiom-beta` — run OTT geometric survey
- `bench` — throughput benchmark
- `smp` — show CPU topology
- `mem` — show memory allocation state

---

## 5. HTTP API

Geodessical exposes an OpenAI-compatible REST API on `localhost:8080`:

### POST /v1/generate

```json
{
  "model": "gemma4-e2b",
  "prompt": "Explain quantum entanglement.",
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

Response:
```json
{
  "text": "Quantum entanglement is...",
  "tokens_generated": 256,
  "decode_tok_s": 92.5,
  "ttft_ms": 348
}
```

### POST /v1/chat

OpenAI-compatible `messages` array. Conversation history tracked via KV-cache prefix reuse.

### GET /v1/models

Lists all loaded models with arch, quantization, param count, and memory footprint.

---

## 6. Build and Usage

### 6.1 Host Mode (Windows/Linux)

```powershell
# Build
.\build_host.ps1

# Run inference
.\build_host\geodessical.exe model.gguf -p "Your prompt here" -n 256

# Run with GPU
.\build_host\geodessical.exe model.gguf -p "prompt" -n 256 --cuda

# Run OTT axiom survey
.\build_host\geodessical.exe model.gguf --axiom-beta-only --axiom-fast --axiom-gpu -v

# Start HTTP API server
.\build_host\geodessical.exe model.gguf --serve --port 8080
```

### 6.2 Bare-Metal (TensorOS)

```powershell
# Build kernel image
.\build.ps1

# Build Raspberry Pi 4 image
.\build_rpi.ps1

# Boot under QEMU
qemu-system-x86_64 -kernel kernel64.bin -m 8G -smp 4 -accel whpx
```

### 6.3 Key CLI Flags

**General inference:**

| Flag | Default | Description |
|------|---------|-------------|
| `-p <text>` | — | Prompt string |
| `-n <N>` | 512 | Max tokens to generate |
| `--temp <f>` | 0.7 | Sampling temperature |
| `--top-k <N>` | 40 | Top-K sampling |
| `--top-p <f>` | 0.9 | Top-P (nucleus) sampling |
| `-t <N>` | auto | Thread count |
| `--ctx-size <N>` | 2048 | Context window size |
| `--serve` | off | Start HTTP API server |
| `--port <N>` | 8080 | HTTP server port |
| `-i` | off | Interactive multi-turn mode |
| `-v` | off | Verbose output |
| `--log-level <N>` | auto | Logging verbosity (0=quiet, 3=trace) |

**Thinking token control (reasoning models):**

| Flag | Description |
|------|-------------|
| `--no-think` | Strip `<think>…</think>` tokens at inference |
| `--force-think` | Prepend think-start token even if model omits it |
| `--show-think` | Output think tokens alongside final answer |

**OTT inference modes:**

| Flag | Description |
|------|-------------|
| `--ott-fast` | Speed-first: geodesic spec-decode batch=16, AttnRes on (max TPS) |
| `--ott-speculative` | Standard spec-decode: batch=2, geodesic drafts + transformer verify |
| `--ott-perfect` | Upper-bound mode: exact greedy rollout, all drafts accepted (100% rate) |
| `--ott-full` | Full OTT pipeline: axiom survey + geodesic-first + AttnRes + OneDecode prep |
| `--ott-theorem` | Theorem mode: all of --ott-full + depth-attn (maximum reasoning quality) |
| `--one-decode` | OneDecode: bake geodesic flow map to `ott_one_decode.bin`, skip Phase 5 |
| `--ott-od` | OTT-OD protocol: use baked OneDecode map as speculative draft source |
| `--ott-swarm <K>` | OD-SWARM: fan out K draft candidates per draft slot |
| `--ott-spec-thresh <f>` | Geodesic confidence pre-filter (default: 0.1; topk verifier is authoritative) |
| `--ott-spec-batch <N>` | Draft batch size for spec-decode (default: 2) |
| `--no-verifier` | Emit geodesic drafts directly without transformer verification |

**Attention modifiers:**

| Flag | Default | Description |
|------|---------|-------------|
| `--attnres` | off | AttnRes depth stabilization residual |
| `--attnres-strength <f>` | 0.35 | AttnRes mixing weight |
| `--depth-attn` | off | Depth-wise residual attention across all layers |
| `--depth-attn-strength <f>` | 0.55 | Depth attention mixing weight |
| `--depth-attn-window <N>` | 16 | Layer history window for depth attention |

**Axiom Beta / OTT survey:**

| Flag | Default | Description |
|------|---------|-------------|
| `--axiom-beta-run` | off | Run 5-phase OTT geometric survey after loading |
| `--axiom-beta-only` | off | Run survey and exit |
| `--axiom-fast` | off | Fast-mode survey (reduced workload) |
| `--axiom-gpu` | auto | Use GPU for Phase 5 scorer |
| `--axiom-samples <N>` | 256 | Phase 1–4 sample count |
| `--axiom-probe <N>` | 1024 | Phase 5 vocab probe token count |
| `--axiom-seed <N>` | 0 | Deterministic seed |
| `--axiom-report <path>` | `axiom_beta_report.json` | JSON report output path |

**Visualization:**

| Flag | Default | Description |
|------|---------|-------------|
| `--vis [dir]` | `axiom_vis/` | Emit Riemannian manifold JSON for the web visualizer |

---

## 7. Design Principles

1. **Zero dependencies**: Core compiles with `zig cc` or any C11 compiler. No BLAS, no protobuf, no Python.
2. **Bare-metal first**: Every feature works in TensorOS ring-0. No OS syscalls in the inference path.
3. **Profile-driven**: Optimize only what measurement proves is a bottleneck. JIT kernels were added after profiling showed 40% of time in SIMD loops.
4. **Correctness > Performance**: CPU AVX2 path is the gold standard. GPU and JIT paths are validated against it.
5. **Fail fast**: Validate at system boundaries (GGUF parser, tokenizer, quantization format). Crash with clear diagnostics on corruption.
6. **Minimal allocation**: Model weights are memory-mapped (zero copy). Scratch buffers pooled and reused across tokens.

---

## 8. Known Limitations and Future Work

### v0.6.0 Known Gaps
- **GPU utilization at long context**: 0–84% vs Ollama's 71–88%. More layers need CUDA dispatch.
- **RoPE JIT blocked for Gemma4**: Gemma4 SWA head_dim (256) ≠ full head_dim triggers guard; JIT RoPE disabled (JIT-TODO-2).
- **Softmax not JIT'd**: Scalar fallback; planned (JIT-TODO-1) but not yet implemented.
- **Q4_K_M quantization**: Not yet supported (blocked on dequant kernel). Q6_K is fully supported.
- **No multi-GPU**: Layer sharding across 2+ GPUs not yet implemented.
- **Gemma4 GPU-resident forward**: Disabled pending ISWA quality validation.

### v0.7 Targets
- GPU-resident forward pass for Gemma4 (ISWA on GPU)
- Q4_K_M dequantization kernel
- Softmax JIT (JIT-TODO-1) + fused RMSNorm+scale (JIT-TODO-4)
- Axiom Beta-4: geodesic inference as default decode proposer (Phase 5 → full proposer)
- ARM NEON backend for Raspberry Pi 4 / Apple Silicon
- LRU GEMV kernel cache for JIT (JIT-TODO-3)

---

*All performance figures measured on AMD Ryzen 9 7940HS + RTX 4070 Laptop, Windows host mode, April 2026. Bare-metal QEMU WHPX measurements differ; see benchmark_results.md for full tables.*
