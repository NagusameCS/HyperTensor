# Changelog

All notable changes to TensorOS are documented in this file.

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
- Multiboot2 bootloader with x86_64 long mode + SSE2
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
