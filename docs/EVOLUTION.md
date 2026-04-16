# Geodessical Evolution Plan

## Vision
Geodessical is a unified AI inference engine that runs on bare metal (TensorOS) and as a hosted runtime. The roadmap below defines the evolution from the current state (v0.5 "Synapse") through future milestones.

---

## Current State — v0.5 "Synapse" (Complete)

### Core Inference
- [x] GGUF model loading (7 architectures: LLaMA, Qwen2, Gemma, Gemma4, Phi-2, Phi-3, SmolLM)
- [x] Multi-format support (GGUF, SafeTensors headers, raw binary)
- [x] Q4_0, Q8_0, F16, F32 quantized weight execution
- [x] AVX2+FMA SIMD vectorized GEMV kernels
- [x] Grouped Query Attention (GQA)
- [x] Rotary Position Embeddings (RoPE, LongRoPE, freq_factors)
- [x] KV-cache for autoregressive decoding
- [x] BPE tokenizer with hash-table O(1) lookup
- [x] Multi-turn chat with context tracking

### Acceleration
- [x] CUDA GPU offload (RTX 4070: 14.5 tok/s decode, 29% speedup over CPU)
- [x] Selective GPU dispatch (out_dim >= 8192 threshold)
- [x] Multi-threaded CPU parallelism
- [x] Flash Attention pattern (online softmax, single-pass)
- [x] Paged Attention (tiled KV cache)

### Architecture
- [x] Backend abstraction (CPU, CUDA, MLIR vtable dispatch)
- [x] MLIR-inspired IR optimizer (op fusion, DCE, buffer planning)
- [x] Tensor bridge (hidden-state capture/inject between models)
- [x] Hidden-state projection (LINEAR, TRUNCATE, PAD)
- [x] Modification packaging (token-space edits without detokenize/retokenize)
- [x] Token-space communication (distributional structure between LLMs)
- [x] BLT tokenizer (byte-level transport)
- [x] HuggingFace auto-download (WinHTTP HTTPS client)

### Advanced
- [x] Speculative execution (5 techniques: pattern, statistical, entropy, chain, cache)
- [x] x86 JIT compiler for hot compute kernels
- [x] Distributed inference skeleton
- [x] HTTP API server (v1/generate, v1/chat, v1/models, v1/version)

---

## Phase 1 — v0.6 "Cortex" (Next)

### Performance
- [ ] Q6_K and Q4_K_M quantization support (broader model compat)
- [ ] INT8 quantized GEMV with dequant-on-the-fly
- [ ] CUDA kernel: fused RMSNorm+GEMV (eliminate memory round-trip)
- [ ] CUDA kernel: fused SiLU+Mul (GeGLU in one launch)
- [ ] Multi-GPU: split model across 2+ GPUs via layer sharding
- [ ] CPU: ARM NEON SIMD backend (for Raspberry Pi / Apple Silicon)

### Multi-Model
- [ ] Dual-model pipeline: draft + verifier in single process
- [ ] Speculative decoding with real draft model (not heuristics)
- [ ] Token-space communication between live model instances
- [ ] Mod-package pipeline: model A generates → model B refines
- [ ] Vocab mapping for cross-tokenizer communication

### Infrastructure
- [ ] Model catalog: list + search HuggingFace models from CLI
- [ ] Download progress bar with ETA
- [ ] Resume interrupted downloads
- [ ] SHA256 model integrity verification
- [ ] Configuration file for persistent settings

---

## Phase 2 — v0.7 "Nexus"

### Inference
- [ ] Continuous batching (serve N concurrent requests)
- [ ] Prompt caching (reuse KV cache for common prefixes)
- [ ] Mixture of Experts (MoE) routing
- [ ] Vision model support (LLaVA-style image + text)
- [ ] Speech-to-text integration (Whisper decoder)

### Optimization
- [ ] MLIR graph optimization in JIT mode (capture + optimize full layer)
- [ ] Auto-tuning: benchmark and select optimal kernel for each op
- [ ] Memory-mapped model weights (no copy, direct mmap into inference)
- [ ] Weight streaming for models larger than VRAM

### Vulkan Compute
- [ ] SPIR-V shader compilation for compute kernels
- [ ] Vulkan GEMV kernel (cross-platform GPU)
- [ ] Vulkan attention kernel
- [ ] Device enumeration and capability detection
- [ ] Fallback: Vulkan → CUDA → CPU cascade

---

## Phase 3 — v0.8 "Synapse II"

### Training
- [ ] LoRA fine-tuning (rank-4/8/16 adapters)
- [ ] LoRA adapter loading + inference
- [ ] QLoRA: quantized base + FP16 adapters
- [ ] Simple gradient descent for small models
- [ ] Loss tracking and convergence reporting

### Distribution
- [ ] Network-distributed inference: split layers across machines
- [ ] P2P model sharing between Geodessical instances
- [ ] Tensor bridge over TCP/IP (send hidden states across network)
- [ ] Token-space communication over network sockets

### Package Manager
- [ ] Full modelpkg install/update/remove cycle
- [ ] Dependency resolution for multi-model pipelines
- [ ] Model optimization: auto-quantize for local hardware
- [ ] Model signature verification (Ed25519)

---

## Phase 4 — v1.0 "Genesis"

### Production Readiness
- [ ] Comprehensive test suite (correctness vs reference implementations)
- [ ] Numeric stability validation (FP16 vs FP32 divergence tracking)
- [ ] Memory leak detection and sanitizer passes
- [ ] Security audit: input validation, buffer overflow protection
- [ ] API stability guarantee (semantic versioning)

### TensorOS Integration
- [ ] Boot-to-inference: load model from disk during kernel init
- [ ] Bare-metal CUDA driver (bypass OS, direct PCIe register access)
- [ ] virtio-blk model streaming (read from disk into inference)
- [ ] Bare-metal network stack → distributed inference
- [ ] Real-time inference scheduling (guaranteed latency)

### Ecosystem
- [ ] Plugin system for custom model architectures
- [ ] Python bindings (ctypes / pybind)
- [ ] C++ header-only client library
- [ ] Web UI for model management and chat
- [ ] Prometheus/Grafana metrics export

---

## Design Principles

1. **Zero dependencies**: Core engine compiles with zig cc, no external libraries
2. **Bare-metal first**: Every feature must work in TensorOS ring-0 mode
3. **Profile-driven**: Optimize only what the profiler proves is a bottleneck
4. **Correctness > Performance**: CPU backend is gold standard; accelerators must match within tolerance
5. **Minimal allocation**: Reuse buffers aggressively; pool all scratch memory
6. **Fail fast**: Validate at system boundaries; crash early with clear diagnostic

---

## Architecture Decision Records

### ADR-001: MLIR-Inspired IR vs Full MLIR
**Decision**: Build lightweight IR optimizer in pure C rather than linking LLVM MLIR.
**Rationale**: Full MLIR requires C++ runtime and 100+ MB of libraries. Our IR captures the essential optimizations (fusion, DCE, buffer planning) that matter for inference in ~500 lines of C.

### ADR-002: Token-Space Communication
**Decision**: Support both hard (discrete tokens) and soft (logit distributions) exchange.
**Rationale**: Hard tokens are simple but lossy. Soft distributions preserve uncertainty information, enabling better speculative decoding and model composition.

### ADR-003: Modification Packaging
**Decision**: Package edits as token-space operations (INSERT/DELETE/REPLACE).
**Rationale**: Text-level edits require detokenize→edit→retokenize which loses information. Token-level edits preserve the exact model representation.

### ADR-004: Backend Dispatch
**Decision**: vtable-based dispatch with optional compile-time backends.
**Rationale**: Allows adding new backends (Vulkan, Metal, NPU) without changing model code. Compile-time flags keep binary size minimal.
