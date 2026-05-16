<!-- :%#@=+-.                              .-=+@#%: -->
<!--  .:+#%@@@@@@@@@@@@@@@@@@@@@@@@#=...:=#@@@@@@@@@@@@@@@@@@@@@@@@%#+:. -->
<!--   .:=#@@@@@@@@@@@@@@@@@@@@@%-.:+%@@@@@#=:=%@@@@@@@@@@@@@@@@@%+-. -->
<!--      :-%@@@@@@@@@@@@@@@@@@@#=#@@@@@@@@@@@%=-@@@@@@@@@@@@@@@. -->
<!--         :+#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+=*@@@@@@@@@@@#: -->
<!--            -*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#=#@@@@@@@@@+. -->
<!--              .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@@*. -->
<!--                .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--                  -%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*. -->
<!--                   :%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%: -->
<!--                    .*@@@@@@@@@@@@@@@@@@@@@@@@@@@@%. -->
<!--                     :#@@@@@@@@@@@@@@@@@@@@@@@@@@@*. -->
<!--                      .#@@@@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                       .*@@@@@@@@@@@@@@@@@@@@@@@@+ -->
<!--                        .@@@@@@@@%+-::=@@@@@@@@: -->
<!--                        .*@@@@@@@.       .+@@@@@@: -->
<!--                        .*@@@@@@@:         :@@@@@@: -->
<!--                        .*@@@@@@@%+-::::-+#@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   HyperTensor -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   universal geometric tensor framework -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   Papers I-XXX -->
<!--                        :#@@@@@@@@@@@@@@@@@@@@@@@*.   15/18 at 100% -->
<!--                       .*@@@@@@@@@@@@@@@@@@@@@@@@%.   Jury-GTC @ 53x -->
<!--                      .+@@@@@@@@@@@@@@@@@@@@@@@@@@#:   AGT @ 50K primes -->
<!--                     -%@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.   External verify 14/14 -->
<!--                   :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+.   COG 10K converged -->
<!--                 .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.  Bilateral 1.5B 0.968 -->
<!--               :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--            .-*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%. -->
<!--         .-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@. -->
<!--      .-*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--   .:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+: -->
<!-- .:+#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%#+:. -->

# Geodessical Evolution Plan

## Vision
Geodessical is a unified AI inference engine that runs on bare metal (TensorOS) and as a hosted runtime. The roadmap below defines the evolution from the current state (v0.5 "Synapse" --- released April 2026) through future milestones.

---

## Current State --- v0.6.0 "Synapse" (Released April 18, 2026)

### Core Inference
- [x] GGUF model loading (7 architectures: LLaMA, Qwen2, Gemma, Gemma4, Phi-2, Phi-3, SmolLM)
- [x] Multi-format support (GGUF, SafeTensors headers, raw binary)
- [x] Q40, Q80, F16, F32 quantized weight execution
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
- [x] Speculative Neural Execution (SNE): 5 microarchitecture-inspired inference techniques:
  - Adaptive Precision Cascade (APC): INT16 fast path, escalate to FP32 on high entropy
  - Speculative Layer Fusion (SLF): skip matmul when activation signature matches cache
  - Entropy-Aware Neuron Pruning (EANP): prune zero-entropy neurons at runtime
  - Compute DAG Scheduling: Tomasulo-inspired dependency scheduling for tensor ops
  - Confidence-Gated Early Exit: exit depth proportional to input difficulty
- [x] 13 JIT-compiled SSE2 kernels (silu, rmsnorm, rope, gelu, layernorm, dot, axpy, q4/q8 gemv, ...)
- [x] CUDA dynamic DLL dispatch (~50 kernel ops): fused QKV, ISWA, batch prefill, CUDA Graph
- [x] Distributed inference skeleton
- [x] HTTP API server (v1/generate, v1/chat, v1/models, v1/version)
- [x] Axiom Beta-3: 5-phase OTT survey pipeline (PCA, symmetry, curvature, axioms, geodesic pilot)
- [x] Phase 2 using real dequantized Q-weight rows (not block statistics)
- [x] Phase 3 full Riemann computation (∂Γ derivative + Γ·Γ algebraic terms)
- [x] Decode-aligned oracle targets for Phase 5 geodesic pilot
- [x] Persistent warp-state storage (`axiomwarpstate.dat`)
- [x] Knowledge injection prototype (local Christoffel warp with Gaussian decay)
- [x] Phase 3 warm-cache (197 s cold -> 0.17 s warm, −99.9%)
- [x] OTT inference modes: `--ott-fast`, `--ott-speculative`, `--ott-perfect`, `--ott-full`, `--ott-theorem`
- [x] OneDecode (`--one-decode`): bake geodesic flow map once -> `ottonedecode.bin`
- [x] OTT-OD (`--ott-od`): OneDecode as speculative draft source
- [x] AttnRes (`--attnres`): attention depth stabilization residual
- [x] depth-attn (`--depth-attn`): depth-wise residual cross-layer attention
- [x] Thinking token control: `--no-think`, `--force-think`, `--show-think`
- [x] Gemma4 ISWA: interleaved sliding-window attention with per-layer ISWA embeddings
- [x] Q6K, BF16, Q41 CUDA upload support
- [x] `--axiom-fast`, `--axiom-gpu`, `--axiom-probe`, `--axiom-samples`, `--axiom-seed` CLI flags

### Performance Highlights (v0.6.0)
- Gemma4 E2B decode: 107.7 tok/s (GPU), 92.5 tok/s end-to-end
- SmolLM2-135M decode: 174--271 tok/s (GPU, long context)
- Axiom fast-mode survey: ~977 ms total (MRR ≈ 0.067)
- vs Ollama gemma3:4b: +22.7% on same hardware

---

## Phase 1 --- v0.7 "Cortex" (Next)

### Performance
- [ ] Q4KM quantization support (blocked on dequant kernel; Q6_K already done)
- [ ] Softmax JIT kernel (JIT-TODO-1)
- [ ] Fused RMSNorm+scale JIT (JIT-TODO-4)
- [ ] GPU-resident forward pass for Gemma4 (ISWA quality validation complete)
- [ ] Multi-GPU: split model across 2+ GPUs via layer sharding
- [ ] CPU: ARM NEON SIMD backend (for Raspberry Pi / Apple Silicon)

### Multi-Model
- [ ] Dual-model pipeline: draft + verifier in single process
- [ ] Speculative decoding with auxiliary draft model (OTT-OD provides geodesic draft; this adds a separate neural draft model)
- [ ] Token-space communication between live model instances
- [ ] Mod-package pipeline: model A generates -> model B refines
- [ ] Vocab mapping for cross-tokenizer communication

### Infrastructure
- [ ] Model catalog: list + search HuggingFace models from CLI
- [ ] Download progress bar with ETA
- [ ] Resume interrupted downloads
- [ ] SHA256 model integrity verification
- [ ] Configuration file for persistent settings

### OTT / Axiom Beta-4 (Geodesic Inference Prototype)
- [ ] Promote Phase 5 from pilot evaluator to candidate token proposer in decode loop
- [ ] LRU hidden-state cache keyed by (tokenid, layer) --- items 21--26 in GEODESSICALPLAN.md
- [ ] Block-level manifold sampling to reduce ottgethidden_state call count (8x)
- [ ] Online softmax merge for Phase 5 probe pool scoring (32x peak memory reduction)
- [ ] RMSNorm normalization of captured hidden state keys (items 22--24)
- [ ] Depth-sink layer detection for optimal hidden state capture layer
- [ ] Two-phase batch I/O for hidden state collection (10x I/O reduction)
- [ ] Layer-wise geodesic trajectory matching vs transformer hidden states

---

## Phase 2 --- v0.7 "Nexus"

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
- [ ] Fallback: Vulkan -> CUDA -> CPU cascade

---

## Phase 3 --- v0.8 "Synapse II"

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

## Phase 4 --- v1.0 "Genesis"

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
- [ ] Bare-metal network stack -> distributed inference
- [ ] Real-time inference scheduling (guaranteed latency)

### Ecosystem
- [ ] Plugin system for custom model architectures
- [ ] Python bindings (ctypes / pybind)
- [ ] C++ header-only client library
- [ ] Web UI for model management and chat
- [ ] Prometheus/Grafana metrics export

---

## Design Principles

1. Zero dependencies: Core engine compiles with zig cc, no external libraries
2. Bare-metal first: Every feature must work in TensorOS ring-0 mode
3. Profile-driven: Optimize only what the profiler proves is a bottleneck
4. Correctness > Performance: CPU backend is gold standard; accelerators must match within tolerance
5. Minimal allocation: Reuse buffers aggressively; pool all scratch memory
6. Fail fast: Validate at system boundaries; crash early with clear diagnostic

---

## Architecture Decision Records

### ADR-001: MLIR-Inspired IR vs Full MLIR
Decision: Build lightweight IR optimizer in pure C rather than linking LLVM MLIR.
Rationale: Full MLIR requires C++ runtime and 100+ MB of libraries. Our IR captures the essential optimizations (fusion, DCE, buffer planning) that matter for inference in ~500 lines of C.

### ADR-002: Token-Space Communication
Decision: Support both hard (discrete tokens) and soft (logit distributions) exchange.
Rationale: Hard tokens are simple but lossy. Soft distributions preserve uncertainty information, enabling better speculative decoding and model composition.

### ADR-003: Modification Packaging
Decision: Package edits as token-space operations (INSERT/DELETE/REPLACE).
Rationale: Text-level edits require detokenize->edit->retokenize which loses information. Token-level edits preserve the exact model representation.

### ADR-004: Backend Dispatch
Decision: vtable-based dispatch with optional compile-time backends.
Rationale: Allows adding new backends (Vulkan, Metal, NPU) without changing model code. Compile-time flags keep binary size minimal.
