
<p align="center">
  <h1 align="center">HyperTensor</h1>
  <p align="center"><b>AI inference runtime, bare-metal OS kernel, and geometric inference research</b></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/lang-C11-blue?logo=c&logoColor=white" alt="Language">
  <img src="https://img.shields.io/badge/arch-x86__64_%7C_ARM64-orange" alt="Architecture">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build">
  <img src="https://img.shields.io/badge/mode-hosted_%7C_bare--metal-informational" alt="Mode">
  <img src="https://img.shields.io/badge/LLM-GGUF_models_working-success" alt="LLM Working">
  <img src="https://img.shields.io/github/last-commit/NagusameCS/HyperTensor?label=last%20commit" alt="Last Commit">
</p>

HyperTensor is the umbrella project for three interconnected systems written in C11:

| Component | Description |
|-----------|-------------|
| **Geodessical** | GGUF inference runtime — runs as a native host application on Windows/Linux. v0.6 "Synapse". |
| **TensorOS** | Bare-metal AI operating system — Multiboot1 kernel, x86_64/ARM64, AI shell, SMP, JIT, GPU drivers. |
| **OTT** | Organic Training Theory — research track treating weight space as a Riemannian manifold and replacing the transformer forward pass with geodesic equation solving. |

All three share the same inference core (GGUF parser, BPE tokenizer, JIT compiler, SMP parallel GEMV). Geodessical adds a HAL to run that core as a host application; TensorOS boots it on bare metal; OTT experiments on top of both.

### Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Evolution Roadmap](docs/EVOLUTION.md)
- [Geodessical Development Plan (OTT)](docs/GEODESSICAL_PLAN.md)
- [Autonomous Axiomatic Subsystem — Beta-3](docs/AUTONOMOUS_AXIOMATIC_SUBSYSTEM_BETA.md)
- [Axiom Beta Internal Design](docs/AXIOM_BETA_INTERNAL_DESIGN.md)
- [OTT Whitepaper](docs/OTT_WHITEPAPER.md)
- [Geodessical System Whitepaper](docs/GEODESSICAL_WHITEPAPER.md)
- [Changelog](CHANGELOG.md)

### Geodessical — Inference Capabilities (v0.6 "Synapse")

- **GGUF model loading** — Qwen2.5, LLaMA 3, Gemma 2/4 (ISWA), SmolLM 2, Mistral, Phi-3/3.5
- **Quantization** — Q4_0, Q4_1, Q8_0, Q6_K, F16, BF16, F32 weight formats
- **JIT-compiled kernels** — 13 SSE2 native kernels (rmsnorm, rope, silu, gelu, dot, axpy, q4/q8 gemv, …); AVX2 for batched GEMV
- **SMP parallel GEMV** — Multi-threaded matrix-vector multiply across all CPU cores via IPI dispatch
- **CUDA GPU offload** — Dynamic DLL dispatch (~50 GPU kernels); fused QKV, ISWA, batch prefill, CUDA Graph capture
- **OTT inference modes** — `--ott-fast`, `--ott-speculative`, `--ott-perfect`, `--ott-full`, `--ott-theorem`
- **OneDecode / OTT-OD** — Bake geodesic flow map once; use as speculative draft source (`--one-decode`, `--ott-od`)
- **AttnRes + depth-attn** — Attention residual stabilization and depth-wise cross-layer attention
- **Thinking token control** — `--no-think`, `--force-think`, `--show-think` for reasoning models (QwQ, DeepSeek-R1, etc.)
- **HTTP API server** — `/v1/generate`, `/v1/chat`, `/v1/models`, `/v1/version`; `--serve --port 8080`
- **Host-mode runtime** — Memory-mapped model loading, native threads, cross-platform (Windows/Linux)
- **Bare-metal mode** — Boots as standalone TensorOS kernel via Multiboot1

### TensorOS — Kernel Capabilities

- **x86_64 bare-metal boot** — Multiboot1, long mode, PAE, 16 GB identity map
- **SMP** — INIT-SIPI-SIPI AP bootstrap, up to 64 CPUs, LAPIC
- **Tensor-aware memory** — Heap + arena + slab, LRU model cache (64 slots), 2 MB JIT W^X pool
- **Pseudocode JIT** — Full lexer/parser/IR/optimizer pipeline
- **Tensor scheduler, IPC, VFS, sandbox** — Kernel-level AI subsystems
- **ARM64 / Raspberry Pi 4** — Boot stub + NEON backend (in progress)

### OTT — Research Status (April 2026)

See [docs/GEODESSICAL_PLAN.md](docs/GEODESSICAL_PLAN.md) and [docs/AXIOM_BETA_INTERNAL_DESIGN.md](docs/AXIOM_BETA_INTERNAL_DESIGN.md) for full phase breakdown.

**Axiom Beta-3** is the current survey/research build integrated into the Geodessical runtime.

| Phase | Status | Detail |
|-------|--------|--------|
| Phase 1 — Geometry Survey | ✅ Stable | PCA + TwoNN, k≈41 measured (Gemma4 E2B) |
| Phase 2 — Symmetry | ✅ Stable | Head weight fingerprint analysis using real dequantized Q-weight rows; 80 invariant pairs, 64 generators |
| Phase 3 — Curvature | ✅ Stable | Full Riemann (∂Γ + Γ·Γ algebraic); Fisher-blended metric, Christoffel symbols; warm-cache 0.17 s (−99.9%) |
| Phase 4 — Axioms | 🟡 Active | Uncertainty-driven oracle loop (12 calls max), MRR improving |
| Phase 5 — Geodesic Pilot | 🟡 Active | Decode-aligned oracle targets, persistent warp state, MRR ≈ 0.067 |
| Knowledge Injection | 🟡 Prototype | Local Christoffel warp with threshold-triggered recompute |
| Geodesic Forward Pass | ❌ Pending | O(n·k²) transformer replacement — core OTT target |

**Overall OTT readiness: ~70%** (geometry foundation, axiom discovery active; geodesic inference not yet default path)

Fast-mode survey timing (Gemma4 E2B, `--axiom-fast`):
- Total: ~977 ms | Phase 4: ~669 ms | Phase 5: ~43 ms

CLI flags: `--axiom-beta-run`, `--axiom-beta-only`, `--axiom-fast`, `--axiom-probe <n>`, `--axiom-gpu`, `--axiom-samples <n>`, `--axiom-seed <n>`

**OTT inference modes** (run on top of axiom survey):

| Mode | Flag | Description |
|------|------|-------------|
| Standard spec-decode | `--ott-speculative` | Geodesic drafts + transformer verify, batch=2 |
| Speed-first spec-decode | `--ott-fast` | Batch=16, fast axiom, AttnRes on |
| Perfect-day upper bound | `--ott-perfect` | Exact greedy rollout, 100% draft acceptance |
| Full OTT pipeline | `--ott-full` | Axiom + geodesic-first + AttnRes + OneDecode |
| Theorem mode | `--ott-theorem` | `--ott-full` + depth-attn (maximum reasoning quality) |
| OneDecode | `--one-decode` | Bake geodesic flow map once → `ott_one_decode.bin` |
| OTT-OD | `--ott-od` | Use baked OneDecode map as speculative draft source |

---

## Performance Snapshot (Measured)

The numbers below are from an actual local run on April 13, 2026.

### Test Hardware

- CPU: AMD Ryzen 9 7940HS (8 cores / 16 threads)
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB class; runtime reported ~7052 MB free)
- RAM: 32 GB
- OS: Windows (host mode)

### Workload

- Prompt: `Write a 500-word explanation of how compilers optimize loops, in plain English.`
- Max generation: 256 tokens
- Single run per engine/model (no averaging in this table)

### Results

| Engine | Model | Throughput Metric | Measured Value |
|---|---|---|---|
| Geodessical | `google_gemma-4-E2B-it-Q4_0.gguf` | End-to-end generation rate | **92.5 tok/s** |
| Geodessical | `google_gemma-4-E2B-it-Q4_0.gguf` | Decode-only rate | **107.7 tok/s** |
| Ollama | `gemma3:4b` | Eval rate (`eval_count / eval_duration`) | **75.36 tok/s** |
| Ollama | `gemma4:latest` | Eval rate (`eval_count / eval_duration`) | **30.21 tok/s** |

### Direct Comparison (same machine, same prompt length)

- Geodessical end-to-end (92.5 tok/s) vs Ollama `gemma3:4b` (75.36 tok/s): **+22.7%**
- Geodessical end-to-end (92.5 tok/s) vs Ollama `gemma4:latest` (30.21 tok/s): **+206.2%**

### Repro Commands

Geodessical:

```powershell
.\build_host\geodessical.exe "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf" -p "Write a 500-word explanation of how compilers optimize loops, in plain English." -n 256
```

Ollama (`gemma3:4b`):

```powershell
$body = @{ model = 'gemma3:4b'; prompt = 'Write a 500-word explanation of how compilers optimize loops, in plain English.'; stream = $false; options = @{ num_predict = 256; temperature = 0.7 } } | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' -Method Post -ContentType 'application/json' -Body $body
[math]::Round(($r.eval_count / ($r.eval_duration / 1e9)), 2)
```

Ollama (`gemma4:latest`):

```powershell
$body = @{ model = 'gemma4:latest'; prompt = 'Write a 500-word explanation of how compilers optimize loops, in plain English.'; stream = $false; options = @{ num_predict = 256; temperature = 0.7 } } | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' -Method Post -ContentType 'application/json' -Body $body
[math]::Round(($r.eval_count / ($r.eval_duration / 1e9)), 2)
```

Notes:

- This is a practical runtime comparison, not a strict model-equivalence benchmark.
- Geodessical and Ollama model packages are not byte-identical here, so use these results as operational guidance, not a canonical leaderboard.

### Demo: Hosted Inference

```
$ ./geodessical phi3.5-mini-q4_0.gguf -p "What is an operating system?"

  Geodessical v0.6.0 "Synapse"
  High-Performance AI Inference Runtime

[CPU] SSE2=1 AVX2=1 FMA=1 AVX512=0
[SMP] 8 CPUs online (7 workers + BSP)
[GD] Loading model: phi3.5-mini-q4_0.gguf
[GD] Mapped 2081 MB
[LLM] Model: Phi 3.5 Mini Instruct (phi3)
[LLM] 32 layers, 3072-dim, 32064 vocab, 32 heads
[GD] Model loaded in 1240 ms
[GD] Prompt: "What is an operating system?"

An operating system (OS) is a complex piece of software that manages...
```

---

## Building

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| `zig` (0.15+) | C compiler | [ziglang.org/download](https://ziglang.org/download/) |

### Host Mode (Windows/Linux)

```powershell
# Build the hosted runtime
.\build_host.ps1

# Run with a GGUF model
.\build_host\geodessical.exe phi3.5.gguf -p "Hello world"

# Interactive chat mode
.\build_host\geodessical.exe phi3.5.gguf -i
```

Or with CMake (if GCC/Clang available):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./geodessical phi3.5.gguf -i
```

### Bare-Metal Mode (QEMU)

```powershell
# Build the bare-metal kernel + run in QEMU
.\build.ps1 -Run

# QEMU flags: -machine q35,accel=whpx -cpu EPYC-v4 -smp 4 -m 8G
#             -drive file=phi3.5.gguf,format=raw,if=virtio
```

---

## Usage

```
Geodessical <model.gguf> [options]

Options:
  -p, --prompt <text>    Prompt text (default: interactive)
  -n, --tokens <num>     Max tokens to generate (default: 128)
  -t, --threads <num>    Thread count (default: all CPUs)
  --temp <float>         Temperature (default: 0.7)
  --top-k <int>          Top-K sampling (default: 40)
  --top-p <float>        Nucleus sampling (default: 0.9)
  -i, --interactive      Interactive chat mode
  -h, --help             Show this help
```

---

## Architecture

Geodessical operates in two modes:

### Host Mode (new)

```
┌─────────────────────────────────────────────────┐
│  geodessical.exe / Geodessical                  │
│  CLI: model load, prompt, interactive chat      │
├─────────────────────────────────────────────────┤
│  HAL (Hardware Abstraction Layer)               │
│  ┌───────────┬───────────┬──────────────────┐   │
│  │ Memory    │ Threading │ CPU Detection    │   │
│  │ malloc    │ Win32/    │ CPUID: SSE2,    │   │
│  │ aligned   │ pthreads  │ AVX2, FMA,      │   │
│  │ mmap      │ workers   │ AVX-512         │   │
│  └───────────┴───────────┴──────────────────┘   │
├─────────────────────────────────────────────────┤
│  Inference Engine (shared with bare-metal)      │
│  ┌──────┬─────────┬──────┬──────┬───────────┐   │
│  │ GGUF │ BPE     │ JIT  │ SMP  │ Forward   │   │
│  │parse │tokenize │ x86  │GEMV  │ pass      │   │
│  └──────┴─────────┴──────┴──────┴───────────┘   │
└─────────────────────────────────────────────────┘
```

### Bare-Metal Mode (original TensorOS)

The full TensorOS kernel boots via Multiboot1, runs on x86_64/ARM64, and includes the AI shell, tensor scheduler, native git, GPU drivers. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full boot sequence and memory layout.

---

## Project Structure

```
HyperTensor/
├── host/                      # Host-mode runtime (Geodessical)
│   ├── hal.h                  # Hardware Abstraction Layer header
│   ├── hal.c                  # Cross-platform HAL implementation
│   ├── main.c                 # CLI entry point
│   └── shims/                 # Include shims (kernel→HAL redirect)
│       └── kernel/...         # Shim headers for all kernel includes
├── runtime/
│   ├── nn/
│   │   ├── llm.c             # Full LLM inference engine
│   │   ├── llm.h             # Model types and API
│   │   └── gguf.c            # GGUF format parser
│   └── jit/
│       ├── x86_jit.c         # x86_64 JIT code emitter
│       └── llm_jit.c         # JIT forward kernels
├── kernel/                    # Bare-metal kernel (TensorOS heritage)
├── boot/                      # Bootloader (Multiboot1, ARM64)
├── build_host.ps1             # Host-mode build script (Zig CC)
├── build.ps1                  # Bare-metal build script
└── CMakeLists.txt             # CMake build (GCC/Clang)
```

---

## How It Works

1. **Model Loading**: Memory-maps the GGUF file (no copy), parses metadata, maps tensor pointers directly into the file.

2. **Tokenization**: BPE tokenizer built from GGUF vocabulary with an O(1) hash table lookup and merge-based encoding.

3. **Forward Pass**: Full transformer forward pass with RMSNorm → QKV projection → RoPE → GQA attention → SwiGLU FFN → LM head.

4. **JIT Compilation**: On first inference, six x86_64 SIMD kernels are JIT-compiled (vadd, dot, axpy, fused_silu_mul, rope, rmsnorm) — eliminating per-element function call overhead.

5. **SMP Dispatch**: Matrix-vector multiplies are partitioned across all CPU cores via the HAL's thread pool.

6. **Sampling**: Temperature-scaled softmax with top-k/top-p nucleus sampling and optional greedy decoding.

---

## Supported Models

Current GGUF coverage in this runtime includes:

| Model | Architecture | Tested |
|-------|-------------|--------|
| Gemma 4 E2B It | gemma4 | ✅ |
| Phi-3.5 Mini Instruct | phi3 | ✅ |
| Qwen2.5 | qwen2 | ✅ |
| LLaMA 3 | llama | ✅ |
| Gemma 2 | gemma | ✅ |
| SmolLM 2 | llama | ✅ |
| Mistral | llama | ✅ |
| Phi-2 | phi2 | ✅ |

Quantization: Q4_0, Q8_0, F16, F32

---

## Origin

HyperTensor grew out of a bare-metal AI OS experiment (TensorOS). The core inference engine — GGUF parser, BPE tokenizer, JIT compiler, SMP parallel GEMV — was first written for the kernel. Geodessical adds the HAL layer to run that same core as a native host application on Windows and Linux. OTT is the current research track exploring whether the transformer forward pass can be replaced entirely by geodesic equation solving on the weight manifold.

---

## License

MIT


