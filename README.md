
<p align="center">
  <h1 align="center">HyperTensor</h1>
  <p align="center"><b>High-Performance AI Inference Runtime</b></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/lang-C11-blue?logo=c&logoColor=white" alt="Language">
  <img src="https://img.shields.io/badge/arch-x86__64_%7C_ARM64-orange" alt="Architecture">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build">
  <img src="https://img.shields.io/badge/mode-hosted_%7C_bare--metal-informational" alt="Mode">
  <img src="https://img.shields.io/badge/LLM-Phi--3.5_working-success" alt="LLM Working">
  <img src="https://img.shields.io/github/last-commit/NagusameCS/HyperTensor?label=last%20commit" alt="Last Commit">
</p>

HyperTensor is a minimal, high-performance AI inference runtime that runs GGUF language models on **any platform** — from bare-metal x86_64 with zero OS overhead to a native Windows/Linux application. Born from [TensorOS](https://github.com/NagusameCS/TensorOS), it inherits a battle-tested GGUF parser, BPE tokenizer, JIT compiler, and SMP-parallel forward pass, now packaged as a portable host-mode runtime.

### Key Features

- **GGUF model loading** — Qwen, LLaMA, Gemma, SmolLM, Mistral, Phi-2/3/3.5
- **Quantization** — Q4_0, Q8_0, F16, F32 weight formats
- **JIT-compiled kernels** — Native x86_64 SSE2/AVX2 forward pass kernels
- **SMP parallel GEMV** — Multi-threaded matrix-vector multiply across all CPU cores
- **Host-mode runtime** — Memory-mapped model loading, native threads, cross-platform
- **Bare-metal mode** — Still boots as a standalone OS via Multiboot1

### Demo: Hosted Inference

```
$ ./hypertensor phi3.5-mini-q4_0.gguf -p "What is an operating system?"

  HyperTensor v0.4.0 "Axon"
  High-Performance AI Inference Runtime

[CPU] SSE2=1 AVX2=1 FMA=1 AVX512=0
[SMP] 8 CPUs online (7 workers + BSP)
[HT] Loading model: phi3.5-mini-q4_0.gguf
[HT] Mapped 2081 MB
[LLM] Model: Phi 3.5 Mini Instruct (phi3)
[LLM] 32 layers, 3072-dim, 32064 vocab, 32 heads
[HT] Model loaded in 1240 ms
[HT] Prompt: "What is an operating system?"

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
.\build_host\hypertensor.exe phi3.5.gguf -p "Hello world"

# Interactive chat mode
.\build_host\hypertensor.exe phi3.5.gguf -i
```

Or with CMake (if GCC/Clang available):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./hypertensor phi3.5.gguf -i
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
hypertensor <model.gguf> [options]

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

HyperTensor operates in two modes:

### Host Mode (new)

```
┌─────────────────────────────────────────────────┐
│  hypertensor.exe / hypertensor                  │
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

The full TensorOS kernel boots via Multiboot1, runs on x86_64/ARM64, and includes the AI shell, tensor scheduler, native git, GPU drivers, and everything documented in the [TensorOS README](https://github.com/NagusameCS/TensorOS).

---

## Project Structure

```
HyperTensor/
├── host/                      # Host-mode runtime (NEW)
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

Any LLaMA-architecture GGUF model works:

| Model | Architecture | Tested |
|-------|-------------|--------|
| Phi-3.5 Mini Instruct | phi3 | ✅ 162 ms/tok |
| Qwen2.5 | qwen2 | ✅ |
| LLaMA 3 | llama | ✅ |
| Gemma 2 | gemma | ✅ |
| SmolLM 2 | llama | ✅ |
| Mistral | llama | ✅ |
| Phi-2 | phi2 | ✅ |

Quantization: Q4_0, Q8_0, F16, F32

---

## Origin

HyperTensor evolved from [TensorOS](https://github.com/NagusameCS/TensorOS), a bare-metal AI operating system. The core inference engine, GGUF parser, BPE tokenizer, JIT compiler, and SMP parallel GEMV are shared between both projects. HyperTensor adds the HAL layer to run the same inference code as a native application on Windows and Linux.

---

## License

MIT


