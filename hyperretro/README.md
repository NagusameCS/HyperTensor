# HyperRetro

**Geometric LLM compression with verifiable quality certificates.**

[![PyPI](https://img.shields.io/pypi/v/hyperretro)](https://pypi.org/project/hyperretro/)
[![Python](https://img.shields.io/pypi/pyversions/hyperretro)](https://pypi.org/project/hyperretro/)
[![License](https://img.shields.io/pypi/l/hyperretro)](LICENSE)

## Install

```bash
pip install hyperretro
hyperretro setup          # interactive guided install
```

Or pick your extras:

```bash
pip install "hyperretro[hf]"               # + HuggingFace (compress, export, distill)
pip install "hyperretro[gguf]"             # + GGUF reader (load llama.cpp / Ollama files)
pip install "hyperretro[vllm]"             # + vLLM adapter
pip install "hyperretro[hf,gguf,vllm,bench]"  # full stack
```

## Quick Start

```bash
# Compress a model
hyperretro compress Qwen/Qwen2.5-1.5B --ffn-rank 1024 --int4 -o compressed/

# Export to GGUF (llama.cpp / Ollama)
hyperretro export compressed/ --format gguf --quantize Q4_K_M

# Load a GGUF file directly
hyperretro compress ./model-q4_k_m.gguf --ffn-rank 256 -o compressed/

# Get a quality certificate
hyperretro certify --model Qwen/Qwen2.5-1.5B --rank 1024 --out cert.json

# Kernel benchmark
hyperretro bench-kernels

# List available model backends
hyperretro list-backends
```

## CLI Commands (13)

| Command | Description |
|---|---|
| `setup` | Interactive install wizard — picks the right extras |
| `compress` | Compress a model (HF, GGUF, OM) — SVD + int4 |
| `export` | Export to GGUF / safetensors / HF format |
| `info` | Inspect a checkpoint: tensors, quant, layers |
| `certify` | Quality certificate: trust tier + jury-proof PPL bounds |
| `bench-kernels` | Fused dual-Q8_0 GEMV vs baselines |
| `benchmark` | Full compression benchmark suite |
| `distill` | GRC light distillation — recover PPL after compression |
| `gauge` | AxiomGauge — diagonal gauge optimization (free quality) |
| `card` | Generate HuggingFace model card |
| `red-team` | Adversarial attack evaluation (GCG/AutoPrompt/PAIR) |
| `list-backends` | Show available model backends |
| `--help` | Full help for any subcommand |

## Python API (32 exports)

### Core compression

```python
import hyperretro

# Load any model (HF, GGUF, OpenMythos, vLLM)
model = hyperretro.load_model("Qwen/Qwen2.5-1.5B")
model = hyperretro.load_model("model-q4_k_m.gguf")          # GGUF auto-detected
model = hyperretro.load_model("mythos_1b", backend="openmythos")

# Compress
compressed = hyperretro.compress(model, ffn_rank=1024, int4=True)

# Export
hyperretro.export_model(compressed, "model.gguf", format="gguf")
hyperretro.export_model(compressed, "compressed/", format="safetensors")
```

### Certificates & benchmarks

```python
cert = hyperretro.certify_compression(state_dict, config, stats, model_id="my-model")
print(cert.summary())
# -> "GOLD: max forward error <= 0.34, 94% spectral efficiency"

bench = hyperretro.run_kernel_bench(rows=4096, in_dim=4096, iters=50)
bench = hyperretro.run_compression_bench("Qwen/Qwen2.5-1.5B", out_dir="/tmp/test")
```

### Geometric tools (requires hypercore)

```python
from hyperretro import (
    AxiomGauge,            # GL(d) diagonal gauge optimizer
    ThermalRankController, # temperature-driven rank scheduler
    OnlineOjaBasis,        # rejection-driven adaptive PCA
    NativeLinear,          # train on compressed Gr(k,d) manifold
    RiemannianAdamW,       # manifold-respecting optimizer
    KExpansionScheduler,   # exponential k-warmup
    TreeDrafter,           # Medusa/EAGLE tree speculative decode
    GCGAttack,             # adversarial prompt attack
)
```

### Kernels

```python
import numpy as np

scales, codes = hyperretro.q8_0_quantize(W)
W_back = hyperretro.q8_0_dequantize(scales, codes)
y_a, y_b = hyperretro.gemv_dual_q8_0(x, Wa, Wb)   # fused dual GEMV
bk = hyperretro.kernels_backend()                   # -> 'gpu' / 'torch' / 'numpy'
```

## Model Backends (4)

| Backend | Load from | Extra |
|---|---|---|
| **HuggingFace** | Repo ID, local dir, safetensors | built-in |
| **GGUF** | `.gguf` files (llama.cpp / Ollama) | `[gguf]` |
| **OpenMythos** | OpenMythos models | `om` |
| **vLLM** | vLLM LLM instances | `[vllm]` |

Auto-detection: `.gguf` extension -> GGUF, `mythos_` prefix -> OpenMythos, `/` in name -> HF.

## Kernel Backends (6-tier)

| # | Backend | Description | Requirements |
|---|---|---|---|
| 1 | `cuda_cext` | Raw CUDA kernel (fastest) | NVCC + host compiler |
| 2 | `cext` | JIT C++ extension | C++ compiler |
| 3 | `cpu_opt` | Pre-compiled AVX2 | x86_64 CPU |
| 4 | `gpu` | Pure-PyTorch CUDA (9.5x numpy) | PyTorch + CUDA |
| 5 | `torch` | Pure-PyTorch CPU | PyTorch |
| 6 | `numpy` | Always available | nothing |

Set `HYPERRETRO_FORCE_FALLBACK=1` to force numpy path.

## Certificate System

HyperRetro is the only compression tool that produces **mathematically verifiable
quality certificates**:

- **Trust tier**: PLATINUM / GOLD / SILVER / BRONZE
- **BP-NS bound**: per-layer forward-error bound (Eckart-Young)
- **Spectral efficiency**: information retained per parameter
- **Frobenius certificates**: relative error in weight space (Q/K/V)
- **Jury-proof PPL bounds**: strict worst-case + concentration bound

## Benchmarks

| Config | PPL | Disk | Shrink |
|---|---|---|---|
| fp16 baseline (Qwen2.5-1.5B) | 2.33 | 2955 MB | 1.00x |
| Aware-factored fp16 | 4.39 | 2581 MB | 1.15x |
| int4 FFN-only + AWQ | 6.04 | 1242 MB | 2.38x |

GPU: RTX 4070, dual-Q8 GEMV 4096x4096 = 21.5ms (9.5x vs CPU).

## Contributing

```bash
git clone https://github.com/NagusameCS/HyperTensor.git
cd HyperTensor
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
