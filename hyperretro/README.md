# HyperRetro

**HyperTensor, retrofitted into the PyTorch / HuggingFace / vLLM ecosystem.**

HyperTensor proper is a standalone runtime. HyperRetro is the *integrated*
sibling project: it takes the same geometric primitives (UGT shared basis,
GRC / sink-aware projection, geodesic speculative draft, fused dual-Q8
GEMV) and exposes them as drop-in pieces of the standard inference stack.

```
hyperretro/
├── kernels/        # PyTorch C++ extension (gemv_dual_q8_0, ...)
├── hf/             # offline HuggingFace compression -> .safetensors
├── vllm/           # speculative-decoding draft adapter
└── bench/          # 3-way benchmark harness (baseline | retro | HyperTensor)
```

## Three retrofits

### 1. Fused kernels as a PyTorch extension

The CUDA kernel `kernel_gemv_dual_q8_0` from
[`runtime/nn/cuda_kernels.cu`](../runtime/nn/cuda_kernels.cu) is wrapped as a
JIT-built `torch.utils.cpp_extension` so users can call it from regular
PyTorch:

```python
import hyperretro
import torch

x = torch.randn(4096)
# Wa, Wb may be float matrices or pre-quantized (scale, codes) tuples
out_a, out_b = hyperretro.gemv_dual_q8_0(x, Wa, Wb)
```

Backend resolution: `cext` (JIT-compiled C extension) → `torch` (pure
torch reference) → `numpy` (always works). Force the fallback with
`HYPERRETRO_FORCE_FALLBACK=1`.

### 2. Offline HuggingFace compression

A single CLI takes a vanilla HF model, runs the GRC projection / sink-aware
GRC pipeline ([Paper E](../ARXIV_SUBMISSIONS/paper-V)), and writes the
result back out as standard `.safetensors` shards that load with stock
`AutoModelForCausalLM.from_pretrained`:

```bash
pip install -e hyperretro[hf]
hyperretro-compress \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --out ./qwen-grc-1024/ \
    --rank 1024 \
    --sink 4
```

The output directory is 100 % HuggingFace-native — no HyperTensor runtime
needed at inference time. A `hyperretro_report.json` is written alongside
recording the per-layer Frobenius rel-err.

### 3. Geodesic speculative draft for vLLM

`hyperretro.vllm.GeodesicDraft` replaces the random / smaller-model draft
proposer in vLLM-style speculative decoding with the geodesic-step
draft from [Paper C](../docs/papers/03-speculative-decoding.html). The
adapter is framework-agnostic (`propose(h_curr, h_prev) -> (token_ids,
confidences)`) and includes a `register_with_vllm()` hook for live
deployments.

## Benchmarks

```bash
hyperretro-bench kernel  --rows 4096 --in-dim 4096
hyperretro-bench spec    --d-model 512 --k 64 --vocab 2048 --steps 64
hyperretro-bench compress --model Qwen/Qwen2.5-0.5B --out /tmp/qwen-retro \
                         --rank 256 --eval-text "The quick brown fox..."
```

Each subcommand emits a JSON report comparing **standard baseline**,
**HyperRetro**, and (where applicable) **standalone HyperTensor**.

## License

MIT for code, CC-BY-4.0 for the accompanying documentation/papers — same
as the parent HyperTensor project.
