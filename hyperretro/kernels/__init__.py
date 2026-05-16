#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""HyperRetro fused kernels — multi-backend with GPU and CPU opt support.

Backend resolution order:
  1. `cuda_cext` — JIT-compiled CUDA kernel (needs NVCC + host compiler)
  2. `cext`      — JIT-compiled C++ extension (needs C++ compiler)
  3. `cpu_opt`   — pre-compiled AVX2 shared library
  4. `gpu`       — pure-PyTorch CUDA tensor ops (no compiler needed, 10-30× numpy)
  5. `torch`     — pure-PyTorch CPU (MKL-accelerated)
  6. `numpy`     — always available

The GPU backend runs all operations on-device via PyTorch tensor ops.
On Windows without Visual Studio, the GPU backend is the recommended path.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple

import numpy as np

# Silence torch's noisy "Error checking compiler" / "Could not find files"
# messages when MSVC is missing on Windows.  We handle fallback gracefully.
import logging as _logging
for _name in ("torch.utils.cpp_extension", "torch", "setuptools"):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

__all__ = ["gemv_dual_q8_0", "backend", "q8_0_quantize", "q8_0_dequantize"]

_BACKEND: str | None = None
_CEXT = None
_CUDA_EXT = None
_CPU_OPT_AVAIL: bool | None = None


def _try_load_cuda_ext():
    """Attempt to JIT-compile the CUDA kernel. Silent fallback.

    Requires NVCC + a host C++ compiler (MSVC/Clang/GCC).
    Falls back gracefully — the pure-PyTorch GPU backend handles CUDA without compilation.
    """
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if os.environ.get("HYPERRETRO_FORCE_FALLBACK"):
        return None
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, load
    except Exception:
        return None
    here = os.path.dirname(__file__)
    cuda_src = os.path.join(here, "csrc", "cuda", "gemv_dual_q8_0.cu")
    if not os.path.exists(cuda_src):
        return None
    # Ensure venv Scripts (ninja) on PATH
    _venv_scripts = os.path.join(sys.prefix, "Scripts") if sys.platform == "win32" else os.path.join(sys.prefix, "bin")
    if _venv_scripts not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _venv_scripts + os.pathsep + os.environ.get("PATH", "")
    try:
        _CUDA_EXT = load(
            name="hyperretro_kernels_cuda",
            sources=[cuda_src],
            extra_cuda_cflags=["-O3", f"--gpu-architecture=sm_{''.join(str(c) for c in torch.cuda.get_device_capability(0))}"],
            verbose=False,
        )
    except Exception as e:
        msg = str(e)
        if "cl.exe" not in msg.lower() and not ("where" in msg.lower() and "cl" in msg.lower()):
            warnings.warn(
                f"hyperretro: CUDA JIT failed ({msg[:120]})",
                stacklevel=2,
            )
        _CUDA_EXT = None
    return _CUDA_EXT


def _has_cpu_opt() -> bool:
    """Check if the compiled CPU AVX2 library is available."""
    global _CPU_OPT_AVAIL
    if _CPU_OPT_AVAIL is not None:
        return _CPU_OPT_AVAIL
    try:
        from .cpu_opt import _has_cpu_opt as _check
        _CPU_OPT_AVAIL = _check()
    except Exception:
        _CPU_OPT_AVAIL = False
    return _CPU_OPT_AVAIL


def _has_cuda() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _try_load_cext():
    """Attempt to JIT-build the C extension. Silently fall back on failure.

    Requires a C++ compiler (gcc/clang/MSVC).  On Windows without Visual Studio,
    the pure-PyTorch GPU backend is used instead — it's 10-30× faster than numpy.
    """
    global _CEXT
    if _CEXT is not None:
        return _CEXT
    if os.environ.get("HYPERRETRO_FORCE_FALLBACK"):
        return None

    # Ensure ninja from the venv is on PATH (pip installs it there)
    _venv_scripts = os.path.join(sys.prefix, "Scripts") if sys.platform == "win32" else os.path.join(sys.prefix, "bin")
    if _venv_scripts not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _venv_scripts + os.pathsep + os.environ.get("PATH", "")

    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load
    except Exception:
        return None
    here = os.path.dirname(__file__)
    src = os.path.join(here, "csrc", "gemv_dual_q8_0.cpp")
    if not os.path.exists(src):
        return None
    try:
        _CEXT = load(
            name="hyperretro_kernels",
            sources=[src],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:
        msg = str(e)
        if "cl.exe" in msg.lower() or ("where" in msg.lower() and "cl" in msg.lower()):
            # Expected on Windows without VS Build Tools — GPU backend works fine
            pass
        else:
            warnings.warn(
                f"hyperretro: failed to JIT-build C extension ({msg[:120]}); "
                "falling back to torch/numpy reference.",
                stacklevel=2,
            )
        _CEXT = None
    return _CEXT


def backend() -> str:
    """Return the backend that will be used for kernel calls.

    Resolution order:
      1. cuda_cext — JIT-compiled CUDA kernel (needs NVCC + MSVC/Clang)
      2. cext      — JIT-compiled C++ extension (needs C++ compiler)
      3. cpu_opt   — pre-compiled AVX2 shared library
      4. gpu       — pure-PyTorch CUDA tensor ops (no compiler needed, 10-30x numpy)
      5. torch     — pure-PyTorch CPU (MKL-accelerated)
      6. numpy     — always available
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    # Try CUDA JIT first (fastest — raw CUDA kernel)
    if _has_cuda() and _try_load_cuda_ext() is not None:
        _BACKEND = "cuda_cext"
        return _BACKEND
    # Try C++ JIT (CPU parallel)
    if _try_load_cext() is not None:
        _BACKEND = "cext"
        return _BACKEND
    # Try pre-compiled AVX2
    if _has_cpu_opt():
        _BACKEND = "cpu_opt"
        return _BACKEND
    # Pure-PyTorch CUDA — fast, no compiler needed
    if _has_cuda() and not os.environ.get("HYPERRETRO_FORCE_CPU") and not os.environ.get("HYPERRETRO_FORCE_FALLBACK"):
        _BACKEND = "gpu"
        return _BACKEND
    # Pure-PyTorch CPU
    try:
        import torch  # noqa: F401
        _BACKEND = "torch"
    except Exception:
        _BACKEND = "numpy"
    return _BACKEND


# -----------------------------------------------------------------------------
# Q8_0 quantization (32-element blocks, scale-per-block fp32 + int8 codes).
# Layout: for each row,  blocks = in_dim/32; per block: float32 scale, int8[32].
# -----------------------------------------------------------------------------

BLOCK = 32


def q8_0_quantize(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a [rows, in_dim] float matrix to Q8_0.

    Returns (scales[rows, n_blocks] float32, codes[rows, n_blocks, 32] int8).
    `in_dim` must be a multiple of 32.
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, in_dim = W.shape
    assert in_dim % BLOCK == 0, "in_dim must be a multiple of 32"
    n_blocks = in_dim // BLOCK
    Wb = W.reshape(rows, n_blocks, BLOCK)
    amax = np.max(np.abs(Wb), axis=-1)            # [rows, n_blocks]
    scale = amax / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    codes = np.round(Wb / scale[..., None]).clip(-128, 127).astype(np.int8)
    return scale.astype(np.float32), codes


def q8_0_dequantize(scale: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Inverse of q8_0_quantize."""
    return (codes.astype(np.float32) * scale[..., None]).reshape(
        scale.shape[0], -1
    )


# -----------------------------------------------------------------------------
# Public kernel: fused dual Q8_0 GEMV.  Mirrors runtime/nn/cuda_kernels.cu
# kernel_gemv_dual_q8_0, but accepts (scale, codes) tuples for portability.
# -----------------------------------------------------------------------------

def _as_q8_pair(W):
    """Accept either a packed (scale, codes) tuple or a float matrix."""
    if isinstance(W, tuple) and len(W) == 2:
        return W
    if hasattr(W, "detach"):
        import torch
        if torch.is_tensor(W):
            W = W.detach().cpu().numpy()
    return q8_0_quantize(np.asarray(W, dtype=np.float32))


def _gemv_dual_q8_numpy(x, W_a, W_b):
    x = np.ascontiguousarray(x, dtype=np.float32)
    sa, ca = _as_q8_pair(W_a)
    sb, cb = _as_q8_pair(W_b)
    rows, n_blocks, _ = ca.shape
    assert x.shape[-1] == n_blocks * BLOCK
    xb = x.reshape(n_blocks, BLOCK)
    # Per row: sum_blocks scale * (codes . xb)
    dot_a = np.einsum("rbn,bn->rb", ca.astype(np.float32), xb)
    dot_b = np.einsum("rbn,bn->rb", cb.astype(np.float32), xb)
    out_a = np.sum(dot_a * sa, axis=1)
    out_b = np.sum(dot_b * sb, axis=1)
    return out_a, out_b


def _gemv_dual_q8_torch(x, W_a, W_b, dtype=None):
    import torch
    # Quantize off-tensor (numpy) for parity; in production these would be
    # pre-quantized weights stored in q8_0 format.
    sa_np, ca_np = _as_q8_pair(W_a)
    sb_np, cb_np = _as_q8_pair(W_b)
    if dtype is None:
        dtype = torch.float32
    if torch.is_tensor(x):
        device = x.device
    else:
        device = torch.device("cpu")
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
    x = x.to(dtype=dtype)
    sa = torch.from_numpy(sa_np).to(device=device, dtype=dtype)
    ca = torch.from_numpy(ca_np).to(device=device, dtype=dtype)
    sb = torch.from_numpy(sb_np).to(device=device, dtype=dtype)
    cb = torch.from_numpy(cb_np).to(device=device, dtype=dtype)
    rows, n_blocks, B = ca.shape
    xb = x.reshape(n_blocks, B)
    out_a = (sa * (ca * xb).sum(dim=-1)).sum(dim=-1)
    out_b = (sb * (cb * xb).sum(dim=-1)).sum(dim=-1)
    return out_a, out_b


def _gemv_dual_q8_cext(x, W_a, W_b):
    import torch
    sa_np, ca_np = _as_q8_pair(W_a)
    sb_np, cb_np = _as_q8_pair(W_b)
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    x = x.contiguous().to(torch.float32).cpu()
    sa = torch.from_numpy(np.ascontiguousarray(sa_np))
    ca = torch.from_numpy(np.ascontiguousarray(ca_np))
    sb = torch.from_numpy(np.ascontiguousarray(sb_np))
    cb = torch.from_numpy(np.ascontiguousarray(cb_np))
    out_a, out_b = _CEXT.gemv_dual_q8_0(x, sa, ca, sb, cb)
    return out_a, out_b


def gemv_dual_q8_0(x, W_a, W_b, dtype=None):
    """Fused dual Q8_0 GEMV.

    Computes simultaneously::

        out_a = W_a @ x
        out_b = W_b @ x

    sharing the input load (HyperTensor's ``gemv_dual_q8_0`` pattern from
    ``runtime/nn/cuda_kernels.cu``, ~16% DRAM-traffic reduction vs two
    separate GEMVs).

    Args:
        x:    [in_dim] activation (np.ndarray or torch.Tensor)
        W_a:  [rows, in_dim] weight, or pre-quantized (scale, codes) tuple
        W_b:  same shape/format as W_a
        dtype: torch.dtype for GPU/torch backends (default float32).
            Use torch.float16 or torch.bfloat16 for speed.

    Returns:
        (out_a, out_b) — type matches the backend (np or torch).
    """
    b = backend()
    if b == "cext":
        return _gemv_dual_q8_cext(x, W_a, W_b)
    if b == "cpu_opt":
        from .cpu_opt import _gemv_dual_q8_cpu_opt
        return _gemv_dual_q8_cpu_opt(x, W_a, W_b)
    if b == "gpu":
        from .gpu import gemv_dual_q8_gpu
        return gemv_dual_q8_gpu(x, W_a, W_b, compute_dtype=dtype)
    if b == "torch":
        return _gemv_dual_q8_torch(x, W_a, W_b, dtype=dtype)
    return _gemv_dual_q8_numpy(x, W_a, W_b)
