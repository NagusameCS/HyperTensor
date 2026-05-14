"""HyperRetro fused kernels — PyTorch C++ extension with fallbacks.

Backend resolution order:
  1. `cext`  — JIT-compiled C extension via torch.utils.cpp_extension.load.
               Mirrors HyperTensor's runtime/nn fused kernel layout.
  2. `torch` — pure-torch reference (fast, exact float math, GPU-capable).
  3. `numpy` — pure-numpy reference, always available.

All three paths return identical numeric results up to fp32 rounding.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple

import numpy as np

__all__ = ["gemv_dual_q8_0", "backend", "q8_0_quantize", "q8_0_dequantize"]

_BACKEND: str | None = None
_CEXT = None


def _try_load_cext():
    """Attempt to JIT-build the C extension. Silently fall back on failure."""
    global _CEXT
    if _CEXT is not None:
        return _CEXT
    if os.environ.get("HYPERRETRO_FORCE_FALLBACK"):
        return None
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
        warnings.warn(
            f"hyperretro: failed to JIT-build C extension ({e!r}); "
            "falling back to torch/numpy reference.",
            stacklevel=2,
        )
        _CEXT = None
    return _CEXT


def backend() -> str:
    """Return the backend that will be used for kernel calls."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if _try_load_cext() is not None:
        _BACKEND = "cext"
        return _BACKEND
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


def _gemv_dual_q8_torch(x, W_a, W_b):
    import torch
    # Quantize off-tensor (numpy) for parity; in production these would be
    # pre-quantized weights stored in q8_0 format.
    sa_np, ca_np = _as_q8_pair(W_a)
    sb_np, cb_np = _as_q8_pair(W_b)
    if torch.is_tensor(x):
        device = x.device
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        x = torch.as_tensor(x, dtype=dtype, device=device)
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


def gemv_dual_q8_0(x, W_a, W_b):
    """Fused dual Q8_0 GEMV.

    Computes simultaneously::

        out_a = W_a @ x
        out_b = W_b @ x

    sharing the input load (HyperTensor's `gemv_dual_q8_0` pattern from
    `runtime/nn/cuda_kernels.cu`, ~16% DRAM-traffic reduction vs two
    separate GEMVs).

    Args:
        x:    [in_dim] activation (np.ndarray or torch.Tensor)
        W_a:  [rows, in_dim] weight, or pre-quantized (scale, codes) tuple
        W_b:  same shape/format as W_a

    Returns:
        (out_a, out_b) — type matches the backend (np or torch).
    """
    b = backend()
    if b == "cext":
        return _gemv_dual_q8_cext(x, W_a, W_b)
    if b == "torch":
        return _gemv_dual_q8_torch(x, W_a, W_b)
    return _gemv_dual_q8_numpy(x, W_a, W_b)
