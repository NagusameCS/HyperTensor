"""HyperRetro: retrofit HyperTensor's geometric optimizations into the
standard PyTorch / HuggingFace / vLLM ecosystem.

Three prongs:
  - hyperretro.kernels:  fused kernels packaged as a PyTorch C++ extension
                         (gemv_dual_q8_0, with a NumPy reference fallback)
  - hyperretro.hf:       offline HuggingFace compression
                         (UGT / GRC / sink-aware projection -> .safetensors)
  - hyperretro.vllm:     vLLM-shaped speculative draft-model adapter
                         (geodesic step in k-space, GRC trajectory cache)

Public re-exports below give the FlashAttention-style ergonomic surface:

    >>> import hyperretro
    >>> y_a, y_b = hyperretro.gemv_dual_q8_0(x, W_a_q8, W_b_q8)

License: MIT (code) + CC-BY-4.0 (papers/docs).
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "gemv_dual_q8_0",
    "kernels_backend",
    "compress_hf_model",
    "GeodesicDraft",
]


def gemv_dual_q8_0(x, W_a, W_b):
    """Lazy proxy so importing the top-level package does not require torch."""
    from .kernels import gemv_dual_q8_0 as _impl

    return _impl(x, W_a, W_b)


def kernels_backend() -> str:
    """Return the active kernels backend: 'cext', 'numpy', or 'torch'."""
    from .kernels import backend

    return backend()


def compress_hf_model(*args, **kwargs):
    from .hf.compress import compress_hf_model as _impl

    return _impl(*args, **kwargs)


class GeodesicDraft:
    """Lazy proxy for hyperretro.vllm.draft.GeodesicDraft."""

    def __new__(cls, *args, **kwargs):
        from .vllm.draft import GeodesicDraft as _impl

        return _impl(*args, **kwargs)
