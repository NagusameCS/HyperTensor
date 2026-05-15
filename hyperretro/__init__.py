"""HyperRetro: retrofit HyperTensor's geometric optimizations into the
standard PyTorch / HuggingFace / vLLM ecosystem.

Three prongs:
  - hyperretro.kernels:  fused kernels (CPU torch + GPU torch + CUDA source)
  - hyperretro.hf:       offline HF compression + GRC light distillation
  - hyperretro.vllm:     speculative draft via compressed-model drafter

Public re-exports below give the FlashAttention-style ergonomic surface:

    >>> import hyperretro
    >>> y_a, y_b = hyperretro.gemv_dual_q8_0(x, W_a_q8, W_b_q8)

License: MIT (code) + CC-BY-4.0 (papers/docs).
"""
from __future__ import annotations

__version__ = "0.3.3"
__all__ = [
    "gemv_dual_q8_0",
    "kernels_backend",
    "q8_0_quantize",
    "q8_0_dequantize",
    "compress_hf_model",
    "distill_hf_model",
    "CompressedDrafter",
    "KSpaceDrafter",
    "GeodesicDraft",  # alias for CompressedDrafter
    "register_hyperretro_drafter",  # vLLM integration
    # Unified model layer (v0.2.1)
    "load_model",
    "compress",
    "export_model",
    "list_backends",
    "list_formats",
    # Certificates & benchmarks (v0.3.0)
    "certify_compression",
    "run_kernel_bench",
    "run_compression_bench",
    # Geometric tools — re-exported from hypercore (v0.3.0)
    "GeodesicMetric",
    "HallucinationGuard",
    "GenerationMetrics",
    "AxiomGauge",
    "ThermalRankController",
    "OnlineOjaBasis",
    "TreeDrafter",
    "EagleFeatureDrafter",
    # Red team (v0.3.0)
    "GCGAttack",
    "AutoPromptAttack",
    "PAIRAttack",
    # Native k-space training (v0.3.0)
    "NativeLinear",
    "RiemannianAdamW",
    "KExpansionScheduler",
]


def gemv_dual_q8_0(x, W_a, W_b):
    """Lazy proxy so importing the top-level package does not require torch."""
    from .kernels import gemv_dual_q8_0 as _impl

    return _impl(x, W_a, W_b)


def kernels_backend() -> str:
    """Return the active kernels backend: 'cext', 'gpu', 'torch', or 'numpy'."""
    from .kernels import backend

    return backend()


def compress_hf_model(*args, **kwargs):
    from .hf.compress import compress_hf_model as _impl

    return _impl(*args, **kwargs)


def distill_hf_model(*args, **kwargs):
    from .hf.distill import distill_hf_model as _impl

    return _impl(*args, **kwargs)


class _LazyDrafter:
    """Lazy proxy that imports the real class on first instantiation."""
    _class_name: str

    def __new__(cls, *args, **kwargs):
        from .vllm import draft
        real_cls = getattr(draft, cls._class_name)
        return real_cls(*args, **kwargs)


class CompressedDrafter(_LazyDrafter):
    """Speculative draft proposer backed by a GRC-compressed HF model (Paper III)."""
    _class_name = "CompressedDrafter"


class KSpaceDrafter(_LazyDrafter):
    """Legacy k-space geodesic drafter — research use only."""
    _class_name = "KSpaceDrafter"


class GeodesicDraft(_LazyDrafter):
    """Alias for CompressedDrafter (backward-compatible name)."""
    _class_name = "CompressedDrafter"


# ---------------------------------------------------------------------------
# Unified model layer (v0.2.1) — lazy proxies
# ---------------------------------------------------------------------------

def load_model(*args, **kwargs):
    """Load any model (HF, OpenMythos, local) via unified abstraction."""
    from hyperretro.models import load_model as _impl
    return _impl(*args, **kwargs)


def compress(*args, **kwargs):
    """Compress a model with HyperRetro (aware-SVD + int4)."""
    from hyperretro.models import compress_model as _impl
    return _impl(*args, **kwargs)


def export_model(*args, **kwargs):
    """Export to GGUF, safetensors, or HF format."""
    from hyperretro.models import export_model as _impl
    return _impl(*args, **kwargs)


def list_backends():
    """List available model backends (huggingface, openmythos)."""
    from hyperretro.models import list_backends as _impl
    return _impl()


def list_formats():
    """List supported export formats."""
    from hyperretro.models import list_formats as _impl
    return _impl()


def register_hyperretro_drafter(*args, **kwargs):
    """Register HyperRetro drafter for vLLM speculative decoding."""
    from .vllm_adapter import register_hyperretro_drafter as _impl
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Kernel quantize/dequantize (v0.3.0)
# ---------------------------------------------------------------------------

def q8_0_quantize(*args, **kwargs):
    """Quantize a float matrix to Q8_0 (block_size=32, int8 codes + fp32 scales)."""
    from .kernels import q8_0_quantize as _impl
    return _impl(*args, **kwargs)


def q8_0_dequantize(*args, **kwargs):
    """Dequantize a Q8_0 matrix back to float."""
    from .kernels import q8_0_dequantize as _impl
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Compression certificates (v0.3.0)
# ---------------------------------------------------------------------------

def certify_compression(*args, **kwargs):
    """Produce a verifiable quality certificate with trust tier and PPL bounds."""
    from .certificates import certify_compression as _impl
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark harness (v0.3.0)
# ---------------------------------------------------------------------------

def run_kernel_bench(*args, **kwargs):
    """Benchmark the fused dual-Q8_0 GEMV kernel vs baselines."""
    from .bench.run import run_kernel_bench as _impl
    return _impl(*args, **kwargs)


def run_compression_bench(*args, **kwargs):
    """Benchmark HyperRetro compression vs HF baseline."""
    from .bench.run import run_compression_bench as _impl
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Geometric tools — lazy re-export from hypercore (v0.3.0)
# ---------------------------------------------------------------------------

def __getattr__(name):
    """Lazy import for hypercore geometric modules.

    HyperRetro's core compression pipeline (compress, export, kernels,
    certificates) works without hypercore.  The geometric tools below
    are optional enhancements — install hypercore to unlock them:

        pip install git+https://github.com/NagusameCS/HyperTensor.git#subdirectory=hypercore
    """
    _HYPERCORE_NAMES = (
        "GeodesicMetric", "HallucinationGuard", "GenerationMetrics",
        "AxiomGauge", "ThermalRankController", "OnlineOjaBasis",
        "TreeDrafter", "EagleFeatureDrafter",
        "GCGAttack", "AutoPromptAttack", "PAIRAttack",
        "NativeLinear", "RiemannianAdamW", "KExpansionScheduler",
    )
    if name not in _HYPERCORE_NAMES:
        raise AttributeError(f"module 'hyperretro' has no attribute '{name}'")

    try:
        import hypercore
    except ImportError:
        raise AttributeError(
            f"'{name}' requires hypercore, which is not installed. "
            f"Install with: pip install git+https://github.com/NagusameCS/"
            f"HyperTensor.git#subdirectory=hypercore"
        ) from None

    if name in ("GeodesicMetric", "HallucinationGuard", "GenerationMetrics"):
        return {
            "GeodesicMetric": hypercore.GeodesicMetric,
            "HallucinationGuard": hypercore.HallucinationGuard,
            "GenerationMetrics": hypercore.GenerationMetrics,
        }[name]
    return getattr(hypercore, name)
