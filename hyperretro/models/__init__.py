"""HyperTensor unified model abstraction layer.

HyperTensor is a universal model compression and optimization layer.
It doesn't care whether your model comes from HuggingFace, OpenMythos,
or a custom architecture — the same compression, quantization,
distillation, and export tools work on all of them.

This module provides the abstract base and registry:

    from hyperretro.models import load_model, compress_model, export_model

    # Auto-detect and load any supported model type
    model = load_model("Qwen/Qwen2.5-1.5B")       # HuggingFace
    model = load_model("mythos_1b")                # OpenMythos
    model = load_model("/path/to/safetensors")     # Local checkpoint

    # Compress (same API for all model types)
    compressed = compress_model(model, ffn_rank=1024, int4=True)

    # Export to industry-standard formats
    export_model(compressed, "model.gguf")         # llama.cpp
    export_model(compressed, "compressed/")         # HF safetensors

Supported backends:
    - HuggingFace transformers (always available)
    - OpenMythos (optional: pip install open-mythos)
    - Local safetensors checkpoints (auto-detected)

Industry infrastructure supported:
    - safetensors (save/load)
    - GGUF (llama.cpp / Ollama)
    - HuggingFace config.json + tokenizer
    - HyperRetro factored format (FactoredLinear)
    - GRC certificates
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import sys


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_SUPPORTED_BACKENDS: dict[str, type] = {}
_BACKEND_LOADERS: dict[str, callable] = {}
_MODEL_CACHE: dict[str, "AbstractModel"] = {}


def register_backend(name: str, adapter_cls: type, loader: callable):
    """Register a model backend.

    Args:
        name: backend name (e.g. "huggingface", "openmythos")
        adapter_cls: the AbstractModel subclass
        loader: function that takes a model_id and returns (model, config_dict)
    """
    _SUPPORTED_BACKENDS[name] = adapter_cls
    _BACKEND_LOADERS[name] = loader


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractModel:
    """Abstract base for all model backends.

    Each backend (HuggingFace, OpenMythos, etc.) implements this interface.
    All HyperTensor tools operate on AbstractModel instances — they never
    need to know the underlying implementation.
    """

    backend: str = "unknown"

    # -- Required interface --

    @property
    def config(self) -> dict:
        """Model configuration as a normalized dict."""
        raise NotImplementedError

    @property
    def state_dict(self) -> dict:
        """Model state dict (parameter_name -> tensor)."""
        raise NotImplementedError

    @property
    def architecture(self) -> str:
        """Architecture identifier (e.g. 'qwen2', 'mythos', 'llama')."""
        raise NotImplementedError

    @property
    def param_count(self) -> int:
        """Total parameter count."""
        raise NotImplementedError

    @property
    def model_obj(self):
        """The underlying model object (torch.nn.Module or similar)."""
        raise NotImplementedError

    def forward(self, input_ids, **kwargs):
        """Run a forward pass. Returns logits."""
        raise NotImplementedError

    def save(self, path: Union[str, Path], **kwargs):
        """Save model to disk in standard format."""
        raise NotImplementedError

    def load_state_dict_from(self, state_dict: dict, strict: bool = False):
        """Load weights into the model."""
        raise NotImplementedError

    # -- Optional interface (backends may override) --

    def get_attention_layers(self) -> list[dict]:
        """Return list of attention layer descriptions for compression."""
        return []

    def get_ffn_layers(self) -> list[dict]:
        """Return list of FFN layer descriptions for compression."""
        return []

    def get_vocab_size(self) -> int:
        return self.config.get("vocab_size", 0)

    def get_hidden_size(self) -> int:
        return self.config.get("hidden_size", self.config.get("dim", 0))

    def get_num_layers(self) -> int:
        return self.config.get("num_hidden_layers", self.config.get("n_layers", 0))

    def to(self, device):
        self.model_obj.to(device)
        return self

    def eval(self):
        self.model_obj.eval()
        return self

    def train(self):
        self.model_obj.train()
        return self


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_model_type(model_id: str) -> str:
    """Detect what kind of model `model_id` refers to.

    Detection order:
    1. OpenMythos variant name (mythos_1b, mythos_3b, etc.)
    2. GGUF file (.gguf extension)
    3. HuggingFace repo ID (contains '/')
    4. Local directory (config.json, safetensors, etc.)
    5. OpenMythos class lookup (via importlib)
    """
    path = Path(model_id)

    # 1. OpenMythos variant names
    if model_id.startswith("mythos_"):
        return "openmythos"

    # 2. HuggingFace repo IDs
    if "/" in model_id and not path.exists():
        return "huggingface"

    # 3. Local directory
    if path.is_dir():
        # Check for HuggingFace config
        if (path / "config.json").exists():
            return "huggingface"
        # Check for HyperRetro factored manifest
        if (path / "hyperretro_factored.json").exists():
            return "huggingface"  # factored checkpoints use HF skeleton
        # Check for safetensors
        if list(path.glob("*.safetensors")):
            return "huggingface"

    # 3b. Local GGUF file
    if path.is_file() and path.suffix in (".gguf", ".GGUF"):
        return "gguf"
    if model_id.endswith(".gguf"):
        return "gguf"

    # 4. Try OpenMythos class lookup (via importlib to avoid heavy import)
    from importlib.util import find_spec
    if find_spec("open_mythos") is not None:
        try:
            from open_mythos import OpenMythos, MythosConfig
        except ImportError:
            pass

    return "huggingface"  # default


# ---------------------------------------------------------------------------
# Public API: load / compress / export
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    *,
    backend: str | None = None,
    device: str = "cpu",
    dtype: str = "float16",
    **kwargs,
) -> AbstractModel:
    """Load a model from any supported source.

    Args:
        model_id: HuggingFace repo ID, OpenMythos variant name,
                  local directory path, or model class.
        backend: force a specific backend ('huggingface' or 'openmythos').
                 Auto-detected if None.
        device: 'cpu' or 'cuda'.
        dtype: 'float16', 'bfloat16', or 'float32'.
        **kwargs: passed to the backend loader.

    Returns:
        An AbstractModel instance.
    """
    if backend is None:
        backend = _detect_model_type(model_id)

    if backend not in _SUPPORTED_BACKENDS:
        # Lazy-register if available
        if backend == "huggingface":
            _register_hf()
        elif backend == "openmythos":
            _register_om()
        elif backend == "gguf":
            _register_gguf()
        elif backend == "vllm":
            _register_vllm()

    if backend not in _BACKEND_LOADERS:
        raise ValueError(
            f"Backend '{backend}' is not available. "
            f"Supported: {list(_BACKEND_LOADERS.keys())}. "
            f"Install extra dependencies if needed."
        )

    loader = _BACKEND_LOADERS[backend]
    raw_model, config_dict = loader(model_id, device=device, dtype=dtype, **kwargs)

    adapter_cls = _SUPPORTED_BACKENDS[backend]
    model = adapter_cls(raw_model, config_dict)
    return model


def compress_model(
    model: AbstractModel,
    *,
    ffn_rank: int = 1024,
    attn_rank: int = 0,
    int4: bool = True,
    int4_block_size: int = 128,
    int4_awq: bool = True,
    activation_corpus: str | None = None,
    **kwargs,
) -> "CompressedModel":
    """Compress a model using HyperRetro.

    Works on any supported model type. The compression strategy is
    automatically adapted to the model architecture.

    Args:
        model: any AbstractModel instance.
        ffn_rank: target SVD rank for FFN matrices.
        attn_rank: GRC rank for attention (0 = skip).
        int4: apply block-wise int4 quantization.
        int4_block_size: quantization block size.
        int4_awq: use AWQ-aware quantization.
        activation_corpus: path to calibration text for AWQ.
        **kwargs: backend-specific options.

    Returns:
        CompressedModel wrapping the compressed state dict + manifest.
    """
    from hyperretro.models._compress import _compress_abstract_model
    return _compress_abstract_model(
        model,
        ffn_rank=ffn_rank,
        attn_rank=attn_rank,
        int4=int4,
        int4_block_size=int4_block_size,
        int4_awq=int4_awq,
        activation_corpus=activation_corpus,
        **kwargs,
    )


class CompressedModel:
    """A compressed model: state_dict + manifest, ready to export.

    Not tied to any specific backend — can be exported to GGUF,
    safetensors, or loaded back into a backend-specific model.
    """

    def __init__(self, state_dict: dict, manifest: dict, source_backend: str,
                 source_config: dict):
        self.state_dict = state_dict
        self.manifest = manifest
        self.source_backend = source_backend
        self.source_config = source_config

    @property
    def total_tensors(self) -> int:
        return len(self.state_dict)

    def export(self, path: str | Path, format: str = "auto", **kwargs):
        """Export to a standard format.

        Args:
            path: output path (file for GGUF, directory for safetensors).
            format: 'gguf', 'safetensors', 'hf', or 'auto' (from extension).
            **kwargs: format-specific options.
        """
        return export_model(self, path, format=format, **kwargs)


def export_model(
    model: AbstractModel | CompressedModel,
    path: str | Path,
    *,
    format: str = "auto",
    **kwargs,
) -> Path:
    """Export a model to an industry-standard format.

    Args:
        model: AbstractModel or CompressedModel.
        path: output path.
        format: 'gguf', 'safetensors', 'hf', or 'auto'.
        **kwargs: format-specific options.

    Returns:
        Output path.
    """
    from hyperretro.models._export import _export_model
    return _export_model(model, path, format=format, **kwargs)


# ---------------------------------------------------------------------------
# Lazy backend registration
# ---------------------------------------------------------------------------

def _register_hf():
    """Register the HuggingFace backend (always available)."""
    from hyperretro.models.hf import HuggingFaceAdapter, _load_hf_model
    register_backend("huggingface", HuggingFaceAdapter, _load_hf_model)


def _register_om():
    """Register the OpenMythos backend (optional)."""
    try:
        from hyperretro.models.om import OpenMythosAdapter, _load_om_model
        register_backend("openmythos", OpenMythosAdapter, _load_om_model)
    except ImportError as e:
        raise ImportError(
            "OpenMythos backend requires 'open-mythos' package. "
            "Install with: pip install open-mythos"
        ) from e


def _register_gguf():
    """Register the GGUF backend (optional — reads llama.cpp / Ollama files)."""
    try:
        from hyperretro.models.gguf import GGUFAdapter, _load_gguf
        register_backend("gguf", GGUFAdapter, _load_gguf)
    except ImportError as e:
        raise ImportError(
            "GGUF backend requires: pip install gguf"
        ) from e


def _register_vllm():
    """Register the vLLM backend (optional — needs GPU + vllm package)."""
    try:
        from hyperretro.models.vllm_model import VLLMAdapter, _load_vllm
        register_backend("vllm", VLLMAdapter, _load_vllm)
    except ImportError as e:
        raise ImportError(
            "vLLM backend requires: pip install vllm"
        ) from e


# Auto-register HF on import
_register_hf()


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def list_backends() -> dict[str, bool]:
    """List available backends and their status."""
    backends = {"huggingface": True}
    try:
        import open_mythos  # noqa: F401
        backends["openmythos"] = True
    except ImportError:
        backends["openmythos"] = False
    try:
        import gguf  # noqa: F401
        backends["gguf"] = True
    except ImportError:
        backends["gguf"] = False
    # vLLM is heavyweight — check via importlib to avoid loading CUDA on import
    from importlib.util import find_spec
    backends["vllm"] = find_spec("vllm") is not None
    return backends


def list_formats() -> list[str]:
    """List supported export formats."""
    return ["safetensors", "gguf", "hf"]
