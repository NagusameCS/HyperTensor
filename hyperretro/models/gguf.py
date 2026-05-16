"""GGUF model backend — load llama.cpp / Ollama GGUF files as AbstractModel.

Enables:  hyperretro compress llama-3b-q4_k_m.gguf --ffn-rank 256

Uses the `gguf` library for metadata + tensor access.  Falls through
gracefully if gguf is not installed.

Supports: Llama, Mistral, Qwen2, Phi, Gemma, GPT-2 GGUF files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import json
import struct

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from hyperretro.models import AbstractModel


class GGUFAdapter(AbstractModel):
    """Wraps a GGUF file as a HyperTensor AbstractModel.

    Lazy-loads tensors on first access to state_dict — reads metadata
    at construction but defers heavy MMAP/decompression until needed.
    """

    backend = "gguf"

    def __init__(self, gguf_path: str | Path):
        try:
            from gguf import GGUFReader
        except ImportError:
            raise ImportError(
                "GGUF backend requires: pip install gguf"
            ) from None

        self._path = Path(gguf_path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self._path}")

        self._reader = GGUFReader(str(self._path))
        self._config = self._extract_config()
        self._state_dict_cache = None
        self._model_obj = None
        self._tensor_info = {}
        self._index_tensors()

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------

    def _extract_config(self) -> dict:
        """Build a HF-compatible config dict from GGUF metadata."""
        f = self._reader.fields

        def _get(key, default=None):
            v = f.get(key)
            if v is None:
                return default
            parts = getattr(v, 'parts', None)
            if not parts:
                return default
            # For GGUF fields: numeric → parts[-1].item(), string → bytes(parts[-1]).decode()
            val = parts[-1]
            # Check if it's a string (array of uint8 bytes)
            if hasattr(val, 'dtype') and val.dtype.kind == 'u' and val.size > 1:
                try:
                    return bytes(val).decode('utf-8', errors='replace').rstrip('\x00')
                except Exception:
                    pass
            # Numeric: extract scalar
            if hasattr(val, 'item'):
                try:
                    return val.item()
                except Exception:
                    pass
            return val

        arch = str(_get("general.architecture", "llama"))
        cfg = {
            "architectures": [self._arch_to_hf(arch)],
            "model_type": arch,
            "hidden_size": int(_get(f"{arch}.embedding_length", _get(f"{arch}.hidden_size", 4096))),
            "num_hidden_layers": int(_get(f"{arch}.block_count", 32)),
            "num_attention_heads": int(_get(f"{arch}.attention.head_count", 32)),
            "num_key_value_heads": int(_get(f"{arch}.attention.head_count_kv",
                                            _get(f"{arch}.attention.head_count", 32))),
            "intermediate_size": int(_get(f"{arch}.feed_forward_length", 14336)),
            "vocab_size": int(_get(f"{arch}.vocab_size", 32000)),
            "max_position_embeddings": int(_get(f"{arch}.context_length",
                                                _get(f"{arch}.rope.dimension_count", 2048))),
            "rms_norm_eps": float(_get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)),
            "rope_theta": float(_get(f"{arch}.rope.freq_base", 10000.0)),
            "torch_dtype": "float16",
            "tie_word_embeddings": bool(_get(f"{arch}.tensor_data_layout")),
        }
        self._arch = arch
        return cfg

    @staticmethod
    def _arch_to_hf(gguf_arch: str) -> str:
        _map = {
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "falcon": "FalconForCausalLM",
            "gpt2": "GPT2LMHeadModel",
            "gptneox": "GPTNeoXForCausalLM",
            "mpt": "MptForCausalLM",
            "baichuan": "BaiChuanForCausalLM",
            "starcoder": "GPTBigCodeForCausalLM",
            "refact": "RefactForCausalLM",
            "bert": "BertForMaskedLM",
            "bloom": "BloomForCausalLM",
            "stablelm": "StableLmForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen2moe": "Qwen2MoeForCausalLM",
            "phi2": "PhiForCausalLM",
            "phi3": "Phi3ForCausalLM",
            "plamo": "PLAMOForCausalLM",
            "codeshell": "CodeShellForCausalLM",
            "orion": "OrionForCausalLM",
            "gemma": "GemmaForCausalLM",
            "gemma2": "Gemma2ForCausalLM",
            "starcoder2": "Starcoder2ForCausalLM",
            "command-r": "CohereForCausalLM",
            "dbrx": "DbrxForCausalLM",
            "olmo": "OlmoForCausalLM",
            "jais": "JaisForCausalLM",
            "openelm": "OpenELMForCausalLM",
            "arctic": "ArcticForCausalLM",
            "deepseek": "DeepseekForCausalLM",
            "deepseek2": "DeepseekV2ForCausalLM",
            "chatglm": "ChatGLMForCausalLM",
            "bitnet": "BitnetForCausalLM",
            "t5": "T5ForConditionalGeneration",
            "t5encoder": "T5EncoderModel",
            "jamba": "JambaForCausalLM",
            "grok": "GrokForCausalLM",
            "exaone": "ExaoneForCausalLM",
            "granite": "GraniteForCausalLM",
            "granitemoe": "GraniteMoeForCausalLM",
            "chameleon": "ChameleonForCausalLM",
        }
        return _map.get(gguf_arch, "LlamaForCausalLM")

    # ------------------------------------------------------------------
    # Tensor index
    # ------------------------------------------------------------------

    def _index_tensors(self):
        """Build a lazy tensor index from GGUF tensor infos."""
        for tensor in self._reader.tensors:
            name = tensor.name
            # GGUF stores shape as numpy memmap of uint64: e.g. [d1, d2, ...]
            raw = tensor.shape
            try:
                if hasattr(raw, 'tolist'):
                    shape = [int(s) for s in raw.tolist()]
                elif raw is None:
                    shape = []
                else:
                    shape = [int(s) for s in list(raw)]
            except Exception:
                shape = []
            self._tensor_info[name] = {
                "name": name,
                "shape": tuple(shape),
                "n_elements": int(np.prod(shape)) if shape else 0,
                "tensor_obj": tensor,
            }

    # ------------------------------------------------------------------
    # AbstractModel interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict:
        return self._config

    @property
    def state_dict(self) -> dict:
        """Lazy-load all tensors into memory.  Warning: decompresses Q4/Q8."""
        if self._state_dict_cache is not None:
            return self._state_dict_cache

        sd = {}
        for name, info in self._tensor_info.items():
            try:
                data = info["tensor_obj"].data
                if data is None:
                    continue
                if _HAS_TORCH:
                    sd[name] = torch.from_numpy(np.array(data))
                else:
                    sd[name] = np.array(data)
            except Exception:
                continue

        self._state_dict_cache = sd
        return sd

    @property
    def architecture(self) -> str:
        return self._arch

    @property
    def param_count(self) -> int:
        return sum(t["n_elements"] for t in self._tensor_info.values())

    @property
    def model_obj(self):
        return None  # GGUF is not a live model

    def forward(self, input_ids, **kwargs):
        raise NotImplementedError(
            "GGUF backend is read-only for compression. "
            "Use hyperretro export to export the compressed model back to GGUF."
        )

    def save(self, path: Union[str, Path], **kwargs):
        from hyperretro.hf.gguf_export import export_gguf
        path = Path(path)
        sd_flat = {}
        for k, v in self.state_dict.items():
            sd_flat[k] = v.cpu().numpy() if _HAS_TORCH and hasattr(v, 'numpy') else v
        export_gguf(sd_flat, self.config, str(path),
                    model_name=kwargs.get("name", "hyperretro"))

    def load_state_dict_from(self, state_dict: dict, strict: bool = False):
        self._state_dict_cache = state_dict

    def get_ffn_layers(self) -> list[dict]:
        arch = self._arch
        layers = []
        n = self.config.get("num_hidden_layers", 0)
        for i in range(n):
            layer_keys = {}
            prefix = f"blk.{i}." if arch in ("llama", "mistral", "qwen2") else f"model.layers.{i}."
            for suffix in ("ffn_gate", "ffn_up", "ffn_down",
                           "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                           "mlp.c_fc", "mlp.c_proj"):
                full = f"{prefix}{suffix}.weight"
                if full in self._tensor_info:
                    layer_keys[suffix] = full
            if layer_keys:
                layers.append({"layer_idx": i, "keys": layer_keys})
        return layers

    def get_attention_layers(self) -> list[dict]:
        arch = self._arch
        layers = []
        n = self.config.get("num_hidden_layers", 0)
        for i in range(n):
            layer_keys = {}
            prefix = f"blk.{i}." if arch in ("llama", "mistral", "qwen2") else f"model.layers.{i}."
            for suffix in ("attn_q", "attn_k", "attn_v",
                           "attn_output", "self_attn.q_proj",
                           "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj"):
                full = f"{prefix}{suffix}.weight"
                if full in self._tensor_info:
                    layer_keys[suffix.split(".")[-1]] = full
            if layer_keys:
                layers.append({"layer_idx": i, "keys": layer_keys})
        return layers


# ---------------------------------------------------------------------------
# Loader for the model registry
# ---------------------------------------------------------------------------

def _load_gguf(model_id: str) -> GGUFAdapter:
    """Factory function for the model registry."""
    return GGUFAdapter(model_id)
