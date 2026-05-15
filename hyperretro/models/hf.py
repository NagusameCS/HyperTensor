"""HuggingFace model adapter for HyperTensor's unified model layer.

Wraps any HuggingFace AutoModelForCausalLM as an AbstractModel,
providing the standard interface that all HyperTensor tools consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

from hyperretro.models import AbstractModel


class HuggingFaceAdapter(AbstractModel):
    """Wraps a HuggingFace model as a HyperTensor AbstractModel."""

    backend = "huggingface"

    def __init__(self, model: "torch.nn.Module", config_dict: dict):
        self._model = model
        self._config = config_dict
        self._state_dict = None  # cached

    @property
    def config(self) -> dict:
        return self._config

    @property
    def state_dict(self) -> dict:
        if self._state_dict is None:
            self._state_dict = self._model.state_dict()
        return self._state_dict

    @property
    def architecture(self) -> str:
        arch_list = self._config.get("architectures", ["unknown"])
        arch = arch_list[0] if arch_list else "unknown"
        # Normalize common arch names
        arch_map = {
            "Qwen2ForCausalLM": "qwen2",
            "LlamaForCausalLM": "llama",
            "MistralForCausalLM": "llama",
            "GPT2LMHeadModel": "gpt2",
            "Phi3ForCausalLM": "phi3",
            "GemmaForCausalLM": "gemma",
            "Gemma2ForCausalLM": "gemma2",
        }
        return arch_map.get(arch, arch.lower())

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    @property
    def model_obj(self):
        return self._model

    def forward(self, input_ids, **kwargs):
        return self._model(input_ids, **kwargs)

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path), safe_serialization=True)
        # Save tokenizer if available
        if "tokenizer" in kwargs and kwargs["tokenizer"] is not None:
            kwargs["tokenizer"].save_pretrained(str(path))

    def load_state_dict_from(self, state_dict: dict, strict: bool = False):
        self._model.load_state_dict(state_dict, strict=strict)
        self._state_dict = None  # invalidate cache

    def get_attention_layers(self) -> list[dict]:
        """Discover attention weight keys for compression."""
        layers = []
        for key, tensor in self.state_dict.items():
            if any(pat in key for pat in [
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".attn.c_attn.weight",
                ".attention.qkv_proj.weight",
            ]):
                m, n = tuple(tensor.shape)
                layers.append({
                    "weight_key": key,
                    "shape": (m, n),
                    "in_features": n,
                    "out_features": m,
                })
        return layers

    def get_ffn_layers(self) -> list[dict]:
        """Discover FFN weight keys for compression."""
        layers = []
        for key, tensor in self.state_dict.items():
            if any(pat in key for pat in [
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
                ".mlp.c_fc.weight",
                ".mlp.c_proj.weight",
            ]):
                if hasattr(tensor, "dim") and tensor.dim() == 2:
                    m, n = tuple(tensor.shape)
                    layers.append({
                        "weight_key": key,
                        "shape": (m, n),
                        "in_features": n,
                        "out_features": m,
                    })
        return layers

    def to(self, device):
        self._model.to(device)
        return self

    def eval(self):
        self._model.eval()
        return self

    def train(self):
        self._model.train()
        return self


def _load_hf_model(
    model_id: str,
    *,
    device: str = "cpu",
    dtype: str = "float16",
    revision: str | None = None,
    **kwargs,
) -> tuple["torch.nn.Module", dict]:
    """Load a HuggingFace model and return (model, config_dict)."""
    from transformers import AutoModelForCausalLM, AutoConfig

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_id or kwargs.get("model_id", ""),
        revision=revision,
        torch_dtype=torch_dtype,
    )

    config = AutoConfig.from_pretrained(
        model_id or kwargs.get("model_id", ""),
        revision=revision,
    )

    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config.__dict__)
    return model, config_dict
