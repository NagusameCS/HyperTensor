"""OpenMythos model adapter for HyperTensor's unified model layer.

Wraps an OpenMythos model as an AbstractModel, providing the standard
interface. OpenMythos is an optional backend — only loaded if the
open-mythos package is installed.

The adapter handles the architectural differences:
- MLA/GQA attention → mapped to standard attention compression targets
- MoE FFN (routed + shared experts) → each expert is a separate FFN target
- Recurrent block → gets special rank allocation (T× amplification)
- LTI injection / ACT halting / LoRA adapters → preserved as-is (tiny params)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import json

import torch
import numpy as np

from hyperretro.models import AbstractModel


class OpenMythosAdapter(AbstractModel):
    """Wraps an OpenMythos model as a HyperTensor AbstractModel."""

    backend = "openmythos"

    def __init__(self, model: "torch.nn.Module", config_dict: dict):
        self._model = model
        self._config = config_dict
        self._state_dict = None

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
        return "mythos"

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    @property
    def model_obj(self):
        return self._model

    def forward(self, input_ids, n_loops=None, **kwargs):
        return self._model(input_ids, n_loops=n_loops, **kwargs)

    def save(self, path: Union[str, Path], **kwargs):
        """Save OM model. Uses safetensors for weights, JSON for config."""
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        (path / "config.json").write_text(json.dumps(self._config, indent=2))

        # Save weights as safetensors
        sd = {}
        for k, v in self.state_dict.items():
            sd[k] = v.contiguous().cpu()
        save_file(sd, str(path / "model.safetensors"))

    def load_state_dict_from(self, state_dict: dict, strict: bool = False):
        self._model.load_state_dict(state_dict, strict=strict)
        self._state_dict = None

    def get_attention_layers(self) -> list[dict]:
        """Discover attention weights for compression.

        For MLA: compress q_down, kv_down projections (largest attn weights).
        For GQA: compress wq, wk, wv projections.
        """
        layers = []
        for key, tensor in self.state_dict.items():
            if ".attn." not in key or not key.endswith(".weight"):
                continue
            if not (hasattr(tensor, "dim") and tensor.dim() == 2):
                continue
            m, n = tuple(tensor.shape)
            # Only compress large projections (>100K elements)
            if m * n > 100_000:
                layers.append({
                    "weight_key": key,
                    "shape": (m, n),
                    "in_features": n,
                    "out_features": m,
                    # Mark if this is in the recurrent block
                    "is_recurrent": "recurrent.block" in key,
                })
        return layers

    def get_ffn_layers(self) -> list[dict]:
        """Discover FFN weights for compression.

        Includes MoE experts (routed + shared) and dense FFNs.
        """
        layers = []
        for key, tensor in self.state_dict.items():
            if not key.endswith(".weight"):
                continue
            if not (hasattr(tensor, "dim") and tensor.dim() == 2):
                continue
            # Match FFN patterns: experts, shared_experts, or .ffn.
            if not any(pat in key for pat in [
                ".experts.", ".shared_experts.", ".ffn.gate.",
                ".ffn.up.", ".ffn.down.",
            ]):
                continue
            m, n = tuple(tensor.shape)
            layers.append({
                "weight_key": key,
                "shape": (m, n),
                "in_features": n,
                "out_features": m,
                "is_recurrent": "recurrent.block" in key,
                "is_expert": ".experts." in key or ".shared_experts." in key,
            })
        return layers

    @property
    def num_recurrent_loops(self) -> int:
        """Default number of recurrent loops (for rank amplification)."""
        return self._config.get("max_loop_iters", 16)

    def to(self, device):
        self._model.to(device)
        return self

    def eval(self):
        self._model.eval()
        return self

    def train(self):
        self._model.train()
        return self


def _load_om_model(
    model_id: str,
    *,
    device: str = "cpu",
    dtype: str = "float16",
    **kwargs,
) -> tuple["torch.nn.Module", dict]:
    """Load an OpenMythos model by variant name or config."""
    from open_mythos import OpenMythos

    # Map variant names
    variant_map = {
        "mythos_1b": "mythos_1b",
        "mythos_3b": "mythos_3b",
        "mythos_10b": "mythos_10b",
        "mythos_50b": "mythos_50b",
        "mythos_100b": "mythos_100b",
        "mythos_500b": "mythos_500b",
        "mythos_1t": "mythos_1t",
    }

    # Check if it's a file path to a config
    config_path = Path(model_id)
    if config_path.is_file() and config_path.suffix == ".json":
        config_dict = json.loads(config_path.read_text())
        from open_mythos import MythosConfig
        cfg = MythosConfig(**config_dict)
    elif model_id in variant_map:
        import open_mythos.variants as variants
        variant_fn = getattr(variants, variant_map[model_id])
        cfg = variant_fn()
    else:
        # Try as a config dict
        raise ValueError(
            f"Unknown OpenMythos model: {model_id}. "
            f"Use a variant name: {list(variant_map.keys())} "
            f"or pass a config dict."
        )

    model = OpenMythos(cfg)
    config_dict = {
        "model_type": "openmythos",
        "architecture": "mythos",
        "vocab_size": cfg.vocab_size,
        "dim": cfg.dim,
        "hidden_size": cfg.dim,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "num_hidden_layers": (cfg.prelude_layers + 1 + cfg.coda_layers),
        "max_loop_iters": cfg.max_loop_iters,
        "prelude_layers": cfg.prelude_layers,
        "coda_layers": cfg.coda_layers,
        "attn_type": cfg.attn_type,
        "n_experts": cfg.n_experts,
        "n_shared_experts": cfg.n_shared_experts,
        "expert_dim": cfg.expert_dim,
        "max_seq_len": cfg.max_seq_len,
        "lora_rank": cfg.lora_rank,
    }

    return model, config_dict
