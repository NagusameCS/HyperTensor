"""vLLM model backend — load models via vLLM for speculative decode.

Enables:  hyperretro load_models("Qwen/Qwen2.5-7B", backend="vllm")

Wraps a vLLM LLM instance as an AbstractModel.  This is primarily useful
for serving compressed models as speculative drafters — the vLLM backend
integrates with vLLM's SpeculativeWorker directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from hyperretro.models import AbstractModel


class VLLMAdapter(AbstractModel):
    """Wraps a vLLM LLM instance as a HyperTensor AbstractModel.

    vLLM models are opaque — they don't expose state_dict or config the
    way HF models do.  This adapter provides the AbstractModel interface
    using vLLM's internal APIs where available, and falls back to the
    HuggingFace config loaded from the model directory.
    """

    backend = "vllm"

    def __init__(self, llm_instance, model_id: str, config_dict: dict = None):
        self._llm = llm_instance
        self._model_id = model_id
        if config_dict is None:
            config_dict = self._extract_vllm_config(llm_instance)
        self._config = config_dict

    @staticmethod
    def _extract_vllm_config(llm_instance) -> dict:
        """Extract config from vLLM's internal model."""
        try:
            # vLLM stores config under llm_engine.model_config.hf_config
            engine = llm_instance.llm_engine
            hf_cfg = engine.model_config.hf_config
            return hf_cfg.to_dict() if hasattr(hf_cfg, 'to_dict') else vars(hf_cfg)
        except Exception:
            # Fallback: try to read config from the model path
            try:
                model_path = getattr(llm_instance, 'model_path', None)
                if model_path:
                    import json
                    cfg_path = Path(model_path) / "config.json"
                    if cfg_path.exists():
                        return json.loads(cfg_path.read_text())
            except Exception:
                pass
            return {"model_type": "vllm", "model_id": str(llm_instance)}

    # ------------------------------------------------------------------
    # AbstractModel interface
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict:
        return self._config

    @property
    def state_dict(self) -> dict:
        raise NotImplementedError(
            "vLLM models do not expose state_dict directly. "
            "Use compress with a HuggingFace or GGUF backend instead."
        )

    @property
    def architecture(self) -> str:
        arch_list = self._config.get("architectures", ["VLLMForCausalLM"])
        return arch_list[0].replace("ForCausalLM", "").lower() if arch_list else "vllm"

    @property
    def param_count(self) -> int:
        try:
            return self._llm.llm_engine.model_config.get_num_params()
        except Exception:
            hidden = self._config.get("hidden_size", 0)
            layers = self._config.get("num_hidden_layers", 0)
            if hidden and layers:
                return 12 * hidden * hidden * layers  # rough estimate
            return 0

    @property
    def model_obj(self):
        return self._llm

    def forward(self, input_ids, **kwargs):
        """vLLM's generation interface, not a raw forward."""
        from vllm import SamplingParams
        params = SamplingParams(temperature=0, max_tokens=1)
        if isinstance(input_ids, str):
            prompt = input_ids
        elif hasattr(input_ids, 'tolist'):
            prompt = input_ids  # token ids — vLLM accepts these
        else:
            prompt = str(input_ids)
        outputs = self._llm.generate([prompt], params)
        return outputs[0].outputs[0].token_ids if outputs else []

    def save(self, path: Union[str, Path], **kwargs):
        raise NotImplementedError(
            "vLLM models cannot be saved via HyperRetro. "
            "Save the underlying HuggingFace model instead."
        )

    def load_state_dict_from(self, state_dict: dict, strict: bool = False):
        raise NotImplementedError(
            "vLLM models cannot be reloaded from state_dict."
        )

    def to(self, device):
        return self  # vLLM manages device placement

    def eval(self):
        return self  # vLLM is always in eval mode

    def train(self):
        raise NotImplementedError("vLLM adapter is inference-only")


# ---------------------------------------------------------------------------
# Loader for the model registry
# ---------------------------------------------------------------------------

def _load_vllm(model_id: str) -> VLLMAdapter:
    """Factory function for the model registry — loads via vLLM."""
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM backend requires: pip install vllm"
        ) from None

    llm = LLM(
        model=model_id,
        dtype="auto",
        gpu_memory_utilization=0.85,
    )
    return VLLMAdapter(llm, model_id)
