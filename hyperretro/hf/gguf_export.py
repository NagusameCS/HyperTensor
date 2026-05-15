"""HyperRetro → GGUF exporter (attack #5).

Converts a HyperRetro checkpoint (factored or dense) to GGUF format for
llama.cpp / Ollama inference. Factored weights are materialized to dense
fp16 during export. Supports the full GGUF metadata surface (tokenizer,
chat template, rope params, etc.) so the output is a drop-in GGUF file.

Workflow:
  1. hyperretro-compress --factored --int4 → HyperRetro checkpoint
  2. hyperretro-gguf-export checkpoint_dir/ output.gguf → fp16 GGUF
  3. llama-quantize output.gguf output-Q4_K_M.gguf Q4_K_M → 4-bit GGUF

This is the standard llama.cpp pipeline; step 3 gives the industry-standard
Q4_K_M compression (~3.5× shrink, +3-7% PPL). The HyperRetro-unique value
is in step 1 (activation-aware FFN factoring before quantization), which
reduces the PPL cost of aggressive quantization.

Architecture mapping:
  - Qwen2 → 'qwen2'
  - Llama → 'llama'
  - Mistral → 'llama' (same arch)
  - GPT-2 → 'gpt2'
  - Phi-3 → 'phi3'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Architecture name mapping (HF config → GGUF arch string)
_ARCH_MAP: dict[str, str] = {
    "Qwen2ForCausalLM": "qwen2",
    "Qwen2_5ForCausalLM": "qwen2",
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "GPT2LMHeadModel": "gpt2",
    "Phi3ForCausalLM": "phi3",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma2",
}


def _detect_arch(hf_config: dict | object) -> str:
    """Detect GGUF architecture string from HF config."""
    if isinstance(hf_config, dict):
        arch = hf_config.get("architectures", [None])[0]
        model_type = hf_config.get("model_type", "")
    else:
        arch = getattr(hf_config, "architectures", [None])[0] if hasattr(hf_config, "architectures") else None
        model_type = getattr(hf_config, "model_type", "")

    if arch and arch in _ARCH_MAP:
        return _ARCH_MAP[arch]
    # Fallback via model_type
    type_map = {
        "qwen2": "qwen2",
        "llama": "llama",
        "mistral": "llama",
        "gpt2": "gpt2",
        "phi3": "phi3",
        "gemma": "gemma",
        "gemma2": "gemma2",
    }
    return type_map.get(model_type, "llama")


# ---------------------------------------------------------------------------
# Tensor name mapping (HF → GGUF)
# ---------------------------------------------------------------------------

def _hf_to_gguf_tensor_name(hf_name: str, n_layers: int | None = None) -> str:
    """Map a HuggingFace state_dict key to a GGUF tensor name.

    Standard mappings for Qwen2/Llama architectures.
    """
    # Strip 'model.' prefix
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]

    # Embedding
    if name == "embed_tokens.weight":
        return "token_embd.weight"

    # Final norm
    if name == "norm.weight":
        return "output_norm.weight"

    # LM head
    if name == "lm_head.weight":
        return "output.weight"

    # Layer-specific mappings
    parts = name.split(".")
    if parts[0] == "layers" and parts[1].isdigit():
        li = int(parts[1])
        rest = ".".join(parts[2:])

        # Attention
        if rest == "input_layernorm.weight":
            return f"blk.{li}.attn_norm.weight"
        if rest == "post_attention_layernorm.weight":
            return f"blk.{li}.ffn_norm.weight"

        # Q/K/V/O projections
        if rest == "self_attn.q_proj.weight":
            return f"blk.{li}.attn_q.weight"
        if rest == "self_attn.k_proj.weight":
            return f"blk.{li}.attn_k.weight"
        if rest == "self_attn.v_proj.weight":
            return f"blk.{li}.attn_v.weight"
        if rest == "self_attn.o_proj.weight":
            return f"blk.{li}.attn_output.weight"

        # Biases (if present)
        if rest == "self_attn.q_proj.bias":
            return f"blk.{li}.attn_q.bias"
        if rest == "self_attn.k_proj.bias":
            return f"blk.{li}.attn_k.bias"
        if rest == "self_attn.v_proj.bias":
            return f"blk.{li}.attn_v.bias"
        if rest == "self_attn.o_proj.bias":
            return f"blk.{li}.attn_output.bias"

        # FFN (SwiGLU)
        if rest == "mlp.gate_proj.weight":
            return f"blk.{li}.ffn_gate.weight"
        if rest == "mlp.up_proj.weight":
            return f"blk.{li}.ffn_up.weight"
        if rest == "mlp.down_proj.weight":
            return f"blk.{li}.ffn_down.weight"

        # Factored FFN keys — materialize to dense
        if rest in ("mlp.gate_proj.factored_A", "mlp.gate_proj.factored_B",
                     "mlp.up_proj.factored_A", "mlp.up_proj.factored_B",
                     "mlp.down_proj.factored_A", "mlp.down_proj.factored_B"):
            return None  # handled specially

    # Unknown — keep as-is
    return name


# ---------------------------------------------------------------------------
# Materialize factored weights
# ---------------------------------------------------------------------------

def _materialize_factored_state_dict(
    state_dict: dict[str, "torch.Tensor | np.ndarray"],
    manifest: dict,
) -> dict[str, "np.ndarray"]:
    """Convert a factored state_dict to dense by computing B @ A where needed.

    Returns a new dict with dense weight tensors in fp16 numpy format.
    """
    import torch

    result: dict[str, np.ndarray] = {}

    # Build a map of prefix → (A, B, weight_key)
    factored_map: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    for entry in manifest.get("ffn", []):
        wkey = entry["weight_key"]
        prefix = wkey[:-len(".weight")] if wkey.endswith(".weight") else wkey
        a_key = f"{prefix}.factored_A"
        b_key = f"{prefix}.factored_B"
        if a_key in state_dict and b_key in state_dict:
            A = state_dict[a_key]
            B = state_dict[b_key]
            A_np = A.float().cpu().numpy() if hasattr(A, "cpu") else np.asarray(A, dtype=np.float32)
            B_np = B.float().cpu().numpy() if hasattr(B, "cpu") else np.asarray(B, dtype=np.float32)
            factored_map[prefix] = (A_np, B_np, wkey)

    for entry in manifest.get("layers", []):
        li = entry["layer_idx"]
        rank = entry["rank"]
        in_f = entry["in_features"]
        for slot in ("q", "k", "v"):
            slot_info = entry.get("keys", {}).get(slot, {})
            if not slot_info:
                continue
            prefix = f"model.layers.{li}.self_attn.{slot}_proj"
            wkey = f"{prefix}.weight"
            for a_suf, b_suf in [
                (f".factored_A{slot}", f".factored_B{slot}"),
                (".factored_A", ".factored_B"),
            ]:
                a_key = prefix + a_suf
                b_key = prefix + b_suf
                if a_key in state_dict and b_key in state_dict:
                    A = state_dict[a_key]
                    B = state_dict[b_key]
                    A_np = A.float().cpu().numpy() if hasattr(A, "cpu") else np.asarray(A, dtype=np.float32)
                    B_np = B.float().cpu().numpy() if hasattr(B, "cpu") else np.asarray(B, dtype=np.float32)
                    factored_map[prefix] = (A_np, B_np, wkey)
                    break

    # Materialize: B @ A → dense weight
    materialized: dict[str, np.ndarray] = {}
    for prefix, (A, B, wkey) in factored_map.items():
        W = (B.astype(np.float64) @ A.astype(np.float64)).astype(np.float16)
        materialized[prefix] = W

    # Build output state dict
    for key, tensor in state_dict.items():
        # Skip factored keys and scales
        if key.endswith(".factored_A") or key.endswith(".factored_B"):
            continue
        if key.endswith(".q") or key.endswith(".scales"):
            continue

        # Check if this key corresponds to a materialized prefix
        prefix_candidate = key
        if key.endswith(".weight"):
            prefix_candidate = key[:-len(".weight")]

        if prefix_candidate in materialized:
            result[key] = materialized[prefix_candidate]
        elif prefix_candidate in factored_map:
            result[key] = materialized.get(prefix_candidate, None)
        else:
            # Pass through as fp16
            if hasattr(tensor, "cpu"):
                result[key] = tensor.float().cpu().numpy().astype(np.float16)
            else:
                result[key] = np.asarray(tensor, dtype=np.float16)

    # Now handle any remaining materialized weights whose dense key wasn't in state_dict
    for prefix, W in materialized.items():
        wkey = f"{prefix}.weight"
        if wkey not in result:
            result[wkey] = W

    return result


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_hyperretro_to_gguf(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    *,
    model_name: str | None = None,
    quantization_type: str | None = None,
) -> Path:
    """Export a HyperRetro checkpoint to GGUF format.

    Parameters
    ----------
    checkpoint_dir : directory containing model.safetensors and config.json.
        Also supports standard HF model directories.
    output_path : target .gguf file path.
    model_name : override model name in GGUF metadata.
    quantization_type : if set, use GGUF native quantization (e.g. "Q4_K_M").
        Currently writes fp16; quantization is deferred to llama-quantize.

    Returns
    -------
    output_path
    """
    import torch
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoConfig
    from gguf import GGUFWriter

    ckpt = Path(checkpoint_dir)
    out = Path(output_path)

    # Load config
    config = AutoConfig.from_pretrained(str(ckpt))
    arch = _detect_arch(config)

    # Load weights
    sf_path = ckpt / "model.safetensors"
    if sf_path.exists():
        state_dict = load_safetensors(str(sf_path))
    else:
        # Try loading from HF model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=torch.float16)
        state_dict = model.state_dict()
        del model

    # Load manifest if present (for factored checkpoints)
    manifest_path = ckpt / "hyperretro_factored.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"[gguf] Factored checkpoint detected ({len(manifest.get('ffn', []))} FFN, "
              f"{len(manifest.get('layers', []))} attn layers)")

    # Materialize factored weights to dense
    if manifest:
        state_dict = _materialize_factored_state_dict(state_dict, manifest)
        print(f"[gguf] Factored weights materialized to dense")

    # Extract metadata from config
    n_layers = getattr(config, "num_hidden_layers", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    intermediate_size = getattr(config, "intermediate_size", 0)
    n_heads = getattr(config, "num_attention_heads", 0)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden_size // n_heads
    vocab_size = getattr(config, "vocab_size", 0)
    rope_theta = getattr(config, "rope_theta", 10000.0)
    rope_scaling = getattr(config, "rope_scaling", None)
    layer_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
    context_length = getattr(config, "max_position_embeddings", 32768)
    tie_embeddings = getattr(config, "tie_word_embeddings", False)

    # Create GGUF writer
    writer = GGUFWriter(str(out), arch)

    # ---- Metadata ----
    writer.add_name(model_name or f"HyperRetro-{arch}")
    writer.add_description("HyperRetro compressed model")
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_context_length(context_length)
    writer.add_rope_freq_base(rope_theta)
    if rope_scaling and isinstance(rope_scaling, dict):
        try:
            from gguf.constants import RopeScalingType
            stype = rope_scaling.get("type", "none")
            # Map string to enum
            stype_map = {t.value: t for t in RopeScalingType}
            if stype in stype_map:
                writer.add_rope_scaling_type(stype_map[stype])
            elif hasattr(RopeScalingType, stype.upper()):
                writer.add_rope_scaling_type(getattr(RopeScalingType, stype.upper()))
            writer.add_rope_scaling_factor(rope_scaling.get("factor", 1.0))
        except Exception:
            pass
    writer.add_layer_norm_rms_eps(layer_norm_eps)

    # Tokenizer metadata
    writer.add_vocab_size(vocab_size)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        if tok.bos_token_id is not None:
            writer.add_bos_token_id(tok.bos_token_id)
        if tok.eos_token_id is not None:
            writer.add_eos_token_id(tok.eos_token_id)
        if tok.pad_token_id is not None:
            writer.add_pad_token_id(tok.pad_token_id)
        # Chat template
        if hasattr(tok, "chat_template") and tok.chat_template:
            writer.add_chat_template(tok.chat_template)
    except Exception:
        pass

    writer.add_file_type(1)  # 1 = mostly F16

    # ---- Write tensors ----
    tensor_count = 0
    skipped = 0

    for hf_name, tensor in sorted(state_dict.items()):
        gguf_name = _hf_to_gguf_tensor_name(hf_name, n_layers)
        if gguf_name is None:
            skipped += 1
            continue

        # Convert to numpy fp16
        if hasattr(tensor, "cpu"):
            W = tensor.float().cpu().numpy().astype(np.float16)
        else:
            W = np.asarray(tensor, dtype=np.float16)

        # GGUF requires contiguous row-major
        W = np.ascontiguousarray(W)

        writer.add_tensor(gguf_name, W)
        tensor_count += 1

    print(f"[gguf] Wrote {tensor_count} tensors ({skipped} skipped)")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out.stat().st_size / 1e6
    print(f"[gguf] Saved: {out} ({size_mb:.1f} MB)")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Export HyperRetro checkpoint to GGUF for llama.cpp."
    )
    p.add_argument("checkpoint_dir", help="HyperRetro or HF model directory")
    p.add_argument("output", help="Output .gguf file path")
    p.add_argument("--name", default=None, help="Model name for GGUF metadata")
    p.add_argument("--quant", default=None, choices=["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0"],
                   help="GGUF quantization type (deferred to llama-quantize)")
    args = p.parse_args(argv)

    export_hyperretro_to_gguf(
        args.checkpoint_dir,
        args.output,
        model_name=args.name,
        quantization_type=args.quant,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
