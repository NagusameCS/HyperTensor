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

"""Int4-quantized factored checkpoint writer & loader (attack #7 realisation).

Stores factored (A, B) matrices in 4-bit packed format inside standard
safetensors, giving 5–14× on-disk shrink vs fp16 while keeping the
factored structure for downstream KV-cache / speculative-decode use.

Format (per factored weight, e.g. model.layers.0.mlp.gate_proj):

    {prefix}.factored_A.q      uint8   (k, (n+1)//2)  packed int4, row-major
    {prefix}.factored_A.scales  float16 (k,)           per-row dequant scales
    {prefix}.factored_B.q      uint8   (m, (k+1)//2)  packed int4
    {prefix}.factored_B.scales  float16 (m,)           per-row dequant scales

The manifest (hyperretro_factored.json) gains ``"quantization": "int4"`` per
entry. Non-factored weights (K/V projections, embeddings, etc.) can also be
stored as int4 via ``{key}.q`` + ``{key}.scales``, or kept as fp16 for
simplicity.

Loader (load_int4_factored_model) dequantises to fp16 at load time, producing
a standard FactoredLinear model. Future work: keep int4 at runtime via custom
CUDA kernels (bitsandbytes-style).

Industry context: bnb nf4 stores 4-bit weights with block-wise scales
(64-weight blocks). This module uses per-row scales for simplicity; the
storage overhead is ~3% (one fp16 scale per row), competitive with bnb's
~3% metadata overhead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Pack / unpack int4
# ---------------------------------------------------------------------------

def pack_int4_rows(W_q: np.ndarray) -> np.ndarray:
    """Pack an (m, n) int8 array (values in [-7, 7]) to uint8 (m, (n+1)//2).

    Each pair of consecutive columns is packed: low nibble = col 2j, high = col 2j+1.
    If n is odd, the last column's high nibble is set to 0.
    """
    m, n = W_q.shape
    # Shift values from [-7, 7] to [0, 15]
    W_u = (W_q + 8).astype(np.uint8)
    n_packed = (n + 1) // 2
    packed = np.zeros((m, n_packed), dtype=np.uint8)
    for j in range(n_packed):
        lo = W_u[:, 2 * j]
        hi = W_u[:, 2 * j + 1] if 2 * j + 1 < n else np.zeros(m, dtype=np.uint8)
        packed[:, j] = lo | (hi << 4)
    return packed


def unpack_int4_rows(packed: np.ndarray, n_cols: int) -> np.ndarray:
    """Reverse of pack_int4_rows. Returns int8 array (m, n_cols) in [-7, 7]."""
    m = packed.shape[0]
    result = np.zeros((m, n_cols), dtype=np.int8)
    for j in range(n_cols):
        byte_idx = j // 2
        if j % 2 == 0:
            result[:, j] = (packed[:, byte_idx] & 0x0F).astype(np.int8) - 8
        else:
            result[:, j] = ((packed[:, byte_idx] >> 4) & 0x0F).astype(np.int8) - 8
    return result


def smart_pack(W_q: np.ndarray, n_bits: int) -> np.ndarray:
    """Pack quantized weights: int4→packed uint8, int8→passthrough int8."""
    if n_bits <= 4:
        return pack_int4_rows(W_q)
    else:
        # int8: no packing needed, store directly
        return W_q.astype(np.int8)


def smart_unpack(packed: np.ndarray, n_cols: int, n_bits: int) -> np.ndarray:
    """Reverse of smart_pack."""
    if n_bits <= 4:
        return unpack_int4_rows(packed, n_cols)
    else:
        # int8: already unpacked
        return packed.astype(np.int8)


# ---------------------------------------------------------------------------
# Quantize factored state_dict
# ---------------------------------------------------------------------------

def quantize_factored_state_dict(
    state_dict: dict[str, "torch.Tensor | np.ndarray"],
    factored_manifest: dict,
    *,
    n_bits: int = 4,
    quantize_non_factored: bool = True,
    non_factored_min_bytes: int = 1024,
    block_size: int = 128,
    activation_norms: dict[str, np.ndarray] | None = None,
    int8_patterns: list[str] | None = None,
) -> dict:
    """Quantize A/B matrices in a factored state_dict to int4 in-place.

    Uses block-wise quantization (llama.cpp Q4_0 style) for much better
    fidelity than per-row symmetric quantization. Optionally AWQ-aware
    when activation_norms are provided.

    Set ``int8_patterns`` to a list of key substrings (e.g. ``['embed']``)
    to quantize matching weights at int8 instead of int4. This is useful
    for large embedding matrices where int4 causes disproportionate PPL
    damage — int8 recovers ~2 PPL points for ~4% size increase.

    Parameters
    ----------
    state_dict : dict of tensor_name → tensor. Must already have
        ``{prefix}.factored_A`` and ``{prefix}.factored_B`` keys.
    factored_manifest : the manifest dict (contents of hyperretro_factored.json).
    n_bits : 4 or 8.
    quantize_non_factored : if True, also int4-quantize non-factored weight
        tensors (K/V/O projections, embeddings) that exceed non_factored_min_bytes.
    non_factored_min_bytes : skip tiny tensors (layer norms, biases).
    block_size : elements per quantization block (128 = Q4_0 style).
    activation_norms : optional dict of weight_key → (in_features,) column norms
        for AWQ-aware quantization. Same format as collect_ffn_input_norms output.

    Returns
    -------
    state_dict : same dict, mutated in-place. Factored_A/B tensors are
        replaced by ``.q`` (packed uint8), ``.scales`` (fp16, 2D block-wise),
        and optionally ``.awq_scales`` (fp16) if AWQ-aware.
    updated_manifest : the manifest dict with ``"quantization": "int4_blockwise"``
        per entry.
    """
    import torch
    from hyperretro.hf.factor_quantize import quantize_blockwise_int4

    factored_keys_seen: set[str] = set()

    # 1. Quantize factored FFN entries
    for entry in factored_manifest.get("ffn", []):
        wkey = entry["weight_key"]
        prefix = wkey[:-len(".weight")] if wkey.endswith(".weight") else wkey
        a_key = f"{prefix}.factored_A"
        b_key = f"{prefix}.factored_B"

        # Get activation norms for this weight (for AWQ-aware quantization)
        col_norms = None
        if activation_norms is not None and wkey in activation_norms:
            col_norms = activation_norms[wkey]

        for mat_key in [a_key, b_key]:
            if mat_key not in state_dict:
                continue
            tensor = state_dict[mat_key]
            W = tensor.float().cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)

            # AWQ-aware only for A matrices (input-side: shape (k, in_features))
            # B matrices (output-side: shape (out_features, k)) use plain block-wise
            use_awq = (col_norms is not None and mat_key == a_key)

            if use_awq:
                from hyperretro.hf.factor_quantize import quantize_matrix_int4_best
                W_q, scales, awq_scales = quantize_matrix_int4_best(
                    W, col_norms=col_norms, block_size=block_size, n_bits=n_bits,
                )
                state_dict[f"{mat_key}.awq_scales"] = torch.from_numpy(awq_scales).to(torch.float16)
            else:
                W_q, scales = quantize_blockwise_int4(
                    W, block_size=block_size, n_bits=n_bits,
                )

            packed = pack_int4_rows(W_q)
            state_dict[f"{mat_key}.q"] = torch.from_numpy(packed)
            # scales is (m, n_blocks) for block-wise
            state_dict[f"{mat_key}.scales"] = torch.from_numpy(scales).to(torch.float16)
            factored_keys_seen.add(mat_key)

        entry["quantization"] = f"int{n_bits}_blockwise"
        if col_norms is not None:
            entry["quantization"] += "_awq"

    # 2. Quantize factored attn entries
    for entry in factored_manifest.get("layers", []):
        li = entry.get("layer_idx", 0)
        for slot in ("q", "k", "v"):
            slot_info = entry.get("keys", {}).get(slot, {})
            if not slot_info:
                continue
            for suffix in [f".factored_A{slot}", f".factored_B{slot}",
                            ".factored_A", ".factored_B"]:
                for prefix_pattern in [
                    f"model.layers.{li}.self_attn.{slot}_proj",
                ]:
                    candidate = prefix_pattern + suffix
                    if candidate in state_dict:
                        tensor = state_dict[candidate]
                        W = tensor.float().cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
                        W_q, scales = quantize_blockwise_int4(
                            W, block_size=block_size, n_bits=n_bits,
                        )
                        packed = smart_pack(W_q, n_bits)
                        state_dict[f"{candidate}.q"] = torch.from_numpy(packed)
                        state_dict[f"{candidate}.scales"] = torch.from_numpy(scales).to(torch.float16)
                        factored_keys_seen.add(candidate)
                        break

        entry["quantization"] = f"int{n_bits}_blockwise"

    # 3. Remove old fp16 factored keys
    for old_key in factored_keys_seen:
        state_dict.pop(old_key, None)

    # 4. Optionally quantize non-factored weights (block-wise)
    if quantize_non_factored:
        non_factored_to_quantize: list[str] = []
        for key, tensor in state_dict.items():
            if key.endswith(".q") or key.endswith(".scales") or key.endswith(".awq_scales"):
                continue
            if key in factored_keys_seen:
                continue
            if not hasattr(tensor, "numel"):
                continue
            n_elem = tensor.numel()
            if n_elem < non_factored_min_bytes // 2:
                continue
            if hasattr(tensor, "dim") and tensor.dim() == 2:
                non_factored_to_quantize.append(key)

        for key in non_factored_to_quantize:
            tensor = state_dict[key]
            W = tensor.float().cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
            # Check for int8 override (e.g. embeddings)
            use_bits = n_bits
            if int8_patterns:
                for pat in int8_patterns:
                    if pat in key:
                        use_bits = max(8, n_bits)  # int8 or higher
                        break
            W_q, scales = quantize_blockwise_int4(
                W, block_size=block_size, n_bits=use_bits,
            )
            packed = smart_pack(W_q, use_bits)
            state_dict[f"{key}.q"] = torch.from_numpy(packed)
            state_dict[f"{key}.scales"] = torch.from_numpy(scales).to(torch.float16)
            del state_dict[key]

    return state_dict, factored_manifest


# ---------------------------------------------------------------------------
# Int4-factored safetensors writer
# ---------------------------------------------------------------------------

def save_int4_factored_checkpoint(
    state_dict: dict,
    attn_entries: list,
    ffn_entries: list,
    *,
    out_dir: str | Path,
    hf_config,
    tokenizer=None,
    n_bits: int = 4,
    quantize_non_factored: bool = True,
    block_size: int = 128,
    activation_norms: dict[str, np.ndarray] | None = None,
    int8_patterns: list[str] | None = None,
) -> dict:
    """Write a HyperRetro int4-factored checkpoint to disk.

    Same API as :func:`hyperretro.hf.factored.save_factored_checkpoint` but
    stores factored matrices as packed int4 instead of fp16/bf16.

    Set ``int8_patterns=['embed']`` to store embedding matrices at int8
    instead of int4 — recovers ~2 PPL for ~4% size increase.

    Returns a report dict.
    """
    import torch
    from safetensors.torch import save_file
    from hyperretro.hf.factored import build_manifest, FactoredEntry

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tied = bool(getattr(hf_config, "tie_word_embeddings", False))

    # Build manifest first (before quantization mutates keys)
    entry_objs: list[FactoredEntry] = []
    for e in attn_entries:
        if isinstance(e, FactoredEntry):
            entry_objs.append(e)
        else:
            entry_objs.append(FactoredEntry(
                layer_idx=e["layer_idx"], rank=e["rank"],
                in_features=e["in_features"], out_features=e["out_features"],
                keys={"q": e.get("q_key", ""), "k": e.get("k_key", ""),
                      "v": e.get("v_key", "")},
                bias_keys=e.get("biases", {}),
            ))

    manifest = build_manifest(entry_objs, shared=True, ffn_entries=ffn_entries)

    # Drop lm_head if tied embeddings
    sd = dict(state_dict)
    if tied and "lm_head.weight" in sd:
        del sd["lm_head.weight"]

    # Quantize
    sd, manifest = quantize_factored_state_dict(
        sd, manifest,
        n_bits=n_bits,
        quantize_non_factored=quantize_non_factored,
        block_size=block_size,
        activation_norms=activation_norms,
        int8_patterns=int8_patterns,
    )

    # Ensure all tensors are on CPU and properly typed
    out_sd: dict = {}
    for k, v in sd.items():
        if hasattr(v, "cpu"):
            v = v.contiguous().cpu()
        if hasattr(v, "numpy"):
            v = v
        else:
            v = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32 if v.dtype == np.dtype('float32') else v.dtype))
        if k.endswith(".q"):
            # Keep int8 as int8, only force uint8 for int4 packed data
            if v.dtype == torch.int8:
                v = v.to(torch.int8)
            else:
                v = v.to(torch.uint8)
        elif k.endswith(".scales") or k.endswith(".awq_scales"):
            v = v.to(torch.float16)
        elif isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
            v = v.to(torch.float16)
        out_sd[k] = v

    save_file(out_sd, str(out / "model.safetensors"))

    # Write manifest
    (out / "hyperretro_factored.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # HF config + tokenizer
    hf_config.save_pretrained(out)
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(out)
        except Exception:
            pass

    # Size stats
    sf_size = (out / "model.safetensors").stat().st_size
    return {
        "out_dir": str(out),
        "attn_factored": len(entry_objs),
        "ffn_factored": len(ffn_entries),
        "tied_lm_head_dropped": tied and "lm_head.weight" in state_dict,
        "quantization": f"int{n_bits}",
        "on_disk_mb": round(sf_size / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Int4 → standard factored converter (dequant on load)
# ---------------------------------------------------------------------------

def dequant_int4_checkpoint(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    *,
    dequant_dtype: str = "float16",
) -> Path:
    """Convert an int4-factored checkpoint back to standard fp16 factored format.

    Reads ``{prefix}.factored_A.q`` + ``{prefix}.factored_A.scales`` and
    writes ``{prefix}.factored_A`` (fp16), similarly for B. Non-factored
    quantized weights (``{key}.q`` / ``{key}.scales``) are also dequantized.

    The output directory is loadable by the standard
    :func:`~hyperretro.hf.factored.load_factored_hf_model`.

    Returns the output directory Path.
    """
    import torch
    from safetensors.torch import load_file as load_sf, save_file as save_sf
    import shutil

    ckpt = Path(checkpoint_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sf_path = ckpt / "model.safetensors"
    manifest_path = ckpt / "hyperretro_factored.json"

    if not sf_path.exists():
        raise FileNotFoundError(f"Safetensors not found: {sf_path}")

    raw_sd = load_sf(str(sf_path))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    target_dtype = dtype_map.get(dequant_dtype, torch.float16)

    q_keys = sorted(k for k in raw_sd if k.endswith(".q"))
    processed_scales: set[str] = set()
    out_sd: dict[str, torch.Tensor] = {}

    for q_key in q_keys:
        base = q_key[:-2]  # strip ".q"
        scales_key = base + ".scales"
        awq_key = base + ".awq_scales"
        if scales_key not in raw_sd:
            print(f"[warn] missing scales for {q_key}, skipping")
            continue

        packed = raw_sd[q_key].cpu().numpy()
        scales = raw_sd[scales_key].float().cpu().numpy()

        # Detect int4 vs int8 from packed data shape BEFORE inferring n_cols
        # int4 packed: width ≈ n_cols/2. int8: width ≈ n_cols (no packing)
        is_int8 = (packed.shape[1] > 0 and packed.shape[0] > 0 and
                   packed.dtype == np.int8)  # smart_pack stores int8 as int8
        
        # Infer column count
        n_cols = _infer_n_cols(base, manifest, packed.shape[0])
        
        if is_int8:
            n_cols = packed.shape[1]  # int8: no packing
            W_q = packed.astype(np.int8)
        elif n_cols <= 0:
            n_cols = packed.shape[1] * 2  # int4 from packed width
            W_q = unpack_int4_rows(packed, n_cols)
        elif packed.shape[1] * 2 >= n_cols * 1.5:
            # Heuristic: if packed width ≥ 75% of inferred n_cols, likely int8
            n_cols = packed.shape[1]
            W_q = packed.astype(np.int8)
        else:
            W_q = unpack_int4_rows(packed, n_cols)
        m, n = W_q.shape

        # Detect format: 1D scales = per-row, 2D scales = block-wise
        if scales.ndim == 1:
            # Legacy per-row format
            W_dq = (W_q.astype(np.float32) * scales[:, None]).astype(np.float32)
        elif scales.ndim == 2:
            # Block-wise format (Q4_0 style)
            n_blocks = scales.shape[1]
            W_dq = np.zeros((m, n), dtype=np.float32)
            # Determine block size from scales shape and n_cols
            blk = (n + n_blocks - 1) // n_blocks
            for b in range(n_blocks):
                start = b * blk
                end = min(start + blk, n)
                W_dq[:, start:end] = W_q[:, start:end].astype(np.float32) * scales[:, b:b+1]
        else:
            raise ValueError(f"Unexpected scales ndim={scales.ndim} for {q_key}")

        # Apply AWQ inverse scaling if present
        if awq_key in raw_sd:
            awq_scales = raw_sd[awq_key].float().cpu().numpy()
            W_dq = W_dq / awq_scales[None, :].astype(np.float32)
            processed_scales.add(awq_key)

        out_sd[base] = torch.from_numpy(W_dq).to(target_dtype)
        processed_scales.add(scales_key)

    # Copy non-quantized tensors
    for key, tensor in raw_sd.items():
        if key.endswith(".q"):
            continue
        if key.endswith(".scales") and key in processed_scales:
            continue
        if key in out_sd:
            continue
        out_sd[key] = tensor.to(target_dtype) if hasattr(tensor, "to") else tensor

    save_sf(out_sd, str(out / "model.safetensors"))

    # Copy config files
    if manifest_path.exists():
        shutil.copy(manifest_path, out / "hyperretro_factored.json")
    for f in ckpt.glob("config.json"):
        shutil.copy(f, out / "config.json")
    for f in ckpt.glob("tokenizer*"):
        shutil.copy(f, out / f.name)
    for f in ckpt.glob("*.jinja"):
        shutil.copy(f, out / f.name)

    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"[dequant] {ckpt.name} → {out.name}  ({size_mb:.1f} MB fp16)")
    return out


def _infer_n_cols(base: str, manifest: dict, n_rows: int) -> int:
    """Infer number of columns for a dequantised matrix from manifest.
    
    For factored matrices, uses the manifest entry's in_features/rank.
    For non-factored quantized weights, falls back to packed_width * 2
    (since packed stores pairs of columns).
    """
    shape = _infer_dequant_shape(base, manifest)
    if shape is not None:
        return shape[1]
    # For non-factored matrices, use the actual packed width
    # packed width = (n_cols + 1) // 2, so n_cols ≈ packed_width * 2
    # But we don't have packed_width here — the caller should handle this.
    # Return a sentinel that the caller can check.
    return -1


def _infer_dequant_shape(base: str, manifest: dict) -> tuple[int, int] | None:
    """Infer (rows, cols) for a dequantised factored matrix from manifest."""
    for entry in manifest.get("ffn", []):
        wkey = entry["weight_key"]
        prefix = wkey[:-len(".weight")] if wkey.endswith(".weight") else wkey
        if base == f"{prefix}.factored_A":
            return (entry["rank"], entry["in_features"])
        if base == f"{prefix}.factored_B":
            return (entry["out_features"], entry["rank"])
    for entry in manifest.get("layers", []):
        li = entry.get("layer_idx", 0)
        rank = entry.get("rank", 0)
        in_f = entry.get("in_features", 0)
        out_f = entry.get("out_features", 0)
        # Try common key patterns
        for slot in ("q", "k", "v"):
            for prefix_pattern in [
                f"model.layers.{li}.self_attn.{slot}_proj",
                f"model.layers.{li}.self_attn.{slot}_proj.weight",
            ]:
                if base == f"{prefix_pattern}.factored_A":
                    return (rank, in_f)
                if base == f"{prefix_pattern}.factored_B":
                    return (out_f, rank)
    return None


# ---------------------------------------------------------------------------
# Convenience: load int4 model
# ---------------------------------------------------------------------------

def load_int4_factored_model(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    dequant_dtype: str = "float16",
) -> "torch.nn.Module":
    """Load an int4-factored checkpoint as a standard HF model.

    Dequantises int4 → fp16 in a temp directory, then delegates to
    :func:`~hyperretro.hf.factored.load_factored_hf_model`.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dequant = dequant_int4_checkpoint(
            checkpoint_dir, tmpdir, dequant_dtype=dequant_dtype,
        )
        from hyperretro.hf.factored import load_factored_hf_model
        model = load_factored_hf_model(str(dequant), device=device)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Int4-factored checkpoint tools."
    )
    p.add_argument("command", choices=["info"], default="info", nargs="?")
    p.add_argument("checkpoint_dir")
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint_dir)
    manifest = json.loads((ckpt / "hyperretro_factored.json").read_text())
    sf_path = ckpt / "model.safetensors"
    sf_size_mb = sf_path.stat().st_size / 1e6 if sf_path.exists() else 0

    n_ffn = len(manifest.get("ffn", []))
    n_attn = len(manifest.get("layers", []))
    quantized = any(e.get("quantization") for e in manifest.get("ffn", []))

    print(f"Checkpoint: {ckpt}")
    print(f"  On-disk:   {sf_size_mb:.1f} MB")
    print(f"  Attn layers: {n_attn}")
    print(f"  FFN matrices: {n_ffn}")
    print(f"  Quantized: {quantized}")
    if quantized:
        ranks = set(e.get("rank", 0) for e in manifest.get("ffn", []))
        print(f"  Ranks: {sorted(ranks)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
