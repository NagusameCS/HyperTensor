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

"""Fast attack #7 validator: measure int4 quantization error on already-factored
FFN checkpoints from round 13.

Loads the factored (A, B) matrices from existing safetensors checkpoints,
applies int4 quantization, and measures reconstruction error vs the original
dense weights. No SVD re-computation needed.

Output: benchmarks/factor_int4_error_fast.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]  # scripts/ → project root
sys.path.insert(0, str(_ROOT))

from hyperretro.hf.factor_quantize import (
    quantize_matrix_int4,
    dequantize_matrix_int4,
    int4_storage_bytes,
    estimate_full_model_shrink,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
CHECKPOINTS = {
    "aware_r384": _ROOT / "outputs" / "_ffn_only_aware_r384",
    "vanilla_r384": _ROOT / "outputs" / "_ffn_only_vanilla_r384",
    "aware_r768": _ROOT / "outputs" / "_ffn_only_aware_r768",
    "vanilla_r768": _ROOT / "outputs" / "_ffn_only_vanilla_r768",
    "aware_r1024": _ROOT / "outputs" / "_ffn_only_aware_r1024",
    "vanilla_r1024": _ROOT / "outputs" / "_ffn_only_vanilla_r1024",
}
# Only sample a few layers to keep it fast
SAMPLE_LAYERS = [0, 7, 14, 21, 27]
FFN_TYPES = ["gate_proj", "up_proj", "down_proj"]
N_BITS = 4
OUT = _ROOT / "benchmarks" / "factor_int4_error_fast.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_dense_weights() -> dict[str, np.ndarray]:
    """Load original dense FFN weights from HF model."""
    from transformers import AutoModelForCausalLM

    print("[load] dense model weights ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    )
    model.to("cpu")
    sd = model.state_dict()
    weights = {}
    for layer in SAMPLE_LAYERS:
        for ffn_type in FFN_TYPES:
            key = f"model.layers.{layer}.mlp.{ffn_type}.weight"
            if key in sd:
                weights[key] = sd[key].float().cpu().numpy().copy()
    del model
    print(f"[load] {len(weights)} dense FFN matrices loaded")
    return weights


def load_factored_matrices(checkpoint_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, int, int]]:
    """Load (A, B) pairs from a factored safetensors checkpoint.

    Returns {weight_key: (A, B, in_features, out_features)}.
    """
    sf_path = checkpoint_dir / "model.safetensors"
    if not sf_path.exists():
        print(f"  [skip] {sf_path} not found")
        return {}

    manifest_path = checkpoint_dir / "hyperretro_factored.json"
    if not manifest_path.exists():
        print(f"  [skip] manifest not found in {checkpoint_dir}")
        return {}

    manifest = json.loads(manifest_path.read_text())
    tensors = load_safetensors(str(sf_path))

    result = {}
    for entry in manifest.get("ffn", []):
        key = entry["weight_key"]
        # Only sample layers we care about
        layer_id = None
        for li in SAMPLE_LAYERS:
            if f"model.layers.{li}." in key:
                layer_id = li
                break
        if layer_id is None:
            continue

        # Safetensors keys strip the ".weight" suffix:
        #   weight_key = "model.layers.0.mlp.gate_proj.weight"
        #   A_key       = "model.layers.0.mlp.gate_proj.factored_A"
        prefix = key[:-len(".weight")] if key.endswith(".weight") else key
        a_key = prefix + ".factored_A"
        b_key = prefix + ".factored_B"
        if a_key not in tensors or b_key not in tensors:
            continue
        A = tensors[a_key].float().cpu().numpy()
        B = tensors[b_key].float().cpu().numpy()
        result[key] = (A, B, entry["in_features"], entry["out_features"])
    return result


def measure(W_dense: np.ndarray, A: np.ndarray, B: np.ndarray) -> dict:
    """Measure int4 quantization error for a factored weight."""
    frob_W = float(np.linalg.norm(W_dense, "fro"))
    if frob_W == 0:
        frob_W = 1.0

    m, n = W_dense.shape
    k = A.shape[0]  # rank

    # 1. fp16 factored error
    W_factor = (B.astype(np.float64) @ A.astype(np.float64)).astype(np.float32)
    err_factor = float(np.linalg.norm(W_dense - W_factor, "fro") / frob_W)

    # 2. int4 quantize A and B
    A_q, A_scales = quantize_matrix_int4(A, axis=0, n_bits=N_BITS)
    B_q, B_scales = quantize_matrix_int4(B, axis=0, n_bits=N_BITS)
    A_dq = dequantize_matrix_int4(A_q, A_scales, axis=0)
    B_dq = dequantize_matrix_int4(B_q, B_scales, axis=0)
    W_int4 = (B_dq.astype(np.float64) @ A_dq.astype(np.float64)).astype(np.float32)
    err_int4 = float(np.linalg.norm(W_dense - W_int4, "fro") / frob_W)

    # 3. Direct int4 of dense W
    W_q, W_scales = quantize_matrix_int4(W_dense, axis=0, n_bits=N_BITS)
    W_dq = dequantize_matrix_int4(W_q, W_scales, axis=0)
    err_dense_int4 = float(np.linalg.norm(W_dense - W_dq, "fro") / frob_W)

    # 4. Quantization penalty: how much ADDITIONAL error does int4 add?
    quant_penalty = (err_int4 / max(err_factor, 1e-12)) if err_factor > 1e-12 else 0.0

    # 5. Storage sizes
    dense_fp16 = m * n * 2
    dense_int4 = int4_storage_bytes(m, n, N_BITS)
    factor_fp16 = k * (m + n) * 2
    factor_int4 = int4_storage_bytes(k, n, N_BITS) + int4_storage_bytes(m, k, N_BITS)

    return {
        "shape": [m, n],
        "rank": k,
        "frob_relerr_factor": round(err_factor, 8),
        "frob_relerr_int4": round(err_int4, 8),
        "frob_relerr_dense_int4": round(err_dense_int4, 8),
        "quant_penalty": round(quant_penalty, 3),
        "dense_fp16_mb": round(dense_fp16 / 1e6, 3),
        "dense_int4_mb": round(dense_int4 / 1e6, 3),
        "factor_fp16_mb": round(factor_fp16 / 1e6, 3),
        "factor_int4_mb": round(factor_int4 / 1e6, 3),
        "shrink_vs_fp16": round(dense_fp16 / max(factor_int4, 1), 2),
        "shrink_vs_dense_int4": round(dense_int4 / max(factor_int4, 1), 2),
    }


def main() -> int:
    t0 = time.time()

    # Load dense weights (only sampled layers)
    dense = load_dense_weights()

    rows: list[dict] = []
    summaries: list[dict] = []

    for tag, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: {ckpt_dir} does not exist")
            continue

        print(f"\n[{tag}] loading factored matrices ...")
        factored = load_factored_matrices(ckpt_dir)
        if not factored:
            print(f"  [skip] no factored matrices found")
            continue

        errs_factor = []
        errs_int4 = []
        errs_dense_int4 = []
        penalties = []
        shrink_vs_fp16_vals = []
        shrink_vs_dense_int4_vals = []

        for key, (A, B, in_f, out_f) in factored.items():
            if key not in dense:
                print(f"  [warn] {key} not in dense weights, skipping")
                continue

            W = dense[key]
            stats = measure(W, A, B)
            stats["key"] = key
            stats["tag"] = tag
            rows.append(stats)

            errs_factor.append(stats["frob_relerr_factor"])
            errs_int4.append(stats["frob_relerr_int4"])
            errs_dense_int4.append(stats["frob_relerr_dense_int4"])
            penalties.append(stats["quant_penalty"])
            shrink_vs_fp16_vals.append(stats["shrink_vs_fp16"])
            shrink_vs_dense_int4_vals.append(stats["shrink_vs_dense_int4"])

            print(f"  {key.split('.mlp.')[1]:20s}  "
                  f"factor_err={stats['frob_relerr_factor']:.6f}  "
                  f"int4_err={stats['frob_relerr_int4']:.6f}  "
                  f"quant_penalty={stats['quant_penalty']:.1f}×  "
                  f"int4_size={stats['factor_int4_mb']:.1f}MB  "
                  f"shrink={stats['shrink_vs_fp16']:.1f}×")

        if errs_factor:
            summaries.append({
                "tag": tag,
                "n_matrices": len(errs_factor),
                "mean_factor_err": round(float(np.mean(errs_factor)), 6),
                "mean_int4_err": round(float(np.mean(errs_int4)), 6),
                "mean_dense_int4_err": round(float(np.mean(errs_dense_int4)), 6),
                "mean_quant_penalty": round(float(np.mean(penalties)), 2),
                "mean_shrink_vs_fp16": round(float(np.mean(shrink_vs_fp16_vals)), 2),
                "mean_shrink_vs_dense_int4": round(float(np.mean(shrink_vs_dense_int4_vals)), 2),
            })
            print(f"  → mean: factor_err={summaries[-1]['mean_factor_err']:.6f}  "
                  f"int4_err={summaries[-1]['mean_int4_err']:.6f}  "
                  f"quant_penalty={summaries[-1]['mean_quant_penalty']:.1f}×  "
                  f"shrink={summaries[-1]['mean_shrink_vs_fp16']:.1f}×")

    # Full-model shrink
    est = estimate_full_model_shrink(
        hidden_dim=1536, intermediate_dim=8960, n_layers=28, vocab_size=151936,
        ffm_rank=1024, attn_rank=640, n_bits=N_BITS, n_kv_heads=2, head_dim=128,
    )

    result = {
        "model": MODEL_ID,
        "n_bits": N_BITS,
        "sample_layers": SAMPLE_LAYERS,
        "dense_fp16_mb_est": round(est.total_fp16_mb, 1),
        "dense_int4_mb_est": round(est.total_int4_mb, 1),
        "factor_int4_mb_est": round(est.total_factor_int4_mb, 1),
        "shrink_vs_fp16_est": round(est.shrink_vs_fp16, 2),
        "shrink_vs_dense_int4_est": round(est.shrink_vs_dense_int4, 2),
        "summaries": summaries,
        "rows": rows,
        "wall_s": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
