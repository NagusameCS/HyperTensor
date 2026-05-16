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

"""Attack #7 validator: factor × int4 reconstruction error on real FFN weights.

Loads Qwen2.5-1.5B's FFN matrices, factors at multiple ranks (vanilla and
activation-aware), int4-quantizes A and B, and measures Frobenius
reconstruction error vs the original weight.

Key question: does int4 quantization of the factored (A, B) pair add
acceptable additional error beyond the factoring error itself?

If the "quantization penalty" (int4_err / factor_err) is < 2×, the
composition is viable — we get the shrink multiplier of int4 on top
of the shrink of factoring, with modest additional PPL cost.

Output: benchmarks/factor_int4_error.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

from hyperretro.hf.factor_quantize import (
    factor_then_quantize,
    estimate_full_model_shrink,
    int4_storage_bytes,
)
from hyperretro.hf.factored import _FFN_SUFFIXES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
RANKS = [256, 384, 512, 640, 768, 896, 1024]
N_BITS = 4
OUT = _ROOT / "benchmarks" / "factor_int4_error.json"
CALIB_PATH = _ROOT / "data" / "wikitext2_train_5k.txt"
DEVICE = "cpu"  # GPU is down

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dense_ffn_weights(model_id: str) -> dict[str, np.ndarray]:
    """Load FFN weight matrices (gate_proj, up_proj, down_proj) from HF."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[load] {model_id} → cpu ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
    )
    model.to("cpu")
    sd = model.state_dict()
    weights = {}
    for key, tensor in sd.items():
        if any(key.endswith(s) for s in _FFN_SUFFIXES):
            weights[key] = tensor.float().cpu().numpy().copy()
    del model
    return weights


def collect_ffn_norms_for_weights(
    weights: dict[str, np.ndarray],
    model_id: str,
    corpus_path: str | Path,
) -> dict[str, np.ndarray]:
    """Run a calibration forward pass to get column norms for the FFN weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hyperretro.hf.activation import collect_ffn_input_norms

    print("[norms] collecting activation column norms ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
    )
    model.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    norms = collect_ffn_input_norms(
        model, tokenizer,
        corpus_path=corpus_path,
        n_batches=4,
        seq_len=256,
        device=DEVICE,
    )
    del model
    return norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()

    # 1. Load weights
    weights = load_dense_ffn_weights(MODEL_ID)
    print(f"[load] {len(weights)} FFN weight matrices loaded")

    # 2. Collect activation norms (CPU: ~2 min)
    norms = {}
    if CALIB_PATH.exists():
        norms = collect_ffn_norms_for_weights(weights, MODEL_ID, CALIB_PATH)
        print(f"[norms] collected norms for {len(norms)} FFN keys")
    else:
        print(f"[norms] corpus not found at {CALIB_PATH}, skipping aware path")

    # 3. For each weight matrix and each rank, compute factor+int4 errors
    rows: list[dict] = []
    weight_keys = sorted(weights.keys())

    for key in weight_keys:
        W = weights[key]
        m, n = W.shape
        frob_W = float(np.linalg.norm(W, "fro"))
        print(f"\n[{key}] shape=({m},{n}) ||W||_F={frob_W:.1f}")

        col_norms = norms.get(key, None)

        # Dense int4 baseline (once per key)
        from hyperretro.hf.factor_quantize import quantize_matrix_int4, dequantize_matrix_int4
        W_q, W_scales = quantize_matrix_int4(W, axis=0, n_bits=N_BITS)
        W_dq_dense = dequantize_matrix_int4(W_q, W_scales, axis=0)
        dense_int4_err = float(np.linalg.norm(W - W_dq_dense, "fro") / frob_W)
        dense_int4_bytes = int4_storage_bytes(m, n, N_BITS)

        for rank in RANKS:
            # Vanilla factor + int4
            r_vanilla = factor_then_quantize(W, rank, col_norms=None, n_bits=N_BITS)
            row_vanilla = {
                "key": key,
                "rank": rank,
                "mode": "vanilla",
                "shape": list(W.shape),
                "frob_relerr_factor": round(r_vanilla.frob_relerr_factor, 8),
                "frob_relerr_int4": round(r_vanilla.frob_relerr_int4, 8),
                "frob_relerr_dense_int4": round(r_vanilla.frob_relerr_dense_int4, 8),
                "quant_penalty": round(
                    (r_vanilla.frob_relerr_int4 / max(r_vanilla.frob_relerr_factor, 1e-12)), 3
                ) if r_vanilla.frob_relerr_factor > 1e-12 else 0.0,
                "dense_fp16_mb": round(r_vanilla.dense_fp16_bytes / 1e6, 3),
                "factor_fp16_mb": round(r_vanilla.factor_fp16_bytes / 1e6, 3),
                "factor_int4_mb": round(r_vanilla.factor_int4_bytes / 1e6, 3),
                "shrink_vs_fp16": round(r_vanilla.dense_fp16_bytes / max(r_vanilla.factor_int4_bytes, 1), 2),
                "shrink_vs_dense_int4": round(dense_int4_bytes / max(r_vanilla.factor_int4_bytes, 1), 2),
            }
            rows.append(row_vanilla)

            # Aware factor + int4
            if col_norms is not None:
                r_aware = factor_then_quantize(W, rank, col_norms=col_norms, n_bits=N_BITS)
                row_aware = {
                    "key": key,
                    "rank": rank,
                    "mode": "aware",
                    "shape": list(W.shape),
                    "frob_relerr_factor": round(r_aware.frob_relerr_factor, 8),
                    "frob_relerr_int4": round(r_aware.frob_relerr_int4, 8),
                    "frob_relerr_dense_int4": round(r_aware.frob_relerr_dense_int4, 8),
                    "quant_penalty": round(
                        (r_aware.frob_relerr_int4 / max(r_aware.frob_relerr_factor, 1e-12)), 3
                    ) if r_aware.frob_relerr_factor > 1e-12 else 0.0,
                    "dense_fp16_mb": round(r_aware.dense_fp16_bytes / 1e6, 3),
                    "factor_fp16_mb": round(r_aware.factor_fp16_bytes / 1e6, 3),
                    "factor_int4_mb": round(r_aware.factor_int4_bytes / 1e6, 3),
                    "shrink_vs_fp16": round(r_aware.dense_fp16_bytes / max(r_aware.factor_int4_bytes, 1), 2),
                    "shrink_vs_dense_int4": round(dense_int4_bytes / max(r_aware.factor_int4_bytes, 1), 2),
                }
                rows.append(row_aware)

            print(f"  r={rank:4d}  "
                  f"vanilla: factor_err={r_vanilla.frob_relerr_factor:.6f}  "
                  f"int4_err={r_vanilla.frob_relerr_int4:.6f}  "
                  f"factor_int4_mb={r_vanilla.factor_int4_bytes/1e6:.2f}  "
                  f"shrink={r_vanilla.dense_fp16_bytes/max(r_vanilla.factor_int4_bytes,1):.1f}×", end="")
            if col_norms is not None:
                print(f"  |  aware: factor_err={r_aware.frob_relerr_factor:.6f}  "
                      f"int4_err={r_aware.frob_relerr_int4:.6f}  "
                      f"shrink={r_aware.dense_fp16_bytes/max(r_aware.factor_int4_bytes,1):.1f}×")
            else:
                print()

    # 4. Full-model shrink estimate
    est = estimate_full_model_shrink(
        hidden_dim=1536, intermediate_dim=8960, n_layers=28, vocab_size=151936,
        ffm_rank=1024, attn_rank=640, n_bits=N_BITS, n_kv_heads=2, head_dim=128,
    )

    # 5. Save
    result = {
        "model": MODEL_ID,
        "n_bits": N_BITS,
        "ranks": RANKS,
        "weight_keys": weight_keys,
        "dense_fp16_mb_est": round(est.total_fp16_mb, 1),
        "dense_int4_mb_est": round(est.total_int4_mb, 1),
        "factor_int4_mb_est": round(est.total_factor_int4_mb, 1),
        "shrink_vs_fp16_est": round(est.shrink_vs_fp16, 2),
        "shrink_vs_dense_int4_est": round(est.shrink_vs_dense_int4, 2),
        "rows": rows,
        "wall_s": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
