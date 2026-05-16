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

"""Round-13 mini-A/B: activation-aware FFN factoring (Wanda/AWQ-style) vs vanilla SVD.

Single axis comparison. No distill. Same attn-k, same FFN rank. Measure PPL
and on-disk MB. Goal: confirm activation-aware shifts truncation error into
low-activation channels and recovers PPL at fixed rank.
"""
from __future__ import annotations
import gc, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.compress import compress_hf_model
from hyperretro.hf.factored import load_factored_hf_model

MODEL_ID = "Qwen/Qwen2.5-1.5B"
RANK_K = 640
SINK = 8
FFN_R = 1024
VANILLA_DIR = Path("outputs/_15B_factored_native_full")  # reused from round 12
AWARE_DIR = Path("outputs/_15B_factored_aware")
OUT = Path("benchmarks/activation_aware_ffn_ladder.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_TEXT = (
    "The 1956 Dartmouth conference marked the founding of artificial "
    "intelligence as a field. Researchers there hoped to capture human "
    "reasoning with symbolic logic. Neural networks fell out of favour "
    "after the perceptron's limitations were exposed in 1969, but were "
    "rehabilitated by the 1986 backpropagation paper. Deep learning's "
    "modern era began with AlexNet on ImageNet in 2012, and Transformers "
    "redefined language modelling in 2017. By 2022, instruction-tuned "
    "models reached general utility, and by 2025 reasoning over long "
    "contexts had become the dominant frontier. "
) * 3
CALIB_PATH = Path("data/wikitext2_train.txt")
FALLBACK_CALIB = Path("outputs/_aware_calib.txt")


def disk_mb(p: Path) -> float:
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1024 / 1024


@torch.inference_mode()
def ppl(m, tok) -> float:
    m.eval()
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids.to(m.device)
    return float(math.exp(m(ids, labels=ids).loss.float()))


def eval_factored(p: Path) -> tuple[float, float, dict]:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m, info = load_factored_hf_model(p, dtype="bfloat16" if DEVICE == "cuda" else "float32")
    m.to(DEVICE)
    val = ppl(m, tok)
    sz = disk_mb(p)
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return val, sz, info


def eval_dense_baseline() -> tuple[float, float]:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(DEVICE)
    val = ppl(m, tok)
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return val, 2955.4


def main() -> None:
    t0 = time.time()
    print(f"=== Round 13: activation-aware FFN factoring A/B on {MODEL_ID} (device={DEVICE}) ===")

    # Calibration corpus for activation collection
    if CALIB_PATH.exists():
        calib = CALIB_PATH
    else:
        FALLBACK_CALIB.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_CALIB.write_text(EVAL_TEXT * 8, encoding="utf-8")
        calib = FALLBACK_CALIB

    print("[rung 0] baseline fp16")
    p0, s0 = eval_dense_baseline()
    print(f"  PPL={p0:.4f}  on-disk={s0:.1f} MB (reference)")

    print("[rung 1] vanilla SVD factored attn+FFN")
    if not VANILLA_DIR.exists():
        compress_hf_model(
            MODEL_ID, str(VANILLA_DIR),
            rank_k=RANK_K, sink_T=SINK,
            ffn_rank_in=FFN_R, ffn_rank_out=FFN_R, ffn_mode="svd",
            dtype="bfloat16", factored=True,
        )
    p1, s1, info1 = eval_factored(VANILLA_DIR)
    print(f"  PPL={p1:.4f}  on-disk={s1:.1f} MB  attn={info1['manifest_layers']} ffn={info1['manifest_ffn']}")

    print("[rung 2] activation-aware SVD factored attn+FFN")
    if not AWARE_DIR.exists():
        compress_hf_model(
            MODEL_ID, str(AWARE_DIR),
            rank_k=RANK_K, sink_T=SINK,
            ffn_rank_in=FFN_R, ffn_rank_out=FFN_R, ffn_mode="svd",
            dtype="bfloat16", factored=True,
            activation_aware=True, activation_corpus_path=str(calib),
            activation_n_batches=4 if DEVICE == "cpu" else 16,
            activation_seq_len=256 if DEVICE == "cpu" else 512,
        )
    p2, s2, info2 = eval_factored(AWARE_DIR)
    print(f"  PPL={p2:.4f}  on-disk={s2:.1f} MB  attn={info2['manifest_layers']} ffn={info2['manifest_ffn']}")

    rows = [
        ("baseline_fp16", p0, s0),
        ("factored_vanilla_svd", p1, s1),
        ("factored_activation_aware", p2, s2),
    ]
    print(f"\n[wall] {time.time()-t0:.1f}s\n")
    print(f"{'config':<30s}{'PPL':>14s}{'PPLx':>10s}{'on-disk MB':>14s}{'shrink':>9s}")
    for name, pp, sz in rows:
        print(f"{name:<30s}{pp:>14.4f}{pp/p0:>10.3f}{sz:>14.1f}{s0/max(sz,1e-6):>9.3f}")

    aware_vs_vanilla = (p1 - p2) / p1 if p1 > 0 else 0.0
    print(f"\n[aware Δ vs vanilla] PPL: {p2:.2f} vs {p1:.2f}  ({aware_vs_vanilla*100:+.2f}%)")

    OUT.write_text(json.dumps({
        "model": MODEL_ID, "rank_k": RANK_K, "sink_T": SINK, "ffn_rank": FFN_R,
        "calib_path": str(calib),
        "rungs": [{"name": n, "ppl": pp, "ppl_x": pp/p0,
                   "on_disk_mb": sz, "shrink_x": s0/max(sz,1e-6)}
                  for n, pp, sz in rows],
        "aware_ppl_delta_pct_vs_vanilla": aware_vs_vanilla * 100.0,
    }, indent=2))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
