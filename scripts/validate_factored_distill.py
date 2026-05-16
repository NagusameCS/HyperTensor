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

"""Round-12 closed loop: native factored attn + FFN + LoRA distill recovery.

Goal: combine round-11's native factored emission (PPL-preserving, real
on-disk bytes saved) with round-8's LoRA distill (PPL recovery after FFN
compression). Result is a checkpoint that lands in factored form on disk
*and* preserves PPL within the round-8 envelope.

Ladder:
  0. baseline fp16
  1. native factored (attn k=640 + FFN r=1024, no distill)
  2. distilled factored (attn k=640 + FFN r=1024 + KL distill 200 steps)

Outputs: benchmarks/factored_distill_ladder.json
"""
from __future__ import annotations
import gc, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.compress import compress_hf_model
from hyperretro.hf.distill import distill_hf_model
from hyperretro.hf.factored import load_factored_hf_model

MODEL_ID = "Qwen/Qwen2.5-1.5B"
NATIVE_DIR = Path("outputs/_15B_factored_native_full")
DISTILL_DIR = Path("outputs/_15B_factored_distilled")
RANK_K = 640
SINK = 8
FFN_R = 1024
DISTILL_STEPS = 200
LORA_R = 16
OUT = Path("benchmarks/factored_distill_ladder.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

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
CALIB_PATH = Path("outputs/_factored_distill_corpus.txt")


def disk_mb(path: Path) -> float:
    if not path.exists(): return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


@torch.inference_mode()
def ppl(model, tok) -> float:
    model.eval()
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, labels=ids)
    return float(math.exp(out.loss.float()))


def eval_dense(path_or_id) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=torch.float16).cuda()
    p = ppl(m, tok)
    del m; gc.collect(); torch.cuda.empty_cache()
    return p


def eval_factored(path: Path) -> tuple[float, dict]:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m, info = load_factored_hf_model(path, dtype="bfloat16")
    m.cuda()
    p = ppl(m, tok)
    del m; gc.collect(); torch.cuda.empty_cache()
    return p, info


def main():
    t0 = time.time()
    print(f"=== Round 12: factored + distill on {MODEL_ID} ===")

    # Calibration corpus for distillation
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(EVAL_TEXT * 6, encoding="utf-8")

    print("[rung 0] baseline fp16 PPL")
    p0 = eval_dense(MODEL_ID)
    s0 = 2955.4  # known fp16 footprint of Qwen2.5-1.5B
    print(f"  PPL={p0:.4f}  on-disk={s0:.1f} MB (reference)")

    print("[rung 1] native factored attn+FFN, no distill")
    if not NATIVE_DIR.exists():
        compress_hf_model(
            MODEL_ID, str(NATIVE_DIR),
            rank_k=RANK_K, sink_T=SINK,
            ffn_rank_in=FFN_R, ffn_rank_out=FFN_R,
            ffn_mode="svd",
            dtype="bfloat16", factored=True,
        )
    p1, info1 = eval_factored(NATIVE_DIR)
    s1 = disk_mb(NATIVE_DIR)
    print(f"  PPL={p1:.4f}  on-disk={s1:.1f} MB  attn={info1['manifest_layers']} ffn={info1['manifest_ffn']}")

    print("[rung 2] distilled factored attn+FFN")
    if not DISTILL_DIR.exists():
        distill_hf_model(
            MODEL_ID, str(DISTILL_DIR),
            rank_k=RANK_K, sink_T=SINK,
            lora_rank=LORA_R, lora_alpha=float(LORA_R * 2),
            steps=DISTILL_STEPS, batch_size=1, seq_len=256,
            learning_rate=1e-4,
            corpus_path=str(CALIB_PATH),
            layers=[],
            device="cuda", dtype="float32",
            loss_type="kl", kl_temperature=4.0,
            ffn_rank_in=FFN_R, ffn_rank_out=FFN_R,
            factored=True,
            factored_ffn_rel_tol=1e-3,
            save_dtype="bfloat16",
        )
    p2, info2 = eval_factored(DISTILL_DIR)
    s2 = disk_mb(DISTILL_DIR)
    print(f"  PPL={p2:.4f}  on-disk={s2:.1f} MB  attn={info2['manifest_layers']} ffn={info2['manifest_ffn']}")

    rows = [
        ("baseline_fp16", p0, s0),
        ("factored_no_distill", p1, s1),
        ("factored_distilled", p2, s2),
    ]
    print(f"\n[wall] {time.time()-t0:.1f}s\n")
    print(f"{'config':<28s}{'PPL':>10s}{'PPLx':>9s}{'on-disk MB':>15s}{'shrink':>10s}")
    for name, pp, sz in rows:
        print(f"{name:<28s}{pp:>10.4f}{pp/p0:>9.3f}{sz:>15.1f}{s0/max(sz,1e-6):>10.3f}")

    OUT.write_text(json.dumps({
        "model": MODEL_ID,
        "rank_k": RANK_K, "sink_T": SINK, "ffn_rank": FFN_R,
        "distill_steps": DISTILL_STEPS, "lora_rank": LORA_R,
        "rungs": [{"name": n, "ppl": pp, "ppl_x": pp/p0,
                   "on_disk_mb": sz, "shrink_x": s0/max(sz,1e-6)}
                  for n, pp, sz in rows],
    }, indent=2))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
