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

"""Validate attack #2b: NATIVE factored emission from compress.py.

Unlike round 10 (which SVD-retrofitted an already-merged dense GRC
checkpoint), this path emits (A, B_*) directly during GRC projection,
preserving the GRC-projected weights to machine precision.

Ladder on Qwen2.5-1.5B:

  rung 0  baseline fp16        from HF hub
  rung 1  dense GRC bf16       attn k=640, sink=8, rematerialised to d×d
  rung 2  native factored bf16 attn k=640, sink=8, factored to disk

Success criteria:
  * rung 2 PPL == rung 1 PPL (within bf16 noise) -- proves no SVD-retrofit loss
  * rung 2 on-disk < rung 1 on-disk -- proves bytes actually move
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.compress import compress_hf_model
from hyperretro.hf.factored import load_factored_hf_model

MODEL_ID = "Qwen/Qwen2.5-1.5B"
DENSE_DIR = Path("outputs/_15B_k640_dense_native")
FACT_DIR = Path("outputs/_15B_k640_factored_native")
FACT_FFN_DIR = Path("outputs/_15B_k640_in1024_factored_native")
RANK_K = 640
SINK = 8

EVAL_TEXT = (
    "Machine learning models trained on large corpora exhibit emergent "
    "capabilities not present at smaller scales The transformer architecture "
    "introduced by Vaswani et al in 2017 revolutionized NLP by replacing "
    "recurrent architectures with self-attention mechanisms Large language "
    "models like GPT BERT and their successors have demonstrated remarkable "
    "capabilities in text generation reasoning and code completion These "
    "models are typically trained on vast corpora of internet text and fine "
    "tuned for specific downstream tasks"
) * 3

tok = AutoTokenizer.from_pretrained(MODEL_ID)


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def disk_mb(path: Path) -> float:
    total = 0
    for f in path.iterdir():
        if f.suffix in (".safetensors", ".bin"):
            total += f.stat().st_size
    return total / (1024.0 * 1024.0)


@torch.inference_mode()
def ppl_dense(model_dir: str, dtype=torch.bfloat16) -> float:
    m = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype
    ).cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m
    _free()
    return p


@torch.inference_mode()
def ppl_factored(model_dir: Path, dtype: str = "bfloat16") -> float:
    m, info = load_factored_hf_model(model_dir, dtype=dtype)
    m = m.cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m
    _free()
    return p, info


@torch.inference_mode()
def ppl_baseline() -> float:
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m
    _free()
    return p


def run():
    t0 = time.time()

    print("[rung 0] baseline fp16")
    p0 = ppl_baseline()
    print(f"  PPL={p0:.4f}")

    print("[rung 1] dense GRC bf16 (rematerialised, native compress)")
    if not DENSE_DIR.exists():
        print("  compressing ...")
        compress_hf_model(
            MODEL_ID, str(DENSE_DIR),
            rank_k=RANK_K, sink_T=SINK,
            dtype="bfloat16", factored=False,
        )
    p1 = ppl_dense(str(DENSE_DIR), dtype=torch.bfloat16)
    s1 = disk_mb(DENSE_DIR)
    print(f"  PPL={p1:.4f}  on-disk={s1:.1f} MB")

    print("[rung 2] native factored GRC bf16")
    if not FACT_DIR.exists():
        print("  compressing ...")
        compress_hf_model(
            MODEL_ID, str(FACT_DIR),
            rank_k=RANK_K, sink_T=SINK,
            dtype="bfloat16", factored=True,
        )
    p2, info = ppl_factored(FACT_DIR, dtype="bfloat16")
    s2 = disk_mb(FACT_DIR)
    print(f"  PPL={p2:.4f}  on-disk={s2:.1f} MB  (patched {info['patched_linears']} attn linears)")

    print("[rung 3] native factored GRC + FFN factored bf16")
    if not FACT_FFN_DIR.exists():
        print("  compressing ...")
        compress_hf_model(
            MODEL_ID, str(FACT_FFN_DIR),
            rank_k=RANK_K, sink_T=SINK,
            ffn_rank_in=1024, ffn_mode="svd",
            dtype="bfloat16", factored=True,
        )
    p3, info3 = ppl_factored(FACT_FFN_DIR, dtype="bfloat16")
    s3 = disk_mb(FACT_FFN_DIR)
    print(f"  PPL={p3:.4f}  on-disk={s3:.1f} MB  "
          f"(patched {info3['patched_linears']}, ffn={info3['manifest_ffn']})")

    # Get baseline disk size from HF cache
    from huggingface_hub import scan_cache_dir
    s0 = 0.0
    info_cache = scan_cache_dir()
    for repo in info_cache.repos:
        if repo.repo_id == MODEL_ID:
            s0 = repo.size_on_disk / (1024.0 * 1024.0)

    print(f"\n[wall] {time.time()-t0:.1f}s")
    print(f"\n{'config':<26} {'PPL':>10} {'PPLx':>8} {'on-disk MB':>14} {'shrink':>10}")
    rows = [
        ("baseline_fp16", p0, s0),
        ("dense_grc_bf16_k640", p1, s1),
        ("factored_grc_bf16_k640", p2, s2),
        ("factored_grc_ffn_bf16", p3, s3),
    ]
    base = rows[0][1]; base_sz = rows[0][2]
    out = []
    for n, p, m in rows:
        rx = p / base
        sx = base_sz / max(m, 1e-9)
        out.append({"config": n, "ppl": p, "ppl_x": rx,
                    "on_disk_mb": m, "shrink_x_vs_baseline": sx})
        print(f"{n:<26} {p:>10.4f} {rx:>8.3f} {m:>14.1f} {sx:>10.3f}")

    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/native_factored_ladder.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\n[saved] benchmarks/native_factored_ladder.json")


if __name__ == "__main__":
    run()
