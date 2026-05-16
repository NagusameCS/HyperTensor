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

"""Validate GRC * nf4 composition vs industry-meta quantization.

Compression ladder on Qwen2.5-1.5B:

  rung 0  baseline fp16          -- industry default
  rung 1  bnb nf4 only            -- bitsandbytes industry meta (QLoRA)
  rung 2  HyperRetro GRC bf16     -- attn k=640 + FFN_in r=1024 (v0.2.1)
  rung 3  HyperRetro GRC * nf4    -- composed (this round's contribution)

Eval: held-out ML-history paragraph (zero overlap with the WikiText-2
calibration corpus used by FFN distill). Reports on-disk size and PPL
ratio vs fp16 baseline.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
GRC_DIR = "outputs/_15B_k640_in1024_bf16"   # v0.2.1 round-8 best config

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


@torch.inference_mode()
def ppl_native(model_path: str, dtype: torch.dtype = torch.float16) -> float:
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m
    _free()
    return p


# back-compat alias
ppl_fp16 = ppl_native


@torch.inference_mode()
def ppl_nf4(model_path: str) -> float:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto"
    ).eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m
    _free()
    return p


def disk_mb(path: str) -> float:
    p = Path(path)
    if p.is_dir():
        total = 0
        for f in p.iterdir():
            if f.suffix in (".safetensors", ".bin"):
                total += f.stat().st_size
        return total / (1024.0 * 1024.0)
    return 0.0


def hf_cache_mb(repo_id: str) -> float:
    """Approx on-disk size of HF cached snapshot for a repo."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_id == repo_id:
            return repo.size_on_disk / (1024.0 * 1024.0)
    return 0.0


def run():
    rows = []
    t0 = time.time()

    print("[rung 0] baseline fp16")
    p0 = ppl_fp16(MODEL_ID)
    s0 = hf_cache_mb(MODEL_ID)
    rows.append(("baseline_fp16", p0, s0))
    print(f"  PPL={p0:.4f}  on-disk~={s0:.0f} MB")

    print("[rung 1] bnb nf4 only (industry meta)")
    p1 = ppl_nf4(MODEL_ID)
    # produce a quantized snapshot to measure on-disk size
    from hyperretro.hf.quantize import quantize_to_nf4
    nf4_dir = "outputs/_15B_nf4_only"
    rep1 = quantize_to_nf4(MODEL_ID, nf4_dir)
    s1 = rep1["on_disk_mb"]
    rows.append(("nf4_only", p1, s1))
    print(f"  PPL={p1:.4f}  on-disk={s1:.0f} MB  ({s0/max(s1,1e-9):.2f}x vs fp16)")

    print("[rung 2] HyperRetro GRC bf16 (k=640, ffn_in=1024)")
    if not Path(GRC_DIR).exists():
        print(f"  SKIP: {GRC_DIR} missing -- run round-8 distill first")
        p2 = float("nan"); s2 = float("nan")
    else:
        p2 = ppl_native(GRC_DIR, dtype=torch.bfloat16)
        s2 = disk_mb(GRC_DIR)
    rows.append(("grc_bf16", p2, s2))
    print(f"  PPL={p2:.4f}  on-disk={s2:.0f} MB")

    print("[rung 3] HyperRetro GRC * nf4 (composed)")
    if not Path(GRC_DIR).exists():
        print("  SKIP")
        p3 = float("nan"); s3 = float("nan")
    else:
        composed_dir = "outputs/_15B_grc_x_nf4"
        rep3 = quantize_to_nf4(GRC_DIR, composed_dir)
        p3 = ppl_nf4(composed_dir)
        s3 = rep3["on_disk_mb"]
    rows.append(("grc_x_nf4", p3, s3))
    print(f"  PPL={p3:.4f}  on-disk={s3:.0f} MB  "
          f"({s0/max(s3,1e-9):.2f}x vs fp16)")

    print(f"\n[wall] {time.time()-t0:.1f}s")
    print(f"\n{'config':<20} {'PPL':>10} {'PPLx':>8} {'size_MB':>10} {'shrink':>8}")
    base_ppl = rows[0][1]; base_sz = rows[0][2]
    out = []
    for name, ppl, sz in rows:
        rx = (ppl / base_ppl) if base_ppl else float("nan")
        sx = (base_sz / sz) if sz else float("nan")
        out.append({"config": name, "ppl": ppl, "ppl_x": rx,
                    "size_mb": sz, "shrink_x": sx})
        print(f"{name:<20} {ppl:>10.4f} {rx:>8.3f} {sz:>10.1f} {sx:>8.2f}")

    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/grc_x_nf4_ladder.json").write_text(json.dumps(out, indent=2))
    print("\n[saved] benchmarks/grc_x_nf4_ladder.json")


if __name__ == "__main__":
    run()
