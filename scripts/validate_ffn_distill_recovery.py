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

"""Validate FFN-compression + distillation recovery on Qwen2.5-0.5B.

v0.2.1 finding: naive low-rank SVD of FFN catastrophically hurts PPL
(e.g. PPL 86 vs 7.18 baseline at ~86% rank). This script tests the
*natural* follow-on hypothesis: a brief LoRA distillation pass should
recover most of the lost PPL, because the rank-r factorization captures
the dominant subspace and LoRA fills the rest.

Pipeline:
  baseline       -> attn-only k=640                  (known-good v0.2.0)
  ffn_naive      -> attn k=640 + FFN SVD,  no LoRA
  ffn_distill    -> attn k=640 + FFN SVD,  short LoRA distill on calib text

Outputs benchmarks/ffn_distill_recovery.json.
"""
import sys; sys.path.insert(0, '.')
import json, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.compress import compress_state_dict, CompressConfig
from hyperretro.hf.ffn_compress import (
    compress_ffn_state_dict, FFNCompressConfig,
)
from hyperretro.hf.distill import distill_hf_model

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("benchmarks/ffn_distill_recovery.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

SAMPLE = (
    "Robert Boulter is an English film, television and theatre actor. He "
    "had a guest-starring role on the television series The Bill in 2000. "
    "This was followed by a starring role in the play Herons written by "
    "Simon Stephens, which was performed in 2001 at the Royal Court "
    "Theatre. He had a guest role in the television series Judge John "
    "Deed in 2002. In 2004 Boulter landed a role as Craig in the episode "
    "Teddy's Story of the television series The Long Firm; he starred "
    "alongside actors Mark Strong and Derek Jacobi. He was cast in the "
    "2005 theatre productions of the Philip Ridley play Mercury Fur. "
) * 8


@torch.inference_mode()
def perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    seq = enc.size(1)
    out = model(enc, labels=enc)
    return float(torch.exp(out.loss.float()))


def eval_state_dict(sd: dict, label: str) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32
    )
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    ppl = perplexity(model, tok, SAMPLE)
    print(f"  [{label}] PPL = {ppl:.3f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return ppl


def eval_model_dir(path: Path, label: str) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32
    )
    model.cuda().eval()
    ppl = perplexity(model, tok, SAMPLE)
    print(f"  [{label}] PPL = {ppl:.3f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return ppl


def main():
    results = {"model": MODEL, "configs": []}

    print("=== Loading baseline state_dict ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32
    )
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    del base_model; gc.collect()

    # ---- 1. Baseline: untouched ----
    ppl_base = eval_state_dict(base_sd, "baseline")
    results["configs"].append({"label": "baseline", "ppl": ppl_base})

    # ---- 2. Attn-only k=640 (known v0.2.0 best for 0.5B) ----
    print("=== attn-only k=640 ===")
    sd_attn = {k: v.clone() for k, v in base_sd.items()}
    cfg = CompressConfig(rank_k=640, sink_T=8, layers=[])
    compress_state_dict(sd_attn, cfg)
    ppl_attn = eval_state_dict(sd_attn, "attn-only k=640")
    results["configs"].append(
        {"label": "attn_only_k640", "ppl": ppl_attn}
    )
    del sd_attn; gc.collect()

    # ---- 3. Attn + FFN-down naive (no distill) ----
    print("=== attn k=640 + FFN-down rank=512 naive ===")
    sd_ffn = {k: v.clone() for k, v in base_sd.items()}
    compress_state_dict(sd_ffn, CompressConfig(
        rank_k=640, sink_T=8, layers=[],
    ))
    ffn_cfg = FFNCompressConfig(rank_in=0, rank_out=512, layers=[], mode="svd")
    compress_ffn_state_dict(sd_ffn, ffn_cfg)
    ppl_naive = eval_state_dict(sd_ffn, "FFN naive r_out=512")
    results["configs"].append(
        {"label": "ffn_naive_rout512", "ppl": ppl_naive}
    )
    del sd_ffn, base_sd; gc.collect()
    torch.cuda.empty_cache()

    # ---- 4. Attn + FFN-down WITH distill recovery ----
    print("=== attn k=640 + FFN-down rank=512 + LoRA distill ===")
    out_dir = Path("outputs/_ffn_distill_recovery")
    if out_dir.exists():
        import shutil; shutil.rmtree(out_dir)

    # Write the calibration corpus to a file the distiller can read.
    corpus = Path("outputs/_ffn_distill_corpus.txt")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(SAMPLE * 4, encoding="utf-8")

    t0 = time.time()
    report = distill_hf_model(
        MODEL, str(out_dir),
        rank_k=640, sink_T=8,
        lora_rank=16, lora_alpha=32.0,
        steps=150, batch_size=1, seq_len=256,
        learning_rate=1e-4,
        corpus_path=str(corpus),
        layers=[],
        device="cuda", dtype="float32",
        loss_type="kl",  # KL won the v0.2.1 race
        kl_temperature=4.0,
        ffn_rank_in=0,
        ffn_rank_out=512,
    )
    distill_secs = time.time() - t0
    print(f"  distill took {distill_secs:.1f}s")

    ppl_distill = eval_model_dir(out_dir, "FFN + distill r_out=512")
    results["configs"].append({
        "label": "ffn_distill_rout512",
        "ppl": ppl_distill,
        "distill_seconds": distill_secs,
        "lora_rank": 16, "steps": 150,
        "n_layers_distilled": report.get("n_layers_distilled"),
    })

    print("\n=== Summary ===")
    for c in results["configs"]:
        print(f"  {c['label']:35s}  PPL {c['ppl']:.3f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
