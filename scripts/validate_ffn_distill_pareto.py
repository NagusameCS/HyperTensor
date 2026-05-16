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

"""FFN-distill Pareto sweep: how aggressive can we compress + recover?

Background: v0.2.1 round 2 showed FFN-down r=512 (vs 896 hidden) recovers
from PPL 23 679 -> 2.12 in 150 steps of LoRA distillation. This script
sweeps r_out down to the point where recovery fails, to find the actual
compression frontier.

For each rank we report:
  - naive PPL (no distill) — should explode
  - distilled PPL (150 steps, r=16 LoRA, ~30s/config)
  - param savings vs baseline FFN weight count

Qwen2.5-0.5B: 24 layers, hidden=896, intermediate=4864.
Each down_proj is 896 x 4864 = 4.36M params. r_out=k stores 896*k + k*4864
= k*(5760) params. Saving vs full: 4.36M - k*5760.
  k=512 -> 2.95M stored, save 1.41M / layer  -> 32% per-layer FFN-down save
  k=256 -> 1.47M stored, save 2.89M / layer  -> 66%
  k=128 -> 0.74M stored, save 3.62M / layer  -> 83%
  k= 64 -> 0.37M stored, save 3.99M / layer  -> 92%
"""
import sys; sys.path.insert(0, '.')
import json, time, gc, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.compress import compress_state_dict, CompressConfig
from hyperretro.hf.ffn_compress import (
    compress_ffn_state_dict, FFNCompressConfig,
)
from hyperretro.hf.distill import distill_hf_model

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("benchmarks/ffn_distill_pareto.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
HIDDEN = 896
INTER = 4864
LAYERS = 24

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
    out = model(enc, labels=enc)
    return float(torch.exp(out.loss.float()))


def eval_state_dict(sd: dict, label: str) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    ppl = perplexity(model, tok, SAMPLE)
    print(f"  [{label}] PPL = {ppl:.3f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return ppl


def eval_model_dir(path: Path, label: str) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.cuda().eval()
    ppl = perplexity(model, tok, SAMPLE)
    print(f"  [{label}] PPL = {ppl:.3f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return ppl


def param_savings(r_out: int) -> tuple[int, float]:
    """Return (saved_params_total, fraction_of_total_model_params)."""
    full = HIDDEN * INTER  # per layer down_proj
    stored = HIDDEN * r_out + r_out * INTER  # factorized
    saved_per_layer = full - stored
    saved_total = saved_per_layer * LAYERS
    # Qwen2.5-0.5B has ~494M params total
    return saved_total, saved_total / 494e6


def run_config(r_out: int, base_sd: dict, corpus_path: Path) -> dict:
    print(f"\n=== r_out = {r_out} ===", flush=True)

    # Naive (no distill)
    sd = {k: v.clone() for k, v in base_sd.items()}
    compress_state_dict(sd, CompressConfig(rank_k=640, sink_T=8, layers=[]))
    compress_ffn_state_dict(sd, FFNCompressConfig(
        rank_in=0, rank_out=r_out, layers=[], mode="svd",
    ))
    ppl_naive = eval_state_dict(sd, f"naive r_out={r_out}")
    del sd; gc.collect(); torch.cuda.empty_cache()

    # Distilled
    out_dir = Path(f"outputs/_ffn_pareto_r{r_out}")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    t0 = time.time()
    report = distill_hf_model(
        MODEL, str(out_dir),
        rank_k=640, sink_T=8,
        lora_rank=16, lora_alpha=32.0,
        steps=150, batch_size=1, seq_len=256,
        learning_rate=1e-4,
        corpus_path=str(corpus_path),
        layers=[], device="cuda", dtype="float32",
        loss_type="kl", kl_temperature=4.0,
        ffn_rank_in=0, ffn_rank_out=r_out,
    )
    distill_secs = time.time() - t0
    ppl_distill = eval_model_dir(out_dir, f"distilled r_out={r_out}")

    saved, frac = param_savings(r_out)
    return {
        "r_out": r_out,
        "ppl_naive": ppl_naive,
        "ppl_distill": ppl_distill,
        "distill_seconds": distill_secs,
        "saved_params": saved,
        "saved_fraction_of_model": frac,
        "n_layers_distilled": report.get("n_layers_distilled"),
    }


def main():
    print("=== Loading baseline ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32
    )
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    del base_model; gc.collect()

    ppl_base = eval_state_dict(base_sd, "baseline")

    # Calibration corpus
    corpus = Path("outputs/_ffn_pareto_corpus.txt")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(SAMPLE * 4, encoding="utf-8")

    results = {
        "model": MODEL,
        "baseline_ppl": ppl_base,
        "hidden": HIDDEN, "intermediate": INTER, "layers": LAYERS,
        "configs": [],
    }

    for r_out in [512, 384, 256, 128, 64]:
        try:
            results["configs"].append(run_config(r_out, base_sd, corpus))
            OUT.write_text(json.dumps(results, indent=2))  # checkpoint
        except Exception as e:
            print(f"  FAILED r_out={r_out}: {e}")
            results["configs"].append({"r_out": r_out, "error": str(e)})

    print("\n=== Pareto Summary ===")
    print(f"  baseline                                  PPL {ppl_base:.3f}")
    for c in results["configs"]:
        if "error" in c:
            print(f"  r_out={c['r_out']:4d}  ERROR: {c['error']}")
            continue
        print(
            f"  r_out={c['r_out']:4d}  naive={c['ppl_naive']:10.2f}  "
            f"distilled={c['ppl_distill']:7.3f}  "
            f"save={c['saved_fraction_of_model']*100:.1f}% of model"
        )

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
