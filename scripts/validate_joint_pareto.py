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

"""Systematic joint Pareto: attn rank × FFN rank × distill recovery.

Finds the compression frontier for Qwen2.5-0.5B: maximum parameter savings
at a target PPL budget (≤1.10× baseline). Tests combinations that prior
rounds never tried together:

  - FFN gate/up compression (ffn_rank_in) WITH distillation recovery
  - Joint attn (rank_k) + FFN gate + FFN down compression
  - Fixed sufficient distill budget (600 steps, LoRA r=32)

Each config takes ~2 min (600 steps). Total ~15-20 min for 8 configs.

Outputs benchmarks/joint_pareto.json
"""
import sys; sys.path.insert(0, '.')
import json, time, gc, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.distill import distill_hf_model

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = Path("benchmarks/joint_pareto.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
HIDDEN = 896; INTER = 4864; LAYERS = 24

SAMPLE = (
    "Robert Boulter is an English film, television and theatre actor. He "
    "had a guest-starring role on the television series The Bill in 2000. "
    "This was followed by a starring role in the play Herons written by "
    "Simon Stephens, which was performed in 2001 at the Royal Court "
    "Theatre. He had a guest role in the television series Judge John "
    "Deed in 2002. In 2004 Boulter landed a role as Craig in the episode "
    "Teddy's Story of the television series The Long Firm; he starred "
    "alongside actors Mark Strong and Derek Jacobi. He was cast in the "
    "2005 theatre productions of the Philip Ridley play Mercury Fur, "
    "which was performed at the Drum Theatre in Plymouth and the Menier "
    "Chocolate Factory in London. He was directed by John Tiffany and "
    "starred alongside Ben Whishaw, Shane Zaza, Harry Kent, Fraser Ayres, "
    "Sophie Stanton and Dominic Hall."
) * 4


@torch.inference_mode()
def perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    out = model(enc, labels=enc)
    return float(torch.exp(out.loss.float()))


def eval_model(path: Path) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    m.cuda().eval()
    ppl = perplexity(m, tok, SAMPLE)
    del m; gc.collect(); torch.cuda.empty_cache()
    return ppl


def param_savings(rank_k: int, ffn_in: int, ffn_out: int) -> dict:
    """Estimate parameter savings vs full model."""
    # Per-layer param counts
    head_dim = HIDDEN  # for Qwen, Q/K/V each hidden x hidden
    # Attn Q/K/V: 3 * hidden^2
    attn_full = 3 * HIDDEN * HIDDEN
    attn_saved = attn_full - 3 * HIDDEN * rank_k  # row+col factoring: 2*k*d per matrix
    # Actually GRC uses shared basis: each Q/K/V stored as k*d + k*d (basis + coeffs)
    # More precisely: per layer, Wq ≈ (Wq @ P^T) @ P stored as (Wq@P^T): d*k + P: k*d
    # So per matrix: 2*k*d. For 3 matrices: 6*k*d
    attn_compressed = 6 * rank_k * HIDDEN

    # FFN gate/up/down: each weights shape
    gate_full = HIDDEN * INTER  # gate_proj: 896 x 4864
    up_full = HIDDEN * INTER    # up_proj
    down_full = HIDDEN * INTER  # down_proj: but transposed dimensions: 896 x 4864

    ffn_compressed = 0
    if ffn_in > 0:
        # gate + up: each stored as rank-r factorization: d*r_in + r_in*inter
        ffn_compressed += 2 * (HIDDEN * ffn_in + ffn_in * INTER)
    else:
        ffn_compressed += gate_full + up_full
    if ffn_out > 0:
        ffn_compressed += (HIDDEN * ffn_out + ffn_out * INTER)
    else:
        ffn_compressed += down_full

    ffn_full = 3 * HIDDEN * INTER
    attn_savings = attn_full - attn_compressed
    ffn_savings = ffn_full - ffn_compressed
    total_savings = attn_savings + ffn_savings
    total_full = attn_full + ffn_full  # per layer

    return {
        "attn_full_per_layer": attn_full,
        "attn_compressed_per_layer": attn_compressed,
        "ffn_full_per_layer": ffn_full,
        "ffn_compressed_per_layer": ffn_compressed,
        "saved_per_layer": total_savings,
        "saved_total": total_savings * LAYERS,
        "saved_fraction_attn_ffn": total_savings / total_full if total_full > 0 else 0,
        "saved_M_params": total_savings * LAYERS / 1e6,
    }


def run_config(tag: str, rank_k: int, ffn_in: int, ffn_out: int,
               corpus: Path, results: list) -> None:
    """Run one config: distill with given ranks, measure PPL, record savings."""
    print(f"\n{'='*60}")
    print(f"  {tag}: rank_k={rank_k} ffn_in={ffn_in} ffn_out={ffn_out}")
    print(f"{'='*60}", flush=True)
    out_dir = Path(f"outputs/_joint_{tag}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.time()
    try:
        report = distill_hf_model(
            MODEL, str(out_dir),
            rank_k=rank_k, sink_T=8,
            lora_rank=32, lora_alpha=64.0,
            steps=600, batch_size=1, seq_len=256,
            learning_rate=1e-4,
            corpus_path=str(corpus),
            layers=[], device="cuda", dtype="float32",
            loss_type="kl", kl_temperature=4.0,
            ffn_rank_in=ffn_in, ffn_rank_out=ffn_out,
        )
    except Exception as e:
        results.append({"tag": tag, "rank_k": rank_k, "ffn_in": ffn_in,
                        "ffn_out": ffn_out, "error": str(e)})
        print(f"  FAILED: {e}")
        return
    secs = time.time() - t0
    ppl = eval_model(out_dir)
    savings = param_savings(rank_k, ffn_in, ffn_out)
    entry = {
        "tag": tag, "rank_k": rank_k, "ffn_in": ffn_in, "ffn_out": ffn_out,
        "ppl": ppl, "distill_seconds": secs,
        **savings,
    }
    results.append(entry)
    print(f"  PPL={ppl:.3f}  saved={savings['saved_M_params']:.1f}M params "
          f"({savings['saved_fraction_attn_ffn']*100:.0f}% of attn+FFN)  "
          f"time={secs:.0f}s")
    OUT.write_text(json.dumps(results, indent=2))


def main():
    corpus = Path("outputs/_joint_pareto_corpus.txt")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(SAMPLE * 4, encoding="utf-8")

    # Baseline
    print("=== Baseline ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ppl_base = perplexity(base_model, tok, SAMPLE)
    print(f"  Baseline PPL = {ppl_base:.3f}")
    del base_model; gc.collect(); torch.cuda.empty_cache()

    results = []

    # Configs to test: [tag, rank_k, ffn_in, ffn_out]
    # Strategy:
    #   A. Lower attn rank with no FFN compression
    #   B. Add FFN down (known working at r=256)
    #   C. Add FFN gate/up (NEW — never tested with distill)
    #   D. Combined: lower attn + FFN down + FFN gate/up
    configs = [
        # --- Attn-only baselines at lower ranks ---
        ("attn_k512",      512, 0, 0),
        ("attn_k448",      448, 0, 0),
        # --- Attn k=640 + FFN down (Pareto from r4) ---
        ("k640_dn384",     640, 0, 384),
        ("k640_dn256",     640, 0, 256),
        ("k640_dn192",     640, 0, 192),
        # --- Attn k=512 + FFN down ---
        ("k512_dn384",     512, 0, 384),
        ("k512_dn256",     512, 0, 256),
        # --- Attn + FFN gate/up (NEW) ---
        ("k640_in512",     640, 512, 0),
        ("k640_in384",     640, 384, 0),
        # --- Combined: attn + gate/up + down ---
        ("k640_in512_dn384", 640, 512, 384),
        ("k640_in384_dn256", 640, 384, 256),
        ("k512_in384_dn256", 512, 384, 256),
    ]

    for tag, rk, fi, fo in configs:
        run_config(tag, rk, fi, fo, corpus, results)

    print(f"\n{'='*60}")
    print(f"  PARETO FRONTIER  (baseline PPL = {ppl_base:.3f})")
    print(f"{'='*60}")
    print(f"  {'Config':<22s} {'PPL':>8s} {'saved_M':>8s} {'%attn+FFN':>9s} {'ΔPPL':>8s}")
    for r in sorted(results, key=lambda x: x.get("ppl", 1e9)):
        if "error" in r:
            print(f"  {r['tag']:<22s}  ERROR: {r['error']}")
        else:
            print(f"  {r['tag']:<22s} {r['ppl']:8.3f} {r['saved_M_params']:8.1f} "
                  f"{r['saved_fraction_attn_ffn']*100:8.1f}% {r['ppl']/ppl_base:8.3f}×")

    OUT.write_text(json.dumps({
        "model": MODEL, "baseline_ppl": ppl_base,
        "hidden": HIDDEN, "intermediate": INTER, "layers": LAYERS,
        "distill_steps": 600, "lora_rank": 32,
        "configs": results,
    }, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
