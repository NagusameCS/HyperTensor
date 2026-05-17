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

"""Find optimal HyperRetro config: honest eval on held-out text.

Key insight from joint Pareto sweep:
  - GRC attn is params-neutral at k=d/2=448 (6*k*d = 3*d^2)
  - FFN gate/up compression + distillation is the most efficient lever
  - This script tests k448 + ffn_in=512 (predicted sweet spot)
"""
import sys; sys.path.insert(0, '.')
import shutil, time, gc, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyperretro.hf.distill import distill_hf_model

MODEL = "Qwen/Qwen2.5-0.5B"
HIDDEN, INTER, LAYERS = 896, 4864, 24

# Held-out eval: ML history (ZERO overlap with calib)
EVAL_TEXT = (
    "The history of machine learning dates back to the 1950s when Arthur Samuel "
    "developed a checkers playing program The term artificial intelligence was "
    "coined at the Dartmouth Conference in 1956 Early neural network research "
    "was pioneered by Frank Rosenblatt who created the perceptron in 1958 "
    "The field experienced several AI winters when funding and interest declined "
    "due to overhyped expectations not being met Deep learning emerged in the "
    "2010s with breakthroughs in image recognition speech processing and "
    "natural language understanding Transformers introduced by Vaswani et al "
    "in 2017 revolutionized NLP by replacing recurrent architectures with "
    "self-attention mechanisms Large language models like GPT BERT and their "
    "successors have demonstrated remarkable capabilities in text generation "
    "reasoning and code completion These models are typically trained on vast "
    "corpora of internet text and fine tuned for specific downstream tasks"
) * 3

# Calibration: actor bio (ZERO overlap with eval)
CALIB_TEXT = (
    "Robert Boulter is an English film television and theatre actor He had a "
    "guest starring role on the television series The Bill in 2000 This was "
    "followed by a starring role in the play Herons written by Simon Stephens "
    "which was performed in 2001 at the Royal Court Theatre He had a guest role "
    "in the television series Judge John Deed in 2002 In 2004 Boulter landed a "
    "role as Craig in the episode Teddy Story of the television series The Long "
    "Firm he starred alongside actors Mark Strong and Derek Jacobi He was cast "
    "in the 2005 theatre productions of the Philip Ridley play Mercury Fur "
    "which was performed at the Drum Theatre in Plymouth and the Menier "
    "Chocolate Factory in London He was directed by John Tiffany and starred "
    "alongside Ben Whishaw Shane Zaza Harry Kent Fraser Ayres Sophie Stanton "
    "and Dominic Hall"
) * 4


def compute_savings(rk, fi, fo):
    """Params saved (millions) vs full attn+FFN."""
    attn_full = 3 * HIDDEN * HIDDEN
    attn_comp = 6 * rk * HIDDEN
    gate_up_full = 2 * HIDDEN * INTER
    down_full = HIDDEN * INTER
    ffn_comp = 0
    ffn_comp += 2 * (HIDDEN * fi + fi * INTER) if fi > 0 else gate_up_full
    ffn_comp += (HIDDEN * fo + fo * INTER) if fo > 0 else down_full
    saved = max(0, (attn_full + gate_up_full + down_full) - (attn_comp + ffn_comp))
    return saved * LAYERS / 1e6


def main():
    corpus = Path("outputs/_optimal_corpus.txt")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(CALIB_TEXT, encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(MODEL)

    # ----- Baseline -----
    print("Loading baseline...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32
    ).cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        ppl_base = float(torch.exp(base(enc, labels=enc).loss.float()))
    print(f"Baseline PPL (held-out) = {ppl_base:.3f}")
    del base; gc.collect(); torch.cuda.empty_cache()

    # ----- Re-eval existing models on held-out text -----
    existing = {
        "attn_k448":        ("outputs/_joint_attn_k448",        448, 0,   0),
        "attn_k512":        ("outputs/_joint_attn_k512",        512, 0,   0),
        "k640_in512":       ("outputs/_joint_k640_in512",       640, 512, 0),
        "k640_in384":       ("outputs/_joint_k640_in384",       640, 384, 0),
        "k640_in512_dn384": ("outputs/_joint_k640_in512_dn384", 640, 512, 384),
    }
    print("\n--- Existing models (held-out eval) ---")
    results = {}
    for tag, (path, rk, fi, fo) in existing.items():
        if not Path(path).exists():
            print(f"  {tag}: NOT FOUND")
            continue
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32
        ).cuda().eval()
        with torch.inference_mode():
            ppl = float(torch.exp(m(enc, labels=enc).loss.float()))
        sv = compute_savings(rk, fi, fo)
        results[tag] = {"ppl": ppl, "savings_M": sv, "rank_k": rk,
                        "ffn_in": fi, "ffn_out": fo}
        print(f"  {tag:<22s} PPL={ppl:.3f} ({ppl/ppl_base:.3f}×)  "
              f"save={sv:.1f}M ({sv/494*100:.1f}% of model)")
        del m; gc.collect(); torch.cuda.empty_cache()

    # ----- Build k448_in512 (predicted sweet spot) -----
    tag = "k448_in512"
    rk, fi, fo = 448, 512, 0
    out = Path(f"outputs/_optimal_{tag}")
    if out.exists():
        shutil.rmtree(out)
    print(f"\n--- Building {tag} (predicted sweet spot) ---")
    t0 = time.time()
    distill_hf_model(
        MODEL, str(out),
        rank_k=rk, sink_T=8,
        lora_rank=32, lora_alpha=64.0,
        steps=600, batch_size=1, seq_len=256,
        learning_rate=1e-4,
        corpus_path=str(corpus),
        layers=[], device="cuda", dtype="float32",
        loss_type="kl", kl_temperature=4.0,
        ffn_rank_in=fi, ffn_rank_out=fo,
    )
    secs = time.time() - t0

    m = AutoModelForCausalLM.from_pretrained(
        out, torch_dtype=torch.float32
    ).cuda().eval()
    with torch.inference_mode():
        ppl = float(torch.exp(m(enc, labels=enc).loss.float()))
    sv = compute_savings(rk, fi, fo)
    results[tag] = {"ppl": ppl, "savings_M": sv, "rank_k": rk,
                    "ffn_in": fi, "ffn_out": fo, "time_s": secs}
    print(f"  {tag:<22s} PPL={ppl:.3f} ({ppl/ppl_base:.3f}×)  "
          f"save={sv:.1f}M ({sv/494*100:.1f}% of model)  time={secs:.0f}s")
    del m; gc.collect(); torch.cuda.empty_cache()

    # ----- Summary -----
    print(f"\n{'='*70}")
    print(f"  OPTIMAL HYPERRETRO — Pareto Frontier")
    print(f"  Baseline PPL = {ppl_base:.3f} (held-out)")
    print(f"{'='*70}")
    print(f"  {'Config':<22s} {'PPL':>7s} {'×base':>6s} {'Saved':>7s} {'%model':>7s}")
    print(f"  {'-'*50}")
    for tag, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"  {tag:<22s} {r['ppl']:7.3f} {r['ppl']/ppl_base:6.3f} "
              f"{r['savings_M']:6.1f}M {r['savings_M']/494*100:6.1f}%")

    # Identify the best ≤1.05× config
    best = min((r for r in results.values() if r["ppl"]/ppl_base <= 1.05),
               key=lambda r: -r["savings_M"], default=None)
    if best:
        print(f"\n   OPTIMAL (≤1.05× PPL): save {best['savings_M']:.1f}M "
              f"({best['savings_M']/494*100:.1f}% of model) "
              f"at {best['ppl']/ppl_base:.3f}× PPL")


if __name__ == "__main__":
    main()
