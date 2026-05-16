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

"""Critical test: does diverse (WikiText-2) calibration fix FFN overfitting?

Tests:
  A. attn k=448 + WikiText distill  → generalize?
  B. attn k=448 + FFN gate/up r=512 + WikiText distill → generalize?
  C. attn k=448 + FFN gate/up r=384 + WikiText distill → generalize?

Held-out eval on ML history text (zero overlap with WikiText).
"""
import sys; sys.path.insert(0, '.')
import shutil, time, gc, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyperretro.hf.distill import distill_hf_model

MODEL = "Qwen/Qwen2.5-0.5B"
HIDDEN, INTER, LAYERS = 896, 4864, 24
CORPUS = "data/wikitext2_train.txt"

# Held-out eval (ML history — zero overlap with WikiText actor bios)
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

tok = AutoTokenizer.from_pretrained(MODEL)

@torch.inference_mode()
def ppl_heldout(model_path):
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m; gc.collect(); torch.cuda.empty_cache()
    return p


def savings(rk, fi, fo):
    attn_f = 3*HIDDEN*HIDDEN; attn_c = 6*rk*HIDDEN
    gate_up_f = 2*HIDDEN*INTER; down_f = HIDDEN*INTER
    ffn_c = 0
    ffn_c += 2*(HIDDEN*fi+fi*INTER) if fi>0 else gate_up_f
    ffn_c += (HIDDEN*fo+fo*INTER) if fo>0 else down_f
    saved = max(0, (attn_f+gate_up_f+down_f)-(attn_c+ffn_c))
    return saved * LAYERS / 1e6


# Baseline
print("=== BASELINE ===")
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32
).cuda().eval()
enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
with torch.inference_mode():
    ppl_base = float(torch.exp(base(enc, labels=enc).loss.float()))
print(f"Baseline held-out PPL = {ppl_base:.3f}")
del base; gc.collect(); torch.cuda.empty_cache()

# Also measure existing attn-only no-distill at k=448 for reference
print("\n=== REFERENCE: attn k=448 NO distill ===")
from hyperretro.hf.compress import compress_state_dict, CompressConfig
ref = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
sd = ref.state_dict()
compress_state_dict(sd, CompressConfig(rank_k=448, sink_T=8, layers=[]))
ref.load_state_dict(sd, strict=False)
with torch.inference_mode():
    ppl_ref = float(torch.exp(ref(enc, labels=enc).loss.float()))
print(f"attn_k448 NO distill: held-out PPL = {ppl_ref:.3f} ({ppl_ref/ppl_base:.3f}x)")
del ref; gc.collect(); torch.cuda.empty_cache()

# Test configs
configs = [
    ("k448_wikitext",     448, 0,   0),
    ("k448_in512_wiki",   448, 512, 0),
    ("k448_in384_wiki",   448, 384, 0),
]

results = []
for tag, rk, fi, fo in configs:
    print(f"\n=== {tag} (rk={rk} fi={fi} fo={fo}) ===")
    out = Path(f"outputs/_wiki_{tag}")
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    try:
        distill_hf_model(
            MODEL, str(out),
            rank_k=rk, sink_T=8,
            lora_rank=32, lora_alpha=64.0,
            steps=600, batch_size=1, seq_len=256,
            learning_rate=1e-4,
            corpus_path=CORPUS,
            layers=[], device="cuda", dtype="float32",
            loss_type="kl", kl_temperature=4.0,
            ffn_rank_in=fi, ffn_rank_out=fo,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({"tag": tag, "error": str(e)})
        continue
    secs = time.time() - t0
    ppl = ppl_heldout(out)
    sv = savings(rk, fi, fo)
    results.append({"tag": tag, "ppl": ppl, "savings_M": sv, "time_s": secs,
                    "rk": rk, "fi": fi, "fo": fo})
    print(f"  Held-out PPL={ppl:.3f} ({ppl/ppl_base:.3f}x)  "
          f"save={sv:.1f}M ({sv/494*100:.1f}% of model)  time={secs:.0f}s")

# Summary
print(f"\n{'='*65}")
print(f"  WIKITEXT-2 DISTILLATION — Generalization Test")
print(f"  Baseline (held-out)        = {ppl_base:.3f}")
print(f"  Attn k=448 NO distill      = {ppl_ref:.3f} ({ppl_ref/ppl_base:.3f}x)")
print(f"{'='*65}")
print(f"  {'Config':<22s} {'Held-out PPL':>12s} {'xbase':>6s} {'Saved':>7s}")
for r in sorted(results, key=lambda x: x.get("ppl", 1e9)):
    if "error" in r:
        print(f"  {r['tag']:<22s} ERROR: {r['error']}")
    else:
        print(f"  {r['tag']:<22s} {r['ppl']:12.3f} {r['ppl']/ppl_base:6.3f}x "
              f"{r['savings_M']:6.1f}M")

# Did it fix the overfitting?
best_ffn = min((r for r in results if r.get("fi", 0) > 0 and "error" not in r),
               key=lambda r: r["ppl"], default=None)
if best_ffn:
    ratio = best_ffn["ppl"] / ppl_base
    if ratio < 2.0:
        print(f"\n  *** DIVERSE CALIBRATION WORKS! Best FFN config: "
              f"{best_ffn['tag']} at {ratio:.3f}x PPL, "
              f"saving {best_ffn['savings_M']:.1f}M params ***")
    else:
        print(f"\n  *** FFN still overfits (best={ratio:.1f}x). "
              f"Need more distill budget or larger LoRA ***")
