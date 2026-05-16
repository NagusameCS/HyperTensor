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

"""Validate FFN compression on Qwen2.5-0.5B.

Sweeps several (rank_attn, rank_ffn_in, rank_ffn_out) configurations
and measures WikiText-2 perplexity.

Qwen2.5-0.5B dims: hidden=896, intermediate=4864, layers=24.

Param budget per layer:
  attn (Q,K,V,O): ~hidden^2 * 4 = ~3.2M
  ffn  (gate,up,down): hidden*intermediate*3 = ~13.1M
  ratio: FFN is ~4.1x attention -> ~80% of layer params.

So a 50% FFN-rank cut saves ~4x more params than a 50% attn cut.
"""
import sys; sys.path.insert(0, '.')
import json, time, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyperretro.hf.compress import compress_state_dict, CompressConfig
from hyperretro.hf.ffn_compress import (
    compress_ffn_state_dict, FFNCompressConfig,
)

MODEL = "Qwen/Qwen2.5-0.5B"
WIKITEXT_SAMPLE = """Robert Boulter is an English film, television and theatre actor. He had a guest-starring role on the television series The Bill in 2000. This was followed by a starring role in the play Herons written by Simon Stephens, which was performed in 2001 at the Royal Court Theatre. He had a guest role in the television series Judge John Deed in 2002. In 2004 Boulter landed a role as "Craig" in the episode "Teddy's Story" of the television series The Long Firm; he starred alongside actors Mark Strong and Derek Jacobi. He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur, which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London. He was directed by John Tiffany and starred alongside Ben Whishaw, Shane Zaza, Harry Kent, Fraser Ayres, Sophie Stanton and Dominic Hall.

In 2006, Boulter starred alongside Whishaw again in the play Citizenship written by Mark Ravenhill. He appeared on a 2006 episode of the television series, Doctors, followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke. How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham. Boulter starred in two films in 2008, Daylight Robbery by filmmaker Paris Leonti, and Donkey Punch directed by Olly Blackburn. In May 2008, Boulter made a guest appearance on a two-part episode arc of the television series Waking the Dead, followed by an appearance on the television series Survivors in November 2008. He had a recurring role in ten episodes of the television series Casualty in 2010, as "Kieron Fletcher". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti.

In 2000 Boulter had a guest-starring role on the television series The Bill; he portrayed "Scott Parry" in the episode, "In Safe Hands". Boulter starred as "Scott" in the play Herons written by Simon Stephens, which was performed in 2001 at the Royal Court Theatre. A review of Boulter's performance in The Independent on Sunday described him as "horribly menacing" in the role, and he received critical reviews in The Herald, and Evening Standard. He appeared in the television series Judge John Deed in 2002. In 2004 Boulter was cast in the role of "Craig" in the episode "Teddy's Story" of the television series The Long Firm; he starred alongside actors Mark Strong and Derek Jacobi. Boulter starred as "Darren", in the 2005 theatre productions of the Philip Ridley play Mercury Fur, which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London. He was directed by John Tiffany and starred alongside Ben Whishaw, Shane Zaza, Harry Kent, Fraser Ayres, Sophie Stanton and Dominic Hall. The Daily Telegraph theatre critic Charles Spencer described Boulter's performance as "horribly compelling".
"""


@torch.inference_mode()
def perplexity(model, tokenizer, text: str, stride: int = 512) -> float:
    enc = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    max_len = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 1024
    max_len = min(max_len, 1024)
    nlls = []
    for begin in range(0, enc.size(1), stride):
        end = min(begin + max_len, enc.size(1))
        ids = enc[:, begin:end]
        if ids.size(1) < 2:
            break
        out = model(ids, labels=ids)
        nlls.append(out.loss.float() * (ids.size(1) - 1))
        if end == enc.size(1):
            break
    total_loss = torch.stack(nlls).sum() / (enc.size(1) - 1)
    return float(torch.exp(total_loss))


def run_one(rank_attn: int, ffn_in: int, ffn_out: int, label: str) -> dict:
    print(f"\n=== {label} (attn_k={rank_attn} ffn_in={ffn_in} ffn_out={ffn_out}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

    sd = model.state_dict()
    t0 = time.time()
    attn_stats: dict = {}
    if rank_attn > 0:
        attn_stats = compress_state_dict(sd, CompressConfig(rank_k=rank_attn, sink_T=4))
    ffn_stats: dict = {}
    if ffn_in > 0 or ffn_out > 0:
        ffn_stats = compress_ffn_state_dict(
            sd, FFNCompressConfig(rank_in=ffn_in, rank_out=ffn_out)
        )
    model.load_state_dict(sd)
    compress_dt = time.time() - t0

    model = model.to("cuda").half()
    model.eval()
    t0 = time.time()
    ppl = perplexity(model, tok, WIKITEXT_SAMPLE)
    eval_dt = time.time() - t0

    # Average errors
    if attn_stats:
        attn_err = np.mean([s["frob_relerr_q"] for s in attn_stats.values()])
    else:
        attn_err = 0.0
    if ffn_stats:
        gate_errs = [s.get("frob_relerr_gate", 0.0) for s in ffn_stats.values() if "frob_relerr_gate" in s]
        down_errs = [s.get("frob_relerr_down", 0.0) for s in ffn_stats.values() if "frob_relerr_down" in s]
        ffn_gate_err = float(np.mean(gate_errs)) if gate_errs else 0.0
        ffn_down_err = float(np.mean(down_errs)) if down_errs else 0.0
    else:
        ffn_gate_err = ffn_down_err = 0.0

    result = {
        "label": label,
        "rank_attn": rank_attn,
        "ffn_in": ffn_in,
        "ffn_out": ffn_out,
        "ppl": ppl,
        "compress_sec": compress_dt,
        "eval_sec": eval_dt,
        "attn_relerr_mean": float(attn_err),
        "ffn_gate_relerr_mean": ffn_gate_err,
        "ffn_down_relerr_mean": ffn_down_err,
    }
    print(f"  PPL={ppl:.3f}  compress={compress_dt:.1f}s eval={eval_dt:.1f}s")
    print(f"  attn_err={attn_err:.3f}  ffn_gate_err={ffn_gate_err:.3f}  ffn_down_err={ffn_down_err:.3f}")

    # Free
    del model, sd
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    # Qwen2.5-0.5B: hidden=896, intermediate=4864
    # Note: For Qwen2.5-0.5B:
    #   gate/up shape [4864, 896] -> max useful SVD rank = 896
    #   down    shape [896, 4864] -> max useful SVD rank = 896
    # so all FFN ranks are over the hidden=896 axis.
    configs = [
        # (rank_attn, ffn_in, ffn_out, label)
        (0,   0,    0,    "baseline fp16"),
        (640, 0,    0,    "attn-only k=640 (prior best)"),
        # down_proj alone — meaningful ranks
        (0,   0,    768,  "down-only r=768 (86%)"),
        (0,   0,    640,  "down-only r=640 (71%)"),
        (0,   0,    512,  "down-only r=512 (57%)"),
        (0,   0,    448,  "down-only r=448 (50%)"),
        (0,   0,    256,  "down-only r=256 (29%)"),
        # gate/up at very mild ranks
        (0,   832,  0,    "gate/up r=832 (93%)"),
        (0,   768,  0,    "gate/up r=768 (86%)"),
        (0,   640,  0,    "gate/up r=640 (71%)"),
        # joint
        (640, 0,    640,  "joint attn=640 down=640"),
        (640, 768,  640,  "joint attn=640 gate/up=768 down=640"),
    ]
    results = []
    for rk, fi, fo, lbl in configs:
        try:
            results.append(run_one(rk, fi, fo, lbl))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"label": lbl, "error": str(e)})

    print("\n\nSummary")
    print(f"{'Config':<40s} {'PPL':>10s} {'Δ vs base':>10s} {'attn_err':>8s} {'ffn_g_err':>10s} {'ffn_d_err':>10s}")
    print("-" * 100)
    base = next((r for r in results if r.get("label") == "baseline fp16"), None)
    base_ppl = base["ppl"] if base and "ppl" in base else None
    for r in results:
        if "error" in r:
            print(f"{r['label']:<40s} FAILED: {r['error']}")
            continue
        delta = (r["ppl"] / base_ppl - 1.0) * 100 if base_ppl else 0.0
        print(f"{r['label']:<40s} {r['ppl']:10.3f} {delta:+9.2f}% "
              f"{r['attn_relerr_mean']:8.3f} {r['ffn_gate_relerr_mean']:10.3f} {r['ffn_down_relerr_mean']:10.3f}")

    out_path = "benchmarks/ffn_compress_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
