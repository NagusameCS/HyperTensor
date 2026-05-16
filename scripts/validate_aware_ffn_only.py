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

"""Round-13 focused FFN-only A/B: vanilla vs activation-aware SVD.

Isolates the activation-aware signal by skipping attention compression and
factoring FFN at a tighter rank where truncation actually bites. This is
the clean test: same rank, same matrices, only the SVD weighting differs.

Outputs benchmarks/activation_aware_ffn_only.json.
"""
from __future__ import annotations
import gc, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.activation import collect_ffn_input_norms
from hyperretro.hf.factored import (
    factor_ffn_state_dict, save_factored_checkpoint,
    load_factored_hf_model,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
FFN_RANKS = [384, 768]  # below intermediate=8960 minor dim 1536
CALIB_PATH = Path("data/wikitext2_train_5k.txt")
OUT = Path("benchmarks/activation_aware_ffn_only.json")
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


def disk_mb(p: Path) -> float:
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1024 / 1024


@torch.inference_mode()
def ppl_of(m, tok) -> float:
    m.eval()
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids.to(m.device)
    return float(math.exp(m(ids, labels=ids).loss.float()))


def make_factored_ffn_only(out_dir: Path, *, rank: int,
                            col_norms: dict | None) -> dict:
    """Load fresh dense model, factor FFN only (no attn), save factored."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.float32
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    sd = m.state_dict()
    # Tight rel_tol forces use of the rank cap; min rank effectively == max_rank
    entries = factor_ffn_state_dict(
        sd, max_rank=rank, rel_tol=0.0,  # rel_tol=0 ⇒ always max_rank
        activation_col_norms=col_norms,
    )
    save_factored_checkpoint(
        sd, attn_entries=[], ffn_entries=entries,
        out_dir=out_dir, hf_config=m.config, dtype="bfloat16", tokenizer=tok,
    )
    del m; gc.collect()
    return {"ffn_factored": len(entries), "rank": rank}


def eval_factored(p: Path) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m, _info = load_factored_hf_model(p, dtype="float32")
    m.to(DEVICE); val = ppl_of(m, tok)
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return val


def collect_norms() -> dict:
    """One-time activation collection on the fresh dense model."""
    print(f"[norms] collecting on device={DEVICE} from {CALIB_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(DEVICE)
    norms = collect_ffn_input_norms(
        m, tok, corpus_path=str(CALIB_PATH),
        n_batches=4 if DEVICE == "cpu" else 16,
        seq_len=256 if DEVICE == "cpu" else 512,
        device=DEVICE,
    )
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    print(f"[norms] got {len(norms)} keys")
    return norms


def main() -> None:
    t0 = time.time()
    print(f"=== Round 13 (FFN-only A/B) on {MODEL_ID} (device={DEVICE}) ===")

    # Baseline PPL once
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(DEVICE)
    p0 = ppl_of(m, tok)
    print(f"[baseline] PPL={p0:.4f}")
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    norms = collect_norms()

    rows = []
    for r in FFN_RANKS:
        for tag, col_norms in (("vanilla", None), ("aware", norms)):
            out_dir = Path(f"outputs/_ffn_only_{tag}_r{r}")
            print(f"[{tag} r={r}] factoring...")
            info = make_factored_ffn_only(out_dir, rank=r, col_norms=col_norms)
            p = eval_factored(out_dir)
            sz = disk_mb(out_dir)
            print(f"  PPL={p:.4f}  on-disk={sz:.1f} MB  ffn={info['ffn_factored']}")
            rows.append({"tag": tag, "rank": r, "ppl": p,
                         "ppl_x": p / p0, "on_disk_mb": sz})

    # Pair up
    summary = []
    by_rank: dict[int, dict] = {}
    for row in rows:
        by_rank.setdefault(row["rank"], {})[row["tag"]] = row
    for r, pair in sorted(by_rank.items()):
        pv = pair["vanilla"]["ppl"]; pa = pair["aware"]["ppl"]
        delta_pct = 100.0 * (pv - pa) / pv if pv > 0 else 0.0
        summary.append({"rank": r, "vanilla_ppl": pv, "aware_ppl": pa,
                        "aware_recovery_pct": delta_pct,
                        "on_disk_mb": pair["aware"]["on_disk_mb"]})
        print(f"[rank={r}] vanilla={pv:.4f}  aware={pa:.4f}  "
              f"recovery={delta_pct:+.2f}%")

    OUT.write_text(json.dumps({
        "model": MODEL_ID, "baseline_ppl": p0,
        "ffn_ranks": FFN_RANKS, "device": DEVICE,
        "rows": rows, "summary": summary,
        "wall_seconds": time.time() - t0,
    }, indent=2))
    print(f"\n[saved] {OUT}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
