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

"""Round-13 follow-up: FFN-only A/B at r=1024 (matches round-12 ladder rank).

Reuses the cached activation norms (none — fast to recollect) and runs both
vanilla and aware factoring at r=1024 only, then merges into
benchmarks/activation_aware_ffn_only.json.
"""
from __future__ import annotations
import gc, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.activation import collect_ffn_input_norms
from hyperretro.hf.factored import (
    factor_ffn_state_dict, save_factored_checkpoint,
    load_factored_hf_model,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
RANK = 1024
CALIB_PATH = Path("data/wikitext2_train_5k.txt")
OUT = Path("benchmarks/activation_aware_ffn_only.json")
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


def make(out_dir: Path, *, col_norms) -> int:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    sd = m.state_dict()
    entries = factor_ffn_state_dict(
        sd, max_rank=RANK, rel_tol=0.0,
        activation_col_norms=col_norms,
    )
    save_factored_checkpoint(
        sd, attn_entries=[], ffn_entries=entries,
        out_dir=out_dir, hf_config=m.config, dtype="bfloat16", tokenizer=tok,
    )
    del m; gc.collect()
    return len(entries)


def eval_factored(p: Path) -> float:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m, _ = load_factored_hf_model(p, dtype="float32")
    m.to(DEVICE); val = ppl_of(m, tok)
    del m; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return val


def main() -> None:
    t0 = time.time()
    print(f"=== Round 13 r=1024 follow-up (device={DEVICE}) ===")

    print("[norms] collecting...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
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

    rows = []
    for tag, col in (("vanilla", None), ("aware", norms)):
        out_dir = Path(f"outputs/_ffn_only_{tag}_r{RANK}")
        print(f"[{tag} r={RANK}] factoring...")
        n = make(out_dir, col_norms=col)
        p = eval_factored(out_dir)
        sz = disk_mb(out_dir)
        print(f"  PPL={p:.4f}  on-disk={sz:.1f} MB  ffn={n}")
        rows.append({"tag": tag, "rank": RANK, "ppl": p,
                     "ppl_x": p / 2.3332, "on_disk_mb": sz})

    # Merge into existing JSON
    existing = json.loads(OUT.read_text())
    existing["rows"].extend(rows)
    by_rank: dict[int, dict] = {}
    for r in existing["rows"]:
        by_rank.setdefault(r["rank"], {})[r["tag"]] = r
    existing["summary"] = []
    for r in sorted(by_rank):
        if "vanilla" not in by_rank[r] or "aware" not in by_rank[r]: continue
        pv = by_rank[r]["vanilla"]["ppl"]; pa = by_rank[r]["aware"]["ppl"]
        existing["summary"].append({
            "rank": r, "vanilla_ppl": pv, "aware_ppl": pa,
            "aware_recovery_pct": 100.0*(pv-pa)/pv if pv > 0 else 0.0,
            "on_disk_mb": by_rank[r]["aware"]["on_disk_mb"],
        })
    existing["ffn_ranks"] = sorted(by_rank)
    existing["wall_seconds_r1024"] = time.time() - t0
    OUT.write_text(json.dumps(existing, indent=2))
    for s in existing["summary"]:
        print(f"[rank={s['rank']}] vanilla={s['vanilla_ppl']:.4f}  "
              f"aware={s['aware_ppl']:.4f}  recovery={s['aware_recovery_pct']:+.2f}%")
    print(f"[saved] {OUT}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
