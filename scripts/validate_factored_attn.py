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

"""Validate attack #2: factored on-disk storage for HyperRetro attn.

Pipeline:

  1. Compress Qwen2.5-1.5B with attn k=640 into a dense d×d checkpoint
     (the v0.2.1 path that produced 0% on-disk shrink).
  2. *Re-factor* that checkpoint into (A, B_*) safetensors form using
     ``factor_attn_state_dict``.
  3. Load with ``load_factored_hf_model`` and measure:
       - on-disk MB (compare to fp16 baseline + dense GRC checkpoint)
       - PPL on the held-out ML paragraph

Success criteria: on-disk attn cost drops from d² to k·(d_in+3·d_out)/3
per layer; PPL within a few percent of dense GRC (no extra approximation
since we're SVD-refactoring an already rank-≤k matrix).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperretro.hf.compress import _group_attn_by_layer
from hyperretro.hf.factored import (
    build_manifest,
    factor_attn_state_dict,
    factor_ffn_state_dict,
    load_factored_hf_model,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
GRC_DIR = Path("outputs/_15B_k640_in1024_bf16")  # dense GRC checkpoint
FACTORED_DIR = Path("outputs/_15B_k640_factored_bf16")
RANK_K = 640  # must match the compression rank used to build GRC_DIR

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
def ppl_native(model, dtype) -> float:
    model = model.to("cuda").eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    loss = model(enc, labels=enc).loss.float()
    return float(torch.exp(loss))


@torch.inference_mode()
def ppl_from_dir(model_dir, dtype=torch.bfloat16) -> float:
    m = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype
    ).cuda().eval()
    enc = tok(EVAL_TEXT, return_tensors="pt").input_ids.cuda()
    p = float(torch.exp(m(enc, labels=enc).loss.float()))
    del m
    _free()
    return p


def refactor_in_place():
    """Read GRC_DIR weights, refactor attn, write FACTORED_DIR."""
    FACTORED_DIR.mkdir(parents=True, exist_ok=True)

    # Copy non-weight artifacts
    import shutil
    for f in GRC_DIR.iterdir():
        if f.suffix in (".json", ".jinja", ".txt"):
            shutil.copy2(f, FACTORED_DIR / f.name)

    # Load all shards into one dict
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted(GRC_DIR.glob("*.safetensors")):
        sd.update(load_file(str(shard)))

    layer_keys = _group_attn_by_layer(sd)
    sd, entries = factor_attn_state_dict(
        sd, rank=RANK_K, layer_keys=layer_keys, rel_tol=1e-3,
    )
    ffn_entries = factor_ffn_state_dict(sd, max_rank=1536, rel_tol=1e-3)
    manifest = build_manifest(entries, shared=True, ffn_entries=ffn_entries)
    (FACTORED_DIR / "hyperretro_factored.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"      attn-factored: {len(entries)} layers, "
          f"ffn-factored: {len(ffn_entries)} linears")

    # Save as single shard for simplicity
    save_file(sd, str(FACTORED_DIR / "model.safetensors"))
    return manifest


def run():
    if not GRC_DIR.exists():
        print(f"[fatal] missing {GRC_DIR} — run round-8 distill first")
        return

    t0 = time.time()
    print("[1/4] re-factoring attn weights ...")
    manifest = refactor_in_place()
    print(f"      {len(manifest['layers'])} layers factored, rank={RANK_K}")

    print("[2/4] measuring on-disk sizes ...")
    dense_mb = disk_mb(GRC_DIR)
    factored_mb = disk_mb(FACTORED_DIR)
    # baseline approx: a vanilla bf16 1.5B is ~2955 MB on disk after save
    print(f"      dense GRC   : {dense_mb:.1f} MB")
    print(f"      factored GRC: {factored_mb:.1f} MB  "
          f"({dense_mb/max(factored_mb,1e-9):.3f}× of dense)")

    print("[3/4] dense GRC PPL on held-out paragraph ...")
    p_dense = ppl_from_dir(GRC_DIR, dtype=torch.bfloat16)
    print(f"      PPL_dense = {p_dense:.4f}")

    print("[4/4] factored GRC PPL ...")
    model, info = load_factored_hf_model(FACTORED_DIR, dtype="bfloat16")
    print(f"      patched {info['patched_linears']} Linears "
          f"(missing dense keys: {info['missing_dense']}, "
          f"unexpected: {info['unexpected']})")
    p_fact = ppl_native(model, dtype=torch.bfloat16)
    print(f"      PPL_factored = {p_fact:.4f}")
    del model
    _free()

    print(f"\n[wall] {time.time()-t0:.1f}s")
    print(f"\n{'config':<20} {'PPL':>10} {'on-disk MB':>14} {'shrink x':>10}")
    rows = [
        ("dense_grc_bf16",   p_dense, dense_mb),
        ("factored_grc_bf16", p_fact, factored_mb),
    ]
    for name, ppl, mb in rows:
        print(f"{name:<20} {ppl:>10.4f} {mb:>14.1f} "
              f"{dense_mb/max(mb,1e-9):>10.3f}")

    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/factored_attn_ladder.json").write_text(json.dumps([
        {"config": n, "ppl": p, "on_disk_mb": m,
         "shrink_x_vs_dense_grc": dense_mb / max(m, 1e-9)}
        for n, p, m in rows
    ], indent=2))
    print("\n[saved] benchmarks/factored_attn_ladder.json")


if __name__ == "__main__":
    run()
