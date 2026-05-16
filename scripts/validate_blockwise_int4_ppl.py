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

"""Measure PPL of block-wise int4 factored checkpoint.

Compares against old per-row int4 PPL (11.84 at r=1024).
Uses the aware_r1024 checkpoint from round 13.

Single config, ~25 min CPU.
"""

import json, shutil, sys, time
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from hyperretro.hf.factor_int4 import (
    save_int4_factored_checkpoint,
    dequant_int4_checkpoint,
)
from hyperretro.hf.factored import load_factored_hf_model
from hyperretro.hf.activation import collect_ffn_input_norms

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SRC = Path("outputs/_ffn_only_aware_r1024")
OUT = _ROOT / "benchmarks" / "factor_int4_blockwise_ppl.json"

EVAL_TEXT = (
    "Machine learning has transformed the way we build software. "
    "Instead of writing explicit rules, we collect examples and let "
    "the optimization process discover patterns. This paradigm shift "
    "began with simple linear models and has now evolved into massive "
    "neural networks with hundreds of billions of parameters. "
    "The key insight is that gradient descent on a sufficiently large "
    "dataset will find structure that human engineers would never "
    "think to encode. However, this power comes at a cost: the models "
    "are opaque, their failures are unpredictable, and their "
    "computational requirements are staggering. "
) * 3

def ppl(model, tokenizer, text, max_tokens=256):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
    return float(torch.exp(loss))

def disk_mb(d):
    total = 0.0
    for sf in d.glob("*.safetensors"):
        total += sf.stat().st_size
    return total / 1e6

def main():
    t0 = time.time()

    manifest = json.loads((SRC / "hyperretro_factored.json").read_text())
    orig_sd = load_file(str(SRC / "model.safetensors"))
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect activation norms for AWQ
    print("[norms] collecting activation column norms ...")
    import torch as _t
    model_dense = _t.nn.Module()  # placeholder
    try:
        from transformers import AutoModelForCausalLM
        model_dense = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=_t.float32,
        ).cpu()
        act_norms = collect_ffn_input_norms(
            model_dense, tokenizer,
            corpus_path=str(_ROOT / "data" / "wikitext2_train_5k.txt"),
            n_batches=4, seq_len=256, device="cpu",
        )
        del model_dense
        print(f"[norms] collected {len(act_norms)} keys")
    except Exception as e:
        print(f"[norms] FAILED: {e}")
        act_norms = None

    results = []

    for label, block_size, use_awq in [
        ("blockwise_128", 128, False),
        ("blockwise_128_awq", 128, True),
        ("blockwise_64", 64, False),
    ]:
        print(f"\n{'='*50}")
        print(f"[{label}] block_size={block_size}, awq={use_awq}")
        print(f"{'='*50}")

        int4_dir = Path(f"outputs/_ppl_test_{label}")
        if int4_dir.exists():
            shutil.rmtree(int4_dir)

        t_save = time.time()
        report = save_int4_factored_checkpoint(
            orig_sd, [], manifest.get("ffn", []),
            out_dir=int4_dir, hf_config=cfg, n_bits=4,
            block_size=block_size,
            quantize_non_factored=False,
            activation_norms=act_norms if use_awq else None,
        )
        int4_mb = report["on_disk_mb"]
        print(f"  Int4: {int4_mb:.1f} MB ({time.time()-t_save:.0f}s)")

        dq_dir = Path(f"outputs/_ppl_test_{label}_dq")
        if dq_dir.exists():
            shutil.rmtree(dq_dir)

        t_dq = time.time()
        dequant_int4_checkpoint(int4_dir, dq_dir)
        print(f"  Dequant: {disk_mb(dq_dir):.0f} MB ({time.time()-t_dq:.0f}s)")

        t_load = time.time()
        model, info = load_factored_hf_model(str(dq_dir), dtype="float16")
        model = model.cpu()
        print(f"  Loaded: {info['patched_linears']} factored linears ({time.time()-t_load:.0f}s)")

        t_ppl = time.time()
        val = ppl(model, tokenizer, EVAL_TEXT)
        print(f"  PPL={val:.4f} ({time.time()-t_ppl:.0f}s)")

        results.append({
            "label": label,
            "block_size": block_size,
            "awq": use_awq,
            "ppl": round(val, 4),
            "ppl_vs_fp16_baseline": round(val / 2.3332, 2),
            "ppl_vs_fp16_factored": round(val / 4.5262, 3),
            "int4_mb": round(int4_mb, 1),
            "dequant_mb": round(disk_mb(dq_dir), 1),
        })

        del model
        shutil.rmtree(int4_dir, ignore_errors=True)
        shutil.rmtree(dq_dir, ignore_errors=True)

    # Add reference points
    results.append({"label": "REF_fp16_factored", "ppl": 4.5262, "ppl_vs_fp16_baseline": 1.94})
    results.append({"label": "REF_per_row_int4", "ppl": 11.8413, "ppl_vs_fp16_baseline": 5.08})
    results.append({"label": "REF_fp16_baseline", "ppl": 2.3332, "ppl_vs_fp16_baseline": 1.0})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {OUT} ({time.time()-t0:.0f}s)")
    print("\nSUMMARY:")
    for r in results:
        print(f"  {r['label']:30s}  PPL={r.get('ppl',0):.4f}  "
              f"vs_fp16={r.get('ppl_vs_fp16_baseline','?'):.2f}x")

    return 0

if __name__ == "__main__":
    sys.exit(main())
