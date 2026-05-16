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

"""Clean end-to-end benchmark: aware-factor → full-model int4 → PPL.

Uses the EXISTING aware_r1024 checkpoint (known PPL=4.53) as the source.
Quantizes ALL weights (factored FFN + dense attn + embeddings) with
block-wise int4, dequants back, loads, measures PPL.

This avoids re-running factoring (already validated) and focuses on
measuring the full-model int4 impact correctly.

Single run, ~15 min CPU.
Output: benchmarks/fullmodel_int4_ppl.json
"""

import json, shutil, sys, time
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from hyperretro.hf.factor_int4 import (
    save_int4_factored_checkpoint,
    dequant_int4_checkpoint,
    quantize_factored_state_dict,
    pack_int4_rows,
)
from hyperretro.hf.factored import load_factored_hf_model
from hyperretro.hf.activation import collect_ffn_input_norms
from hyperretro.hf.factor_quantize import quantize_blockwise_int4

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SRC = Path("outputs/_ffn_only_aware_r1024")
OUT = _ROOT / "benchmarks" / "fullmodel_int4_ppl.json"
CALIB = _ROOT / "data" / "wikitext2_train_5k.txt"

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


def ppl_eval(model, tokenizer, text, max_tokens=256):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return float(torch.exp(loss)) if loss is not None else float("inf")


def disk_mb(d):
    total = 0.0
    for sf in d.glob("*.safetensors"):
        total += sf.stat().st_size
    return total / 1e6


def main():
    t0 = time.time()
    results = []

    # Load reference
    manifest = json.loads((SRC / "hyperretro_factored.json").read_text())
    orig_sd = load_file(str(SRC / "model.safetensors"))
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orig_mb = disk_mb(SRC)
    print(f"Source checkpoint: {orig_mb:.1f} MB, {len(orig_sd)} tensors, "
          f"{len(manifest.get('ffn',[]))} FFN factored")

    # Reference: fp16 factored PPL
    print("\n[REF] Loading fp16 factored model ...")
    model_ref, info = load_factored_hf_model(str(SRC), dtype="float16")
    model_ref = model_ref.cpu()
    ppl_ref = ppl_eval(model_ref, tokenizer, EVAL_TEXT)
    print(f"  PPL={ppl_ref:.4f}  patched={info['patched_linears']}")
    del model_ref
    results.append({"label": "fp16_factored_ref", "ppl": round(ppl_ref, 4), "disk_mb": round(orig_mb, 1)})

    # Collect activation norms for AWQ
    print("\n[norms] Collecting for AWQ-aware int4 ...")
    try:
        from transformers import AutoModelForCausalLM
        model_dense = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32,
        ).cpu()
        act_norms = collect_ffn_input_norms(
            model_dense, tokenizer,
            corpus_path=str(CALIB), n_batches=4, seq_len=256, device="cpu",
        )
        del model_dense
        print(f"  Collected {len(act_norms)} keys")
    except Exception as e:
        print(f"  FAILED: {e}, continuing without AWQ")
        act_norms = None

    # Test configs
    configs = [
        ("block128", 128, False, False),
        ("block128_awq", 128, True, False),
        ("block128_full", 128, True, True),  # full-model = quantize non-factored too
    ]

    for label, blk, awq, full_model in configs:
        print(f"\n{'='*50}")
        print(f"[{label}] blk={blk}, awq={awq}, full_model={full_model}")
        print(f"{'='*50}")

        int4_dir = Path(f"outputs/_bench_full_{label}")
        if int4_dir.exists():
            shutil.rmtree(int4_dir)

        # Build manifest copy
        import copy
        man_copy = copy.deepcopy(manifest)

        # Quantize
        t_q = time.time()
        sd_q, man_q = quantize_factored_state_dict(
            dict(orig_sd), man_copy,
            n_bits=4, block_size=blk,
            quantize_non_factored=full_model,
            activation_norms=act_norms if awq else None,
        )

        # Write quantized safetensors
        out_sd = {}
        for k, v in sd_q.items():
            if hasattr(v, "cpu"):
                v = v.contiguous().cpu()
            if hasattr(v, "numpy"):
                pass
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(np.ascontiguousarray(v))
            if k.endswith(".q"):
                v = v.to(torch.uint8)
            elif k.endswith(".scales") or k.endswith(".awq_scales"):
                v = v.to(torch.float16)
            elif isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
                v = v.to(torch.float16)
            out_sd[k] = v

        int4_dir.mkdir(parents=True, exist_ok=True)
        save_file(out_sd, str(int4_dir / "model.safetensors"))
        (int4_dir / "hyperretro_factored.json").write_text(json.dumps(man_q, indent=2))
        # Copy HF config
        import shutil as sh
        for f in SRC.glob("config.json"):
            sh.copy(f, int4_dir / "config.json")
        for f in SRC.glob("tokenizer*"):
            sh.copy(f, int4_dir / f.name)
        for f in SRC.glob("*.jinja"):
            sh.copy(f, int4_dir / f.name)

        int4_mb = disk_mb(int4_dir)
        q_time = time.time() - t_q
        print(f"  Int4 saved: {int4_mb:.1f} MB ({q_time:.0f}s)  "
              f"shrink={orig_mb/int4_mb:.1f}x")

        # Dequant
        dq_dir = Path(f"outputs/_bench_full_{label}_dq")
        if dq_dir.exists():
            shutil.rmtree(dq_dir)

        t_dq = time.time()
        dequant_int4_checkpoint(int4_dir, dq_dir)
        dq_time = time.time() - t_dq
        dq_mb = disk_mb(dq_dir)
        print(f"  Dequant: {dq_mb:.0f} MB ({dq_time:.0f}s)")

        # Load and eval
        t_load = time.time()
        model, info = load_factored_hf_model(str(dq_dir), dtype="float16")
        model = model.cpu()
        load_time = time.time() - t_load

        ppl_val = ppl_eval(model, tokenizer, EVAL_TEXT)
        print(f"  PPL={ppl_val:.4f}  loaded={info['patched_linears']} linears ({load_time:.0f}s)")

        results.append({
            "label": label,
            "block_size": blk,
            "awq": awq,
            "full_model_quantize": full_model,
            "ppl": round(ppl_val, 4),
            "ppl_vs_ref": round(ppl_val / max(ppl_ref, 1e-12), 3),
            "ppl_vs_baseline": round(ppl_val / 2.3332, 2),
            "int4_mb": round(int4_mb, 1),
            "dequant_mb": round(dq_mb, 1),
            "shrink_vs_fp16_factored": round(orig_mb / int4_mb, 2),
            "shrink_vs_fp16_baseline": round(2955.4 / int4_mb, 2),
        })

        del model
        shutil.rmtree(int4_dir, ignore_errors=True)
        shutil.rmtree(dq_dir, ignore_errors=True)

    # Save results
    results.append({"label": "REF_fp16_baseline", "ppl": 2.3332})
    results.append({"label": "REF_nf4_industry", "ppl": 2.5318, "shrink": 2.71})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    wall = time.time() - t0

    print(f"\n[saved] {OUT} ({wall:.0f}s)")
    print("\n" + "=" * 60)
    print("FULL-MODEL INT4 RESULTS")
    print("=" * 60)
    for r in results:
        s = r.get("shrink_vs_fp16_baseline", 0)
        print(f"  {r['label']:25s}  PPL={r.get('ppl',0):.4f}  "
              f"shrink={s:.2f}x" if s else f"  {r['label']:25s}  PPL={r.get('ppl',0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
