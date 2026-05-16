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

"""Measure actual PPL of factor+int4: quantize existing factored checkpoint
to int4, dequant back, load model, eval PPL.

This is the first hard measurement of the factor×int4 composition's
end-to-end PPL impact (no distill). Compares against the known fp16
factored PPL from round 13.

GPU not required — runs on CPU (~20 min per config).
"""

import json, shutil, sys, time
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
from safetensors.torch import load_file

from hyperretro.hf.factor_int4 import (
    save_int4_factored_checkpoint,
    dequant_int4_checkpoint,
)
from hyperretro.hf.factored import load_factored_hf_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
CHECKPOINTS = {
    "aware_r1024": Path("outputs/_ffn_only_aware_r1024"),
    "aware_r768": Path("outputs/_ffn_only_aware_r768"),
}
OUT = _ROOT / "benchmarks" / "factor_int4_ppl.json"

# Held-out eval text (same 3× repeated ML-history paragraph as rounds 11-13)
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
    "Machine learning has transformed the way we build software. "
    "Instead of writing explicit rules, we collect examples and let "
    "the optimization process discover patterns. This paradigm shift "
    "began with simple linear models and has now evolved into massive "
    "neural networks with hundreds of billions of parameters. "
    "The key insight is that gradient descent on a sufficiently large "
    "dataset will find structure that human engineers would never "
    "think to encode. However, this power comes at a cost: the models "
    "are opaque, their failures are unpredictable, and their "
    "computational requirements are staggering."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ppl(model, tokenizer, text: str, max_tokens: int = 256) -> float:
    """Compute perplexity on a text snippet."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_tokens)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        if loss is None:
            # Fallback: compute manually
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
    return float(torch.exp(loss))


def get_disk_mb(directory: Path) -> float:
    """Total safetensors bytes in directory."""
    total = 0.0
    for sf in directory.glob("*.safetensors"):
        total += sf.stat().st_size
    return total / 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    results = []

    # Load tokenizer once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for tag, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: {ckpt_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"[{tag}] measuring factor+int4 PPL")
        print(f"{'='*60}")

        # 1. Load manifest
        manifest_path = ckpt_dir / "hyperretro_factored.json"
        manifest = json.loads(manifest_path.read_text())
        orig_sd = load_file(str(ckpt_dir / "model.safetensors"))

        # 2. Save as int4
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID)

        int4_dir = Path(f"outputs/_ppl_test_{tag}_int4")
        if int4_dir.exists():
            shutil.rmtree(int4_dir)

        print(f"  Saving int4 checkpoint ...")
        t_save = time.time()
        report = save_int4_factored_checkpoint(
            orig_sd, [], manifest.get("ffn", []),
            out_dir=int4_dir, hf_config=cfg, n_bits=4,
            quantize_non_factored=False,
        )
        int4_mb = report["on_disk_mb"]
        print(f"  Int4 size: {int4_mb:.1f} MB  ({time.time() - t_save:.0f}s)")

        # 3. Dequant back to fp16 factored
        dq_dir = Path(f"outputs/_ppl_test_{tag}_dq")
        if dq_dir.exists():
            shutil.rmtree(dq_dir)

        print(f"  Dequantizing ...")
        t_dq = time.time()
        dequant_int4_checkpoint(int4_dir, dq_dir)
        dq_mb = get_disk_mb(dq_dir)
        print(f"  Dequant size: {dq_mb:.1f} MB  ({time.time() - t_dq:.0f}s)")

        # 4. Load model and measure PPL
        print(f"  Loading model for PPL eval ...")
        t_load = time.time()
        model, _info = load_factored_hf_model(str(dq_dir), dtype="float16")
        model = model.cpu()
        print(f"  Model loaded ({time.time() - t_load:.0f}s)")

        print(f"  Computing PPL ...")
        t_ppl = time.time()
        val = ppl(model, tokenizer, EVAL_TEXT)
        ppl_time = time.time() - t_ppl
        print(f"  PPL={val:.4f}  ({ppl_time:.0f}s)")

        # 5. Compare with known fp16 factored PPL
        known_ppl = {
            "aware_r1024": 4.5262,
            "aware_r768": 57.0122,
        }.get(tag, None)

        results.append({
            "tag": tag,
            "known_fp16_ppl": known_ppl,
            "int4_dequant_ppl": round(val, 4),
            "ppl_ratio_vs_fp16": round(val / known_ppl, 4) if known_ppl else None,
            "int4_disk_mb": round(int4_mb, 1),
            "dequant_disk_mb": round(dq_mb, 1),
            "wall_s": round(time.time() - t0, 1),
        })

        # Clean up model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if known_ppl:
            ratio = val / known_ppl
            print(f"\n  >>> PPL ratio vs fp16 factored: {ratio:.3f}x "
                  f"({'REGRESSION' if ratio > 1.05 else 'WITHIN NOISE'})")
            print(f"  >>> fp16 baseline PPL: 2.33")
            print(f"  >>> int4-factor PPL vs fp16 baseline: {val/2.3332:.2f}x")

    # 6. Save results
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        if not isinstance(existing, list):
            existing = [existing]
    existing.extend(results)
    OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}  ({time.time() - t0:.0f}s total)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    for r in results:
        tag = r["tag"]
        k = r.get("known_fp16_ppl")
        v = r["int4_dequant_ppl"]
        ratio = r.get("ppl_ratio_vs_fp16")
        print(f"  {tag}: fp16_factored={k:.4f} → int4_factor={v:.4f}  "
              f"({ratio:.3f}x)  |  int4_size={r['int4_disk_mb']:.0f}MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
