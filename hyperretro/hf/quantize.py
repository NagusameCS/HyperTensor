"""Compose HyperRetro GRC-compressed weights with bitsandbytes nf4 quantization.

Industry meta (May 2026) for 4-bit weight quantization is bitsandbytes nf4
(QLoRA, Dettmers et al. 2023) and AWQ / GPTQ. HyperRetro's geometric
compression and 4-bit quantization are *orthogonal* levers:

    raw  fp16  : 1.00x
    GRC  bf16  : ~1.13x  (13% param drop, attn rank + FFN low-rank)
    nf4         : ~4.0x  (4-bit weights, +1-3% PPL typical)
    GRC * nf4   : ~4.5x  (composed, this module)

This module exposes:

* :func:`quantize_to_nf4` -- load an HF model directory (compressed or not),
  re-save it with nf4 weights using the standard ``BitsAndBytesConfig`` path
  (transformers >= 4.42 supports serialisation of 4-bit checkpoints).
* :func:`measure_size_mb` -- on-disk size of all weight shards in a dir.

The output remains 100% inside the HuggingFace ecosystem -- consumers load
it with vanilla ``AutoModelForCausalLM.from_pretrained`` without any
HyperTensor runtime dependency.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuantizeConfig:
    quant_type: str = "nf4"          # "nf4" or "fp4"
    compute_dtype: str = "bfloat16"  # bf16 is safer than fp16 for 1.5B+
    double_quant: bool = True        # QLoRA double-quant
    bnb_4bit_storage: str = "uint8"  # serialisation storage type


def _bnb_config(cfg: QuantizeConfig):
    import torch
    from transformers import BitsAndBytesConfig

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_compute_dtype=dtype_map[cfg.compute_dtype],
        bnb_4bit_use_double_quant=cfg.double_quant,
    )


def quantize_to_nf4(
    model_id_or_path: str,
    out_dir: str | Path,
    *,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
    revision: str | None = None,
) -> dict:
    """Load a HF model and re-save with nf4 4-bit quantized weights.

    Parameters
    ----------
    model_id_or_path : HF repo id or local directory (e.g. a HyperRetro
        GRC-compressed checkpoint produced by :func:`compress_hf_model`).
    out_dir : output directory for the quantized model.
    quant_type : "nf4" (default) or "fp4".
    compute_dtype : matmul accumulation dtype.
    double_quant : QLoRA double quantization (recommended).
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "quantize_to_nf4 requires the 'hf' extras: "
            "pip install hyperretro[hf] bitsandbytes accelerate"
        ) from e

    qcfg = QuantizeConfig(
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        double_quant=double_quant,
    )
    bnb_cfg = _bnb_config(qcfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        revision=revision,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model.save_pretrained(out, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision)
        tok.save_pretrained(out)
    except Exception:
        pass

    # Propagate the upstream hyperretro_report.json / certificate if present
    src = Path(model_id_or_path)
    if src.is_dir():
        for f in ("hyperretro_report.json", "hyperretro_certificate.json"):
            p = src / f
            if p.exists():
                shutil.copy2(p, out / f)

    size_mb = measure_size_mb(out)
    report = {
        "source": str(model_id_or_path),
        "out_dir": str(out),
        "quant": {
            "quant_type": qcfg.quant_type,
            "compute_dtype": qcfg.compute_dtype,
            "double_quant": qcfg.double_quant,
        },
        "on_disk_mb": size_mb,
    }
    (out / "hyperretro_quant_report.json").write_text(json.dumps(report, indent=2))
    return report


def measure_size_mb(model_dir: str | Path) -> float:
    """Sum of .safetensors / .bin shard sizes in a model dir, in MB."""
    p = Path(model_dir)
    total = 0
    for f in p.iterdir():
        if f.suffix in (".safetensors", ".bin"):
            total += f.stat().st_size
    return total / (1024.0 * 1024.0)


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hyperretro-quantize",
        description=(
            "Quantize a HF (or HyperRetro-compressed) model to 4-bit nf4 "
            "weights via bitsandbytes. Composes with attn-GRC / FFN-LoRA "
            "for stacked compression."
        ),
    )
    p.add_argument("--model", required=True,
                   help="HF model id OR path to a HyperRetro-compressed dir")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--quant-type", choices=["nf4", "fp4"], default="nf4")
    p.add_argument("--compute-dtype",
                   choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--no-double-quant", action="store_true",
                   help="Disable QLoRA double-quantization (slightly larger).")
    p.add_argument("--revision", default=None)
    args = p.parse_args(argv)

    report = quantize_to_nf4(
        args.model,
        args.out,
        quant_type=args.quant_type,
        compute_dtype=args.compute_dtype,
        double_quant=not args.no_double_quant,
        revision=args.revision,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
