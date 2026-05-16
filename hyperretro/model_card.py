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

"""HyperRetro model card generator — self-documenting checkpoints.

Produces a README.md and model_card.json inside each compressed checkpoint
with complete metadata: compression ratios, quantization details, PPL
estimates, architecture info, and deployment instructions.

Usage:
    python -m hyperretro.cli card compressed/

The generated model card is compatible with HuggingFace model card standards
and includes all HyperRetro-specific compression metadata.
"""

from __future__ import annotations

import json, time
from pathlib import Path
from typing import Optional


def generate_model_card(
    checkpoint_dir: str | Path,
    *,
    source_model: str | None = None,
    ppl_baseline: float | None = None,
    ppl_compressed: float | None = None,
    extra_notes: str | None = None,
) -> dict:
    """Generate a model card for a HyperRetro-compressed checkpoint.

    Reads the checkpoint's manifest and config to produce a comprehensive
    model card with all relevant metadata.

    Returns the model card as a dict (also written to disk).
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        raise ValueError(f"Not a directory: {ckpt}")

    manifest_path = ckpt / "hyperretro_factored.json"
    config_path = ckpt / "config.json"
    st_path = ckpt / "model.safetensors"

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    # --- Size ---
    size_mb = 0.0
    param_count = 0
    if st_path.exists():
        size_mb = st_path.stat().st_size / 1e6
        try:
            from safetensors.torch import load_file
            sd = load_file(str(st_path))
            # Estimate params from tensor shapes
            for v in sd.values():
                if hasattr(v, "numel"):
                    param_count += v.numel()
        except Exception:
            pass

    # --- Compression details ---
    n_attn = len(manifest.get("layers", []))
    n_ffn = len(manifest.get("ffn", []))
    ffn_ranks = sorted(set(e.get("rank", 0) for e in manifest.get("ffn", [])))
    attn_ranks = sorted(set(e.get("rank", 0) for e in manifest.get("layers", [])))
    quant = "none"
    for e in manifest.get("ffn", [])[:1]:
        quant = e.get("quantization", "none")

    # --- Architecture ---
    arch = config.get("architectures", config.get("model_type", "unknown"))
    if isinstance(arch, list):
        arch = arch[0] if arch else "unknown"
    hidden = config.get("hidden_size", config.get("dim", "?"))
    vocab = config.get("vocab_size", "?")
    n_layers = config.get("num_hidden_layers", config.get("n_layers", "?"))
    n_heads = config.get("num_attention_heads", config.get("n_heads", "?"))
    n_kv = config.get("num_key_value_heads", config.get("n_kv_heads", "?"))

    # --- Shrink estimates ---
    fp16_est = param_count * 2 / 1e6 if param_count > 0 else None
    shrink = fp16_est / size_mb if fp16_est and size_mb > 0 else None

    # --- Build card ---
    card = {
        "model_card_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "compression": {
            "tool": "HyperRetro",
            "version": "0.2.1",
            "techniques": [],
        },
        "model": {
            "source": source_model or "unknown",
            "architecture": str(arch),
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_kv_heads": n_kv,
        },
        "compression_details": {
            "attention_layers_factored": n_attn,
            "ffn_matrices_factored": n_ffn,
            "ffn_ranks": ffn_ranks,
            "attn_ranks": attn_ranks,
            "quantization": quant,
            "shared_basis": manifest.get("shared_basis", False),
        },
        "size": {
            "on_disk_mb": round(size_mb, 1),
            "estimated_fp16_mb": round(fp16_est, 1) if fp16_est else None,
            "estimated_params": param_count,
            "shrink_vs_fp16": round(shrink, 2) if shrink else None,
        },
        "performance": {
            "ppl_baseline_fp16": ppl_baseline,
            "ppl_compressed": ppl_compressed,
            "ppl_ratio": round(ppl_compressed / ppl_baseline, 3) if ppl_baseline and ppl_compressed else None,
        },
        "deployment": {
            "formats": ["safetensors", "gguf"],
            "load_with": "hyperretro.models.load_model() or load_factored_hf_model()",
            "gguf_export": "hyperretro export <checkpoint> --format gguf",
            "llama_cpp_quantize": "llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M",
        },
    }

    # Populate techniques
    techniques = card["compression"]["techniques"]
    if n_ffn > 0:
        techniques.append("activation-aware FFN SVD factoring")
    if n_attn > 0:
        techniques.append("GRC shared-basis attention compression")
    if "int4" in quant or "int8" in quant:
        techniques.append(f"block-wise {quant} quantization")
    if "awq" in quant:
        techniques.append("AWQ-aware quantization scaling")
    if not techniques:
        techniques.append("none")

    # --- Write files ---
    # model_card.json
    (ckpt / "model_card.json").write_text(json.dumps(card, indent=2))

    # README.md
    readme = _generate_readme(card, extra_notes)
    (ckpt / "README.md").write_text(readme)

    print(f"[card] Generated model_card.json + README.md for {ckpt.name}")
    return card


def _generate_readme(card: dict, extra_notes: str | None = None) -> str:
    """Generate a HuggingFace-compatible README.md from a model card."""
    c = card["compression_details"]
    s = card["size"]
    p = card["performance"]
    m = card["model"]

    lines = [
        f"---",
        f"language: en",
        f"license: mit",
        f"tags:",
        f"- hyperretro",
        f"- compressed",
        f"- {m['architecture']}",
    ]
    for t in card["compression"]["techniques"]:
        tag = t.lower().replace(" ", "-").replace("(", "").replace(")", "")
        lines.append(f"- {tag}")
    lines.append("---")

    lines += [
        "",
        f"# HyperRetro-Compressed {m['architecture']}",
        "",
        f"This is a **HyperRetro-compressed** version of **{card['model']['source']}**.",
        "",
        "## Compression Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Original model | {card['model']['source']} |",
        f"| Architecture | {m['architecture']} |",
        f"| Hidden size | {m['hidden_size']} |",
        f"| Layers | {m['num_layers']} |",
        f"| On-disk size | {s['on_disk_mb']:.1f} MB |",
    ]

    if s.get("shrink_vs_fp16"):
        lines.append(f"| Shrink vs fp16 | {s['shrink_vs_fp16']:.1f}× |")

    lines += [
        f"| FFN matrices factored | {c['ffn_matrices_factored']} |",
        f"| FFN ranks | {c['ffn_ranks']} |",
        f"| Attention layers factored | {c['attention_layers_factored']} |",
        f"| Quantization | {c['quantization']} |",
    ]

    if p.get("ppl_compressed") and p.get("ppl_baseline"):
        lines += [
            f"| PPL (baseline fp16) | {p['ppl_baseline']:.2f} |",
            f"| PPL (compressed) | {p['ppl_compressed']:.2f} |",
            f"| PPL ratio | {p.get('ppl_ratio', 1.0):.3f}× |",
        ]

    lines += [
        "",
        "## Compression Techniques",
        "",
    ]
    for i, tech in enumerate(card["compression"]["techniques"], 1):
        lines.append(f"{i}. {tech}")

    lines += [
        "",
        "## Usage",
        "",
        "### Python API",
        "",
        "```python",
        "from hyperretro.models import load_model",
        "model = load_model('<checkpoint_path>')",
        "```",
        "",
        "### GGUF Export (for llama.cpp / Ollama)",
        "",
        "```bash",
        "hyperretro export <checkpoint_path> --format gguf -o model.gguf",
        "llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M",
        "```",
        "",
        "### CLI",
        "",
        "```bash",
        "hyperretro info <checkpoint_path>",
        "```",
        "",
        "## Generated by",
        "",
        f"HyperRetro v{card['compression']['version']} — universal model compression",
        f"Generated: {card['generated_at']}",
    ]

    if extra_notes:
        lines += ["", "## Notes", "", extra_notes]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def cmd_card(args):
    """Generate model card for a checkpoint."""
    card = generate_model_card(
        args.checkpoint,
        source_model=args.source,
        ppl_baseline=args.ppl_baseline,
        ppl_compressed=args.ppl_compressed,
        extra_notes=args.notes,
    )
    if args.json:
        print(json.dumps(card, indent=2))
    return 0


def register_cli(subparsers):
    """Register the 'card' subcommand."""
    p = subparsers.add_parser("card", help="Generate model card for a checkpoint")
    p.add_argument("checkpoint", help="Path to checkpoint directory")
    p.add_argument("--source", default=None, help="Original model name")
    p.add_argument("--ppl-baseline", type=float, default=None)
    p.add_argument("--ppl-compressed", type=float, default=None)
    p.add_argument("--notes", default=None, help="Extra notes for the model card")
    p.add_argument("--json", action="store_true", help="Output JSON only")
    p.set_defaults(func=cmd_card)
    return p
