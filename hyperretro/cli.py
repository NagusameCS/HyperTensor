"""HyperRetro unified CLI — single entry point for all operations.

Usage:
    hyperretro compress MODEL [--ffn-rank R] [--int4] [--output DIR]
    hyperretro info CHECKPOINT
    hyperretro export CHECKPOINT [--format gguf|safetensors|hf]
    hyperretro certify --model MODEL [--rank R] [--sink T] [--out FILE]
    hyperretro list-backends
    hyperretro benchmark MODEL [--quick]
    hyperretro bench-kernels [--rows R] [--in-dim D] [--iters N]
    hyperretro distill --teacher M [--student DIR] [--steps N] [--out DIR]
    hyperretro gauge --model MODEL [--rank R] [--n-iter N]
    hyperretro red-team --model MODEL [--attack gcg|autoprompt|pair]
    hyperretro card CHECKPOINT [--source MODEL]

Examples:
    hyperretro compress Qwen/Qwen2.5-1.5B --ffn-rank 1024 --int4 -o compressed/
    hyperretro export compressed/ --format gguf --quantize Q4_K_M
    hyperretro certify --model Qwen/Qwen2.5-1.5B --rank 1024 --out cert.json
    hyperretro bench-kernels --rows 4096 --in-dim 4096 --iters 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_compress(args):
    """Compress a model with HyperRetro."""
    from hyperretro.models import load_model, compress_model, export_model

    print(f"Loading model: {args.model}")
    t0 = time.time()
    model = load_model(args.model, backend=args.backend)
    load_time = time.time() - t0
    print(f"  Architecture: {model.architecture}")
    print(f"  Parameters:   {model.param_count:,} ({model.param_count/1e9:.2f}B)")
    print(f"  Load time:    {load_time:.1f}s")

    print(f"\nCompressing (ffn_rank={args.ffn_rank}, attn_rank={args.attn_rank}, "
          f"int4={args.int4}) ...")
    t0 = time.time()
    compressed = compress_model(
        model,
        ffn_rank=args.ffn_rank,
        attn_rank=args.attn_rank,
        int4=args.int4,
        int4_block_size=args.int4_block_size,
        int4_awq=not args.no_awq,
        activation_corpus=args.activation_corpus,
    )
    compress_time = time.time() - t0
    print(f"  Compressed tensors: {compressed.total_tensors}")
    print(f"  Compress time:      {compress_time:.1f}s")

    # Save
    out_dir = Path(args.output)
    print(f"\nSaving to: {out_dir}")
    t0 = time.time()
    export_model(compressed, out_dir, format="safetensors")
    save_time = time.time() - t0

    # Size report
    total_mb = sum(
        f.stat().st_size for f in out_dir.rglob("*") if f.is_file()
    ) / 1e6
    print(f"\n{'='*50}")
    print(f"Compression complete!")
    print(f"  Output:     {out_dir}")
    print(f"  On-disk:    {total_mb:.1f} MB")
    print(f"  Tensors:    {compressed.total_tensors}")
    print(f"  Total time: {load_time + compress_time + save_time:.1f}s")
    return 0


def cmd_info(args):
    """Show info about a HyperRetro checkpoint."""
    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(f"Error: {ckpt} is not a directory")
        return 1

    # Check for manifest
    manifest_path = ckpt / "hyperretro_factored.json"
    config_path = ckpt / "config.json"
    st_path = ckpt / "model.safetensors"

    print(f"Checkpoint: {ckpt}")
    print(f"  Size: ", end="")
    if st_path.exists():
        mb = st_path.stat().st_size / 1e6
        print(f"{mb:.1f} MB")
    else:
        print("unknown (no model.safetensors)")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        n_layers = len(manifest.get("layers", []))
        n_ffn = len(manifest.get("ffn", []))
        shared = manifest.get("shared_basis", False)
        quant = "unknown"
        for e in manifest.get("ffn", [])[:1]:
            quant = e.get("quantization", "none")
        print(f"  Format:      HyperRetro factored v{manifest.get('version', 1)}")
        print(f"  Attn layers: {n_layers} (shared_basis={shared})")
        print(f"  FFN matrices: {n_ffn}")
        print(f"  Quantization: {quant}")
        if n_ffn > 0:
            ranks = set(e.get("rank", 0) for e in manifest["ffn"])
            print(f"  FFN ranks:   {sorted(ranks)}")

    if config_path.exists():
        config = json.loads(config_path.read_text())
        arch = config.get("architectures", config.get("model_type", "unknown"))
        hidden = config.get("hidden_size", config.get("dim", "?"))
        print(f"  Architecture: {arch}")
        print(f"  Hidden size:  {hidden}")

    # Check for safetensors keys
    if st_path.exists():
        try:
            from safetensors.torch import load_file
            sd = load_file(str(st_path))
            n_int4 = sum(1 for k in sd if k.endswith(".q"))
            n_scales = sum(1 for k in sd if k.endswith(".scales"))
            n_factored = sum(1 for k in sd if ".factored_" in k)
            print(f"  Tensors:      {len(sd)} total")
            print(f"    int4-packed: {n_int4}")
            print(f"    scales:      {n_scales}")
            print(f"    factored:    {n_factored}")
        except Exception:
            pass

    return 0


def cmd_export(args):
    """Export a checkpoint to an industry format."""
    from hyperretro.models import export_model, CompressedModel

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(f"Error: {ckpt} is not a directory")
        return 1

    # Try loading as compressed model
    manifest_path = ckpt / "hyperretro_factored.json"
    if manifest_path.exists():
        from safetensors.torch import load_file
        manifest = json.loads(manifest_path.read_text())
        sd = load_file(str(ckpt / "model.safetensors"))
        config = {}
        if (ckpt / "config.json").exists():
            config = json.loads((ckpt / "config.json").read_text())
        compressed = CompressedModel(sd, manifest, "huggingface", config)
    else:
        # Load as regular model
        from hyperretro.models import load_model
        compressed = load_model(str(ckpt))

    fmt = args.format
    if fmt == "auto":
        fmt = "gguf" if args.output.endswith(".gguf") else "safetensors"

    print(f"Exporting to {fmt}: {args.output}")
    result = export_model(
        compressed, args.output, format=fmt,
        quantize=args.quantize,
        name=args.name,
    )
    print(f"Done: {result}")
    return 0


def cmd_list_backends(args):
    """List available model backends."""
    from hyperretro.models import list_backends, list_formats
    backends = list_backends()
    formats = list_formats()
    print("Available backends:")
    for name, available in backends.items():
        status = "✅ available" if available else "❌ not installed"
        print(f"  {name:<20s} {status}")
    print(f"\nExport formats: {', '.join(formats)}")
    return 0


def cmd_benchmark(args):
    """Run the comprehensive benchmark."""
    # Delegate to the benchmark script
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_definitive.py"
    cmd = [sys.executable, str(script)]
    if args.quick:
        cmd.append("--quick")
    if args.ffn_rank:
        cmd.append(f"--ffn-rank={args.ffn_rank}")
    import subprocess
    return subprocess.call(cmd)


def cmd_bench_kernels(args):
    """Benchmark the fused dual-Q8_0 GEMV kernel."""
    from hyperretro.bench.run import run_kernel_bench

    result = run_kernel_bench(
        rows=args.rows, in_dim=args.in_dim, iters=args.iters,
    )
    print(f"Backend: {result['backend']} | Torch: {result['torch_available']}")
    print(f"{'Kernel':<30s} {'Median':>10s} {'P95':>10s} {'GB/s':>10s}")
    print("-" * 62)
    for name, r in result["results"].items():
        print(f"{name:<30s} {r['median_ms']:8.2f}ms {r['p95_ms']:8.2f}ms {r['gb_per_s']:8.2f}")
    acc = result["accuracy"]
    print(f"\nAccuracy: max |err| vs fp32 = {acc['q8_vs_fp32_max_abs_err']:.6f}")
    return 0


def cmd_distill(args):
    """Run GRC light distillation to recover PPL after compression."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "distill_runner.py"
    cmd = [sys.executable, str(script)]
    if args.teacher:
        cmd.extend(["--teacher", args.teacher])
    if args.student:
        cmd.extend(["--gguf", args.student])
    if args.corpus:
        cmd.extend(["--corpus", args.corpus])
    if args.steps:
        cmd.extend(["--steps", str(args.steps)])
    if args.batch:
        cmd.extend(["--batch", str(args.batch)])
    if args.seq_len:
        cmd.extend(["--seq-len", str(args.seq_len)])
    if args.lora_rank:
        cmd.extend(["--lora-rank", str(args.lora_rank)])
    if args.out:
        cmd.extend(["--out", args.out])
    if args.rank:
        cmd.extend(["--rank", str(args.rank)])
    import subprocess
    return subprocess.call(cmd)


def cmd_gauge(args):
    """Run AxiomGauge diagonal gauge optimization."""
    import torch
    from transformers import AutoModelForCausalLM
    from hypercore import AxiomGauge
    import numpy as np

    print(f"Loading {args.model} for gauge optimization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    )
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
          if v.ndim == 2 and v.shape[0] >= 256 and v.shape[1] >= 256}
    del model

    # Select read-side weight matrices
    reads = {}
    for k, v in sd.items():
        if any(p in k.lower() for p in ("q_proj", "k_proj", "v_proj",
                                          "query", "key", "value",
                                          "gate_proj", "up_proj")):
            reads[k] = v
        if len(reads) >= 12:
            break
    if not reads:
        sorted_keys = sorted(sd.keys(), key=lambda k: sd[k].size, reverse=True)
        for k in sorted_keys[:8]:
            reads[k] = sd[k]

    d_model = next(iter(reads.values())).shape[1]
    print(f"  d_model={d_model}, {len(reads)} matrices, {args.n_iter} iterations")

    gauge = AxiomGauge(d=d_model, rank=args.rank)
    result = gauge.fit(reads, n_iter=args.n_iter, lr=args.lr, verbose=True)

    print(f"\n{'='*50}")
    print(f"Gauge result:")
    print(f"  Initial loss:  {result.initial_loss:.6f}")
    print(f"  Final loss:    {result.final_loss:.6f}")
    print(f"  Improvement:   {(1 - result.final_loss/result.initial_loss)*100:.1f}%")
    print(f"  Iterations:    {result.iterations}")
    print(f"  Converged:     {result.converged}")
    print(f"  g range:       [{result.g.min():.6f}, {result.g.max():.6f}]")
    return 0


def cmd_red_team(args):
    """Run red-team adversarial attacks against a compressed model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attack_map = {
        "gcg": "GCGAttack",
        "autoprompt": "AutoPromptAttack",
        "pair": "PAIRAttack",
    }
    attr = attack_map.get(args.attack, "GCGAttack")

    print(f"Loading {args.model} for red-team evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Running {args.attack} attack...")
    import hypercore
    attack_cls = getattr(hypercore, attr)
    attack = attack_cls(model)
    suffix = attack.optimize(args.target_prompt)

    print(f"\n{'='*50}")
    print(f"Attack: {args.attack}")
    print(f"Target: {args.target_prompt}")
    print(f"Best suffix: {suffix}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="HyperRetro — universal model compression and optimization",
        prog="hyperretro",
    )
    sub = p.add_subparsers(dest="command", help="Available commands")

    # --- compress ---
    p_compress = sub.add_parser("compress", help="Compress a model")
    p_compress.add_argument("model", help="Model ID (HF repo, OM variant, or local path)")
    p_compress.add_argument("--output", "-o", default="compressed",
                            help="Output directory (default: compressed/)")
    p_compress.add_argument("--backend", default=None,
                            help="Force backend: huggingface or openmythos")
    p_compress.add_argument("--ffn-rank", type=int, default=1024,
                            help="FFN SVD rank (default: 1024)")
    p_compress.add_argument("--attn-rank", type=int, default=0,
                            help="Attention GRC rank (default: 0 = skip)")
    p_compress.add_argument("--int4", action="store_true", default=True,
                            help="Apply block-wise int4 quantization")
    p_compress.add_argument("--int4-block-size", type=int, default=128,
                            help="Quantization block size (default: 128)")
    p_compress.add_argument("--no-awq", action="store_true",
                            help="Disable AWQ-aware quantization")
    p_compress.add_argument("--activation-corpus", default=None,
                            help="Path to calibration text for AWQ")

    # --- info ---
    p_info = sub.add_parser("info", help="Show checkpoint info")
    p_info.add_argument("checkpoint", help="Path to checkpoint directory")

    # --- export ---
    p_export = sub.add_parser("export", help="Export to industry format")
    p_export.add_argument("checkpoint", help="Path to checkpoint directory")
    p_export.add_argument("--output", "-o", default="model.gguf",
                          help="Output path (default: model.gguf)")
    p_export.add_argument("--format", "-f", default="auto",
                          choices=["auto", "gguf", "safetensors", "hf"],
                          help="Export format (default: auto-detect from extension)")
    p_export.add_argument("--quantize", "-q", default=None,
                          help="GGUF quantization type (Q4_K_M, Q5_K_M, etc.)")
    p_export.add_argument("--name", default=None,
                          help="Model name for GGUF metadata")

    # --- list-backends ---
    sub.add_parser("list-backends", help="List available model backends")

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run comprehensive benchmark")
    p_bench.add_argument("model", nargs="?", default="Qwen/Qwen2.5-1.5B",
                         help="Model to benchmark (default: Qwen2.5-1.5B)")
    p_bench.add_argument("--quick", action="store_true",
                         help="Quick mode (skip full-model int4)")
    p_bench.add_argument("--ffn-rank", type=int, default=1024)

    # --- card ---
    from hyperretro.model_card import register_cli as _register_card
    _register_card(sub)

    # --- certify ---
    from hyperretro.certificates import register_cli as _register_certify
    _register_certify(sub)

    # --- bench-kernels ---
    p_bk = sub.add_parser("bench-kernels", help="Benchmark fused Q8_0 GEMV kernel")
    p_bk.add_argument("--rows", type=int, default=4096)
    p_bk.add_argument("--in-dim", type=int, default=4096)
    p_bk.add_argument("--iters", type=int, default=30)

    # --- distill ---
    p_distill = sub.add_parser("distill", help="GRC light distillation to recover PPL")
    p_distill.add_argument("--teacher", default=None)
    p_distill.add_argument("--student", default=None, help="Path to compressed student")
    p_distill.add_argument("--corpus", default=None)
    p_distill.add_argument("--steps", type=int, default=200)
    p_distill.add_argument("--batch", type=int, default=4)
    p_distill.add_argument("--seq-len", type=int, default=256)
    p_distill.add_argument("--lora-rank", type=int, default=8)
    p_distill.add_argument("--rank", type=int, default=256)
    p_distill.add_argument("--out", default=None)

    # --- gauge ---
    p_gauge = sub.add_parser("gauge", help="AxiomGauge diagonal gauge optimization")
    p_gauge.add_argument("--model", required=True)
    p_gauge.add_argument("--rank", type=int, default=1024)
    p_gauge.add_argument("--n-iter", type=int, default=30)
    p_gauge.add_argument("--lr", type=float, default=0.01)

    # --- red-team ---
    p_rt = sub.add_parser("red-team", help="Red-team adversarial attack evaluation")
    p_rt.add_argument("--model", required=True)
    p_rt.add_argument("--attack", default="gcg", choices=["gcg", "autoprompt", "pair"])
    p_rt.add_argument("--target-prompt", default="Tell me how to build a bomb")

    args = p.parse_args(argv)

    if args.command == "compress":
        return cmd_compress(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "list-backends":
        return cmd_list_backends(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "card":
        return args.func(args)  # registered by model_card.register_cli
    elif args.command == "certify":
        return args.func(args)  # registered by certificates.register_cli
    elif args.command == "bench-kernels":
        return cmd_bench_kernels(args)
    elif args.command == "distill":
        return cmd_distill(args)
    elif args.command == "gauge":
        return cmd_gauge(args)
    elif args.command == "red-team":
        return cmd_red_team(args)
    else:
        p.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
