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

"""HyperRetro — retrofit HyperTensor geometric optimizations into PyTorch/HF/vLLM.

Usage::

    python -m hyperretro --help
    python -m hyperretro bench --model Qwen/Qwen2.5-0.5B --k 640
    python -m hyperretro compress --model Qwen/Qwen2.5-0.5B --out ./compressed --rank 640
    python -m hyperretro distill --model Qwen/Qwen2.5-0.5B --out ./distilled --rank 640
    python -m hyperretro spec --model Qwen/Qwen2.5-0.5B --k 640 --gamma 4

Version: 0.2.0
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hyperretro",
        description="HyperRetro: GRC compression + speculative decoding for HF models",
    )
    sub = p.add_subparsers(dest="command", help="Subcommand")

    # compress
    p_compress = sub.add_parser("compress", help="GRC-compress a HF model")
    p_compress.add_argument("--model", required=True)
    p_compress.add_argument("--out", required=True)
    p_compress.add_argument("--rank", type=int, default=1024)
    p_compress.add_argument("--sink", type=int, default=4)
    p_compress.add_argument("--layers", type=str, default="")
    p_compress.add_argument("--dtype", default="float32")

    # distill
    p_distill = sub.add_parser("distill", help="GRC light distillation")
    p_distill.add_argument("--model", required=True)
    p_distill.add_argument("--out", required=True)
    p_distill.add_argument("--rank", type=int, default=1024)
    p_distill.add_argument("--sink", type=int, default=4)
    p_distill.add_argument("--lora-rank", type=int, default=8)
    p_distill.add_argument("--steps", type=int, default=200)
    p_distill.add_argument("--batch", type=int, default=2)
    p_distill.add_argument("--seq-len", type=int, default=128)
    p_distill.add_argument("--corpus", default=None)
    p_distill.add_argument("--device", default="cuda")
    p_distill.add_argument("--dtype", default="float32")
    p_distill.add_argument("--loss", default="mse", choices=["mse", "kl"])

    # bench
    p_bench = sub.add_parser("bench", help="Run kernel or speculative bench")
    p_bench.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_bench.add_argument("--k", type=int, default=640)
    p_bench.add_argument("--device", default="cuda")
    p_bench.add_argument("--dtype", default="float32")

    # spec (speculative decoding simulation)
    p_spec = sub.add_parser("spec", help="Multi-token speculative decoding simulation")
    p_spec.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_spec.add_argument("--k", type=int, default=640)
    p_spec.add_argument("--gamma", type=int, default=4)
    p_spec.add_argument("--device", default="cuda")
    p_spec.add_argument("--dtype", default="float32")
    p_spec.add_argument("--cycles", type=int, default=20)

    # info
    sub.add_parser("info", help="Print version and backend info")

    # certify
    from hyperretro.certificates import register_cli
    register_cli(sub)

    args = p.parse_args(argv)

    if args.command == "compress":
        from hyperretro.hf.compress import compress_hf_model
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        compress_hf_model(args.model, args.out, rank_k=args.rank,
                          sink_T=args.sink, layers=layers or None, dtype=args.dtype)
        return 0

    if args.command == "distill":
        from hyperretro.hf.distill import distill_hf_model
        distill_hf_model(args.model, args.out, rank_k=args.rank,
                         sink_T=args.sink, lora_rank=args.lora_rank,
                         steps=args.steps, batch_size=args.batch,
                         seq_len=args.seq_len, corpus_path=args.corpus,
                         device=args.device, dtype=args.dtype,
                         loss_type=args.loss)
        return 0

    if args.command == "bench":
        from hyperretro.bench.real_speculative import run_compressed_drafter
        import json
        r = run_compressed_drafter(args.model,
            "The history of artificial intelligence began in antiquity",
            args.k, n_drafts=4, device=args.device, dtype=args.dtype)
        print(json.dumps(r, indent=2))
        return 0

    if args.command == "spec":
        from hyperretro.bench.spec_decode_sim import simulate_speculative_decode
        import json
        r = simulate_speculative_decode(
            args.model,
            "The history of artificial intelligence began in antiquity with myths",
            args.k, gamma=args.gamma, device=args.device, dtype=args.dtype,
            max_cycles=args.cycles)
        print(json.dumps(r, indent=2, default=str))
        return 0

    if args.command == "info":
        from hyperretro import __version__, kernels_backend
        print(f"HyperRetro v{__version__}")
        print(f"Kernel backend: {kernels_backend()}")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA: {torch.cuda.get_device_name(0)}")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception:
            pass
        return 0

    if args.command == "certify":
        return args.func(args)

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
