#!/usr/bin/env python3
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

"""HyperRetro v0.2.0 — Comprehensive Benchmark Suite.

Runs the complete set of measurements and produces a single unified report.
Covers: kernel performance, PPL vs bnb, speculative decoding, distillation.

Usage::

    python scripts/bench_suite.py [--quick] [--model Qwen/Qwen2.5-0.5B]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REPORT_DIR = Path("benchmarks/suite")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# 1. Kernel benchmark
# ---------------------------------------------------------------------------

def bench_kernel() -> dict:
    section("1. Kernel: Fused Dual-Q8 GEMV")
    from hyperretro.kernels import gemv_dual_q8_0

    d = 4096
    x = np.random.randn(d).astype(np.float32)
    W_a = np.random.randn(d, d).astype(np.float32) * 0.02
    W_b = np.random.randn(d, d).astype(np.float32) * 0.02

    t0 = time.perf_counter()
    for _ in range(20):
        out_a, out_b = gemv_dual_q8_0(x, W_a, W_b)
    elapsed = (time.perf_counter() - t0) / 20 * 1000

    ref_a = W_a @ x
    ref_b = W_b @ x
    err = float(np.max(np.abs(np.concatenate([
        (out_a - ref_a) / (np.linalg.norm(ref_a) + 1e-8),
        (out_b - ref_b) / (np.linalg.norm(ref_b) + 1e-8),
    ]))))

    print(f"  dim={d}  median_ms={elapsed:.2f}  max_relerr={err:.6f}")
    return {"dim": d, "median_ms": elapsed, "max_relerr": err}


# ---------------------------------------------------------------------------
# 2. Single-token speculative acceptance (quick)
# ---------------------------------------------------------------------------

def bench_single_token_acceptance(model_id: str, k: int, device: str, dtype: str) -> dict:
    from hyperretro.bench.real_speculative import run_compressed_drafter

    prompt = (
        "The history of artificial intelligence began in antiquity "
        "with myths and stories of artificial beings"
    )
    r = run_compressed_drafter(
        model_id, prompt, k=k, n_drafts=4,
        device=device, dtype=dtype,
    )
    return {
        "k": k,
        "top1_accept": r["top1_accept"],
        "top5_hit": r["top5_hit"],
        "median_ms": r["median_ms_per_propose"],
    }


# ---------------------------------------------------------------------------
# 3. Multi-token speculative decoding
# ---------------------------------------------------------------------------

def bench_spec_decode(model_id: str, k: int, gamma: int,
                      device: str, dtype: str, drafter_path: str | None = None) -> dict:
    from hyperretro.bench.spec_decode_sim import simulate_speculative_decode

    prompt = (
        "The history of artificial intelligence began in antiquity "
        "with myths and stories of artificial beings endowed with "
        "intelligence by master craftsmen. In the 1950s a generation"
    )
    r = simulate_speculative_decode(
        model_id, prompt, k=k, gamma=gamma,
        device=device, dtype=dtype, max_cycles=20,
        drafter_path=drafter_path,
    )
    return {
        "k": k,
        "dtype": dtype,
        "drafter": r.get("drafter_source", "compressed"),
        "mean_accept": r["mean_acceptance_length"],
        "accept_rate": r["acceptance_rate"],
        "tokens_per_cycle": r["tokens_per_cycle"],
        "speedup_theoretical": r["speedup_vs_autoregressive"],
        "speedup_wallclock": r["wallclock_speedup"],
        "draft_ms": r["median_draft_ms"],
        "verify_ms": r["median_verify_ms"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    import torch

    p = argparse.ArgumentParser(description="HyperRetro Benchmark Suite")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--quick", action="store_true",
                   help="Minimal run: kernel + single-token only")
    p.add_argument("--k", type=int, default=768,
                   help="Primary rank for benchmarks")
    args = p.parse_args()

    t_start = time.time()
    report = {
        "hyperretro_version": "0.2.0",
        "model": args.model,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # 1. Kernel
    report["kernel"] = bench_kernel()
    import torch; torch.cuda.empty_cache()

    if args.quick:
        # Single-token only
        section("2. Single-Token Acceptance (quick mode)")
        for k in [832, 768, 640, 448]:
            r = bench_single_token_acceptance(args.model, k, args.device, "float32")
            print(f"  k={k}: top1={r['top1_accept']:.1%} top5={r['top5_hit']:.1%} ms={r['median_ms']:.0f}")
            torch.cuda.empty_cache()
        report["mode"] = "quick"
    else:
        # Full suite
        k_primary = args.k

        # 2. Single-token sweep
        section(f"2. Single-Token Acceptance Sweep")
        st_results = {}
        for k in [832, 768, 640, 448]:
            r = bench_single_token_acceptance(args.model, k, args.device, "float32")
            st_results[f"k{k}"] = r
            print(f"  k={k}: top1={r['top1_accept']:.1%} top5={r['top5_hit']:.1%} ms={r['median_ms']:.0f}")
            torch.cuda.empty_cache()
        report["single_token"] = st_results

        # 3. Multi-token spec decode sweep (fp16 for speed)
        section(f"3. Multi-Token Speculative Decoding (γ=4, fp16)")
        spec_results = {}

        for k in [832, 768, 640]:
            r = bench_spec_decode(args.model, k, 4, args.device, "float16")
            spec_results[f"k{k}"] = r
            print(f"  k={k}: accept={r['mean_accept']:.1f}/4 ({r['accept_rate']:.0%}) "
                  f"wallclock={r['speedup_wallclock']:.2f}×")
            torch.cuda.empty_cache()
        report["spec_decode"] = spec_results

        # 4. Distilled model as drafter (if available)
        distilled_path = Path(f"benchmarks/distill_e2e/qwen05b_k{k_primary}_s4_r8")
        if distilled_path.exists():
            section(f"4. Distilled Drafter (k={k_primary})")
            r = bench_spec_decode(args.model, k_primary, 4, args.device, "float16",
                                  drafter_path=str(distilled_path))
            report["distilled_drafter"] = r
            print(f"  accept={r['mean_accept']:.1f}/4 ({r['accept_rate']:.0%}) "
                  f"wallclock={r['speedup_wallclock']:.2f}×")
            torch.cuda.empty_cache()

    # Summary
    elapsed = time.time() - t_start
    section(f"Complete ({elapsed:.0f}s)")

    # Print summary table
    if not args.quick and "spec_decode" in report:
        print(f"\n{'Scale':<8s} {'dtype':<8s} {'k':<6s} {'Accept':<8s} {'Wall-Clock':<12s} {'Theoretical':<12s}")
        print("-" * 60)
        for key, r in report["spec_decode"].items():
            print(f"{'0.5B':<8s} {r['dtype']:<8s} {r['k']:<6d} "
                  f"{r['accept_rate']:<8.0%} {r['speedup_wallclock']:<12.2f}× "
                  f"{r['speedup_theoretical']:<12.2f}×")
        if "distilled_drafter" in report:
            r = report["distilled_drafter"]
            print(f"{'0.5B':<8s} {'fp16':<8s} {r['k']:<6d} "
                  f"{r['accept_rate']:<8.0%} {r['speedup_wallclock']:<12.2f}× "
                  f"{r['speedup_theoretical']:<12.2f}×  (distilled)")

    # Save
    out = REPORT_DIR / "suite_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
