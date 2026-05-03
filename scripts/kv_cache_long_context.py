#!/usr/bin/env python3
"""
KV-Cache Compression at Scale (Tier 2).

The geodessical binary supports --axex-kv to compress the KV cache
using the same per-layer basis used for W_K and W_V.  At short context
(≤2K tokens) this is a non-issue; at 32K+ tokens the KV cache becomes
the dominant VRAM consumer, and k=1024 projection cuts it by ~75%.

This script benchmarks decode throughput and VRAM usage at increasing
context lengths, with KV-cache compression on/off.

Usage:
  python scripts/kv_cache_long_context.py \
    --model models/smollm2-135m-instruct-q8_0.gguf \
    --out benchmarks/kv_cache_long \
    --contexts 2048,4096,8192,16384,32768 \
    --rank 256

The script generates a synthetic long-context prompt (repeated text)
and measures:
  - Peak VRAM usage (via nvidia-smi polling)
  - Decode throughput (tok/s)
  - KV-cache size estimate
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def detect_exe() -> Optional[Path]:
    for p in [
        ROOT / "build_host" / "geodessical2.exe",
        ROOT / "build_host" / "geodessical.exe",
    ]:
        if p.exists():
            return p
    return None


def get_vram_mb() -> int:
    """Get current GPU VRAM usage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(float(out.strip()))
    except Exception:
        return -1


def generate_long_prompt(base_text: str, target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens by repeating base_text."""
    # Rough heuristic: 4 chars per token
    chars_needed = target_tokens * 4
    repeats = max(1, chars_needed // len(base_text) + 1)
    return base_text * repeats


def run_decode_with_kv(exe: Path, model: Path, prompt: str,
                       n_tokens: int = 64, k: Optional[int] = None,
                       kv_compress: bool = False, ctx_size: int = 4096,
                       kv_threshold: float = 0.15) -> dict:
    """Run decode and measure throughput + VRAM."""
    args = [
        str(exe), str(model),
        "--ctx-size", str(ctx_size),
        "--ott-full", "--no-verifier",
        "-p", prompt, "-n", str(n_tokens),
        "--temp", "0",
    ]
    if k is not None:
        args += ["--axex-compress", "--axex-compress-rank", str(k)]
    if kv_compress:
        args += ["--axex-kv", "--axex-kv-threshold", str(kv_threshold)]

    vram_before = get_vram_mb()

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=300,
                              encoding='utf-8', errors='replace')
        stdout = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tok_per_s": 0.0, "vram_delta_mb": 0}
    except Exception as e:
        return {"status": f"error: {e}", "tok_per_s": 0.0, "vram_delta_mb": 0}

    vram_after = get_vram_mb()

    # Parse throughput
    tps = None
    for line in stdout.split("\n"):
        m = re.search(r"([\d.]+)\s*tok(?:ens?)?/s", line)
        if m:
            tps = float(m.group(1))
            break
        m = re.search(r"tps=([\d.]+)", line)
        if m:
            tps = float(m.group(1))
            break

    return {
        "status": "ok" if tps else "parse_error",
        "tok_per_s": tps or 0.0,
        "vram_delta_mb": vram_after - vram_before if vram_before > 0 and vram_after > 0 else 0,
        "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
    }


def compute_kv_cache_size(model_d: int, n_layers: int, n_kv_heads: int,
                          head_dim: int, ctx_len: int, bytes_per_elem: int = 2) -> int:
    """
    Compute KV cache size in bytes.
    KV cache = 2 (K+V)  n_layers  n_kv_heads  head_dim  ctx_len  bytes_per_elem
    """
    return 2 * n_layers * n_kv_heads * head_dim * ctx_len * bytes_per_elem


def main():
    ap = argparse.ArgumentParser(description="KV-Cache Compression at Scale")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/kv_cache_long")
    ap.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--d-model", type=int, default=576,
                    help="Model dimension (576 for SmolLM2, 4096 for Llama-8B)")
    ap.add_argument("--n-layers", type=int, default=30)
    ap.add_argument("--n-kv-heads", type=int, default=9)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--n-tokens", type=int, default=64,
                    help="Decode tokens to generate per run")
    ap.add_argument("--kv-threshold", type=float, default=0.15)
    args = ap.parse_args()

    exe = Path(args.exe) if args.exe else detect_exe()
    if not exe or not exe.exists():
        print("ERROR: geodessical binary not found.", file=sys.stderr)
        sys.exit(2)
    model = Path(args.model)

    contexts = [int(x) for x in args.contexts.split(",")]
    base_text = (
        "The transformer architecture has become the dominant paradigm for "
        "natural language processing tasks. Attention mechanisms allow the model "
        "to weigh the importance of different tokens in the input sequence. "
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"d={args.d_model} L={args.n_layers} KV-heads={args.n_kv_heads} h={args.head_dim}")
    print(f"Contexts: {contexts}")
    print(f"GRC k={args.rank}")
    print()

    csv_path = out_dir / "kv_cache_results.csv"
    results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ctx_len", "kv_cache_est_mb", "kv_compress", "grc_k",
                         "tok_per_s", "vram_delta_mb", "status"])

        for ctx_len in contexts:
            prompt = generate_long_prompt(base_text, ctx_len)
            kv_est_mb = compute_kv_cache_size(
                args.d_model, args.n_layers, args.n_kv_heads,
                args.head_dim, ctx_len
            ) / (1024 * 1024)

            for kv_mode in [False, True]:
                label = f"kv={'on' if kv_mode else 'off'}"
                print(f"  ctx={ctx_len:>5d}  {label:>6s}  "
                      f"kv_est={kv_est_mb:.0f}MB  ...", end=" ", flush=True)

                t0 = time.time()
                result = run_decode_with_kv(
                    exe, model, prompt, args.n_tokens, args.rank,
                    kv_mode, ctx_len, args.kv_threshold,
                )
                elapsed = time.time() - t0

                row = [ctx_len, round(kv_est_mb, 1), kv_mode, args.rank,
                       result["tok_per_s"], result["vram_delta_mb"],
                       result["status"]]
                writer.writerow(row)
                results.append(row)
                f.flush()

                status = "" if result["status"] == "ok" else ""
                print(f"{result['tok_per_s']:.1f} tok/s  "
                      f"vram_Δ={result['vram_delta_mb']}MB  "
                      f"{elapsed:.0f}s  {status}", flush=True)

            time.sleep(10)  # cooldown between contexts

    # Summary
    from collections import defaultdict
    by_ctx_kv = defaultdict(lambda: {"on": [], "off": []})
    for row in results:
        ctx, _est, kv_mode, _k, tps, vram, _st = row
        key = "on" if kv_mode else "off"
        if tps > 0:
            by_ctx_kv[ctx][key].append(tps)

    print("\n=== KV-Cache Results ===")
    print(f"{'ctx':>6s}  {'KV est':>7s}  {'tok/s (off)':>12s}  {'tok/s (on)':>12s}  {'speedup':>8s}")
    for ctx_len in contexts:
        off_tps = sum(by_ctx_kv[ctx_len]["off"]) / max(len(by_ctx_kv[ctx_len]["off"]), 1)
        on_tps = sum(by_ctx_kv[ctx_len]["on"]) / max(len(by_ctx_kv[ctx_len]["on"]), 1)
        kv_est = compute_kv_cache_size(
            args.d_model, args.n_layers, args.n_kv_heads, args.head_dim, ctx_len
        ) / (1024 * 1024)
        speedup = on_tps / off_tps if off_tps > 0 else 0.0
        print(f"{ctx_len:>6d}  {kv_est:7.0f}MB  {off_tps:12.1f}  {on_tps:12.1f}  {speedup:8.3f}")

    summary = {
        "model_d": args.d_model, "n_layers": args.n_layers,
        "n_kv_heads": args.n_kv_heads, "head_dim": args.head_dim,
        "grc_k": args.rank,
        "contexts": contexts,
    }
    with open(out_dir / "kv_cache_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {csv_path}")


if __name__ == "__main__":
    main()
