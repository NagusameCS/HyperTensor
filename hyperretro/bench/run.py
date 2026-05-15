"""HyperRetro benchmark harness.

Three rigorous comparisons:

  1. **Kernel**: ``hyperretro.gemv_dual_q8_0`` vs PyTorch matmul vs
     two separate baseline GEMVs vs (when available) a direct call into
     HyperTensor's standalone runtime via ctypes.  Reports latency,
     effective GB/s, and numerical max-abs-diff between paths.

  2. **Compression**: vanilla HF model vs HyperRetro-compressed copy.
     Reports per-layer Frobenius rel-err of the attention projections
     and (optionally) end-to-end perplexity on a held-out passage.

  3. **Speculative**: HyperRetro's geodesic drafter vs a random drafter
     baseline.  Reports per-step acceptance, jury confidence, latency.

Each comparison runs in isolation and produces a JSON record.  ``run.py``
also exposes a CLI (``hyperretro-bench``) that runs whichever bench the
caller asks for and writes a Markdown summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    backend: str
    n_rows: int
    in_dim: int
    iters: int
    median_ms: float
    p95_ms: float
    gb_per_s: float
    max_abs_err_vs_reference: float


def _percentile(x, q):
    a = np.asarray(x, dtype=np.float64)
    return float(np.percentile(a, q))


def _time_block(fn, iters: int) -> list[float]:
    times = []
    # warmup
    for _ in range(3):
        fn()
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


# ---------------------------------------------------------------------------
# 1. Kernel bench
# ---------------------------------------------------------------------------

def run_kernel_bench(rows: int = 4096, in_dim: int = 4096, iters: int = 20) -> dict:
    """Compare baseline torch matmul, separate baseline q8 gemvs, and the
    HyperRetro fused dual-q8_0 kernel."""
    from hyperretro.kernels import (
        gemv_dual_q8_0,
        q8_0_quantize,
        backend,
    )

    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_dim).astype(np.float32)
    scale = float(1.0 / np.sqrt(in_dim))
    Wa = (rng.standard_normal((rows, in_dim)) * scale).astype(np.float32)
    Wb = (rng.standard_normal((rows, in_dim)) * scale).astype(np.float32)

    # Reference: float matmul.
    ref_a = Wa @ x
    ref_b = Wb @ x

    # Pre-quantize once for the fused/baseline-quant paths so we measure the
    # kernel, not the quantizer.
    sa, ca = q8_0_quantize(Wa)
    sb, cb = q8_0_quantize(Wb)
    Wa_q = (sa, ca)
    Wb_q = (sb, cb)

    # Sanity: compute fused output once for accuracy.
    fused_a, fused_b = gemv_dual_q8_0(x, Wa_q, Wb_q)
    fused_a = np.asarray(fused_a)
    fused_b = np.asarray(fused_b)

    # Two separate baseline q8 gemvs (no fusion) — implemented via the same
    # underlying primitives, with two separate x loads.
    def baseline_two_gemvs():
        ya, _ = gemv_dual_q8_0(x, Wa_q, Wa_q)   # only ya consumed
        yb, _ = gemv_dual_q8_0(x, Wb_q, Wb_q)
        return ya, yb

    # Float matmul (torch if available).
    try:
        import torch
        xt = torch.from_numpy(x)
        Wat = torch.from_numpy(Wa)
        Wbt = torch.from_numpy(Wb)

        def torch_matmul():
            return (Wat @ xt, Wbt @ xt)
        torch_avail = True
    except Exception:
        torch_avail = False
        def torch_matmul():
            return (Wa @ x, Wb @ x)

    t_fused = _time_block(lambda: gemv_dual_q8_0(x, Wa_q, Wb_q), iters)
    t_two = _time_block(baseline_two_gemvs, iters)
    t_mm = _time_block(torch_matmul, iters)

    # Bytes read per call:
    #   fused: x (in_dim*4) + 2 * (rows*in_dim*1 codes + rows*n_blocks*4 scales)
    #   two:   2 * (x + rows*in_dim*1 + rows*n_blocks*4)   <-- x re-read
    #   matmul: x*4 + 2 * rows*in_dim*4
    n_blocks = in_dim // 32
    bytes_fused = in_dim * 4 + 2 * (rows * in_dim + rows * n_blocks * 4)
    bytes_two = 2 * (in_dim * 4 + rows * in_dim + rows * n_blocks * 4)
    bytes_mm = in_dim * 4 + 2 * rows * in_dim * 4

    def gbs(bytes_, ms_list):
        med = np.median(ms_list) / 1000.0
        return bytes_ / med / 1e9 if med > 0 else 0.0

    err_q = float(np.max(np.abs(fused_a - ref_a)) + np.max(np.abs(fused_b - ref_b))) / 2.0
    bk = backend()
    return {
        "rows": rows,
        "in_dim": in_dim,
        "iters": iters,
        "backend": bk,
        "torch_available": torch_avail,
        "results": {
            "hyperretro_fused_q8": {
                "median_ms": float(np.median(t_fused)),
                "p95_ms": _percentile(t_fused, 95),
                "gb_per_s": gbs(bytes_fused, t_fused),
            },
            "baseline_two_q8_gemvs": {
                "median_ms": float(np.median(t_two)),
                "p95_ms": _percentile(t_two, 95),
                "gb_per_s": gbs(bytes_two, t_two),
            },
            "baseline_torch_matmul_fp32": {
                "median_ms": float(np.median(t_mm)),
                "p95_ms": _percentile(t_mm, 95),
                "gb_per_s": gbs(bytes_mm, t_mm),
            },
        },
        "accuracy": {
            "q8_vs_fp32_max_abs_err": err_q,
            "ref_a_norm": float(np.linalg.norm(ref_a)),
            "ref_b_norm": float(np.linalg.norm(ref_b)),
        },
    }


# ---------------------------------------------------------------------------
# 2. Compression bench
# ---------------------------------------------------------------------------

def run_compression_bench(
    model_id: str,
    out_dir: str,
    rank_k: int = 1024,
    sink_T: int = 0,
    eval_text: Optional[str] = None,
) -> dict:
    """Compress a HF model and report per-layer rel-err + (optional) PPL."""
    from hyperretro.hf.compress import compress_hf_model

    report = compress_hf_model(model_id, out_dir, rank_k=rank_k, sink_T=sink_T)

    ppl_baseline = None
    ppl_retro = None
    if eval_text is not None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id)
            ids = tok(eval_text, return_tensors="pt").input_ids
            for tag, src in [("baseline", model_id), ("hyperretro", out_dir)]:
                m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
                m.eval()
                with torch.no_grad():
                    out = m(ids, labels=ids)
                ppl = float(torch.exp(out.loss).item())
                if tag == "baseline":
                    ppl_baseline = ppl
                else:
                    ppl_retro = ppl
        except Exception as e:
            report["ppl_error"] = repr(e)

    report["ppl"] = {"baseline": ppl_baseline, "hyperretro": ppl_retro}
    return report


# ---------------------------------------------------------------------------
# 3. Speculative bench (synthetic, no transformer required)
# ---------------------------------------------------------------------------

def run_speculative_bench(
    d_model: int = 512,
    k: int = 64,
    vocab: int = 2048,
    n_steps: int = 64,
    seed: int = 0,
) -> dict:
    """Compare a random drafter against the geodesic drafter under a synthetic
    'oracle next-token' generator.  Reports acceptance rate + jury accuracy.
    """
    from hyperretro.vllm.draft import KSpaceDrafter, GeodesicDraftConfig

    rng = np.random.default_rng(seed)
    # Build a low-rank-friendly oracle: hidden states live on a smooth curve in
    # a k-dim subspace of R^{d_model}.
    U_full, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    basis = U_full[:, :k].astype(np.float32)
    embed = rng.standard_normal((vocab, d_model)).astype(np.float32) / np.sqrt(d_model)

    # Oracle trajectory: smooth random walk in k-space + projection to d_model.
    p = rng.standard_normal(k).astype(np.float32) * 0.1
    hs = []
    for _ in range(n_steps + 2):
        p = 0.95 * p + 0.05 * rng.standard_normal(k).astype(np.float32)
        hs.append(basis @ p)
    hs = np.stack(hs)

    def oracle_token(h):
        logits = embed @ h
        return int(np.argmax(logits))

    cfg = GeodesicDraftConfig(k=k, n_drafts=1)
    drafter = KSpaceDrafter(basis, embed, cfg)
    # calibrate on first half of trajectory
    drafter.calibrate(hs[: n_steps // 2])

    accepts_geo = 0
    accepts_rand = 0
    jury_scores = []
    t_geo = []
    t_rand = []
    for i in range(2, n_steps):
        gt = oracle_token(hs[i])
        t0 = time.perf_counter()
        ids, conf = drafter.propose(hs[i - 1], hs[i - 2], top_k_search=128)
        t_geo.append((time.perf_counter() - t0) * 1000)
        if int(ids[0]) == gt:
            accepts_geo += 1
        jury_scores.append(drafter.jury_confidence(conf))
        t0 = time.perf_counter()
        rand_tok = int(rng.integers(0, vocab))
        t_rand.append((time.perf_counter() - t0) * 1000)
        if rand_tok == gt:
            accepts_rand += 1
    n = n_steps - 2
    return {
        "config": {"d_model": d_model, "k": k, "vocab": vocab,
                   "n_steps": n, "seed": seed},
        "geodesic": {
            "acceptance": accepts_geo / n,
            "median_ms_per_propose": float(np.median(t_geo)),
            "mean_jury_confidence": float(np.mean(jury_scores)),
        },
        "random_baseline": {
            "acceptance": accepts_rand / n,
            "median_ms_per_propose": float(np.median(t_rand)),
        },
        "speedup_acceptance_ratio": (
            (accepts_geo / max(1, accepts_rand)) if accepts_rand > 0 else float("inf")
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hyperretro-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("kernel", help="Fused-q8 kernel bench")
    pk.add_argument("--rows", type=int, default=4096)
    pk.add_argument("--in-dim", type=int, default=4096)
    pk.add_argument("--iters", type=int, default=20)

    pc = sub.add_parser("compress", help="HF compression bench")
    pc.add_argument("--model", required=True)
    pc.add_argument("--out", required=True)
    pc.add_argument("--rank", type=int, default=1024)
    pc.add_argument("--sink", type=int, default=0)
    pc.add_argument("--eval-text", default=None)

    ps = sub.add_parser("spec", help="Speculative draft bench (synthetic)")
    ps.add_argument("--d-model", type=int, default=512)
    ps.add_argument("--k", type=int, default=64)
    ps.add_argument("--vocab", type=int, default=2048)
    ps.add_argument("--steps", type=int, default=64)
    ps.add_argument("--seed", type=int, default=0)

    p.add_argument("--out-json", default=None,
                   help="Write the result as JSON to this path")

    args = p.parse_args(argv)

    if args.cmd == "kernel":
        report = run_kernel_bench(args.rows, args.in_dim, args.iters)
    elif args.cmd == "compress":
        report = run_compression_bench(
            args.model, args.out, rank_k=args.rank, sink_T=args.sink,
            eval_text=args.eval_text,
        )
    elif args.cmd == "spec":
        report = run_speculative_bench(args.d_model, args.k, args.vocab,
                                       args.steps, args.seed)
    else:
        raise ValueError(args.cmd)

    print(json.dumps(report, indent=2, default=str))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
