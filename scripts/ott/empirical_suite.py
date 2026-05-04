#!/usr/bin/env python3
"""
scripts/ott/empirical_suite.py  ---  Comprehensive OTT Empirical Evidence Suite
==============================================================================
Papers C + D empirical anchor generator.

Runs geodessical.exe in three modes across all available models and collects:

  Mode A  --- Baseline autoregressive (no OTT)
  Mode B  --- OTT speculative (--ott-full --ott-speculative)
  Mode C  --- OTT speculative + GRC compression (--axex-compress + spec)

Measures per prompt:
  - tok/s (verifier throughput)
  - acceptance_rate (α)
  - geo_accepted vs xfmr_accepted counts  <- geodesic hit rate
  - od_drafts count
  - speedup vs autoregressive baseline

Grid: thresholds  batch sizes (same as calibration_sweep but for all models).

Outputs:
  benchmarks/ott_empirical/<model_tag>/
    raw.csv           --- one row per (mode, thresh, batch, prompt, rep)
    summary.json      --- per-mode aggregate: mean α, mean tok/s, CI
    speedup_table.md  --- human-readable for Paper C §results
    rejection_log.tsv --- Type I/II/III breakdown (requires --ott-rejection-log)

Usage:
    python scripts/ott/empirical_suite.py [--models smollm2 8b] [--reps 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# geodessical2.exe: proper decode output, full OTT+axex support, no auto-PPL mode
EXE  = ROOT / "build_host" / "geodessical2.exe"

#  Model registry 
MODELS = {
    "smollm2": {
        "path": ROOT / "models" / "smollm2-135m-instruct-q8_0.gguf",
        "tag": "SmolLM2-135M",
        "batch_default": 4,
    },
    "llama8b": {
        "path": Path(r"C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF"
                     r"\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067"
                     r"\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        "tag": "Llama-3.1-8B",
        "batch_default": 2,
    },
}

#  Locked validation prompts 
PROMPTS = [
    "Explain the water cycle in three sentences.",
    "What is the capital of France and why is it culturally significant?",
    "Write a Python function that checks if a number is prime.",
    "Describe how a transformer decoder generates tokens step by step.",
    "Summarise Newton's three laws of motion in plain language.",
    "How does gradient descent work in neural network training?",
    "Compare HTTP/1.1 and HTTP/2 in terms of connection multiplexing.",
    "What is the cosmic microwave background radiation?",
    "Explain why quicksort has O(n log n) average-case complexity.",
    "What causes the northern lights (aurora borealis)?",
]

N_TOKENS = 64
CTX_SIZE = 512

#  Sweep grids 
# Targeted grid: best-known operating point from attres calibration
THRESH_GRID = [0.45]
BATCH_GRID  = [4]

# Regex patterns
# Match generation tok/s from "[GD] N tokens in ... (X tok/s)" line specifically,
# falling back to tps=N (PPL mode) or general tok/s if that line is absent.
RE_TOKS  = re.compile(r"(?:tps=([\d.]+)|\[GD\]\s+\d+\s+tokens\s+in\s+\d+\s+ms\s+\(([\d.]+)\s+tok/s\))")
RE_SPEC  = re.compile(
    r"\[SPEC\]\s+Done:\s+(\d+)\s+tokens\s+"
    r"\(geo_accepted=(\d+)\s+xfmr=(\d+)\s+od_drafts=(\d+)"
    r".*?acceptance_rate=([\d.]+)%"
)

OUT_BASE = ROOT / "benchmarks" / "ott_empirical"


@dataclass
class RunResult:
    model_tag: str
    mode: str       # "baseline", "spec", "spec_grc"
    thresh: float
    batch: int
    prompt_idx: int
    rep: int
    tok_s: float | None
    acceptance_rate: float | None
    geo_accepted: int
    xfmr_accepted: int
    od_drafts: int
    total_tokens: int
    exit_code: int


def run_once(model_path: Path, extra_args: list[str], prompt: str) -> dict:
    cmd = [
        str(EXE), str(model_path),
        "--ctx-size", str(CTX_SIZE),
        "-p", prompt,
        "-n", str(N_TOKENS),
        "--temp", "0",
    ] + extra_args

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600
        )
        text = (r.stdout or "") + "\n" + (r.stderr or "")
        tok_m = RE_TOKS.search(text)
        spec_m = RE_SPEC.search(text)
        def _extract_toks(m):
            if not m:
                return None
            return float(m.group(1) if m.group(1) else m.group(2))
        return {
            "tok_s":           _extract_toks(tok_m),
            "acceptance_rate": float(spec_m.group(5)) if spec_m else None,
            "geo_accepted":    int(spec_m.group(2)) if spec_m else 0,
            "xfmr_accepted":   int(spec_m.group(3)) if spec_m else 0,
            "od_drafts":       int(spec_m.group(4)) if spec_m else 0,
            "total_tokens":    int(spec_m.group(1)) if spec_m else 0,
            "exit_code":       r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"tok_s": None, "acceptance_rate": None, "geo_accepted": 0,
                "xfmr_accepted": 0, "od_drafts": 0, "total_tokens": 0, "exit_code": -1}


def mean_ci(vals: list[float], z: float = 1.96) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, 0.0, 0.0
    sd = statistics.stdev(vals)
    ci = z * sd / (len(vals) ** 0.5)
    return m, sd, ci


def run_model(model_key: str, model_cfg: dict, reps: int, out_dir: Path,
              csv_writer: "csv.DictWriter") -> list[RunResult]:
    model_path = model_cfg["path"]
    tag        = model_cfg["tag"]
    batch_def  = model_cfg["batch_default"]

    if not model_path.exists():
        print(f"[ott] SKIP {tag}: model file not found at {model_path}", file=sys.stderr)
        return []

    results: list[RunResult] = []

    #  Mode A: baseline autoregressive 
    print(f"\n[ott] {tag}  mode=BASELINE ({len(PROMPTS)} prompts  {reps} reps)")
    for pi, prompt in enumerate(PROMPTS):
        for rep in range(reps):
            r = run_once(model_path, [], prompt)
            row = RunResult(
                model_tag=tag, mode="baseline", thresh=0.0, batch=0,
                prompt_idx=pi, rep=rep,
                tok_s=r["tok_s"], acceptance_rate=None,
                geo_accepted=0, xfmr_accepted=0, od_drafts=0,
                total_tokens=r["total_tokens"], exit_code=r["exit_code"]
            )
            results.append(row)
            _write_csv(csv_writer, row)
            if r["tok_s"]:
                print(f"  baseline p={pi} r={rep}  {r['tok_s']:.1f} tok/s")

    #  Mode B: OTT speculative, best-known thresh/batch first 
    # Full grid search to map α vs tok/s Pareto frontier
    print(f"\n[ott] {tag}  mode=SPEC  grid {len(THRESH_GRID)}{len(BATCH_GRID)}")
    for thresh in THRESH_GRID:
        for batch in BATCH_GRID:
            spec_args = [
                "--ott-full", "--ott-speculative",
                "--ott-spec-batch", str(batch),
                "--ott-spec-thresh", str(thresh),
            ]
            for pi, prompt in enumerate(PROMPTS):
                for rep in range(reps):
                    r = run_once(model_path, spec_args, prompt)
                    row = RunResult(
                        model_tag=tag, mode="spec",
                        thresh=thresh, batch=batch,
                        prompt_idx=pi, rep=rep,
                        tok_s=r["tok_s"],
                        acceptance_rate=r["acceptance_rate"],
                        geo_accepted=r["geo_accepted"],
                        xfmr_accepted=r["xfmr_accepted"],
                        od_drafts=r["od_drafts"],
                        total_tokens=r["total_tokens"],
                        exit_code=r["exit_code"]
                    )
                    results.append(row)
                    _write_csv(csv_writer, row)
                    if r["tok_s"] and r["acceptance_rate"] is not None:
                        geo_rate = (r["geo_accepted"] / max(r["total_tokens"], 1)) * 100
                        print(f"  spec t={thresh} b={batch} p={pi}  "
                              f"{r['tok_s']:.1f} tok/s  "
                              f"α={r['acceptance_rate']:.1f}%  "
                              f"geo={geo_rate:.1f}%")

    #  Mode C: spec + GRC compression 
    # Only run at best-known settings to isolate GRC  spec interaction
    print(f"\n[ott] {tag}  mode=SPEC+GRC  thresh=0.45 batch={batch_def}")
    grc_spec_args = [
        "--ott-full", "--ott-speculative",
        "--ott-spec-batch", str(batch_def),
        "--ott-spec-thresh", "0.45",
        "--axex-compress", "--axex-attn-svd",
        "--axex-compress-rank", "1024",
        "--axex-compress-max-err", "0",
    ]
    for pi, prompt in enumerate(PROMPTS):
        for rep in range(reps):
            r = run_once(model_path, grc_spec_args, prompt)
            row = RunResult(
                model_tag=tag, mode="spec_grc",
                thresh=0.45, batch=batch_def,
                prompt_idx=pi, rep=rep,
                tok_s=r["tok_s"],
                acceptance_rate=r["acceptance_rate"],
                geo_accepted=r["geo_accepted"],
                xfmr_accepted=r["xfmr_accepted"],
                od_drafts=r["od_drafts"],
                total_tokens=r["total_tokens"],
                exit_code=r["exit_code"]
            )
            results.append(row)
            _write_csv(csv_writer, row)
            if r["tok_s"]:
                print(f"  spec+grc p={pi} r={rep}  {r['tok_s']:.1f} tok/s  "
                      f"α={r['acceptance_rate']}")

    return results


def _write_csv(w: "csv.DictWriter", row: RunResult) -> None:
    w.writerow({
        "model":           row.model_tag,
        "mode":            row.mode,
        "thresh":          row.thresh,
        "batch":           row.batch,
        "prompt_idx":      row.prompt_idx,
        "rep":             row.rep,
        "tok_s":           row.tok_s if row.tok_s is not None else "",
        "acceptance_rate": row.acceptance_rate if row.acceptance_rate is not None else "",
        "geo_accepted":    row.geo_accepted,
        "xfmr_accepted":   row.xfmr_accepted,
        "od_drafts":       row.od_drafts,
        "total_tokens":    row.total_tokens,
        "exit_code":       row.exit_code,
    })


def build_summary(results: list[RunResult]) -> dict:
    from collections import defaultdict
    agg: dict[tuple, list] = defaultdict(lambda: {"toks": [], "alphas": [], "geo_fracs": []})

    for r in results:
        key = (r.model_tag, r.mode, r.thresh, r.batch)
        if r.tok_s is not None:
            agg[key]["toks"].append(r.tok_s)
        if r.acceptance_rate is not None:
            agg[key]["alphas"].append(r.acceptance_rate)
        if r.total_tokens > 0:
            agg[key]["geo_fracs"].append(r.geo_accepted / r.total_tokens)

    summary = []
    for (model, mode, thresh, batch), v in sorted(agg.items()):
        m_tok, sd_tok, ci_tok = mean_ci(v["toks"])
        m_alpha, sd_alpha, ci_alpha = mean_ci(v["alphas"])
        m_geo, _, _ = mean_ci(v["geo_fracs"])
        summary.append({
            "model": model, "mode": mode,
            "thresh": thresh, "batch": batch,
            "n": len(v["toks"]),
            "mean_tok_s": round(m_tok, 2),
            "sd_tok_s": round(sd_tok, 2),
            "ci95_tok_s": round(ci_tok, 2),
            "mean_alpha_pct": round(m_alpha, 2),
            "sd_alpha_pct": round(sd_alpha, 2),
            "ci95_alpha_pct": round(ci_alpha, 2),
            "mean_geo_frac": round(m_geo, 4),
        })
    return {"entries": summary}


def build_speedup_md(summary: dict, baseline_by_model: dict[str, float]) -> str:
    lines = [
        "# OTT Speculative Decode --- Empirical Speedup Table",
        "",
        "Paper C empirical anchor data
        "Rows show mean tok/s ± 95% CI and acceptance rate α across 10 locked prompts.",
        "Speedup = mean_tok_s / baseline_tok_s for the same model.",
        "",
        "| Model | Mode | thresh | batch | tok/s | ±CI | α (%) | ±CI | geo_frac | Speedup |",
        "|-------|------|--------|-------|-------|-----|-------|-----|----------|---------|",
    ]

    for e in summary["entries"]:
        model   = e["model"]
        mode    = e["mode"]
        base    = baseline_by_model.get(model, 1.0)
        speedup = round(e["mean_tok_s"] / base, 3) if base > 0 else "---"
        alpha   = f"{e['mean_alpha_pct']:.1f}" if e["mean_alpha_pct"] else "---"
        ci_a    = f"{e['ci95_alpha_pct']:.1f}" if e["ci95_alpha_pct"] else "---"
        geo     = f"{e['mean_geo_frac']*100:.1f}%" if e["mean_geo_frac"] else "---"
        lines.append(
            f"| {model} | {mode} | {e['thresh']} | {e['batch']} "
            f"| {e['mean_tok_s']} | ±{e['ci95_tok_s']} "
            f"| {alpha} | ±{ci_a} | {geo} | {speedup} |"
        )

    lines += [
        "",
        "## Key observations",
        "",
        "- Geodesic hit rate (`geo_frac`) shows what fraction of accepted tokens came",
        "  from the Riemannian geodesic draft vs. the transformer verifier correction path.",
        "- Speedup > 1.0 confirms the speculative path outperforms autoregressive decode",
        "  on this hardware. Speedup < 1.0 means the verifier overhead dominates.",
        "- spec_grc rows test whether GRC compression at k=1024 affects alpha.",
        "  Significant alphalpha drop would indicate the compressed attention manifold diverges",
        "  from the uncompressed verifier's predicted distribution.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["smollm2"],
                    choices=list(MODELS.keys()),
                    help="Which models to run (default: smollm2)")
    ap.add_argument("--reps", type=int, default=1,
                    help="Repetitions per cell (default 1; use 3 for publishable CI)")
    ap.add_argument("--out-dir", default=str(OUT_BASE),
                    help="Output root directory")
    args = ap.parse_args()

    if not EXE.exists():
        print(f"[ott] ERROR: geodessical.exe not found at {EXE}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "raw.csv"
    fieldnames = ["model","mode","thresh","batch","prompt_idx","rep",
                  "tok_s","acceptance_rate","geo_accepted","xfmr_accepted",
                  "od_drafts","total_tokens","exit_code"]

    all_results: list[RunResult] = []

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for mkey in args.models:
            if mkey not in MODELS:
                print(f"[ott] Unknown model key: {mkey}", file=sys.stderr)
                continue
            model_dir = out_dir / mkey
            model_dir.mkdir(exist_ok=True)
            rows = run_model(mkey, MODELS[mkey], args.reps, model_dir, writer)
            all_results.extend(rows)

    summary = build_summary(all_results)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # build baseline lookup
    from collections import defaultdict
    base_toks: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        if r.mode == "baseline" and r.tok_s is not None:
            base_toks[r.model_tag].append(r.tok_s)
    baseline_by_model = {m: statistics.mean(v) for m, v in base_toks.items() if v}

    md = build_speedup_md(summary, baseline_by_model)
    md_path = out_dir / "speedup_table.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\n[ott] Complete.")
    print(f"  raw.csv       -> {csv_path}")
    print(f"  summary.json  -> {summary_path}")
    print(f"  speedup_table -> {md_path}")

    # Print headline results
    for e in summary["entries"]:
        if e["mode"] in ("spec", "spec_grc") and e["thresh"] == 0.45:
            base = baseline_by_model.get(e["model"], 1.0)
            su = round(e["mean_tok_s"] / base, 3) if base > 0 else "?"
            print(f"  [{e['model']}] mode={e['mode']} batch={e['batch']}  "
                  f"{e['mean_tok_s']} tok/s  α={e['mean_alpha_pct']}%  speedup={su}")


if __name__ == "__main__":
    main()
