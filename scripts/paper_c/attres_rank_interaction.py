#!/usr/bin/env python3
"""
scripts/paper_c/attres_rank_interaction.py  —  Paper C AttnRes  GRC rank sweep
=================================================================================
The "central empirical unknown" of Paper C §attnres:

  Does GRC attention compression (rank k) change the OTT speculative
  acceptance rate α?

  If α drops significantly at lower k, it means the compressed attention
  manifold diverges from the full-rank verifier's predicted distribution —
  making speculative decode less effective precisely when you most need it
  (on memory-constrained hardware).

Methodology:
  Fix: SmolLM2-135M Q8_0, thresh=0.45, batch=4, 10 locked prompts
  Vary: k ∈ {64, 128, 256, 512, 768, 1024, ∞ (no compress)}
  Measure: α (acceptance rate), tok/s, geo_frac per k

  Each cell: reps runs for statistical stability.
  Output: paired (k, α) table + regression line coefficient for §attnres.

Outputs:
  benchmarks/paper_c_attres/
    attres_raw.csv
    attres_summary.json
    attres_report.md          ← copy-paste into Paper C §attnres
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
EXE   = ROOT / "build_host" / "geodessical2.exe"
MODEL = ROOT / "models" / "smollm2-135m-instruct-q8_0.gguf"
OUT   = ROOT / "benchmarks" / "paper_c_attres"

PROMPTS = [
    "Explain the water cycle in three sentences.",
    "What is the capital of France and why is it famous?",
    "Write a Python function that checks if a number is prime.",
    "Describe how a transformer decoder generates tokens step by step.",
    "Summarise Newton's three laws of motion.",
    "How does gradient descent work in training neural networks?",
    "Compare HTTP/1.1 and HTTP/2 in terms of multiplexing.",
    "What is the cosmic microwave background radiation?",
    "Explain why quicksort has O(n log n) average-case complexity.",
    "What causes the aurora borealis?",
]

K_VALUES = [64, 128, 256, 512, 768, 1024, 0]   # 0 = no compression (baseline)
THRESH   = 0.45
BATCH    = 4
N_TOKENS = 64
CTX_SIZE = 512
REPS     = 2

RE_TOKS = re.compile(r"(?:tps=([\d.]+)|([\d.]+)\s+tok(?:ens)?/s)")
RE_SPEC = re.compile(
    r"\[SPEC\]\s+Done:\s+(\d+)\s+tokens\s+"
    r"\(geo_accepted=(\d+)\s+xfmr=(\d+)\s+od_drafts=(\d+)"
    r".*?acceptance_rate=([\d.]+)%"
)


def run_once(k: int, prompt: str) -> dict:
    base_args = [
        str(EXE), str(MODEL),
        "--ctx-size", str(CTX_SIZE),
        "--ott-full", "--ott-speculative",
        "--ott-spec-batch", str(BATCH),
        "--ott-spec-thresh", str(THRESH),
        "-p", prompt, "-n", str(N_TOKENS), "--temp", "0",
    ]
    if k > 0:
        base_args += [
            "--axex-compress", "--axex-attn-svd",
            "--axex-compress-rank", str(k),
            "--axex-compress-max-err", "0",
        ]
    try:
        r = subprocess.run(base_args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        text = (r.stdout or "") + "\n" + (r.stderr or "")
        tok_m  = RE_TOKS.search(text)
        spec_m = RE_SPEC.search(text)
        return {
            "tok_s":           float(tok_m.group(1) or tok_m.group(2)) if tok_m else None,
            "acceptance_rate": float(spec_m.group(5)) if spec_m else None,
            "geo_accepted":    int(spec_m.group(2)) if spec_m else 0,
            "xfmr_accepted":   int(spec_m.group(3)) if spec_m else 0,
            "total_tokens":    int(spec_m.group(1)) if spec_m else 0,
            "exit_code":       r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"tok_s": None, "acceptance_rate": None, "geo_accepted": 0,
                "xfmr_accepted": 0, "total_tokens": 0, "exit_code": -1}


def main() -> None:
    if not EXE.exists():
        print(f"ERROR: {EXE} not found", file=sys.stderr); sys.exit(1)
    if not MODEL.exists():
        print(f"ERROR: model not found at {MODEL}", file=sys.stderr); sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "attres_raw.csv"

    fieldnames = ["k", "prompt_idx", "rep", "tok_s", "acceptance_rate",
                  "geo_accepted", "xfmr_accepted", "total_tokens", "exit_code"]
    rows_all = []

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for k in K_VALUES:
            label = str(k) if k > 0 else "∞ (no compress)"
            print(f"\n[attres] k={label}  ({len(PROMPTS)} prompts  {REPS} reps)")
            for pi, prompt in enumerate(PROMPTS):
                for rep in range(REPS):
                    r = run_once(k, prompt)
                    row = {"k": k, "prompt_idx": pi, "rep": rep,
                           "tok_s": r["tok_s"], "acceptance_rate": r["acceptance_rate"],
                           "geo_accepted": r["geo_accepted"], "xfmr_accepted": r["xfmr_accepted"],
                           "total_tokens": r["total_tokens"], "exit_code": r["exit_code"]}
                    writer.writerow(row)
                    rows_all.append(row)
                    if r["tok_s"] and r["acceptance_rate"] is not None:
                        print(f"  k={k} p={pi} r={rep}  {r['tok_s']:.1f} tok/s  "
                              f"α={r['acceptance_rate']:.1f}%")

    # aggregate per k
    from collections import defaultdict
    agg: dict[int, dict] = defaultdict(lambda: {"toks": [], "alphas": [], "geo_fracs": []})
    for row in rows_all:
        k = row["k"]
        if row["tok_s"] is not None:
            agg[k]["toks"].append(row["tok_s"])
        if row["acceptance_rate"] is not None:
            agg[k]["alphas"].append(row["acceptance_rate"])
        tot = row["total_tokens"]
        if tot > 0:
            agg[k]["geo_fracs"].append(row["geo_accepted"] / tot)

    summary_rows = []
    for k in K_VALUES:
        v = agg[k]
        n = len(v["toks"])
        if not n:
            continue
        m_tok = statistics.mean(v["toks"])
        sd_tok = statistics.stdev(v["toks"]) if n > 1 else 0.0
        ci_tok = 1.96 * sd_tok / n**0.5 if n > 1 else 0.0
        m_alpha = statistics.mean(v["alphas"]) if v["alphas"] else 0.0
        sd_alpha = statistics.stdev(v["alphas"]) if len(v["alphas"]) > 1 else 0.0
        ci_alpha = 1.96 * sd_alpha / len(v["alphas"])**0.5 if len(v["alphas"]) > 1 else 0.0
        m_geo = statistics.mean(v["geo_fracs"]) if v["geo_fracs"] else 0.0
        summary_rows.append({
            "k": k, "n": n,
            "mean_tok_s": round(m_tok, 2), "sd_tok_s": round(sd_tok, 2), "ci95_tok_s": round(ci_tok, 2),
            "mean_alpha_pct": round(m_alpha, 2), "sd_alpha_pct": round(sd_alpha, 2), "ci95_alpha_pct": round(ci_alpha, 2),
            "mean_geo_frac": round(m_geo, 4),
        })

    summary_path = OUT / "attres_summary.json"
    summary_path.write_text(json.dumps({"model": str(MODEL.name), "thresh": THRESH, "batch": BATCH,
                                         "rows": summary_rows}, indent=2), encoding="utf-8")

    # find no-compression baseline row (k=0)
    base_row = next((r for r in summary_rows if r["k"] == 0), None)
    base_alpha = base_row["mean_alpha_pct"] if base_row else None
    base_toks  = base_row["mean_tok_s"] if base_row else None

    md_lines = [
        "# Paper C — AttnRes  GRC Rank Interaction",
        "",
        f"**Model**: {MODEL.name}  ",
        f"**Settings**: thresh={THRESH}, batch={BATCH}, n_tokens={N_TOKENS}, prompts={len(PROMPTS)}, reps={REPS}",
        "",
        "| k | mean α (%) | ±CI | Δα vs uncompressed | mean tok/s | ±CI | geo_frac |",
        "|---|-----------|-----|-------------------|-----------|-----|----------|",
    ]
    for r in sorted(summary_rows, key=lambda x: x["k"] if x["k"] > 0 else 999999):
        label = str(r["k"]) if r["k"] > 0 else "∞"
        delta_alpha = f"{r['mean_alpha_pct'] - base_alpha:+.2f}" if base_alpha is not None and r["k"] != 0 else "—"
        md_lines.append(
            f"| {label} | {r['mean_alpha_pct']:.2f} | ±{r['ci95_alpha_pct']:.2f} "
            f"| {delta_alpha} | {r['mean_tok_s']} | ±{r['ci95_tok_s']} "
            f"| {r['mean_geo_frac']*100:.1f}% |"
        )

    # simple linear regression of α on k
    valid = [(r["k"], r["mean_alpha_pct"]) for r in summary_rows if r["k"] > 0 and r["mean_alpha_pct"] > 0]
    if len(valid) >= 3:
        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        xm = statistics.mean(xs)
        ym = statistics.mean(ys)
        slope = sum((x - xm)*(y - ym) for x, y in zip(xs, ys)) / sum((x - xm)**2 for x in xs)
        slope_per_1000 = round(slope * 1000, 3)
        md_lines += [
            "",
            f"**Linear regression** (α vs k): slope = {slope:.4f} ppt/rank unit  ",
            f"({slope_per_1000:+.3f} ppt per 1000 rank units)",
            "",
            "Interpretation:",
            f"- Slope ≈ 0 → GRC rank does not significantly affect acceptance rate  ",
            f"- Slope > 0 → higher k (less compression) → higher α (as expected if compressed manifold diverges)  ",
            f"- Slope < 0 → unexpected, would suggest lower-k manifold is *closer* to verifier",
            "",
        ]

    md_path = OUT / "attres_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[attres] Complete.")
    print(f"  raw     -> {csv_path}")
    print(f"  summary -> {summary_path}")
    print(f"  report  -> {md_path}")
    if base_alpha:
        print(f"  Uncompressed baseline α = {base_alpha:.2f}%  tok/s = {base_toks}")
    for r in summary_rows:
        if r["k"] > 0:
            delta = r["mean_alpha_pct"] - (base_alpha or 0)
            print(f"  k={r['k']:5d}  α={r['mean_alpha_pct']:.2f}%  Δα={delta:+.2f}  "
                  f"{r['mean_tok_s']} tok/s")


if __name__ == "__main__":
    main()
