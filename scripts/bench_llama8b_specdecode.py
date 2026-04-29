"""Llama-3.1-8B spec-decode acceptance-rate sweep for Paper C.

Runs `geodessical.exe --ott-speculative` on a fixed prompt set, parses the
[SPEC] Done lines, and emits per-prompt + aggregate acceptance numbers.
"""
import csv
import re
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build_host" / "geodessical.exe"
MODEL = Path(r"C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
OUT = ROOT / "docs" / "figures" / "paper-c"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the autumn moon.",
    "What is the capital of Australia and why was it chosen?",
    "Describe how a transformer decoder generates tokens.",
    "Summarize the plot of Hamlet in three sentences.",
    "How does spaced-repetition learning work?",
    "Compare HTTP/1.1 and HTTP/2 in terms of multiplexing.",
    "What is the cosmic microwave background?",
]
N_TOKENS = 64
SPEC_RE = re.compile(
    r"\[SPEC\] Done:\s+(\d+)\s+tokens\s+\(geo_accepted=(\d+)\s+xfmr=(\d+)\s+od_drafts=(\d+)"
    r".*?acceptance_rate=([\d.]+)%"
)


def run_once(prompt: str) -> dict | None:
    cmd = [
        str(EXE), str(MODEL),
        "--ott-speculative", "--ott-spec-batch", "2", "--ott-spec-thresh", "0.45",
        "--axiom-skip-geodesic",
        "-p", prompt, "-n", str(N_TOKENS), "--temp", "0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    m = SPEC_RE.search(text)
    if not m:
        print(f"  no SPEC line in output", file=sys.stderr)
        print(text[-1000:], file=sys.stderr)
        return None
    tot, geo, xfmr, od, acc = m.groups()
    return {
        "prompt": prompt,
        "tokens": int(tot),
        "geo_accepted": int(geo),
        "xfmr": int(xfmr),
        "od_drafts": int(od),
        "acceptance_rate_pct": float(acc),
    }


def main() -> None:
    results = []
    for i, p in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {p[:50]}...")
        r = run_once(p)
        if r:
            print(f"  α={r['acceptance_rate_pct']:.1f}% (geo={r['geo_accepted']}, xfmr={r['xfmr']})")
            results.append(r)

    csv_path = OUT / "llama8b_specdecode.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "tokens", "geo_accepted", "xfmr", "od_drafts", "acceptance_rate_pct"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    if results:
        accs = [r["acceptance_rate_pct"] for r in results]
        agg = {
            "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
            "n_prompts": len(results),
            "tokens_per_run": N_TOKENS,
            "mean_acceptance_pct": sum(accs) / len(accs),
            "min_acceptance_pct": min(accs),
            "max_acceptance_pct": max(accs),
            "stdev_pct": (sum((a - sum(accs) / len(accs)) ** 2 for a in accs) / len(accs)) ** 0.5,
            "raw": results,
        }
        (OUT / "llama8b_specdecode_summary.json").write_text(json.dumps(agg, indent=2))
        print(f"\nMean α = {agg['mean_acceptance_pct']:.2f}% (min {agg['min_acceptance_pct']:.1f}, max {agg['max_acceptance_pct']:.1f}, σ {agg['stdev_pct']:.2f})")


if __name__ == "__main__":
    main()
