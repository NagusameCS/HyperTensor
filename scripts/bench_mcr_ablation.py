"""Llama-8B MCR (Mix-Compress-Refine) ablation for Paper B §7.

Measures decode tok/s for:
  baseline: --axex-compress --axex-pca-k 1024 (uniform rank)
  mcr     : --axex-mcr --axex-pca-k 1024 (per-phase scaled rank)

Both attn-only, skip-O, Q4_K_M, n=64 decode budget, 4 reps each.
Emits docs/figures/paper-b/mcr_ablation.csv and _summary.json.
"""
from __future__ import annotations
import json, re, statistics, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXE  = REPO / "build_host" / "geodessical.exe"
MODEL = Path(r"C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
OUT_DIR = REPO / "docs" / "figures" / "paper-b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Explain the second law of thermodynamics in one paragraph."
N_DECODE = 64
N_REPS = 4

# Match: "[GD] Decode-only: prefill 290.0 ms, 38.6 tok/s"
TOK_RE = re.compile(r"\[GD\]\s+Decode-only:.*?([\d.]+)\s*tok/s", re.IGNORECASE)
# Fallback: any line ending with tok/s
FALLBACK_RE = re.compile(r"([\d.]+)\s+tok/s")

def parse_tps(text: str):
    for line in reversed(text.splitlines()):
        m = TOK_RE.search(line)
        if m:
            return float(m.group(1))
    # fallback: last tok/s number anywhere
    matches = FALLBACK_RE.findall(text)
    if matches:
        # prefer the [GD] Decode-only one if it appears in a [GD] line
        for line in reversed(text.splitlines()):
            if "[GD]" in line and "Decode" in line:
                m = FALLBACK_RE.search(line)
                if m: return float(m.group(1))
        return float(matches[-1])
    return None

def run(extra_args: list[str], tag: str, rep: int) -> dict:
    cmd = [str(EXE), str(MODEL), "-p", PROMPT, "-n", str(N_DECODE), "--temp", "0"] + extra_args
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dur = time.time() - t0
    out = p.stdout + "\n" + p.stderr
    tps = parse_tps(out)
    print(f"  [{tag} rep{rep}] tps={tps} dur={dur:.1f}s rc={p.returncode}")
    return {"tag": tag, "rep": rep, "tps": tps, "duration_s": dur, "returncode": p.returncode}

def main():
    if not EXE.exists():
        print(f"missing {EXE}", file=sys.stderr); sys.exit(1)
    if not MODEL.exists():
        print(f"missing {MODEL}", file=sys.stderr); sys.exit(1)

    runs = []
    base_args = ["--axex-compress", "--axex-compress-rank", "1024", "--axex-skip-o"]
    mcr_args  = ["--axex-mcr",      "--axex-compress-rank", "1024", "--axex-skip-o"]

    # Cold cache warmup (first run populates wproj cache and is discarded)
    print("warmup baseline (cold cache)...")
    run(base_args, "warmup_baseline", 0)
    print("warmup mcr (cold cache)...")
    run(mcr_args, "warmup_mcr", 0)

    for rep in range(1, N_REPS + 1):
        runs.append(run(base_args, "baseline", rep))
    for rep in range(1, N_REPS + 1):
        runs.append(run(mcr_args, "mcr", rep))

    csv = OUT_DIR / "mcr_ablation.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("tag,rep,tps,duration_s,returncode\n")
        for r in runs:
            f.write(f"{r['tag']},{r['rep']},{r['tps']},{r['duration_s']:.2f},{r['returncode']}\n")
    print(f"wrote {csv}")

    def stats(tag):
        ts = [r["tps"] for r in runs if r["tag"] == tag and r["tps"] is not None]
        if not ts: return {"n": 0}
        return {"n": len(ts), "mean": statistics.mean(ts), "stdev": statistics.stdev(ts) if len(ts) > 1 else 0.0, "min": min(ts), "max": max(ts)}

    summary = {
        "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
        "prompt": PROMPT,
        "n_decode": N_DECODE,
        "baseline": stats("baseline"),
        "mcr": stats("mcr"),
    }
    if summary["baseline"]["n"] and summary["mcr"]["n"]:
        summary["mcr_over_baseline"] = summary["mcr"]["mean"] / summary["baseline"]["mean"]
    out_json = OUT_DIR / "mcr_ablation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
