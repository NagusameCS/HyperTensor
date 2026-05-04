#!/usr/bin/env python3
"""Quick compressed throughput sweep for AttnRes --- minimal and robust."""
import subprocess, re, sys, time, json
from pathlib import Path

EXE = "build_host/geodessical2.exe"
MODEL = "models/smollm2-135m-instruct-q8_0.gguf"
OUT = Path("benchmarks/attnres_quick")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Explain how a transformer attention mechanism works.",
    "Write a Python function that sorts a list using merge sort.",
    "What is the difference between TCP and UDP?",
]
K_VALUES = [144, 201, 259, 316, 374]  # 0.25, 0.35, 0.45, 0.55, 0.65  576

results = []

for k in K_VALUES:
    for attnres in [False, True]:
        for pi, prompt in enumerate(PROMPTS):
            args = [EXE, MODEL, "--ctx-size", "512", "--ott-full", "--no-verifier",
                    "--axex-compress", "--axex-compress-rank", str(k),
                    "-p", prompt, "-n", "64", "--temp", "0"]
            if attnres:
                args += ["--attnres", "--attnres-strength", "0.35"]

            label = f"k={k}_attnres={attnres}_p={pi}"
            print(f"  {label}...", end=" ", flush=True)
            t0 = time.time()
            try:
                proc = subprocess.run(args, capture_output=True, text=True,
                                      timeout=180, encoding='utf-8', errors='replace')
                stdout = proc.stdout + proc.stderr
                elapsed = time.time() - t0
                tps = None
                for line in stdout.split('\n'):
                    m = re.search(r"([\d.]+)\s*tok/s", line)
                    if m and tps is None:
                        tps = float(m.group(1))
                    m = re.search(r"TPS=([\d.]+)", line)
                    if m and tps is None:
                        tps = float(m.group(1))
                if tps:
                    results.append({"k": k, "attnres": attnres, "prompt": pi,
                                    "tok_per_s": round(tps, 1), "elapsed_s": round(elapsed, 1)})
                    print(f"{tps:.1f} tok/s ({elapsed:.0f}s)", flush=True)
                else:
                    results.append({"k": k, "attnres": attnres, "prompt": pi,
                                    "tok_per_s": 0, "elapsed_s": round(elapsed, 1), "error": "no tps"})
                    print(f"FAIL (no TPS, {elapsed:.0f}s)", flush=True)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT", flush=True)
                results.append({"k": k, "attnres": attnres, "prompt": pi, "tok_per_s": 0, "error": "timeout"})
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                results.append({"k": k, "attnres": attnres, "prompt": pi, "tok_per_s": 0, "error": str(e)})

# Summary
from collections import defaultdict
by_cond = defaultdict(list)
for r in results:
    if r.get("tok_per_s", 0) > 0:
        by_cond[(r["k"], r["attnres"])].append(r["tok_per_s"])

print("\n=== Results ===")
print(f"{'k':>5s} {'k/d':>6s} {'AttnRes':>8s} {'Mean tok/s':>12s} {'n':>4s}")
import statistics
for k in K_VALUES:
    for attnres in [False, True]:
        vals = by_cond.get((k, attnres), [])
        mean_val = statistics.mean(vals) if vals else 0
        print(f"{k:5d} {k/576:6.2f} {str(attnres):>8s} {mean_val:12.1f} {len(vals):4d}")

with open(OUT / "attnres_quick.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[done] {OUT / 'attnres_quick.json'}")
