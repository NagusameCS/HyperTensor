"""Llama-8B T_V(k) rank sweep: spec-decode α and throughput at varying compression ranks.

For each k in K_VALUES, runs spec-decode (--ott-speculative) with
--axex-compress --axex-pca-k k --axex-attn-only --axex-skip-o
on a fixed prompt set, measuring acceptance rate and decode tok/s.

Emits docs/figures/paper-c/tv_of_k_sweep.csv and _summary.json.
"""
from __future__ import annotations
import csv, json, re, statistics, subprocess, sys, time
from pathlib import Path

REPO  = Path(__file__).resolve().parents[1]
EXE   = REPO / "build_host" / "geodessical.exe"
MODEL = Path(r"C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
OUT   = REPO / "docs" / "figures" / "paper-c"
OUT.mkdir(parents=True, exist_ok=True)

# Smaller prompt set than spec-decode bench (8) — 4 to keep runtime bounded
# across 6 ranks × 4 prompts × ~30 s/run ≈ 12 min plus cold-cache warmups.
PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "What is the capital of Australia and why was it chosen?",
    "Describe how a transformer decoder generates tokens.",
    "How does spaced-repetition learning work?",
]
K_VALUES = [128, 256, 512, 1024, 1536, 2048]
N_TOKENS = 64

SPEC_RE = re.compile(
    r"\[SPEC\] Done:\s+(\d+)\s+tokens\s+\(geo_accepted=(\d+)\s+xfmr=(\d+)\s+od_drafts=(\d+)"
    r".*?acceptance_rate=([\d.]+)%"
)
TPS_RE = re.compile(r"\[GD\]\s+Decode-only:.*?([\d.]+)\s*tok/s", re.IGNORECASE)
FALLBACK_RE = re.compile(r"([\d.]+)\s+tok/s")

def parse_tps(text: str):
    for line in reversed(text.splitlines()):
        m = TPS_RE.search(line)
        if m: return float(m.group(1))
    for line in reversed(text.splitlines()):
        if "[GD]" in line and "Decode" in line:
            m = FALLBACK_RE.search(line)
            if m: return float(m.group(1))
    matches = FALLBACK_RE.findall(text)
    return float(matches[-1]) if matches else None

def run_once(k: int, prompt: str, warmup: bool = False) -> dict | None:
    cmd = [
        str(EXE), str(MODEL),
        "--axex-compress", "--axex-compress-rank", str(k),
        "--axex-skip-o",
        "--ott-speculative", "--ott-spec-batch", "2", "--ott-spec-thresh", "0.45",
        "--axiom-skip-geodesic",
        "-p", prompt, "-n", str(N_TOKENS), "--temp", "0",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    dur = time.time() - t0
    if warmup:
        return None
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    m = SPEC_RE.search(text)
    tps = parse_tps(text)
    if not m:
        print(f"  k={k} no SPEC line (rc={p.returncode}, dur={dur:.1f}s)")
        return {"k": k, "prompt": prompt, "tps": tps, "duration_s": dur, "returncode": p.returncode}
    tot, geo, xfmr, od, acc = m.groups()
    return {
        "k": k, "prompt": prompt,
        "tokens": int(tot),
        "geo_accepted": int(geo),
        "xfmr": int(xfmr),
        "od_drafts": int(od),
        "acceptance_rate_pct": float(acc),
        "tps": tps,
        "duration_s": dur,
        "returncode": p.returncode,
    }

def main():
    if not EXE.exists():
        print(f"missing {EXE}", file=sys.stderr); sys.exit(1)

    rows = []
    for k in K_VALUES:
        # cold-cache warmup at this k (populates wproj cache for the rank)
        print(f"warmup k={k}…")
        run_once(k, PROMPTS[0], warmup=True)
        for prompt in PROMPTS:
            r = run_once(k, prompt)
            if r:
                rows.append(r)
                a = r.get("acceptance_rate_pct")
                t = r.get("tps")
                print(f"  k={k}  α={a}%  tps={t}")

    csv_path = OUT / "tv_of_k_sweep.csv"
    fields = ["k","prompt","tokens","geo_accepted","xfmr","od_drafts","acceptance_rate_pct","tps","duration_s","returncode"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {csv_path}")

    summary = {"model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M", "k_values": K_VALUES, "per_k": {}}
    for k in K_VALUES:
        a = [r["acceptance_rate_pct"] for r in rows if r["k"] == k and r.get("acceptance_rate_pct") is not None]
        t = [r["tps"] for r in rows if r["k"] == k and r.get("tps") is not None]
        summary["per_k"][str(k)] = {
            "n": len(a),
            "alpha_mean": (sum(a)/len(a)) if a else None,
            "alpha_stdev": (statistics.stdev(a) if len(a) > 1 else 0.0) if a else None,
            "tps_mean": (sum(t)/len(t)) if t else None,
            "tps_stdev": (statistics.stdev(t) if len(t) > 1 else 0.0) if t else None,
        }
    out_json = OUT / "tv_of_k_sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
