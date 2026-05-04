#!/usr/bin/env python3
"""
POST-L SEQUENTIAL RUNNER --- Automatically executes after Model L completes.
1. Export Model L to Ollama as HORI
2. Run CECI cross-model splice at k=128
3. Run remaining GPU experiments (A1 if not done, etc.)
"""

import subprocess, sys, time, json
from pathlib import Path

STEPS = [
    {
        "name": "Export Model L -> Ollama (HORI)",
        "cmd": [sys.executable, "scripts/export_to_ollama.py",
                "--model", "outputs/pure_models/smollm2-135m-language-pure/final",
                "--name", "HORI", "--skill", "language",
                "--output", "outputs/ollama/language-pure"],
        "timeout_min": 15,
        "critical": False,  # Non-critical --- CECI splice can proceed without Ollama
    },
    {
        "name": "CECI Cross-Model Splice (k=128)",
        "cmd": [sys.executable, "scripts/ceci_cross_model.py",
                "--math", "outputs/pure_models/smollm2-135m-math-pure/final",
                "--language", "outputs/pure_models/smollm2-135m-language-pure/final",
                "--k", "128", "--out", "benchmarks/ceci_final_cross_model"],
        "timeout_min": 30,
        "critical": True,
    },
    {
        "name": "CECI Cross-Model Splice (k=256) --- verification",
        "cmd": [sys.executable, "scripts/ceci_cross_model.py",
                "--math", "outputs/pure_models/smollm2-135m-math-pure/final",
                "--language", "outputs/pure_models/smollm2-135m-language-pure/final",
                "--k", "256", "--out", "benchmarks/ceci_final_cross_model_k256"],
        "timeout_min": 30,
        "critical": False,
    },
]

def main():
    print("=" * 70)
    print("POST-L SEQUENTIAL RUNNER")
    print(f"  {len(STEPS)} steps queued")
    print("=" * 70)
    
    results = []
    for i, step in enumerate(STEPS):
        print(f"\n[{i+1}/{len(STEPS)}] {step['name']}")
        print(f"  Timeout: {step['timeout_min']} min")
        t0 = time.perf_counter()
        
        try:
            proc = subprocess.run(
                step["cmd"],
                capture_output=True, text=True,
                timeout=step["timeout_min"] * 60
            )
            elapsed = time.perf_counter() - t0
            ok = proc.returncode == 0
            
            results.append({
                "step": step["name"],
                "success": ok,
                "elapsed_min": round(elapsed / 60, 1),
                "stdout_tail": proc.stdout.split('\n')[-10:] if proc.stdout else [],
            })
            
            status = "" if ok else f" (exit {proc.returncode})"
            print(f"  {status} in {elapsed/60:.1f} min")
            
            if not ok and step["critical"]:
                print(f"  CRITICAL STEP FAILED --- stopping pipeline")
                break
        except subprocess.TimeoutExpired:
            results.append({"step": step["name"], "success": False, "error": "timeout"})
            print(f"  ⏰ TIMEOUT after {step['timeout_min']} min")
    
    # Summary
    print(f"\n{'='*70}")
    ok = sum(1 for r in results if r.get("success"))
    print(f"POST-L PIPELINE COMPLETE: {ok}/{len(results)} succeeded")
    for r in results:
        icon = "" if r.get("success") else ""
        print(f"  {icon} {r['step']}")
    
    Path("benchmarks/post_l_results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
