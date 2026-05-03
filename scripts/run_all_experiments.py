#!/usr/bin/env python3
"""
OVERNIGHT EXPERIMENT RUNNER — Sequential GPU experiment execution.
Runs ALL pending experiments in optimal order after Model M+L complete.

Order: A1 (PPL vs k) → G2 (FFN cluster PPL) → E2 (distillation PPL) → F1 (task benchmarks)

Usage: python scripts/run_all_experiments.py
"""

import subprocess, sys, time, json
from pathlib import Path
from datetime import datetime

EXPERIMENTS = [
    {"id": "A1", "script": "scripts/experiment_a1_ppl_vs_k.py", 
     "desc": "PPL vs GRC compression rank (k=32-576)", "est_min": 120},
    {"id": "G2", "script": "scripts/experiment_g2_ffn_cluster_ppl.py",
     "desc": "FFN cluster compression PPL (C=2/4/8, k_frac=0.25/0.50/0.75)", "est_min": 120},
    {"id": "E2", "script": "scripts/experiment_e2_distill_ppl.py",
     "desc": "GRC Light Distillation PPL recovery (k=256/512/1024)", "est_min": 180},
    {"id": "F1", "script": "scripts/experiment_f1_task_benchmarks.py",
     "desc": "Task-level asymmetric degradation (MMLU+GSM8K)", "est_min": 180},
]

def main():
    print("=" * 70)
    print("OVERNIGHT EXPERIMENT RUNNER")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    total_est = sum(e["est_min"] for e in EXPERIMENTS)
    print(f"Est. total: {total_est} min (~{total_est/60:.1f} hrs)")
    print("=" * 70)
    
    results = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(EXPERIMENTS)}] {exp['id']}: {exp['desc']}")
        print(f"  Script: {exp['script']}")
        print(f"  Est: {exp['est_min']} min")
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
        
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, exp["script"]],
            capture_output=True, text=True, timeout=exp["est_min"] * 120
        )
        elapsed = time.perf_counter() - t0
        
        success = proc.returncode == 0
        result = {
            "id": exp["id"],
            "success": success,
            "elapsed_min": round(elapsed / 60, 1),
            "stdout_last_5": proc.stdout.split('\n')[-5:] if proc.stdout else [],
            "stderr_last_5": proc.stderr.split('\n')[-5:] if proc.stderr else [],
        }
        results.append(result)
        
        status = "" if success else f" (exit {proc.returncode})"
        print(f"  Done: {status} in {elapsed/60:.1f} min")
        
        if not success:
            print(f"  STDERR: {proc.stderr[-500:]}")
    
    # Summary
    print(f"\n{'='*70}")
    print("RUNNER COMPLETE")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success_count = sum(1 for r in results if r["success"])
    print(f"Results: {success_count}/{len(EXPERIMENTS)} passed")
    
    for r in results:
        icon = "" if r["success"] else ""
        print(f"  {icon} {r['id']}: {r['elapsed_min']} min")
    
    Path("benchmarks/runner_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: benchmarks/runner_results.json")

if __name__ == "__main__":
    main()
