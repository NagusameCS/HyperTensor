#!/usr/bin/env python3
"""
QUEUED EXPERIMENT RUNNER.
Executes all pending GPU experiments on local RTX 4070 or EC2 L40S.

P1: PPL validation at predicted k_safe on 2 model sizes
P2 Phase 2-4: FFN activation-weighted compression + LoRA distillation
F2: GSM8K separate benchmark on Llama-8B
F3: HumanEval bimodal benchmark

Usage:
  python scripts/queued_experiments.py --gpu local  # RTX 4070
  python scripts/queued_experiments.py --gpu ec2    # L40S via SSH
"""

import json, os, subprocess, sys, time
from pathlib import Path

QUEUE = [
    {
        "id": "P1",
        "name": "PPL validation at predicted k_safe",
        "script": "scripts/pivot_p1_safe_kd_calibration.py",
        "args": ["--validate-ppl", "--models", "SmolLM2-360M,Gemma-4-2B"],
        "timeout_min": 120,
        "needs_gpu": True,
        "needs_ec2": False,  # Can run on RTX 4070
    },
    {
        "id": "P2_PHASE2",
        "name": "FFN activation collection",
        "script": "scripts/pivot_p2_ffn_activation_weighted.py",
        "args": ["--phase", "2", "--model", "SmolLM2-135M"],
        "timeout_min": 60,
        "needs_gpu": True,
        "needs_ec2": False,
    },
    {
        "id": "P2_PHASE3",
        "name": "FFN weighted compression PPL",
        "script": "scripts/pivot_p2_ffn_activation_weighted.py",
        "args": ["--phase", "3", "--model", "SmolLM2-135M"],
        "timeout_min": 120,
        "needs_gpu": True,
        "needs_ec2": False,
    },
    {
        "id": "P2_PHASE4",
        "name": "FFN LoRA distillation",
        "script": "scripts/pivot_p2_ffn_activation_weighted.py",
        "args": ["--phase", "4", "--model", "SmolLM2-135M"],
        "timeout_min": 180,
        "needs_gpu": True,
        "needs_ec2": True,  # Needs L40S for BF16 training
    },
    {
        "id": "F2",
        "name": "GSM8K separate benchmark on Llama-8B",
        "script": "scripts/experiment_f1_task_benchmarks.py",
        "args": ["--benchmark", "gsm8k", "--model", "Llama-3.1-8B", "--ranks", "256,512,1024,1536,full"],
        "timeout_min": 360,
        "needs_gpu": True,
        "needs_ec2": True,  # Llama-8B needs 24GB+ VRAM
    },
    {
        "id": "F3",
        "name": "HumanEval bimodal benchmark",
        "script": "scripts/experiment_f1_task_benchmarks.py",
        "args": ["--benchmark", "humaneval", "--model", "Llama-3.1-8B", "--ranks", "256,512,1024,1536,full"],
        "timeout_min": 480,
        "needs_gpu": True,
        "needs_ec2": True,
    },
]

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "local"
    
    print("=" * 70)
    print(f"QUEUED EXPERIMENT RUNNER --- target: {target}")
    print(f"  {len(QUEUE)} experiments queued")
    print("=" * 70)
    
    for i, exp in enumerate(QUEUE):
        print(f"\n[{i+1}/{len(QUEUE)}] {exp['id']}: {exp['name']}")
        print(f"  Script: {exp['script']}")
        print(f"  Args: {' '.join(exp['args'])}")
        print(f"  GPU: {'EC2 L40S' if exp['needs_ec2'] else 'RTX 4070'}")
        print(f"  ETA: ~{exp['timeout_min']} min")
        
        if target == "ec2" and exp['needs_ec2']:
            print(f"  -> Queued for EC2 execution")
        elif target == "local" and not exp['needs_ec2']:
            print(f"  -> Ready for local execution (GPU free?)")
        elif target == "local" and exp['needs_ec2']:
            print(f"   Needs EC2 --- skipped for local")
    
    # Save queue
    out = Path("benchmarks/queued_experiments.json")
    with open(out, 'w') as f:
        json.dump(QUEUE, f, indent=2)
    
    print(f"\nQueue saved: {out}")
    print("\nTo execute locally: python scripts/queued_experiments.py local --run")
    print("To execute on EC2:   python scripts/queued_experiments.py ec2 --run")

if __name__ == '__main__':
    main()
