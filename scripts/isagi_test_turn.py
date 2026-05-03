"""Quick single-turn ISAGI test on EC2."""
import sys, os, torch
sys.path.insert(0, '/tmp')
from isagi_chat import build_isagi, isagi_turn

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
use_4bit = "--4bit" in sys.argv

print(f"Building ISAGI with {model_id}...")
system = build_isagi(model_id, use_4bit=use_4bit)

prompt = sys.argv[2] if len(sys.argv) > 2 else "ISAGI, introduce yourself and explain what you can do."
print(f"\n{'='*60}")
print(f"USER: {prompt}")
print(f"{'='*60}")

result = isagi_turn(system, prompt)
print(f"\nISAGI: {result['response']}")
print(f"\n  [COG:{result['cog']} sim:{result['sim']:.2f} metric:{result['metric']:.3f} traj:{result['traj']} {result['ms']}ms]")
print(f"  GTC: {system['gtc'].stats()}")
