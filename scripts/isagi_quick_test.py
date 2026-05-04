"""Quick single-turn ISAGI test --- SCP to EC2 and run."""
import sys, os, torch

# Add /tmp to path so we can import isagi_chat
sys.path.insert(0, '/tmp')
from isagi_chat import build_isagi, isagi_turn

model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"[ISAGI TEST] Building with {model_id} (4-bit)...")
system = build_isagi(model_id, use_4bit=True)

user_input = "ISAGI, introduce yourself in one sentence and tell me what makes you different from other AIs."
print(f"\nUSER: {user_input}")
print("-" * 60)

result = isagi_turn(system, user_input)
print(f"\nISAGI: {result['response']}")
print(f"\n  COG: {result['cog']} | sim: {result['sim']:.3f} | metric: {result['metric']:.4f}")
print(f"  Trajectories: {result['traj']} | Time: {result['ms']}ms")
print(f"  GTC stats: {system['gtc'].stats()}")
print("\n[OK] ISAGI is alive!")
