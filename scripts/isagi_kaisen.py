#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
KAISEN — The Autonomous Manifold Scout
=======================================
ISAGI variant that deliberately ventures OUTSIDE the instinct horizon,
generates ideas in novel territory, verifies them, and expands the
manifold outward. Like a scout mapping uncharted terrain.

  "Kaisen" () — to open/expand + battle/advance
  The manifold is the known world. Kaisen explores beyond it.

LOOP:
  1. USER gives a topic to explore
  2. KAISEN ventures outside current manifold (J < 0.5 = novel)
  3. KAISEN generates a response in unfamiliar territory
  4. VERIFY: internal consistency check + user feedback
  5. If verified → EXPAND manifold (trajectory added, J rises)
  6. If not verified → DISCARD, try different approach
  7. REPEAT: the manifold frontier advances

Commands:
  /scout TOPIC      — Explore a topic autonomously (N rounds)
  /depth N          — Set exploration depth (default: 3)
  /frontier         — Show frontier status (J, d_h, expansion count)
  /push             — Manually expand manifold with current response

Usage:
  python scripts/isagi_kaisen.py --model Qwen/Qwen2.5-1.5B-Instruct --4bit

William "Nagusame" Stewart — HyperTensor 2026
"""
import torch, json, time, os, sys, argparse, math, random
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import OrderedDict

# Import ISAGI components
from isagi_chat import (
    ISAGI_SYSTEM_PROMPT, GTCCache, GRCProjector, 
    _CallbackStreamer, K_UGT, N_CAL_PROMPTS, MAX_NEW,
    DELTA_NOVEL, ETA_METRIC, TEMPERATURE, TOP_P, STATE_DIR, CACHE_DIR
)
from ott_engine import OTTEngine
from instinct_horizon import InstinctHorizon

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_grad_enabled(False)

# 
# KAISEN SCOUT ENGINE
# 

class KaisenScout:
    """Autonomous manifold explorer. Ventures outside, verifies, expands."""
    
    def __init__(self, system, depth=3, verify_strictness=0.3):
        self.s = system
        self.depth = depth
        self.verify_strictness = verify_strictness  # low = aggressive expansion
        self.expansions = []  # track frontier advances
        self.scout_log = []
        
        # Initialize instinct horizon from trajectories
        self.horizon = InstinctHorizon(creativity=0.3)
        self.horizon.calibrate(system["trajectories"])
    
    def scout(self, topic, verbose=True):
        """Autonomous scout loop: explore topic beyond manifold, verify, expand.
        
        Returns: dict with exploration results, frontier status, expansion log
        """
        if verbose:
            print(f"\n  ")
            print(f"  KAISEN SCOUT: '{topic[:60]}'")
            print(f"  Depth: {self.depth} | Frontier J threshold: {self.horizon.J_threshold:.2f}")
            print(f"  Manifold: {len(self.s['trajectories'])} trajectories | R={self.horizon.R:.3f}")
            print(f"  ")
        
        results = []
        frontier_advanced = False
        
        for round_n in range(self.depth):
            if verbose:
                print(f"\n  --- Round {round_n+1}/{self.depth} ---")
            
            # Step 1: Assess current position relative to manifold
            h_topic = self.s["get_h"](topic)
            h_safe = self.s["safe_h"](h_topic)
            hk = self.s["to_k"](h_safe)
            
            J_before, sim_before = self.horizon.jury_confidence(hk)
            inside_before = J_before >= self.horizon.J_threshold
            
            if verbose:
                status = "INSIDE manifold" if inside_before else "OUTSIDE manifold (scouting)"
                print(f"  Topic J={J_before:.3f} — {status}")
            
            # Step 2: Generate response — push OUTSIDE if inside
            if inside_before and round_n == 0:
                # First round inside: push the model to extrapolate
                scout_prompt = (
                    f"{topic}\n\n"
                    f"Think beyond what you know. Explore the edges of this topic. "
                    f"What might be true that hasn't been proven? What connections "
                    f"exist that aren't obvious? Be creative but rigorous."
                )
            elif round_n > 0:
                # Subsequent rounds: build on previous findings
                prev_summary = results[-1].get("response", "")[:200]
                scout_prompt = (
                    f"Building on what we found: {prev_summary}\n\n"
                    f"Now go deeper. What follows from this? What are the implications? "
                    f"Verify your reasoning step by step."
                )
            else:
                scout_prompt = topic
            
            # Generate
            response = self._generate(scout_prompt)
            
            # Step 3: Assess the generated response — is it still outside?
            h_resp = self.s["get_h"](response[:200])
            h_resp_safe = self.s["safe_h"](h_resp)
            h_resp_k = self.s["to_k"](h_resp_safe)
            
            J_after, sim_after = self.horizon.jury_confidence(h_resp_k)
            
            if verbose:
                print(f"  Response J={J_after:.3f} | sim={sim_after:.3f}")
            
            # Step 4: Verify — internal consistency check
            verify_score = self._verify(response, topic)
            
            if verbose:
                print(f"  Verify score: {verify_score:.2f}")
            
            # Step 5: Decision — expand or discard
            should_expand = (
                verify_score >= self.verify_strictness or  # passes verification
                (J_after < 0.3 and verify_score >= 0.15)   # novel but plausible
            )
            
            if should_expand:
                self.s["expand"](h_resp_safe, f"kaisen:{topic[:40]}")
                self.expansions.append({
                    "round": round_n + 1,
                    "topic": topic[:60],
                    "J_before": round(J_before, 3),
                    "J_after": round(J_after, 3),
                    "verify_score": round(verify_score, 3),
                })
                frontier_advanced = True
                if verbose:
                    print(f"  [EXPAND] Manifold grown — {len(self.s['trajectories'])} trajectories now")
            else:
                if verbose:
                    print(f"  [SKIP] Not verified — manifold unchanged")
            
            results.append({
                "round": round_n + 1,
                "topic": topic[:100],
                "response": response[:500],
                "J_before": round(J_before, 3),
                "J_after": round(J_after, 3),
                "verify_score": round(verify_score, 3),
                "expanded": should_expand,
                "inside_before": inside_before,
            })
        
        # Recalibrate horizon after expansion
        if frontier_advanced:
            self.horizon.calibrate(self.s["trajectories"])
        
        # Final frontier status
        hk_final = self.s["to_k"](self.s["safe_h"](self.s["get_h"](topic)))
        J_final, _ = self.horizon.jury_confidence(hk_final)
        
        if verbose:
            print(f"\n  ")
            print(f"  SCOUT COMPLETE")
            print(f"  Expansions: {len([e for e in self.expansions if e['topic'][:40] in topic[:40]])}")
            print(f"  Final J for topic: {J_final:.3f} (was {results[0]['J_before']:.3f})")
            print(f"  Frontier d_h: {self.horizon.instinct_horizon_distance:.3f}")
            print(f"  ")
        
        return {
            "topic": topic,
            "depth": self.depth,
            "results": results,
            "expansions": self.expansions[-self.depth:],
            "J_initial": results[0]["J_before"] if results else None,
            "J_final": round(J_final, 3),
            "frontier_advanced": frontier_advanced,
            "n_expansions_total": len(self.expansions),
            "coverage_radius": round(self.horizon.R, 3),
            "instinct_horizon": round(self.horizon.instinct_horizon_distance, 3),
        }
    
    def _generate(self, prompt, max_tokens=None):
        """Generate a response using the full ISAGI stack."""
        mt = max_tokens or MAX_NEW
        s = self.s
        
        # Build prompt with ISAGI persona
        full_prompt = f"<|im_start|>system\n{ISAGI_SYSTEM_PROMPT}<|im_end|>\n"
        if s["conv_log"]:
            for t in s["conv_log"][-2:]:
                full_prompt += f"\n<|im_start|>user\n{t['user'][:200]}<|im_end|>\n"
                full_prompt += f"<|im_start|>assistant\n{t['response'][:300]}<|im_end|>\n"
        full_prompt += f"\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        enc = s["tok"](full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(s["model"].device)
        np_tok = enc.input_ids.shape[1]
        
        out = s["model"].generate(
            **enc, max_new_tokens=mt, do_sample=True,
            temperature=TEMPERATURE, top_p=TOP_P,
            pad_token_id=s["tok"].eos_token_id,
        )
        return s["tok"].decode(out[0, np_tok:], skip_special_tokens=True).strip()
    
    def _verify(self, response, topic):
        """Internal verification: consistency, coherence, groundedness.
        
        Checks:
          1. Response length (too short = evasive)
          2. Self-consistency (doesn't contradict itself)
          3. Topic relevance (stays on topic)
          4. Jury confidence (is it in familiar territory?)
        """
        score = 0.0
        
        # 1. Length check (0-0.25)
        if len(response) > 50:
            score += 0.15
        if len(response) > 200:
            score += 0.10
        
        # 2. Self-consistency: generate a check prompt and see if response is consistent
        consistency_prompt = (
            f"Here is a claim: {response[:300]}\n\n"
            f"Verify whether this claim is internally consistent. "
            f"Does it contradict itself? Answer YES if consistent, NO if contradictory."
        )
        check = self._generate(consistency_prompt, max_tokens=32)
        if "YES" in check.upper() and "NO" not in check.upper():
            score += 0.30
        elif "YES" in check.upper():
            score += 0.15
        
        # 3. Topic relevance: does response relate to topic?
        topic_words = set(topic.lower().split()[:5])
        resp_words = set(response.lower().split()[:50])
        overlap = len(topic_words & resp_words) / max(len(topic_words), 1)
        score += min(0.25, overlap * 0.5)
        
        # 4. Jury confidence (J > 0.3 = somewhat grounded)
        h_resp = self.s["get_h"](response[:200])
        h_resp_k = self.s["to_k"](self.s["safe_h"](h_resp))
        J_resp, _ = self.horizon.jury_confidence(h_resp_k)
        score += min(0.30, J_resp * 0.6)
        
        return min(1.0, score)
    
    def frontier_status(self):
        """Return current frontier status."""
        return {
            "n_trajectories": len(self.s["trajectories"]),
            "coverage_radius_R": round(self.horizon.R, 4),
            "instinct_horizon_d_h": round(self.horizon.instinct_horizon_distance, 4),
            "J_threshold": round(self.horizon.J_threshold, 3),
            "creativity": self.horizon.creativity,
            "total_expansions": len(self.expansions),
            "recent_expansions": [
                {"topic": e["topic"][:50], "J_after": e["J_after"]}
                for e in self.expansions[-5:]
            ],
        }


# 
# BUILD KAISEN SYSTEM (shared ISAGI init)
# 

def build_kaisen(model_id, use_4bit=True, scout_depth=3):
    """Build the Kaisen system — ISAGI + autonomous scout capability."""
    from isagi_chat import build_isagi
    
    print("=" * 70)
    print("  KAISEN — The Autonomous Manifold Scout")
    print(f"  Base: {model_id} | Depth: {scout_depth}")
    print("  Stack: ISAGI + Instinct Horizon + Autonomous Expansion")
    print("=" * 70)
    
    system = build_isagi(model_id, use_4bit=use_4bit)
    
    # Initialize scout
    scout = KaisenScout(system, depth=scout_depth)
    
    system["scout"] = scout
    
    return system


# 
# MAIN — Interactive Kaisen Chat
# 

def main():
    parser = argparse.ArgumentParser(description="Kaisen — Autonomous Manifold Scout")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--4bit", action="store_true", default=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--load", help="Load .miku state")
    args = parser.parse_args()
    
    print("\n[BUILD] Initializing Kaisen system...")
    system = build_kaisen(args.model, use_4bit=args._4bit if hasattr(args, '_4bit') else True,
                          scout_depth=args.depth)
    
    print("\n" + "=" * 70)
    print("  KAISEN COMMANDS:")
    print("    /scout TOPIC    — Explore a topic autonomously")
    print("    /depth N        — Set exploration depth (default: 3)")
    print("    /frontier       — Show frontier status")
    print("    /push           — Manually expand with last response")
    print("    /status         — Show all stats")
    print("    /save PATH      — Save .miku state")
    print("    /quit           — Exit")
    print("=" * 70 + "\n")
    
    scout = system["scout"]
    last_response = ""
    
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("/quit", "/exit"):
            break
        
        if user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            save_path = parts[1] if len(parts) > 1 else f"{STATE_DIR}/kaisen_state.miku"
            from isagi_chat import save_isagi_state
            save_isagi_state(save_path, system)
            continue
        
        if user_input.startswith("/depth"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    scout.depth = int(parts[1])
                    print(f"  Scout depth set to {scout.depth}")
                except ValueError:
                    print(f"  Usage: /depth N")
            continue
        
        if user_input.startswith("/frontier"):
            fs = scout.frontier_status()
            print(f"  === FRONTIER STATUS ===")
            print(f"  Trajectories: {fs['n_trajectories']} | R={fs['coverage_radius_R']:.3f} | d_h={fs['instinct_horizon_d_h']:.3f}")
            print(f"  J threshold: {fs['J_threshold']:.3f} | Expansions: {fs['total_expansions']}")
            if fs['recent_expansions']:
                print(f"  Recent expansions:")
                for e in fs['recent_expansions']:
                    print(f"    J={e['J_after']:.3f} | {e['topic']}")
            continue
        
        if user_input.startswith("/push"):
            if last_response:
                h = system["get_h"](last_response[:200])
                h_safe = system["safe_h"](h)
                hk = system["to_k"](h_safe)
                J, _ = scout.horizon.jury_confidence(hk)
                system["expand"](h_safe, f"kaisen:manual")
                print(f"  Manifold expanded — J was {J:.3f}, {len(system['trajectories'])} trajectories now")
            else:
                print(f"  No response to expand. Generate one first.")
            continue
        
        if user_input.startswith("/scout"):
            parts = user_input.split(maxsplit=1)
            topic = parts[1] if len(parts) > 1 else input("Topic to scout: ").strip()
            if topic:
                result = scout.scout(topic)
                last_response = result["results"][-1]["response"] if result["results"] else ""
                # Show the response
                if result["results"]:
                    print(f"\n  [KAISEN RESPONSE]")
                    print(f"  {result['results'][-1]['response'][:800]}")
            continue
        
        if user_input.startswith("/status"):
            from isagi_chat import build_isagi  # reuse status logic
            fs = scout.frontier_status()
            gtc_s = system["gtc"].stats()
            print(f"  === KAISEN STATUS ===")
            print(f"  COG: {len(system['trajectories'])} trajectories | d_h={fs['instinct_horizon_d_h']:.3f}")
            print(f"  GTC: {gtc_s['size']} cached | {gtc_s['hit_rate']}% hit rate")
            print(f"  Scout: {fs['total_expansions']} expansions | depth={scout.depth}")
            print(f"  Safety: {len(system['forbidden'])} forbidden | {len(system['snipe_coords'])} snipe")
            continue
        
        # Default: regular ISAGI turn
        from isagi_chat import isagi_turn
        result = isagi_turn(system, user_input)
        last_response = result.get("response", "")
        print(f"\nKaisen > {last_response[:800]}")


if __name__ == "__main__":
    main()
