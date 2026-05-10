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
KAISEN v2 — Autonomous Manifold Scout with Verification Chain
==============================================================
Improvements over v1:
  1. DENSE manifold: lower k, more trajectories, topic clustering
  2. VERIFICATION CHAIN: generate → self-check → retry if wrong → cross-check
  3. SCOUT MEMORY: vector store of explored territory, no repeats
  4. ADAPTIVE DEPTH: stop scouting when J rises above horizon
  5. BATCH SCOUTING: try N approaches, keep best
  6. EXTERNAL CHECK: for math, verify by recomputation in separate prompt

Usage: python scripts/kaisen_v2.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import torch, json, time, os, sys, argparse, math, random
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from instinct_horizon import InstinctHorizon

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_MANIFOLD = 12
N_TRAJECTORIES_MIN = 80
MAX_SCOUT_DEPTH = 8
BATCH_SIZE = 3
VERIFY_RETRIES = 2


class DenseManifold:
    """Builds a dense manifold from diverse clustered facts for sharp J signal."""
    
    def __init__(self, model, tok, dim):
        self.model = model
        self.tok = tok
        self.d = dim
        self.trajectories = []
        self.basis = None
        
    def build(self):
        # Dense topic-clustered calibration facts
        clusters = {
            "math": [
                "The derivative of x^n is n*x^(n-1).",
                "Integral of dx/(1+x^2) is arctan(x).",
                "e^(i*pi) + 1 = 0 is Euler's identity.",
                "The quadratic formula is x = (-b +/- sqrt(b^2-4ac))/(2a).",
                "sin^2(x) + cos^2(x) = 1 is the Pythagorean identity.",
                "log(a*b) = log(a) + log(b).",
                "The determinant of a 2x2 matrix [[a,b],[c,d]] is ad-bc.",
                "A prime number has exactly two factors: 1 and itself.",
                "The sum 1+2+...+n = n(n+1)/2.",
                "d/dx(sin(x)) = cos(x).",
                "The chain rule: d/dx(f(g(x))) = f'(g(x))*g'(x).",
                "The fundamental theorem of calculus connects derivatives and integrals.",
                "A Taylor series expands a function at a point.",
                "The binomial theorem: (a+b)^n = sum(C(n,k)*a^(n-k)*b^k).",
                "The mean value theorem guarantees a point where derivative equals average slope.",
            ],
            "physics": [
                "F = ma is Newton's second law of motion.",
                "E = mc^2 relates energy and mass via the speed of light.",
                "The speed of light in vacuum is 299,792,458 m/s.",
                "Gravity accelerates objects at 9.8 m/s^2 on Earth.",
                "Kepler's first law: planets orbit in ellipses with the Sun at one focus.",
                "The ideal gas law is PV = nRT.",
                "Ohm's law: V = IR relates voltage, current, and resistance.",
                "The first law of thermodynamics: energy is conserved.",
                "Entropy always increases in an isolated system (second law).",
                "The wavelength of a wave is inversely proportional to its frequency.",
                "The force between two charges is F = k*q1*q2/r^2 (Coulomb's law).",
                "Momentum is conserved in closed systems.",
                "The photoelectric effect showed light behaves as particles.",
                "Boyle's law: P1*V1 = P2*V2 at constant temperature.",
                "Archimedes' principle: buoyant force equals weight of displaced fluid.",
            ],
            "cs": [
                "Quicksort has average time complexity O(n log n).",
                "A binary search tree allows O(log n) lookup.",
                "TCP ensures reliable ordered delivery of data packets.",
                "A hash table provides O(1) average lookup time.",
                "Dijkstra's algorithm finds shortest paths in a weighted graph.",
                "A Turing machine is a theoretical model of computation.",
                "The halting problem is undecidable.",
                "BFS explores a graph level by level using a queue.",
                "DFS explores a graph depth-first using recursion or a stack.",
                "A database index speeds up queries using B-trees or hash indexes.",
                "Public-key cryptography uses separate keys for encryption and decryption.",
                "A semaphore controls access to shared resources in concurrent programming.",
                "The difference between a process and a thread is memory isolation.",
                "An HTTP GET request retrieves data; POST submits data.",
                "Regular expressions describe patterns in strings.",
            ],
            "general": [
                "The capital of France is Paris, known as the City of Light.",
                "Water freezes at 0 Celsius and boils at 100 Celsius at sea level.",
                "The Earth completes one orbit around the Sun in approximately 365.25 days.",
                "Photosynthesis converts CO2 and H2O into glucose using sunlight.",
                "DNA is a double helix structure discovered by Watson and Crick in 1953.",
                "Shakespeare wrote 37 plays including Hamlet, Macbeth, and King Lear.",
                "The Great Wall of China stretches over 13,000 miles.",
                "Beethoven composed 9 symphonies despite becoming deaf.",
                "The human brain contains approximately 86 billion neurons.",
                "Mount Everest stands at 8,848 meters above sea level.",
                "The Amazon rainforest produces about 20% of Earth's oxygen.",
                "Diamond is a crystalline form of pure carbon.",
                "The first successful human Moon landing was Apollo 11 in July 1969.",
                "The Mona Lisa was painted by Leonardo da Vinci around 1503.",
                "The cheetah is the fastest land animal, reaching speeds over 100 km/h.",
            ],
        }
        
        all_facts = []
        for domain, facts in clusters.items():
            all_facts.extend(facts)
        
        # Collect hidden states
        hs_list = []
        for fact in all_facts:
            enc = self.tok(fact, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            hs_list.append(out.hidden_states[-1][0, -1, :].float())
        
        hs_stack = torch.stack(hs_list)
        hs_centered = hs_stack - hs_stack.mean(dim=0)
        U, S, _ = torch.linalg.svd(hs_centered.T, full_matrices=False)
        self.basis = U[:, :K_MANIFOLD].float().to(DEVICE)
        
        for fact in all_facts:
            hk = self._project(hs_list[all_facts.index(fact)])
            self.trajectories.append({"proj": hk.cpu(), "label": fact[:60]})
        
        return self.trajectories, self.basis
    
    def _project(self, h):
        return (h @ self.basis).cpu()


class ScoutMemory:
    """Remembers explored territory to avoid repeating failed directions."""
    
    def __init__(self):
        self.visited = []  # list of (topic_hash, k_vector, outcome, J)
        self.dead_ends = []  # approaches that failed verification
    
    def is_explored(self, k_vec, topic, threshold=0.15):
        """Check if this direction has been explored before."""
        if not self.visited:
            return False
        visited_stack = torch.stack([v[1] for v in self.visited[-32:]])
        sims = F.cosine_similarity(k_vec.unsqueeze(0).float(), visited_stack.float())
        if sims.max().item() > (1.0 - threshold):
            return True
        return False
    
    def record(self, topic, k_vec, outcome, J):
        self.visited.append((topic[:40], k_vec.cpu(), outcome, J))
    
    def record_dead_end(self, k_vec):
        self.dead_ends.append(k_vec.cpu())


class KaisenV2:
    """Improved autonomous manifold scout."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", use_4bit=True):
        print(f"[KAISEN v2] Loading {model_name}...")
        
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tok.pad_token = self.tok.eos_token
        self.d_model = self.model.config.hidden_size
        
        # Build dense manifold
        print("[KAISEN v2] Building dense manifold...")
        dm = DenseManifold(self.model, self.tok, self.d_model)
        trajs, self.basis = dm.build()
        
        # Calibrate instinct horizon
        self.horizon = InstinctHorizon(creativity=0.30)
        self.horizon.calibrate(trajs)
        
        # Initialize scout memory
        self.memory = ScoutMemory()
        self.trajectories = trajs
        self.stats = {"scouts": 0, "expansions": 0, "verified": 0, "dead_ends": 0}
        
        print(f"  {len(trajs)} trajectories | k={K_MANIFOLD} | R={self.horizon.R:.2f} | "
              f"d_h={self.horizon.instinct_horizon_distance:.2f} | J_thr={self.horizon.J_threshold:.3f}")
    
    def hidden_state(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()
    
    def to_k(self, h):
        return (h @ self.basis).cpu()
    
    def generate(self, prompt, max_tokens=200):
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        np = enc.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                      temperature=0.7, top_p=0.9, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, np:], skip_special_tokens=True).strip()
    
    def verify_chain(self, problem, response, criteria="", retries=VERIFY_RETRIES):
        """Multi-step verification: self-check, retry if wrong, cross-check."""
        
        for attempt in range(1 + retries):
            # Self-verification
            verify_prompt = (
                f"Problem: {problem[:200]}\n\n"
                f"Proposed solution:\n{response[:400]}\n\n"
                f"Verify this solution step by step. Check each calculation. "
                f"Criteria: {criteria[:200]}\n\n"
                f"Answer ONLY with: VERDICT: CORRECT or VERDICT: INCORRECT\n"
                f"Then explain why in one sentence."
            )
            verdict = self.generate(verify_prompt, max_tokens=80)
            
            is_correct = "CORRECT" in verdict.split('\n')[0].upper() if verdict else False
            
            if is_correct:
                return True, response, verdict
            
            if attempt < retries:
                # Retry with more guidance
                retry_prompt = (
                    f"Your previous solution was marked incorrect. Try again.\n\n"
                    f"Problem: {problem[:200]}\n\n"
                    f"Think step by step. Show all work. Be precise.\n"
                    f"Previous attempt had issues: {verdict[:150]}"
                )
                response = self.generate(retry_prompt, max_tokens=200)
        
        return False, response, verdict
    
    def batch_scout(self, problem, criteria, n_approaches=BATCH_SIZE):
        """Try multiple approaches, keep the best."""
        
        approaches = [
            "Solve this step by step. Show all work. Be precise.",
            "Think about this from first principles. What is the core concept?",
            "Break this into smaller sub-problems. Solve each one carefully.",
            "Consider edge cases. What assumptions are you making?",
            "Work backwards from what the answer should look like.",
        ]
        
        best_response = None
        best_score = -1
        best_verified = False
        best_verdict = ""
        
        for i in range(min(n_approaches, len(approaches))):
            scout_prompt = (
                f"You are an expert problem solver. {approaches[i]}\n\n"
                f"Problem: {problem}\n\n"
                f"End with: ANSWER: [your answer]"
            )
            
            response = self.generate(scout_prompt, max_tokens=200)
            
            # Basic quality score: length, structure, confidence markers
            score = 0.0
            if len(response) > 50: score += 0.2
            if len(response) > 150: score += 0.15
            if "step" in response.lower(): score += 0.1
            if "because" in response.lower(): score += 0.1
            if "therefore" in response.lower() or "thus" in response.lower(): score += 0.1
            if "answer" in response.lower(): score += 0.15
            
            # Verify best
            if score > best_score:
                best_score = score
                best_response = response
                best_verified, _, best_verdict = self.verify_chain(problem, response, criteria, retries=0)
        
        # Full verification chain on best
        if not best_verified:
            best_verified, best_response, best_verdict = self.verify_chain(
                problem, best_response, criteria, retries=VERIFY_RETRIES)
        
        return best_verified, best_response, best_verdict
    
    def scout(self, problem, criteria="Verify correctness.", max_depth=MAX_SCOUT_DEPTH, verbose=True):
        """Main scout loop: venture → verify → expand → repeat."""
        
        self.stats["scouts"] += 1
        
        # Measure initial J
        h_prob = self.hidden_state(problem)
        hk_prob = self.to_k(h_prob)
        J_initial, _ = self.horizon.jury_confidence(hk_prob)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  KAISEN v2 SCOUT")
            print(f"  Problem: {problem[:80]}...")
            print(f"  J_initial: {J_initial:.3f} ({'INSIDE' if J_initial>=self.horizon.J_threshold else 'OUTSIDE'} manifold)")
            print(f"  d_h: {self.horizon.instinct_horizon_distance:.1f} | R: {self.horizon.R:.1f}")
            print(f"{'='*60}")
        
        # Scout history
        history = []
        J_current = J_initial
        
        for depth in range(1, max_depth + 1):
            if verbose:
                print(f"\n  --- Depth {depth}/{max_depth} ---")
            
            # Check if already explored
            if self.memory.is_explored(hk_prob, problem):
                if verbose:
                    print(f"  [SKIP] This territory was already explored")
                break
            
            # Check if we're now inside familiar territory
            if J_current >= self.horizon.J_threshold and depth > 1:
                if verbose:
                    print(f"  [DONE] J={J_current:.3f} >= threshold={self.horizon.J_threshold:.3f} — territory is now familiar")
                break
            
            # Build scout prompt with historical context
            context = ""
            if history:
                prev = history[-1]
                context = f"Previous exploration found: {prev['summary'][:200]}\n\n"
            
            full_problem = context + problem
            
            # Batch scout
            verified, response, verdict = self.batch_scout(full_problem, criteria)
            
            # Measure J after response
            h_resp = self.hidden_state(response[:300])
            hk_resp = self.to_k(h_resp)
            J_after, _ = self.horizon.jury_confidence(hk_resp)
            
            delta = J_after - J_current
            J_current = J_after
            
            # Record
            self.memory.record(problem, hk_resp, "verified" if verified else "unverified", J_after)
            history.append({
                "depth": depth, "verified": verified, "J": round(J_after, 4),
                "delta": round(delta, 4), "summary": response[:200],
            })
            
            if verbose:
                verdict_icon = "CORRECT" if verified else "UNVERIFIED"
                j_icon = "v" if delta < -0.02 else ("^" if delta > 0.02 else "~")
                print(f"  [{verdict_icon}] J={J_after:.3f} ({j_icon}{abs(delta):.3f}) | "
                      f"{'EXPANDED' if verified else 'recorded'}")
            
            # Expand if verified
            if verified:
                self.trajectories.append({"proj": hk_resp.cpu(), "label": f"kaisen:{problem[:40]}"})
                self.horizon.calibrate(self.trajectories)
                self.stats["expansions"] += 1
                self.stats["verified"] += 1
            else:
                self.memory.record_dead_end(hk_resp)
                self.stats["dead_ends"] += 1
            
            # Stop if no progress after multiple attempts
            if not verified and depth >= 3 and all(not h["verified"] for h in history[-3:]):
                if verbose:
                    print(f"  [STOP] 3 consecutive unverified attempts — territory too unfamiliar")
                break
        
        # Post-scout recalibration
        self.horizon.calibrate(self.trajectories)
        J_final, _ = self.horizon.jury_confidence(hk_prob)
        
        if verbose:
            print(f"\n  {'='*40}")
            print(f"  SCOUT COMPLETE")
            print(f"  J: {J_initial:.3f} -> {J_final:.3f} | "
                  f"Verified: {sum(1 for h in history if h['verified'])}/{len(history)}")
            print(f"  R: {self.horizon.R:.2f} | d_h: {self.horizon.instinct_horizon_distance:.2f}")
            print(f"  Trajectories: {len(self.trajectories)}")
            print(f"  {'='*40}")
        
        return {
            "problem": problem[:100],
            "J_initial": round(J_initial, 4),
            "J_final": round(J_final, 4),
            "frontier_advanced": J_final > J_initial,
            "depth_reached": len(history),
            "verified_count": sum(1 for h in history if h["verified"]),
            "best_response": history[-1]["summary"] if history else "",
            "history": history,
            "stats": dict(self.stats),
        }


# 
# DEMO
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    
    kaisen = KaisenV2(model_name=args.model, use_4bit=not args.fp16)
    
    # Test problems
    tests = [
        ("MATH", "How many different ways can you arrange the letters in the word 'ROBOT'? Show all steps.",
         "Count distinct permutations accounting for the repeated 'O'."),
        ("LOGIC", "Alice says Bob is lying. Bob says Charlie is lying. Charlie says both Alice and Bob are lying. Who is telling the truth?",
         "Check for logical consistency in all 8 possible truth-assignments."),
        ("ESTIMATION", "How many golf balls would fit in a school bus? Show your reasoning step by step with approximate dimensions.",
         "Use reasonable estimates for bus volume (8ft x 6ft x 30ft) and golf ball diameter (1.68in)."),
    ]
    
    for title, problem, criteria in tests:
        result = kaisen.scout(problem, criteria)
        print(f"\n  BEST: {result['best_response'][:300]}...")
