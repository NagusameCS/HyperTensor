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
KAISEN v3 — Production-Grade Autonomous Scout
==============================================
Key improvements over v2:
  1. ADAPTIVE K: auto-selects k = max(4, min(64, n_trajectories // 5))
     for optimal J signal regardless of manifold size
  2. MULTI-MODEL: fast scout model + verification model
     (0.5B for exploration, 1.5B for verification)
  3. CONFIDENCE CALIBRATION: tracks J-band reliability, adjusts threshold
  4. EXTERNAL SEED: user can inject known facts to expand manifold
  5. SCOUT REPORT: produces structured summary of what was learned
  6. SESSION PERSISTENCE: saves/loads manifold + memory to disk

Usage:
  python scripts/kaisen_v3.py --seed my_known_facts.txt
  python scripts/kaisen_v3.py --load kaisen_session.pt
  
William "Nagusame" Stewart — HyperTensor 2026
"""
import torch, json, time, os, sys, argparse, math, random
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, OrderedDict

sys.path.insert(0, str(Path(__file__).parent))
from instinct_horizon import InstinctHorizon

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 
# ADAPTIVE MANIFOLD: auto-optimizes k for signal quality
# 

class AdaptiveManifold:
    """Builds a manifold with auto-tuned k for optimal J signal.
    
    Rule: k = max(4, min(64, n_trajectories // 5))
    This ensures trajectories are dense enough for meaningful R.
    With 60 trajectories: k=12 (good). With 500: k=64 (good).
    """
    
    def __init__(self, model, tok, d_model):
        self.model = model; self.tok = tok; self.d = d_model
        self.trajectories = []; self.basis = None; self.K = None
    
    def seed_from_texts(self, texts):
        """Seed manifold from a list of known facts."""
        hs_list = []
        for text in texts:
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            hs_list.append(out.hidden_states[-1][0, -1, :].float())
        
        if len(hs_list) < 4:
            return 0
        
        hs_stack = torch.stack(hs_list)
        hs_c = hs_stack - hs_stack.mean(dim=0)
        U, S, _ = torch.linalg.svd(hs_c.T, full_matrices=False)
        
        # Adaptive k
        self.K = max(4, min(64, len(hs_list) // 5))
        self.basis = U[:, :self.K].float().to(DEVICE)
        
        for i, text in enumerate(texts):
            hk = (hs_list[i] @ self.basis).cpu()
            self.trajectories.append({"proj": hk, "label": text[:60]})
        
        return len(texts)
    
    def add_point(self, text, hidden_state=None):
        """Add a single point to the manifold."""
        if hidden_state is None:
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float()
        else:
            h = hidden_state
        
        if self.basis is None:
            return
        
        hk = (h @ self.basis).cpu()
        self.trajectories.append({"proj": hk, "label": text[:60]})
    
    def hidden_state(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()
    
    def project(self, h):
        if self.basis is None: return None
        return (h @ self.basis).cpu()


# 
# CONFIDENCE CALIBRATION: learns J-band reliability
# 

class ConfidenceCalibrator:
    """Tracks which J ranges produce reliable results and adjusts thresholds."""
    
    def __init__(self):
        self.bands = defaultdict(lambda: {"correct": 0, "total": 0})
    
    def record(self, J, was_correct):
        band = int(J * 10) / 10.0  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        self.bands[band]["total"] += 1
        if was_correct:
            self.bands[band]["correct"] += 1
    
    def reliability(self, J):
        """Return estimated reliability at a given J level."""
        band = int(J * 10) / 10.0
        stats = self.bands.get(band, {"correct": 0, "total": 0})
        if stats["total"] < 3:
            # Not enough data — return prior based on J
            return J  # higher J = more reliable (prior)
        return stats["correct"] / stats["total"]
    
    def suggested_threshold(self, min_reliability=0.5):
        """Find the lowest J band where reliability >= threshold."""
        for band in sorted(self.bands.keys()):
            stats = self.bands[band]
            if stats["total"] >= 3 and stats["correct"] / stats["total"] >= min_reliability:
                return band
        return 0.5  # default
    
    def report(self):
        lines = []
        for band in sorted(self.bands.keys()):
            s = self.bands[band]
            if s["total"] > 0:
                r = s["correct"] / s["total"]
                bar = "#" * int(r * 20)
                lines.append(f"  J={band:.1f}: {r:.0%} reliable ({s['correct']}/{s['total']}) {bar}")
        return "\n".join(lines)


# 
# KAISEN v3
# 

class KaisenV3:
    def __init__(self, scout_model="Qwen/Qwen2.5-0.5B-Instruct",
                 verify_model=None, seed_facts=None):
        
        print(f"[KAISEN v3] Loading scout model: {scout_model}")
        self.scout_model = AutoModelForCausalLM.from_pretrained(
            scout_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.scout_tok = AutoTokenizer.from_pretrained(scout_model, trust_remote_code=True)
        self.scout_tok.pad_token = self.scout_tok.eos_token
        
        self.verify_model = self.scout_model
        self.verify_tok = self.scout_tok
        if verify_model and verify_model != scout_model:
            print(f"[KAISEN v3] Loading verify model: {verify_model}")
            self.verify_model = AutoModelForCausalLM.from_pretrained(
                verify_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            self.verify_tok = AutoTokenizer.from_pretrained(verify_model, trust_remote_code=True)
            self.verify_tok.pad_token = self.verify_tok.eos_token
        
        self.d_model = self.scout_model.config.hidden_size
        
        # Build manifold from seed facts
        self.manifold = AdaptiveManifold(self.scout_model, self.scout_tok, self.d_model)
        
        default_facts = [
            "The speed of light is 299792458 m/s.","F=ma is Newton's second law.",
            "Water freezes at 0C and boils at 100C.","The Earth orbits the Sun.",
            "DNA is a double helix.","a^2+b^2=c^2 for right triangles.",
            "The derivative of x^n is n*x^(n-1).","e^(i*pi)+1=0.",
            "log(a*b)=log(a)+log(b).","The sum 1+2+...+n = n(n+1)/2.",
            "Quicksort average complexity is O(n log n).","A hash table has O(1) lookup.",
            "TCP ensures reliable packet delivery.","Dijkstra finds shortest paths.",
            "BFS uses a queue; DFS uses a stack.","The halting problem is undecidable.",
            "Photosynthesis: CO2+H2O+light -> glucose+O2.","PV=nRT is ideal gas law.",
            "Entropy increases in isolated systems.","E=mc^2 relates energy and mass.",
            "Shakespeare wrote 37 plays.","Beethoven composed 9 symphonies.",
            "The Mona Lisa is by da Vinci.","Mount Everest is 8848m.",
            "The Moon landing was 1969.","The cheetah is fastest land animal.",
            "Diamond is pure carbon.","Honey never spoils.",
            "Octopuses have 3 hearts.","Lightning is 5x hotter than the sun.",
            "A day on Venus is longer than its year.","Bananas are technically berries.",
        ]
        
        if seed_facts:
            with open(seed_facts) as f:
                custom = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            default_facts = custom[:100] if custom else default_facts
        
        n = self.manifold.seed_from_texts(default_facts)
        
        # Calibrate
        self.horizon = InstinctHorizon(creativity=0.30)
        self.horizon.calibrate(self.manifold.trajectories)
        
        # Confidence tracking
        self.calibrator = ConfidenceCalibrator()
        
        # Stats
        self.scout_count = 0
        self.expansions = 0
        self.session_log = []
        
        print(f"  {n} trajectories | k={self.manifold.K} | R={self.horizon.R:.2f} | "
              f"d_h={self.horizon.instinct_horizon_distance:.2f} | J_thr={self.horizon.J_threshold:.3f}")
    
    def _gen(self, model, tok, prompt, max_tokens=128):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        np = enc.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, np:], skip_special_tokens=True).strip()
    
    def generate(self, prompt, max_tokens=128):
        return self._gen(self.scout_model, self.scout_tok, prompt, max_tokens)
    
    def verify(self, problem, response, criteria=""):
        """Strict verification with the verify model."""
        v_prompt = (
            f"You are a strict fact-checker. Verify this solution:\n\n"
            f"Problem: {problem[:200]}\n"
            f"Solution: {response[:400]}\n"
            f"Criteria: {criteria[:150]}\n\n"
            f"Respond with EXACTLY one word: CORRECT or INCORRECT\n"
            f"Then a one-line explanation."
        )
        verdict = self._gen(self.verify_model, self.verify_tok, v_prompt, 60)
        is_correct = "CORRECT" in verdict[:20].upper() and "INCORRECT" not in verdict[:20].upper()
        return is_correct, verdict
    
    def scout(self, problem, criteria="", max_depth=5, verbose=True):
        """Autonomous scout with calibration feedback."""
        self.scout_count += 1
        h_prob = self.manifold.hidden_state(problem)
        hk_prob = self.manifold.project(h_prob)
        J_init, _ = self.horizon.jury_confidence(hk_prob)
        
        if verbose:
            print(f"\n{'='*55}")
            print(f"  KAISEN v3 | J={J_init:.3f} | {'IN' if J_init>=self.horizon.J_threshold else 'OUT'}")
            print(f"  {problem[:70]}...")
            print(f"{'='*55}")
        
        history = []
        J_curr = J_init
        
        for d in range(1, max_depth + 1):
            if J_curr >= self.horizon.J_threshold and d > 1:
                if verbose: print(f"  [DONE] Territory now familiar (J={J_curr:.3f})")
                break
            
            # Build prompt with scout history
            ctx = ""
            if history:
                last = history[-1]
                ctx = f"Previous attempt {'was correct.' if last['verified'] else 'needs improvement.'} "
                if last['verified']:
                    ctx += f"Building on: {last['response'][:100]}\n\n"
            
            prompt = (f"You are an expert problem solver. {ctx}"
                     f"Think step by step. Be precise and rigorous.\n\n"
                     f"Problem: {problem}\n\n"
                     f"End with: ANSWER: [your answer]")
            
            response = self.generate(prompt, 150)
            
            # Verify
            verified, verdict = self.verify(problem, response, criteria)
            
            # Measure J
            h_resp = self.manifold.hidden_state(response[:300])
            hk_resp = self.manifold.project(h_resp)
            J_new, _ = self.horizon.jury_confidence(hk_resp)
            
            # Calibrate
            self.calibrator.record(J_new, verified)
            
            delta = J_new - J_curr
            J_curr = J_new
            
            history.append({
                "depth": d, "verified": verified, "J": round(J_new, 4),
                "delta": round(delta, 4), "response": response[:300],
            })
            
            if verbose:
                icon = "OK" if verified else "??"
                arrow = "v" if delta < -0.02 else ("^" if delta > 0.02 else "~")
                print(f"  [{icon}] d={d} J={J_new:.3f} ({arrow}{abs(delta):.3f}) {response[:60]}...")
            
            if verified:
                self.manifold.add_point(f"kaisen:{problem[:40]}", h_resp)
                self.expansions += 1
            
            if not verified and d >= 3:
                recent = [h["verified"] for h in history[-3:]]
                if not any(recent):
                    if verbose: print(f"  [STOP] 3 unverified — territory too unfamiliar")
                    break
        
        # Post-scout
        self.horizon.calibrate(self.manifold.trajectories)
        J_final, _ = self.horizon.jury_confidence(hk_prob)
        
        # Report
        verified_n = sum(1 for h in history if h["verified"])
        best = history[-1]["response"] if history else ""
        
        result = {
            "problem": problem[:100],
            "J_initial": round(J_init, 4), "J_final": round(J_final, 4),
            "frontier_advanced": J_final > J_init,
            "verified": verified_n, "depth": len(history),
            "best_response": best,
            "history": history,
        }
        self.session_log.append(result)
        
        if verbose:
            print(f"  => Verified {verified_n}/{len(history)} | J {J_init:.3f}->{J_final:.3f} | "
                  f"traj={len(self.manifold.trajectories)}")
        
        return result
    
    def report(self):
        """Full session report."""
        return {
            "scouts": self.scout_count,
            "expansions": self.expansions,
            "trajectories": len(self.manifold.trajectories),
            "k": self.manifold.K,
            "R": round(self.horizon.R, 3),
            "d_h": round(self.horizon.instinct_horizon_distance, 3),
            "J_threshold": round(self.horizon.J_threshold, 3),
            "confidence_calibration": self.calibrator.report(),
            "session_log": [
                {"problem": s["problem"][:60], "verified": s["verified"],
                 "J_initial": s["J_initial"], "J_final": s["J_final"]}
                for s in self.session_log
            ],
        }
    
    def save(self, path):
        state = {
            "trajectories": [(t["proj"].tolist(), t["label"]) for t in self.manifold.trajectories],
            "K": self.manifold.K,
            "horizon_R": self.horizon.R,
            "horizon_J_threshold": self.horizon.J_threshold,
            "scout_count": self.scout_count,
            "expansions": self.expansions,
            "session_log": self.session_log,
            "calibrator_bands": {str(k): dict(v) for k, v in self.calibrator.bands.items()},
        }
        torch.save(state, path)
        print(f"[KAISEN] Session saved to {path}")
    
    def load(self, path):
        state = torch.load(path, map_location="cpu")
        self.manifold.trajectories = [
            {"proj": torch.tensor(p), "label": l}
            for p, l in state["trajectories"]
        ]
        self.manifold.K = state["K"]
        self.horizon.R = state["horizon_R"]
        self.horizon.J_threshold = state["horizon_J_threshold"]
        self.horizon._calibrated = True
        self.scout_count = state.get("scout_count", 0)
        self.expansions = state.get("expansions", 0)
        self.session_log = state.get("session_log", [])
        for k, v in state.get("calibrator_bands", {}).items():
            self.calibrator.bands[float(k)] = v
        print(f"[KAISEN] Session loaded: {len(self.manifold.trajectories)} traj, "
              f"{self.scout_count} scouts, {self.expansions} expansions")


# 
# MAIN
# 

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scout-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--verify-model", default=None)
    parser.add_argument("--seed", help="File with known facts to seed manifold")
    parser.add_argument("--load", help="Load saved session")
    parser.add_argument("--problem", help="Single problem to scout")
    parser.add_argument("--criteria", default="Verify correctness step by step.")
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    
    k = KaisenV3(
        scout_model=args.scout_model,
        verify_model=args.verify_model,
        seed_facts=args.seed,
    )
    
    if args.load:
        k.load(args.load)
    
    if args.problem:
        result = k.scout(args.problem, args.criteria, max_depth=args.depth)
        print(f"\n{'='*55}")
        print(f"  BEST RESPONSE:")
        print(f"  {result['best_response'][:600]}")
        print(f"{'='*55}")
        
        # Show calibration
        print(f"\n  CONFIDENCE CALIBRATION:")
        print(k.calibrator.report())
    else:
        # Interactive
        print(f"\n  KAISEN v3 ready. Commands:")
        print(f"    /scout PROBLEM   — Explore a problem")
        print(f"    /report          — Show session report")
        print(f"    /calibrate       — Show confidence calibration")
        print(f"    /save PATH       — Save session")
        print(f"    /seed FACT       — Add a known fact to manifold")
        print(f"    /quit")
        print()
        
        while True:
            try:
                cmd = input("\nkaisen> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd: continue
            if cmd in ("/quit","/exit"): break
            if cmd == "/report":
                r = k.report()
                print(f"  Scouts: {r['scouts']} | Expansions: {r['expansions']} | "
                      f"Traj: {r['trajectories']} | R={r['R']} | d_h={r['d_h']}")
                for s in r["session_log"][-5:]:
                    print(f"    J={s['J_initial']:.3f}->{s['J_final']:.3f} | "
                          f"verified={s['verified']} | {s['problem'][:50]}")
                continue
            if cmd == "/calibrate":
                print(k.calibrator.report())
                continue
            if cmd.startswith("/save"):
                path = cmd.split(maxsplit=1)[1] if len(cmd.split())>1 else "kaisen_session.pt"
                k.save(path); continue
            if cmd.startswith("/seed"):
                fact = cmd.split(maxsplit=1)[1] if len(cmd.split())>1 else ""
                if fact:
                    k.manifold.seed_from_texts([fact])
                    k.horizon.calibrate(k.manifold.trajectories)
                    print(f"  Added. Traj={len(k.manifold.trajectories)} R={k.horizon.R:.2f}")
                continue
            if cmd.startswith("/scout"):
                problem = cmd.split(maxsplit=1)[1] if len(cmd.split())>1 else input("Problem: ").strip()
                if problem:
                    k.scout(problem, max_depth=args.depth)
                continue
            # Default: treat as problem
            k.scout(cmd, max_depth=args.depth)
