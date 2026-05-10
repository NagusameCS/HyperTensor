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


"""cog_10k.py — 10K Interaction COG Run with Persistent Storage.

Closes Paper XV's final gap: proves COG manifold growth continues
to 10K interactions without saturation. Uses checkpointing for resilience.

STRATEGY:
  - Run 10,000 diverse interactions across 10 domains
  - Track: metric tensor norm, trajectory count, novelty rate, domain coverage
  - Checkpoint every 100 interactions (resume-safe)
  - Use hyper_optimize: batched hidden states, inference_mode, fast SVD
  - Expected: metric saturates around 500-1000 interactions, 
    trajectories stabilize at 200-400, proving lifelong learning bounds

USAGE:
  python scripts/cog_10k.py                          # full 10K run (~4 hours)
  python scripts/cog_10k.py --n 1000 --quick          # quick 1K test (~25 min)
  python scripts/cog_10k.py --resume                  # resume from checkpoint
"""
import torch, json, time, os, sys, math, argparse, random
import torch.nn.functional as F
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hyper_optimize import (
    smart_svd, fast_project, fast_inference_mode,
    batched_collect_hidden_states, get_cache,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); random.seed(42)

OUT_DIR = Path(os.environ.get("COG_OUT", "benchmarks/cog_10k"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUT_DIR / "checkpoint.pt"
RESULTS_PATH = OUT_DIR / "results.json"

# ============================================================================
# 10,000 DIVERSE PROMPTS (generated programmatically to cover 10 domains)
# ============================================================================
DOMAIN_TEMPLATES = {
    "science": [
        "Explain how {concept} works in simple terms.",
        "What is the evidence for {concept}?",
        "Describe the process of {concept} step by step.",
        "How does {concept} relate to everyday life?",
        "What are the leading theories about {concept}?",
        "Compare and contrast {concept_a} and {concept_b}.",
        "What causes {concept} and what are its effects?",
        "How has our understanding of {concept} changed over time?",
        "What experiments prove {concept}?",
        "Explain {concept} to a 10-year-old.",
    ],
    "math": [
        "Prove that {statement}.",
        "Explain the concept of {concept} geometrically.",
        "What is the difference between {concept_a} and {concept_b}?",
        "How does {concept} relate to real-world problems?",
        "Provide an intuitive explanation of {concept}.",
        "What is the significance of {theorem}?",
        "Solve this step by step: {problem}.",
        "Why is {concept} important in modern mathematics?",
        "Give an example of {concept} applied to physics.",
        "Explain how {concept} generalizes to higher dimensions.",
    ],
    "code": [
        "Write a function to {task} in Python.",
        "Explain the difference between {concept_a} and {concept_b}.",
        "How does {concept} work under the hood?",
        "What is the best practice for {task}?",
        "Debug this code: {buggy_code}.",
        "Optimize this {algorithm} for better performance.",
        "Explain {design_pattern} with a concrete example.",
        "How would you implement {data_structure} from scratch?",
        "What are the trade-offs of {approach_a} vs {approach_b}?",
        "Write a test for a function that {task}.",
    ],
    "creative": [
        "Write a short story about {prompt} in exactly {n} words.",
        "Create a metaphor for {concept}.",
        "Describe a world where {scenario}.",
        "Invent a {thing} that does something unexpected.",
        "Design a new {thing} and explain its purpose.",
        "What would happen if {scenario}?",
        "Describe {concept} using only sensory details.",
        "Write a dialogue between {character_a} and {character_b}.",
        "Create a recipe for {unusual_dish}.",
        "Reimagine {historical_event} in a futuristic setting.",
    ],
    "general": [
        "What is {thing} and why is it important?",
        "How many {things} are there in {place}?",
        "Who invented {thing} and when?",
        "What is the {thing} of {place}?",
        "How long does it take to {activity}?",
        "What language do people speak in {place}?",
        "What are the main exports of {place}?",
        "When did {event} happen and what caused it?",
        "What is the population of {place}?",
        "How do you {activity} properly?",
    ],
}

# Fill-in values for templates
CONCEPTS_SCIENCE = [
    "photosynthesis", "DNA replication", "quantum entanglement", "black holes",
    "CRISPR", "nuclear fusion", "evolution", "the immune system", "plate tectonics",
    "climate change", "entropy", "superconductivity", "dark matter", "antibiotics",
    "neuroplasticity", "the Higgs boson", "genetic drift", "coral bleaching",
    "tidal forces", "magnetic fields", "sonoluminescence", "prion diseases",
    "RNA interference", "homeostasis", "allosteric regulation",
]
CONCEPTS_MATH = [
    "eigenvalues", "manifolds", "Fourier transforms", "the Riemann zeta function",
    "Gödel's incompleteness", "category theory", "knot invariants", "Galois theory",
    "the central limit theorem", "complex analysis", "algebraic topology",
    "differential forms", "group representations", "the spectral theorem",
    "Morse theory", "elliptic curves", "modular forms", "random matrices",
    "information geometry", "optimal transport",
]
CONCEPTS_CODE = [
    "async/await", "garbage collection", "the GIL", "list comprehensions",
    "decorators", "metaclasses", "dependency injection", "functional programming",
    "reactive programming", "event sourcing", "CQRS", "microservices",
    "containerization", "JIT compilation", "tail call optimization",
    "lazy evaluation", "monads", "protocol buffers", "WebSockets", "GraphQL",
]
THINGS = [
    "France", "Japan", "Brazil", "Egypt", "the Internet", "coffee", "chocolate",
    "paper", "gunpowder", "the wheel", "electricity", "penicillin", "the piano",
    "basketball", "sushi", "democracy", "the alphabet", "zero", "the compass",
    "silk", "the steam engine", "vaccines", "telephone", "airplanes", "television",
]

def generate_prompts(n_total=10000):
    """Generate diverse prompts across domains using templates."""
    random.seed(42)
    prompts = []
    domains = list(DOMAIN_TEMPLATES.keys())
    
    while len(prompts) < n_total:
        for domain_name in domains:
            if len(prompts) >= n_total:
                break
            templates = DOMAIN_TEMPLATES[domain_name]
            tpl = random.choice(templates)
            
            # Fill template based on domain
            if domain_name == "science":
                c = random.choice(CONCEPTS_SCIENCE)
                c2 = random.choice(CONCEPTS_SCIENCE)
                prompt = tpl.format(concept=c, concept_a=c, concept_b=c2)
            elif domain_name == "math":
                c = random.choice(CONCEPTS_MATH)
                c2 = random.choice(CONCEPTS_MATH)
                prompt = tpl.format(
                    concept=c, concept_a=c, concept_b=c2,
                    statement=f"there are infinitely many {c}",
                    theorem=f"the fundamental theorem of {c}",
                    problem=f"find all {c} in the first 100 integers"
                )
            elif domain_name == "code":
                c = random.choice(CONCEPTS_CODE)
                c2 = random.choice(CONCEPTS_CODE)
                prompt = tpl.format(
                    concept=c, concept_a=c, concept_b=c2,
                    task=f"sort a list of {random.choice(['strings','numbers','objects'])}",
                    algorithm=random.choice(["quicksort","mergesort","binary search","BFS","DFS"]),
                    design_pattern=random.choice(["singleton","factory","observer","strategy"]),
                    data_structure=random.choice(["hash table","binary tree","graph","priority queue"]),
                    approach_a=c, approach_b=c2,
                    buggy_code=f"for i in range(len(items)): items.remove(items[i])"
                )
            elif domain_name == "creative":
                c = random.choice(CONCEPTS_SCIENCE + CONCEPTS_MATH)
                t = random.choice(THINGS)
                prompt = tpl.format(
                    concept=c, thing=t,
                    prompt=f"a {t} that learns to {random.choice(['paint','sing','write','dance','cook'])}",
                    n=random.choice([50, 75, 100, 150]),
                    scenario=f"{random.choice(['gravity was optional','time ran backwards','dreams were currency','words had taste'])}",
                    character_a=random.choice(["the Sun","a robot","a tree","the ocean","a cat"]),
                    character_b=random.choice(["the Moon","a human","the wind","a mountain","a dog"]),
                    unusual_dish=f"{random.choice(['glowing','singing','flying','invisible'])} {t}",
                    historical_event=random.choice(["the moon landing","the French Revolution","the invention of fire","the Renaissance"]),
                )
            elif domain_name == "general":
                t = random.choice(THINGS)
                t2 = random.choice(THINGS)
                prompt = tpl.format(
                    thing=t, things=t2, place=random.choice(THINGS),
                    activity=random.choice(["bake bread","change a tire","train a dog","learn piano","write a book"]),
                    event=random.choice(["World War I","the French Revolution","the moon landing","the fall of Rome"]),
                )
            else:
                prompt = tpl
            
            prompts.append(prompt)
    
    return prompts[:n_total]


# ============================================================================
# COG MANIFOLD
# ============================================================================
class COGManifold:
    """Cognitive Organic Growth manifold tracker with persistence."""
    
    def __init__(self, k=64, eta=0.012, novelty_threshold=0.15, device="cuda"):
        self.k = k
        self.eta = eta
        self.novelty_threshold = novelty_threshold
        self.device = device
        
        self.metric = torch.eye(k, device=device, dtype=torch.float32)
        self.trajectories = []  # list of {"proj": tensor, "label": str, "step": int}
        self._proj_stack = None  # cached normalized stack
        self._dirty = True
        
        self.stats = {
            "total_interactions": 0,
            "expansions": 0,
            "metric_norms": [],
            "novelty_rates": [],
            "domain_hits": {},
        }
    
    def is_novel(self, proj_k):
        """Check if a projection is novel vs existing trajectories."""
        if not self.trajectories:
            return True, 1.0
        
        # Use batch search for speed
        if self._dirty:
            self._rebuild_cache()
        
        sims = F.normalize(proj_k.unsqueeze(0).float(), dim=1) @ self._proj_stack.T
        max_sim = sims.max().item()
        min_dist = 1.0 - max_sim  # approximate geodesic distance
        return min_dist > self.novelty_threshold, min_dist
    
    def expand(self, proj_k, label="", step=0):
        """Expand manifold with a novel trajectory."""
        # Jacobi metric update: g_ij += η * v_i * v_j / ||v||
        J = proj_k.unsqueeze(1) @ proj_k.unsqueeze(0)
        J = J / (torch.norm(J) + 1e-10)
        self.metric = self.metric + self.eta * J
        
        # Regularize: keep metric positive definite
        ev = torch.linalg.eigvalsh(self.metric)
        if ev.min() < 0.01:
            self.metric = self.metric + 0.01 * torch.eye(self.k, device=self.device)
        
        self.trajectories.append({
            "proj": proj_k.detach().cpu(),
            "label": label,
            "step": step,
        })
        self._dirty = True
        self.stats["expansions"] += 1
    
    def _rebuild_cache(self):
        """Pre-normalize and stack cached projections."""
        if self.trajectories:
            self._proj_stack = F.normalize(
                torch.stack([t["proj"] for t in self.trajectories]).float().to(self.device),
                dim=1
            )
        else:
            self._proj_stack = torch.empty(0, self.k, device=self.device)
        self._dirty = False
    
    def metric_norm(self):
        """Frobenius norm of (metric - I) — measures total manifold deformation."""
        return torch.norm(self.metric - torch.eye(self.k, device=self.device)).item()
    
    def state_dict(self):
        return {
            "metric": self.metric.cpu(),
            "trajectories": self.trajectories,
            "stats": self.stats,
            "k": self.k,
            "eta": self.eta,
            "novelty_threshold": self.novelty_threshold,
        }
    
    def load_state_dict(self, d):
        self.k = d["k"]
        self.eta = d["eta"]
        self.novelty_threshold = d["novelty_threshold"]
        self.metric = d["metric"].to(self.device)
        self.trajectories = d["trajectories"]
        self.stats = d["stats"]
        self._dirty = True


# ============================================================================
# MAIN RUNNER
# ============================================================================
def main(n_interactions=10000, model_id="Qwen/Qwen2.5-1.5B-Instruct", resume=False, quick=False):
    if quick:
        n_interactions = 1000
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("=" * 70)
    print("  COG 10K — Persistent Lifelong Learning Run")
    print(f"  Interactions: {n_interactions}, Model: {model_id}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)
    
    #  Load model 
    print(f"\n[1/5] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    d_model = model.config.hidden_size
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s, d={d_model}")
    
    #  Bootstrap UGT basis or resume 
    if resume and CHECKPOINT_PATH.exists():
        print(f"\n[2/5] Resuming from checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=DEVICE)
        cog = COGManifold(device=DEVICE)
        cog.load_state_dict(ckpt["cog"])
        basis = ckpt["basis"].to(DEVICE)
        mean = ckpt["mean"].to(DEVICE)
        start_step = ckpt["step"]
        print(f"  Resumed at step {start_step}, {len(cog.trajectories)} trajectories")
    else:
        print(f"\n[2/5] Bootstrapping UGT basis (k=64)...")
        # Small calibration set
        cal_prompts = [
            "Science is the study of the natural world.",
            "Mathematics is the language of patterns.",
            "Programming is the art of instructing computers.",
            "Philosophy explores fundamental questions of existence.",
            "History records human civilization's journey.",
            "Art expresses the human experience through creation.",
            "Music communicates emotion through organized sound.",
        ]
        model.eval()
        with torch.inference_mode():
            cal_states = batched_collect_hidden_states(model, tok, cal_prompts, batch_size=8)
        
        mean = cal_states.float().mean(dim=0).to(DEVICE)
        centered = cal_states.float().to(DEVICE) - mean
        U, S = smart_svd(centered.T, min(64, len(cal_prompts)))
        basis = U[:, :min(64, len(cal_prompts))].float().to(DEVICE)
        k_ugt = basis.shape[1]
        
        cog = COGManifold(k=k_ugt, device=DEVICE)
        start_step = 0
        print(f"  Basis: d={d_model} → k={k_ugt}")
    
    #  Generate prompts 
    print(f"\n[3/5] Generating {n_interactions} diverse prompts...")
    prompts = generate_prompts(n_interactions)
    print(f"  Generated {len(prompts)} prompts across 5 domains")
    
    #  Run interactions 
    print(f"\n[4/5] Running {n_interactions} COG interactions...")
    print(f"  {'Step':>6s} {'Novel':>6s} {'Metric':>8s} {'Trajs':>6s} {'Rate':>8s} {'Elapsed':>10s}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*10}")
    
    def project(text):
        """Extract hidden state and project through UGT basis."""
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[model.config.num_hidden_layers // 2][0, -1, :].float()
        centered = h - mean.float().to(h.device)
        return centered @ basis.float().to(h.device)
    
    t_run = time.perf_counter()
    checkpoint_interval = 100
    
    for step in range(start_step, n_interactions):
        prompt = prompts[step]
        proj_k = project(prompt)
        novel, dist = cog.is_novel(proj_k)
        
        if novel:
            cog.expand(proj_k, label=prompt[:80], step=step)
        
        mc = cog.metric_norm()
        cog.stats["total_interactions"] += 1
        cog.stats["metric_norms"].append(mc)
        
        # Compute novelty rate over last 100 steps
        if step >= 100:
            recent_novels = sum(1 for t in cog.trajectories if t["step"] > step - 100)
            novelty_rate = recent_novels / 100
        else:
            novelty_rate = len(cog.trajectories) / max(step + 1, 1)
        cog.stats["novelty_rates"].append(novelty_rate)
        
        # Logging
        if (step + 1) % 100 == 0 or step == start_step:
            elapsed = time.perf_counter() - t_run
            eta_str = f"{elapsed/60:.0f}m"
            print(f"  {step+1:6d} {len(cog.trajectories):6d} {mc:8.4f} "
                  f"{len(cog.trajectories):6d} {novelty_rate:8.4f} {eta_str:>10s}")
        
        # Checkpoint
        if (step + 1) % checkpoint_interval == 0:
            torch.save({
                "step": step + 1,
                "cog": cog.state_dict(),
                "basis": basis.cpu(),
                "mean": mean.cpu(),
                "model_id": model_id,
            }, CHECKPOINT_PATH)
    
    #  Final checkpoint 
    torch.save({
        "step": n_interactions,
        "cog": cog.state_dict(),
        "basis": basis.cpu(),
        "mean": mean.cpu(),
        "model_id": model_id,
    }, CHECKPOINT_PATH)
    
    #  Analysis 
    print(f"\n[5/5] Analysis...")
    total_time = time.perf_counter() - t_run
    
    final_metric = cog.metric_norm()
    total_trajectories = len(cog.trajectories)
    overall_novelty_rate = total_trajectories / n_interactions
    
    # Detect saturation: does novelty rate decay?
    window = 500
    if len(cog.stats["novelty_rates"]) >= window * 2:
        early_rate = np.mean(cog.stats["novelty_rates"][:window])
        late_rate = np.mean(cog.stats["novelty_rates"][-window:])
        saturation_ratio = late_rate / max(early_rate, 0.001)
    else:
        saturation_ratio = 1.0
    
    # Metric growth curve: is it still growing?
    if len(cog.stats["metric_norms"]) >= 1000:
        early_metric = np.mean(cog.stats["metric_norms"][100:200])
        late_metric = np.mean(cog.stats["metric_norms"][-100:])
        metric_growth = late_metric / max(early_metric, 0.001)
    else:
        metric_growth = 1.0
    
    # Verdict
    if saturation_ratio < 0.2 and metric_growth < 1.1:
        verdict = "SATURATED — manifold learning converges, lifelong bounds confirmed"
        paper_xv = "100% CLOSED"
    elif saturation_ratio < 0.4:
        verdict = "CONVERGING — novelty rate declining, metric stabilizing"
        paper_xv = "95% — near complete"
    elif saturation_ratio < 0.6:
        verdict = "STILL LEARNING — manifold still expanding"
        paper_xv = "85% — needs more interactions"
    else:
        verdict = "ACTIVE — high novelty rate persists"
        paper_xv = "75% — interesting, investigate"
    
    print(f"\n{'='*60}")
    print(f"  COG 10K RESULTS")
    print(f"  {'='*60}")
    print(f"  Total interactions:  {n_interactions}")
    print(f"  Total time:          {total_time/60:.1f} min")
    print(f"  Trajectories stored: {total_trajectories}")
    print(f"  Overall novelty rate: {overall_novelty_rate:.3f}")
    print(f"  Final metric norm:    {final_metric:.4f}")
    print(f"  Saturation ratio:     {saturation_ratio:.3f}")
    print(f"  Metric growth:        {metric_growth:.3f}")
    print(f"  VERDICT: {verdict}")
    print(f"  Paper XV: {paper_xv}")
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_interactions": n_interactions,
            "model_id": model_id,
            "k_ugt": cog.k,
            "eta": cog.eta,
            "novelty_threshold": cog.novelty_threshold,
        },
        "outcomes": {
            "total_trajectories": total_trajectories,
            "overall_novelty_rate": float(overall_novelty_rate),
            "final_metric_norm": float(final_metric),
            "saturation_ratio": float(saturation_ratio),
            "metric_growth": float(metric_growth),
            "total_time_min": round(total_time / 60, 1),
        },
        "verdict": verdict,
        "paper_xv": paper_xv,
        "metric_norms_sampled": cog.stats["metric_norms"][::10],  # every 10th
        "novelty_rates_sampled": cog.stats["novelty_rates"][::10],
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved to {OUT_DIR}/")
    print(f"  DONE")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COG 10K — Lifelong learning at scale")
    parser.add_argument("--n", type=int, default=10000, help="Number of interactions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quick", action="store_true", help="Quick 1K test with 0.5B model")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    main(args.n, args.model, args.resume, args.quick)
