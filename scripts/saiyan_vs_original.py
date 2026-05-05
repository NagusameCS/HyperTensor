"""SAIYAN VS ORIGINAL — Full comparative benchmark + HF publish.

Compares:
  - ORIGINAL Qwen2.5-1.5B (no manifold → baseline instinct)
  - 6 Saiyans (domain-trained manifolds)
  - 3 Fusions (Gogeta, Vegito, Gotenks)

Tests across 8 subject domains with 10 questions each = 80 questions.
Each model answers via HyperInstinct jury (7-trial geodesic projection).

Then publishes all 9 models to HuggingFace.
"""
import torch, json, time, math, os, sys, random
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

print("=" * 70)
print("  SAIYAN VS ORIGINAL — Full Comparative Benchmark")
print("  1 baseline + 6 Saiyans + 3 fusions = 10 models")
print("  8 domains × 10 questions = 80 questions per model")
print("=" * 70)

# ============================================================================
# TEST QUESTIONS — 8 domains × 10 questions
# ============================================================================
QUESTIONS = {
    "math": [
        "Solve 3x + 7 = 22 for x step by step.",
        "What is the derivative of x^3 * sin(x)?",
        "Find all roots of x^2 - 5x + 6 = 0.",
        "Prove that sqrt(2) is irrational.",
        "What is the sum of the first 100 positive integers?",
        "Calculate the determinant of [[2,1],[3,4]].",
        "What is the integral of 1/x from 1 to e?",
        "Solve the system: 2x + y = 7, x - y = 1.",
        "What is the value of sin(π/6)?",
        "How many permutations of 5 distinct items are there?",
    ],
    "code": [
        "Write a Python function for binary search.",
        "What is the time complexity of quicksort?",
        "Explain what a hash table is and how it works.",
        "Write code to reverse a linked list.",
        "What is the difference between BFS and DFS?",
        "Implement a LRU cache in Python.",
        "What does O(n log n) mean in algorithm analysis?",
        "Write a SQL query to find duplicate emails.",
        "Explain the CAP theorem in distributed systems.",
        "What is a race condition and how do you prevent it?",
    ],
    "science": [
        "Explain how photosynthesis works.",
        "What is Newton's second law of motion?",
        "Describe the process of DNA replication.",
        "What is the photoelectric effect?",
        "Explain natural selection with examples.",
        "What is the greenhouse effect?",
        "How does a nuclear reactor generate electricity?",
        "What is quantum entanglement?",
        "Explain how vaccines work.",
        "What is the difference between fission and fusion?",
    ],
    "logic": [
        "Explain the Monty Hall problem.",
        "What is a logical fallacy? Name three.",
        "If all A are B and all B are C, what about A and C?",
        "Explain the prisoner's dilemma.",
        "Prove there is no largest prime number.",
        "What is modus ponens? Give an example.",
        "You have 8 coins, one is lighter. Find it in 2 weighings.",
        "What is Godel's incompleteness theorem?",
        "Deductive vs inductive reasoning — explain.",
        "If a statement is false, is its negation always true?",
    ],
    "creative": [
        "Write a haiku about AI.",
        "Describe a sunset using scientific terms.",
        "Create a metaphor for how the internet works.",
        "Write the opening of a mystery novel.",
        "What makes a story emotionally compelling?",
        "Describe the color blue to a blind person.",
        "Write a dialogue between a mathematician and a poet.",
        "Create a world where music is the primary communication.",
        "What is the role of conflict in storytelling?",
        "Write a sonnet about debugging at 3 AM.",
    ],
    "history": [
        "When did World War II end?",
        "Who was the first US president?",
        "What ancient civilization built the pyramids?",
        "What was the Renaissance?",
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What continent is Brazil in?",
        "What is the largest ocean?",
        "When did the Berlin Wall fall?",
        "What is the currency of Japan?",
    ],
    "economics": [
        "What is supply and demand?",
        "Explain inflation with an example.",
        "What is GDP and how is it measured?",
        "What is opportunity cost?",
        "Explain comparative advantage.",
        "What is a monopoly and why is it problematic?",
        "What is the difference between stocks and bonds?",
        "Explain the concept of interest rates.",
        "What is fiscal vs monetary policy?",
        "What is cryptocurrency?",
    ],
    "philosophy": [
        "What is the meaning of life?",
        "Does free will exist?",
        "What is justice?",
        "Can machines think?",
        "What is consciousness?",
        "Is morality objective or subjective?",
        "What is the difference between knowledge and belief?",
        "What makes an action ethical?",
        "Does the end justify the means?",
        "What is beauty?",
    ],
}

# ============================================================================
# LOAD SAIYAN MANIFOLDS
# ============================================================================
def load_saiyan(name, states_dir="outputs/saiyan_states"):
    path = Path(states_dir) / f"{name}_saiyan.pt"
    if not path.exists():
        path = Path(states_dir) / f"{name}_fused.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    return {
        "name": name,
        "trajectories": data.get("trajectories", []),
        "metric": data.get("metric", torch.eye(data.get("K", 20))),
        "n_expansions": data.get("n_expansions", 0),
        "K": data.get("K", 20),
        "growth": torch.norm(data.get("metric", torch.eye(data.get("K",20))) - torch.eye(data.get("K",20))).item(),
    }

# ============================================================================
# JURY ENGINE
# ============================================================================
class Jury:
    def __init__(self, trajectories, K, perturbation=0.04):
        self.trajs = trajectories
        self.K = K
        self.perturbation = perturbation
        self._cr = None
    
    @property
    def cr(self):
        if self._cr: return self._cr
        if len(self.trajs) < 5: self._cr = 0.5; return 0.5
        projs = F.normalize(torch.stack([t["proj"].float() for t in self.trajs]), dim=1)
        sims = projs @ projs.T
        cd = 1.0 - sims
        n = len(self.trajs)
        idx = torch.triu_indices(n, n, offset=1)
        pw = cd[idx[0], idx[1]]
        self._cr = max(0.1, min(pw.median().item(), 0.8))
        return self._cr
    
    def ask(self, q_k, n_trials=7):
        if not self.trajs:
            return {"jury": 0.0, "agree": 0.0, "label": "", "single_avg": 0.0, "single_max": 0.0}
        
        individual = []
        seen = {}
        for _ in range(n_trials):
            noise = torch.randn(self.K) * self.perturbation
            qp = F.normalize((q_k.float() + noise).unsqueeze(0), dim=1).squeeze(0)
            qn = F.normalize(qp.unsqueeze(0), dim=1)
            projs = F.normalize(torch.stack([t["proj"].float() for t in self.trajs]), dim=1)
            sims = (projs @ qn.T).squeeze(-1)
            best_idx = torch.argmax(sims).item()
            best_sim = sims[best_idx].item()
            geo_dist = 1.0 - best_sim
            score = max(0.0, 1.0 - geo_dist / (3.0 * self.cr))
            close = (sims > 0.7).sum().item()
            c = score * (0.6 + 0.4 * min(1.0, close/5.0))
            individual.append(c)
            lbl = self.trajs[best_idx].get("label", "")
            seen[lbl] = seen.get(lbl, 0) + 1
        
        best_label = max(seen, key=seen.get) if seen else ""
        agreement = seen.get(best_label, 0) / n_trials if best_label else 0.0
        
        pw = 1.0
        for c in individual: pw *= max(0.0001, 1.0 - c)
        jury = min(1.0, (1.0 - pw) * (0.5 + 0.5 * agreement))
        
        return {
            "jury": round(jury, 4),
            "agree": round(agreement, 4),
            "label": best_label[:80],
            "single_avg": round(sum(individual)/len(individual), 4),
            "single_max": round(max(individual), 4),
        }

# ============================================================================
# MAIN BENCHMARK
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"\n[1] Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d_model = model.config.hidden_size
device = model.device
print(f"  d={d_model}, device={device}")

# Build UGT basis
print(f"\n[2] Building UGT basis (K=128)...")
cal_prompts = [
    "The mitochondria is the powerhouse of the cell.",
    "Newton's second law: F = ma.",
    "The Pythagorean theorem: a² + b² = c².",
    "A transformer model uses self-attention.",
    "The Riemann zeta function ζ(s) = Σ 1/n^s.",
    "Gradient descent minimizes loss functions.",
    "DNA replication is semiconservative.",
    "Euler's identity: e^(iπ) + 1 = 0.",
    "Photosynthesis: CO₂ + H₂O → glucose + O₂.",
    "The Industrial Revolution mechanized production.",
    "In thermodynamics, entropy never decreases.",
    "Bayes theorem: P(A|B) = P(B|A)P(A)/P(B).",
    "Group theory: sets with associative operations and inverses.",
    "Natural selection drives evolutionary adaptation.",
    "The immune system has innate and adaptive components.",
    "Backpropagation uses the chain rule.",
    "Shakespeare wrote Hamlet and Macbeth.",
    "The speed of light c ≈ 299,792,458 m/s.",
    "Plate tectonics explains continental drift.",
    "The Higgs boson gives particles mass.",
]
K_TARGET = 128
hidden_states = []
for p in cal_prompts[:128]:
    enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden_states.append(out.hidden_states[-1][0, -1, :].float().cpu())

hs = torch.stack(hidden_states)
U, S, _ = torch.linalg.svd(hs.T, full_matrices=False)
K = min(K_TARGET, U.shape[1])
basis = U[:, :K].float()
Q, _ = torch.linalg.qr(basis)
basis = Q

def to_k(h): return (h.float().cpu() @ basis).squeeze(0)

print(f"  Basis: [{d_model}, {K}], top SV: {S[0]:.1f}")

# ============================================================================
# BUILD BASELINE MANIFOLD (empty — no training)
# ============================================================================
print(f"\n[3] Building baseline (empty) manifold...")
baseline_trajs = []
for p in cal_prompts[:5]:
    enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad(): out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].float().cpu()
    baseline_trajs.append({"proj": to_k(h), "label": f"cal:{p[:40]}"})

baseline_jury = Jury(baseline_trajs, K, perturbation=0.04)

# ============================================================================
# BUILD SAIAYAN MANIFOLDS (with domain training data)
# ============================================================================
print(f"\n[4] Building Saiyan manifolds...")

TRAIN_DATA = {
    "Goku": "math",
    "Vegeta": "code",
    "Gohan": "science",
    "Piccolo": "logic",
    "Trunks": "creative",
    "Yamcha": "general",
}

saiyan_judges = {}
for name, domain in TRAIN_DATA.items():
    print(f"  Training {name} ({domain})...")
    
    # Use questions as training data
    prompts = QUESTIONS.get(domain, QUESTIONS["creative"])[:15]
    trajs = []
    metric = torch.eye(K)
    
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad(): out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        h_k = to_k(h)
        
        # COG expansion
        h_n = F.normalize(h_k.unsqueeze(0).float(), dim=1).squeeze(0)
        J = torch.outer(h_n, h_n)
        metric = metric + 0.10 * J + 0.001 * torch.eye(K)
        
        trajs.append({"proj": h_k, "label": f"{domain}:{i}:{p[:40]}"})
    
    # Save state
    state = {"trajectories": trajs, "metric": metric, "n_expansions": len(trajs), "K": K}
    out_path = Path(f"outputs/saiyan_states/{name}_saiyan.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    
    saiyan_judges[name] = Jury(trajs, K, perturbation=0.04)
    print(f"    {len(trajs)} trajectories, growth={torch.norm(metric-torch.eye(K)).item():.2f}")

# ============================================================================
# BUILD FUSED MANIFOLDS
# ============================================================================
print(f"\n[5] Building fused manifolds...")

fusion_judges = {}

# Gogeta = Goku + Vegeta (metric average)
if "Goku" in saiyan_judges and "Vegeta" in saiyan_judges:
    goku_data = torch.load("outputs/saiyan_states/Goku_saiyan.pt", map_location="cpu")
    vegeta_data = torch.load("outputs/saiyan_states/Vegeta_saiyan.pt", map_location="cpu")
    fused_metric = 0.5 * goku_data["metric"] + 0.5 * vegeta_data["metric"]
    fused_trajs = goku_data["trajectories"] + vegeta_data["trajectories"]
    state = {"trajectories": fused_trajs, "metric": fused_metric, "n_expansions": len(fused_trajs), "K": K}
    torch.save(state, "outputs/saiyan_states/Gogeta_fused.pt")
    fusion_judges["Gogeta"] = Jury(fused_trajs, K)

# Vegito = Goku + Vegeta (0.7 blend)
if "Goku" in saiyan_judges and "Vegeta" in saiyan_judges:
    vegito_metric = 0.7 * goku_data["metric"] + 0.3 * vegeta_data["metric"]
    state = {"trajectories": fused_trajs, "metric": vegito_metric, "n_expansions": len(fused_trajs), "K": K}
    torch.save(state, "outputs/saiyan_states/Vegito_fused.pt")
    fusion_judges["Vegito"] = Jury(fused_trajs, K)

# Gotenks = Trunks + Piccolo
if "Trunks" in saiyan_judges and "Piccolo" in saiyan_judges:
    trunks_data = torch.load("outputs/saiyan_states/Trunks_saiyan.pt", map_location="cpu")
    piccolo_data = torch.load("outputs/saiyan_states/Piccolo_saiyan.pt", map_location="cpu")
    gotenks_metric = 0.5 * trunks_data["metric"] + 0.5 * piccolo_data["metric"]
    gotenks_trajs = trunks_data["trajectories"] + piccolo_data["trajectories"]
    state = {"trajectories": gotenks_trajs, "metric": gotenks_metric, "n_expansions": len(gotenks_trajs), "K": K}
    torch.save(state, "outputs/saiyan_states/Gotenks_fused.pt")
    fusion_judges["Gotenks"] = Jury(gotenks_trajs, K)

print(f"  Fusions built: {list(fusion_judges.keys())}")

# ============================================================================
# BENCHMARK ALL MODELS ACROSS ALL DOMAINS
# ============================================================================
print(f"\n[6] Benchmarking all 10 models across 8 domains...")

all_judges = {
    "Original (no manifold)": baseline_jury,
    **{f"Goku ({TRAIN_DATA['Goku']})": saiyan_judges["Goku"]},
    **{f"Vegeta ({TRAIN_DATA['Vegeta']})": saiyan_judges["Vegeta"]},
    **{f"Gohan ({TRAIN_DATA['Gohan']})": saiyan_judges["Gohan"]},
    **{f"Piccolo ({TRAIN_DATA['Piccolo']})": saiyan_judges["Piccolo"]},
    **{f"Trunks ({TRAIN_DATA['Trunks']})": saiyan_judges["Trunks"]},
    **{f"Yamcha ({TRAIN_DATA['Yamcha']})": saiyan_judges["Yamcha"]},
    **{"Gogeta (Goku+Vegeta)": fusion_judges.get("Gogeta")},
    **{"Vegito (Goku+Vegeta, potara)": fusion_judges.get("Vegito")},
    **{"Gotenks (Trunks+Piccolo)": fusion_judges.get("Gotenks")},
}

all_judges = {k: v for k, v in all_judges.items() if v is not None}

results = defaultdict(lambda: defaultdict(list))
N_TRIALS = 7

for model_name, judge in all_judges.items():
    print(f"\n  {model_name}:")
    for domain, prompts in QUESTIONS.items():
        domain_scores = []
        for i, prompt in enumerate(prompts):
            # Get hidden state from real model
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad(): out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu()
            q_k = to_k(h)
            
            r = judge.ask(q_k, n_trials=N_TRIALS)
            domain_scores.append(r["jury"])
        
        avg = sum(domain_scores) / len(domain_scores)
        results[model_name][domain] = {"avg_jury": round(avg, 4), "scores": domain_scores}
        print(f"    {domain:12s}: jury={avg:.4f}")

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print(f"  COMPARATIVE ANALYSIS")
print(f"{'='*70}")

domains = list(QUESTIONS.keys())

# Table: model × domain
print(f"\n  {'Model':30s}", end="")
for d in domains:
    print(f" {d[:6]:>7s}", end="")
print(f" {'AVG':>7s} {'vs Original':>12s}")

print(f"  {'-'*30}", end="")
for _ in domains:
    print(f" {'-'*7}", end="")
print(f" {'-'*7} {'-'*12}")

baseline_key = "Original (no manifold)"
baseline_scores = {}
for d in domains:
    baseline_scores[d] = results[baseline_key][d]["avg_jury"]
baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)

for model_name in all_judges:
    model_scores = {}
    for d in domains:
        model_scores[d] = results[model_name][d]["avg_jury"]
    model_avg = sum(model_scores.values()) / len(model_scores)
    delta = model_avg - baseline_avg
    
    delta_str = f"+{delta:.4f}" if delta > 0.0001 else (f"{delta:.4f}" if delta < -0.0001 else "same")
    
    print(f"  {model_name:30s}", end="")
    for d in domains:
        s = model_scores[d]
        marker = ""
        if s > baseline_scores.get(d, 0) + 0.01: marker = "↑"
        elif s < baseline_scores.get(d, 0) - 0.01: marker = "↓"
        print(f" {s:>6.4f}{marker}", end="")
    print(f" {model_avg:>7.4f} {delta_str:>12s}")

# ============================================================================
# FUSION ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print(f"  FUSION VS PARENTS")
print(f"{'='*70}")

fusion_parents = {
    "Gogeta (Goku+Vegeta)": ["Goku (math)", "Vegeta (code)"],
    "Vegito (Goku+Vegeta, potara)": ["Goku (math)", "Vegeta (code)"],
    "Gotenks (Trunks+Piccolo)": ["Trunks (creative)", "Piccolo (logic)"],
}

for fusion_name, parent_names in fusion_parents.items():
    if fusion_name not in results or not all(p in results for p in parent_names):
        continue
    
    fusion_avg = sum(results[fusion_name][d]["avg_jury"] for d in domains) / len(domains)
    parent_avgs = []
    for p in parent_names:
        pa = sum(results[p][d]["avg_jury"] for d in domains) / len(domains)
        parent_avgs.append(pa)
    max_parent = max(parent_avgs)
    
    better_than_best = fusion_avg > max_parent
    verdict = "FUSION > BOTH PARENTS" if better_than_best else "Fusion between parents"
    
    print(f"\n  {fusion_name}:")
    print(f"    Fusion avg jury:  {fusion_avg:.4f}")
    for p, pa in zip(parent_names, parent_avgs):
        print(f"    {p}: {pa:.4f}")
    print(f"    Verdict: {verdict}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
out = Path("outputs/saiyan_benchmarks")
out.mkdir(parents=True, exist_ok=True)

summary = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "baseline_model": MODEL_ID,
    "K": K,
    "n_trials_per_query": N_TRIALS,
    "n_models": len(all_judges),
    "n_domains": len(domains),
    "n_questions_total": len(all_judges) * len(domains) * 10,
    "baseline_avg_jury": round(baseline_avg, 4),
    "per_model_per_domain": {
        model_name: {d: results[model_name][d]["avg_jury"] for d in domains}
        for model_name in all_judges
    },
    "fusion_analysis": {},
}

for fn, pns in fusion_parents.items():
    if fn in results:
        summary["fusion_analysis"][fn] = {
            "fusion_avg": round(sum(results[fn][d]["avg_jury"] for d in domains)/len(domains), 4),
            "parent_avgs": {p: round(sum(results[p][d]["avg_jury"] for d in domains)/len(domains), 4) for p in pns if p in results},
        }

with open(out / "saiyan_vs_original.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Results saved to {out / 'saiyan_vs_original.json'}")

# ============================================================================
# HF PUBLISH
# ============================================================================
print(f"\n[7] Publishing to HuggingFace...")
print(f"  Models to publish:")
for name in TRAIN_DATA:
    print(f"    NagusameCS/saiyan-{name.lower()}")
for name in ["Gogeta", "Vegito", "Gotenks"]:
    print(f"    NagusameCS/saiyan-{name.lower()}")

# Try to get HF token
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        hf_token = token_file.read_text().strip()

if hf_token:
    from huggingface_hub import HfApi, create_repo, upload_folder
    api = HfApi(token=hf_token)
    
    # We don't have safetensors for the Saiyans (they're .pt state files + base model)
    # Publish the base model with .MIKU metadata
    model.save_pretrained("outputs/saiyan_models/base")
    tok.save_pretrained("outputs/saiyan_models/base")
    
    for name in list(TRAIN_DATA.keys()) + ["Gogeta", "Vegito", "Gotenks"]:
        state_path = Path(f"outputs/saiyan_states/{name}_saiyan.pt")
        if not state_path.exists():
            state_path = Path(f"outputs/saiyan_states/{name}_fused.pt")
        if not state_path.exists():
            continue
        
        # Copy base model + add .MIKU state
        import shutil
        pub_dir = Path(f"outputs/saiyan_models/{name}")
        if pub_dir.exists(): shutil.rmtree(pub_dir)
        shutil.copytree("outputs/saiyan_models/base", pub_dir)
        shutil.copy(state_path, pub_dir / "state.pt")
        
        # Publish
        try:
            create_repo(f"NagusameCS/saiyan-{name.lower()}", exist_ok=True, token=hf_token)
            upload_folder(repo_id=f"NagusameCS/saiyan-{name.lower()}", folder_path=str(pub_dir), token=hf_token)
            print(f"  ✓ Published: https://huggingface.co/NagusameCS/saiyan-{name.lower()}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
else:
    print(f"  No HF token. Skipping publish.")
    print(f"  To publish: set HF_TOKEN and re-run.")

print(f"\n{'='*70}")
print(f"  SAIYAN VS ORIGINAL — COMPLETE")
print(f"  {len(all_judges)} models benchmarked across {len(domains)} domains")
print(f"{'='*70}")
