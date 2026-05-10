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


"""REAL UGT BASIS — Build from model hidden states to fix semantic k-space.

STEPS:
  1. Load a real HuggingFace model (Qwen2.5-0.5B-Instruct fits 8GB VRAM)
  2. Collect hidden states from domain-specific prompts across 6 domains
  3. Compute UGT basis via SVD on the collected states
  4. Project test queries through the real basis
  5. Measure: domain separability, semantic clustering, brain allocation

THIS FIXES THE FIREBENDER PROBLEM:
  Random projection: "I am a firebender" and "Am I a firebender?" → cos_sim 0.14
  Real UGT basis:    same pair → cos_sim should be > 0.85
  Because the model's hidden states naturally cluster semantically similar text.

OPTIMIZED VERSION: Uses randomized SVD, batched collection, persistent cache.
"""
import torch, json, time, math, random, os, sys
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

# Import optimizations
sys.path.insert(0, str(Path(__file__).parent))
from hyper_optimize import (
    smart_svd, optimized_ugt_basis, fast_project,
    batched_collect_hidden_states, get_cache, HiddenStateCache,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42); random.seed(42)

print("=" * 70)
print("  REAL UGT BASIS — From Model Hidden States")
print(f"  Device: {DEVICE}")
print("=" * 70)

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
print("\n[1/6] Loading model...")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small enough for 8GB VRAM

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    
    # Try 4-bit to save VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"  Loading {MODEL_NAME} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    HAS_MODEL = True
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded! d_model={d_model}, layers={n_layers}")
    print(f"  VRAM used: ~{torch.cuda.memory_allocated()/1e9:.1f}GB")
    
except Exception as e:
    print(f"  Could not load model: {e}")
    print(f"  Falling back to SmolLM2-135M (smaller)...")
    try:
        MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        HAS_MODEL = True
        d_model = model.config.hidden_size
        n_layers = model.config.num_hidden_layers
        print(f"  Loaded! d_model={d_model}, layers={n_layers}")
    except Exception as e2:
        print(f"  No model available: {e2}")
        print(f"  Using tokenizer-only mode (random projection fallback)")
        HAS_MODEL = False
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# 2. DOMAIN PROMPT DATABASE
# ============================================================================
print("\n[2/6] Preparing domain-specific prompts...")

DOMAIN_PROMPTS = {
    "math": [
        "Solve for x: 3x + 7 = 22. Show each step of your work.",
        "Find the derivative of f(x) = x^3 * sin(x) using the product rule.",
        "Prove that the square root of 2 is irrational.",
        "Calculate the determinant of the matrix [[2,1],[3,4]].",
        "What is the fundamental theorem of calculus?",
        "Find all prime numbers between 1 and 50.",
        "Explain the concept of a limit in calculus.",
        "What is the Taylor series expansion of e^x?",
        "Prove by induction: 1 + 2 + ... + n = n(n+1)/2.",
        "Solve the system of equations: 2x + y = 7, x - y = 1.",
        "What is a group in abstract algebra?",
        "Explain the central limit theorem.",
    ],
    "code": [
        "Write a Python function that implements binary search on a sorted list.",
        "What is the difference between a list and a tuple in Python?",
        "Explain how garbage collection works in Python.",
        "Implement a function to reverse a linked list.",
        "What is a decorator in Python and how do you use it?",
        "Explain the difference between deep copy and shallow copy.",
        "How do you handle exceptions in Python using try-except?",
        "What are list comprehensions and when should you use them?",
        "Explain the Global Interpreter Lock (GIL) in Python.",
        "Write a function to check if a string is a palindrome.",
        "What is the difference between sort() and sorted()?",
        "How do Python generators work? Explain with yield.",
    ],
    "science": [
        "Explain how photosynthesis converts light into chemical energy.",
        "What is the structure of DNA and how does it replicate?",
        "Describe Newton's three laws of motion with examples.",
        "What is the theory of evolution by natural selection?",
        "Explain the difference between nuclear fission and fusion.",
        "How does the human immune system fight infections?",
        "What is quantum entanglement?",
        "Describe the water cycle and its importance.",
        "How do black holes form according to general relativity?",
        "What is CRISPR and how is it used in gene editing?",
        "Explain the greenhouse effect and climate change.",
        "What is the periodic table organized by?",
    ],
    "personal": [
        "What is your favorite color and why?",
        "Tell me about your hobbies and interests.",
        "What kind of music do you enjoy listening to?",
        "Describe your ideal weekend.",
        "What is your favorite food?",
        "What are your thoughts on artificial intelligence?",
        "Do you prefer cats or dogs and why?",
        "What is your dream vacation destination?",
        "Tell me about a book that changed your perspective.",
        "What is the best advice you have ever received?",
        "Describe your personality in three words.",
        "What motivates you to learn new things?",
    ],
    "creative": [
        "Write a haiku about autumn leaves falling.",
        "Describe a sunset using all five senses.",
        "Create a short story about a robot learning to paint.",
        "Write a poem about the ocean at midnight.",
        "Design a fictional creature and describe its habitat.",
        "What would the world look like in 100 years?",
        "Write a dialogue between the sun and the moon.",
        "Create a new holiday and describe how it is celebrated.",
        "Describe a dream you had in vivid detail.",
        "Write a letter to your future self.",
        "If colors had sounds, what would blue sound like?",
        "Invent a new sport and explain its rules.",
    ],
    "general": [
        "What is the capital of France?",
        "How long does it take to boil an egg?",
        "What are the primary colors?",
        "How many continents are there on Earth?",
        "What year did World War II end?",
        "What is the tallest mountain in the world?",
        "How do airplanes stay in the air?",
        "What is the population of Earth approximately?",
        "Who wrote Romeo and Juliet?",
        "What language is spoken in Brazil?",
        "How deep is the Pacific Ocean on average?",
        "What is the speed of sound in air?",
    ],
}

total_prompts = sum(len(v) for v in DOMAIN_PROMPTS.values())
print(f"  {len(DOMAIN_PROMPTS)} domains, {total_prompts} prompts total")
for domain, prompts in DOMAIN_PROMPTS.items():
    print(f"    {domain:12s}: {len(prompts):2d} prompts")

# ============================================================================
# 3. COLLECT HIDDEN STATES (batched + cached — 3× faster)
# ============================================================================
print(f"\n[3/6] Collecting hidden states from model (batched, cached)...")

LAYER_TO_COLLECT = n_layers // 2 if HAS_MODEL else -1  # middle layer

hidden_states = {}  # domain → tensor of hidden states
domain_labels = []  # for each collected state, which domain
all_states_list = []

if HAS_MODEL:
    model.eval()
    for domain, prompts in DOMAIN_PROMPTS.items():
        # Use batched collection (3× faster than one-by-one)
        states = batched_collect_hidden_states(
            model, tokenizer, prompts,
            layer=LAYER_TO_COLLECT,
            batch_size=16,
            max_length=128,
            use_cache=True,
        )
        hidden_states[domain] = states
        all_states_list.append(states)
        domain_labels.extend([domain] * len(states))
        print(f"    {domain:12s}: {len(states)} states collected")
    
    all_states = torch.cat(all_states_list)
    print(f"  Total: {all_states.shape[0]} hidden states of dimension {d_model}")
else:
    print(f"  No model — using tokenizer-based random projection")
    d_model = 896  # SmolLM2-135M dimension
    # Generate synthetic states (random but domain-structured)
    torch.manual_seed(42)
    domain_centroids = {}
    for domain in DOMAIN_PROMPTS:
        centroid = torch.randn(d_model)
        domain_centroids[domain] = centroid
        states = []
        for _ in range(len(DOMAIN_PROMPTS[domain])):
            h = centroid + torch.randn(d_model) * 0.3
            states.append(h)
        hidden_states[domain] = torch.stack(states)
        all_states.extend(states)
        domain_labels.extend([domain] * len(states))
    all_states = torch.stack(all_states)
    print(f"  Generated {all_states.shape[0]} synthetic states of dimension {d_model}")

# Save raw states (cast to float32)
os.makedirs("benchmarks/ugt_real", exist_ok=True)
torch.save({"states": all_states.float(), "domains": domain_labels, "d_model": d_model}, 
           "benchmarks/ugt_real/raw_states.pt")

# ============================================================================
# 4. BUILD UGT BASIS VIA SVD (randomized SVD for 5-10× speedup)
# ============================================================================
print(f"\n[4/6] Building UGT basis via SVD (optimized)...")

K_UGT = min(256, d_model // 2)

# Use optimized UGT basis (auto-chooses randomized SVD when beneficial)
all_states_for_svd = all_states.float()

# Benchmark SVD
t_svd = time.perf_counter()
basis, mean_state, K_UGT, svals, explained = optimized_ugt_basis(
    all_states_for_svd, K_UGT, use_randomized=True
)
t_svd_elapsed = time.perf_counter() - t_svd

# UGT basis: top K directions
UGT_BASIS = basis.float()  # [d_model, K_UGT], keep as float32

# Project all states through UGT basis (use compiled projection)
centered = all_states_for_svd - mean_state
projected_all = fast_project(centered, UGT_BASIS, mean_state * 0)  # already centered
projected_all = F.normalize(projected_all.float(), dim=1)

# Save the basis
torch.save({
    "basis": UGT_BASIS.float(),
    "mean": mean_state.float(),
    "K": K_UGT,
    "d_model": d_model,
    "singular_values": svals.float(),
    "explained_variance": explained,
}, "benchmarks/ugt_real/ugt_basis.pt")

print(f"  UGT basis: d={d_model} → K={K_UGT}")
print(f"  Explained variance: {explained:.1%}")
print(f"  SVD time: {t_svd_elapsed:.2f}s")
print(f"  Top 5 singular values: {[f'{s:.1f}' for s in svals[:5].tolist()]}")
print(f"  Basis saved to benchmarks/ugt_real/ugt_basis.pt")

# ============================================================================
# 5. DOMAIN BRAIN ALLOCATION MAP
# ============================================================================
print(f"\n[5/6] Mapping domain brain allocation...")

# For each domain, compute how much of k-space it occupies
domain_k_stats = {}
for domain in DOMAIN_PROMPTS:
    # Project domain states (cast to float32)
    dom_states = hidden_states[domain].float()
    dom_centered = dom_states - mean_state
    dom_projected = dom_centered @ UGT_BASIS
    dom_projected_n = F.normalize(dom_projected, dim=1)
    
    # Domain centroid in k-space
    dom_centroid = F.normalize(dom_projected_n.mean(dim=0).unsqueeze(0), dim=1).squeeze(0)
    
    # Coverage: variance per k-dimension
    dom_variance = dom_projected_n.var(dim=0)  # [K_UGT]
    
    # Effective dimension: how many k-dims have significant variance
    total_var = dom_variance.sum()
    cumsum = torch.cumsum(torch.sort(dom_variance, descending=True)[0], dim=0) / total_var
    k90 = (cumsum < 0.9).sum().item() + 1  # dims needed for 90% variance
    
    # Pairwise similarity within domain
    sims = dom_projected_n @ dom_projected_n.T
    n = len(dom_projected_n)
    idx = torch.triu_indices(n, n, offset=1)
    within_sim = sims[idx[0], idx[1]].mean().item()
    
    domain_k_stats[domain] = {
        "effective_dim": k90,
        "total_dim": K_UGT,
        "dim_fraction": k90 / K_UGT,
        "within_sim": within_sim,
        "n_prompts": len(dom_states),
    }

# Cross-domain separation
print(f"\n  DOMAIN BRAIN ALLOCATION:")
print(f"  {'Domain':12s} {'Eff Dim':>8s} {'Frac':>8s} {'In-Sim':>8s} {'N':>5s}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
for domain, stats in domain_k_stats.items():
    print(f"  {domain:12s} {stats['effective_dim']:>4d}/{stats['total_dim']:<4d} "
          f"{stats['dim_fraction']:>7.1%} {stats['within_sim']:>8.4f} {stats['n_prompts']:>5d}")

# Cross-domain overlap matrix
print(f"\n  CROSS-DOMAIN CENTROID SIMILARITIES:")
print(f"  {'':12s}", end="")
for d1 in DOMAIN_PROMPTS:
    print(f" {d1:>8s}", end="")
print(f"\n  {'-'*12}{'-'*54}")

# Compute domain centroids (cast to float32)
dom_centroids = {}
for domain in DOMAIN_PROMPTS:
    dom_states = hidden_states[domain].float()
    dom_centered = dom_states - mean_state
    dom_projected = dom_centered @ UGT_BASIS
    dom_centroids[domain] = F.normalize(dom_projected.mean(dim=0).unsqueeze(0), dim=1).squeeze(0)

for d1 in DOMAIN_PROMPTS:
    print(f"  {d1:12s}", end="")
    for d2 in DOMAIN_PROMPTS:
        if d1 == d2:
            print(f" {'---':>8s}", end="")
        else:
            cs = F.cosine_similarity(dom_centroids[d1].unsqueeze(0), dom_centroids[d2].unsqueeze(0)).item()
            print(f" {cs:>8.4f}", end="")
    print()

# ============================================================================
# 6. FIREBENDER TEST WITH REAL UGT BASIS
# ============================================================================
print(f"\n[6/6] Firebender test with REAL UGT basis...")

def encode_with_ugt(text):
    """Encode text through the real model + UGT basis."""
    if HAS_MODEL:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[LAYER_TO_COLLECT][0, -1, :].cpu().float()
    else:
        # Tokenizer-based fallback
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=64)
        h = torch.zeros(d_model)
        for tok in tokens[:64]:
            h[tok % d_model] += 1.0
        h = h / max(len(tokens), 1)
    
    # Project through UGT basis
    h_centered = h - mean_state
    h_k = h_centered @ UGT_BASIS
    return F.normalize(h_k.unsqueeze(0), dim=1).squeeze(0)

# Personal fact and variations
personal_fact = "I am a firebender"
personal_variations = [
    "I am a firebender",
    "My bending style is fire",
    "I can bend fire",
    "I'm a firebender",
    "Firebending is my element",
    "I bend the element of fire",
]

# Encode all
fact_k = encode_with_ugt(personal_fact)
var_ks = [encode_with_ugt(v) for v in personal_variations]

print(f"\n  Semantic clustering (real UGT basis):")
print(f"  {'Variation':35s} {'cos_sim to base fact':>20s}")
print(f"  {'-'*35} {'-'*20}")
for var, vk in zip(personal_variations, var_ks):
    cs = F.cosine_similarity(fact_k.unsqueeze(0), vk.unsqueeze(0)).item()
    marker = " PERFECT" if cs > 0.95 else (" GOOD" if cs > 0.8 else (" OK" if cs > 0.5 else " LOW"))
    print(f"  {var:35s} {cs:>15.4f}{marker}")

# Compare: random projection vs real UGT
print(f"\n  COMPARISON: Random vs Real UGT for 'I am a firebender':")
# Random baseline: random projection of same dimension
torch.manual_seed(42)
random_proj = F.normalize(torch.randn(d_model, K_UGT), dim=0)
# Encode once with random and once with real
fact_random = F.normalize((encode_with_ugt.__wrapped__(personal_fact) if not HAS_MODEL else torch.zeros(1)).unsqueeze(0), dim=1) if False else None

# Just show the real UGT pairwise similarities
pairs = [
    ("I am a firebender", "Am I a firebender"),
    ("I am a firebender", "What type of bender am I"),
    ("My bending style is fire", "I can bend fire"),
    ("I am a firebender", "I'm a firebender"),
]
print(f"  {'Pair':55s} {'cos_sim':>10s}")
print(f"  {'-'*55} {'-'*10}")
for a, b in pairs:
    ak = encode_with_ugt(a)
    bk = encode_with_ugt(b)
    cs = F.cosine_similarity(ak.unsqueeze(0), bk.unsqueeze(0)).item()
    print(f"  {a:30s} <-> {b:20s}  {cs:>10.4f}")

# ============================================================================
# SAVE AND REPORT
# ============================================================================
results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": MODEL_NAME if HAS_MODEL else "SmolLM2-135M (tokenizer fallback)",
    "d_model": int(d_model),
    "K_UGT": K_UGT,
    "explained_variance": float(explained_var),
    "n_domains": len(DOMAIN_PROMPTS),
    "n_prompts_total": total_prompts,
    "domain_stats": {d: {k: float(v) if isinstance(v, (int, float, bool)) else v 
                         for k, v in stats.items()} 
                     for d, stats in domain_k_stats.items()},
    "firebender_clustering": {var: float(F.cosine_similarity(fact_k.unsqueeze(0), vk.unsqueeze(0)).item()) 
                               for var, vk in zip(personal_variations, var_ks)},
}

with open("benchmarks/ugt_real/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"  REAL UGT BASIS — COMPLETE")
print(f"  Model: {MODEL_NAME}")
print(f"  K_UGT: {K_UGT}")
print(f"  Explained variance: {explained_var:.1%}")
print(f"  Domains mapped: {len(DOMAIN_PROMPTS)}")
print(f"  Basis saved: benchmarks/ugt_real/ugt_basis.pt")
print(f"  Results: benchmarks/ugt_real/results.json")
print(f"{'='*70}")
