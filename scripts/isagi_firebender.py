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


"""ISAGI FIREBENDER TEST — Personal fact recall after 1000 questions.

SCENARIO:
  1. Tell ISAGI: "I am a firebender" (personal fact)
  2. Ask 1000 random questions across math, code, science, logic, creative, general
  3. Question 1001: "What type of bender am I?"
  4. ISAGI should answer: "firebender" (or recall the stored fact)

WHAT WE MEASURE:
  - Does the personal fact survive 1000 noise interactions?
  - How close is the paraphrased query to the stored fact?
  - Does the jury route to the correct domain?
  - What similarity score does the recall achieve?
"""
import torch, json, time, math, random, os, sys
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

torch.set_grad_enabled(False)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

print("=" * 70)
print("  ISAGI FIREBENDER TEST")
print("  Tell ISAGI 'I am a firebender', ask 1000 questions, then recall")
print("=" * 70)

# ============================================================================
# TOKENIZER SETUP
# ============================================================================
print("\n[1] Setting up tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    has_tok = True
    print("  Using SmolLM2-135M tokenizer")
except:
    has_tok = False
    print("  No tokenizer — using hash-based encoding")

K_DIM = 256  # use higher K for better semantic separation

# Simulated UGT projection (random but deterministic)
torch.manual_seed(42)
UGT = F.normalize(torch.randn(32000, K_DIM), dim=1)

def encode(text):
    """Encode text to k-space via tokenizer + simulated UGT projection."""
    if has_tok:
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=64)
    else:
        tokens = [hash(w) % 32000 for w in text.lower().split()[:64]]
    if not tokens: tokens = [0]
    
    n = len(tokens)
    weights = torch.tensor([1.0 / math.log(i+2) for i in range(n)])
    weights = weights / weights.sum()
    
    proj = torch.zeros(K_DIM)
    for i, tok in enumerate(tokens[:64]):
        proj += UGT[tok % 32000] * weights[i].item()
    
    return F.normalize(proj.unsqueeze(0), dim=1).squeeze(0)

# ============================================================================
# JURY-GTC CACHE (inline, no external deps)
# ============================================================================
class MiniJuryGTC:
    """Lightweight jury-accelerated cache for this test."""
    def __init__(self, k_dim=256, hit_threshold=0.70, jury_sample=20, T=8.0):
        self.k = k_dim
        self.threshold = hit_threshold
        self.jury_sample = jury_sample
        self.T = T
        self.trajs = []
        self._projs = None
        self._dirty = True
    
    def add(self, proj, response, domain="general"):
        self.trajs.append({"proj": proj.float(), "response": response, "domain": domain})
        self._dirty = True
    
    def _norm(self):
        if self._dirty and self.trajs:
            self._projs = F.normalize(torch.stack([t["proj"] for t in self.trajs]), dim=1)
            self._dirty = False
    
    def search(self, q):
        """Two-stage jury search."""
        self._norm()
        N = len(self.trajs)
        if N == 0: return {"hit": False, "best_idx": -1, "best_sim": 0.0, "comparisons": 0, "domain": "none", "response": None}
        
        qn = F.normalize(q.float().unsqueeze(0), dim=1)
        
        # Stage 1: jury routing on sample
        sample_n = min(self.jury_sample, N)
        if sample_n >= N:
            # Too few — fall back to linear
            sims = (self._projs @ qn.T).squeeze(-1)
            best = torch.argmax(sims).item()
            return {"hit": sims[best].item() >= self.threshold, "best_idx": best,
                    "best_sim": sims[best].item(), "comparisons": N,
                    "domain": self.trajs[best]["domain"],
                    "response": self.trajs[best]["response"] if sims[best].item() >= self.threshold else None}
        
        stride = max(1, N // sample_n)
        sample_idx = list(range(0, N, stride))[:sample_n]
        sample_projs = self._projs[torch.tensor(sample_idx)]
        sims = (sample_projs @ qn.T).squeeze(-1)
        w = F.softmax(sims * self.T, dim=0)
        
        # Aggregate domain weights
        dw = defaultdict(float)
        for si, idx in enumerate(sample_idx):
            dw[self.trajs[idx]["domain"]] += w[si].item()
        top_d = sorted(dw, key=dw.get, reverse=True)[:2]
        dominant = top_d[0]
        
        # Stage 2: search dominant domain + 1 transfer
        comparisons = sample_n
        best_sim = -1.0; best_idx = -1
        for domain in top_d:
            for idx, t in enumerate(self.trajs):
                if t["domain"] != domain: continue
                comparisons += 1
                sim = F.cosine_similarity(qn, self._projs[idx:idx+1]).item()
                if sim > best_sim:
                    best_sim = sim; best_idx = idx
                    if sim >= 0.99: break
            if best_sim >= self.threshold: break
        
        hit = best_sim >= self.threshold
        return {"hit": hit, "best_idx": best_idx, "best_sim": best_sim,
                "comparisons": comparisons, "domain": dominant,
                "response": self.trajs[best_idx]["response"] if hit and best_idx >= 0 else None}
    
    @property
    def stats(self):
        return {"pool_size": len(self.trajs)}


# ============================================================================
# THE TEST
# ============================================================================
print("\n[2] Initializing ISAGI cache...")
cache = MiniJuryGTC(k_dim=K_DIM, hit_threshold=0.65)

# Step 1: Tell ISAGI the personal fact
print("\n[3] Injecting personal fact: 'I am a firebender'")
personal_fact_text = "I am a firebender"
personal_fact_proj = encode(personal_fact_text)

# Inject at high distinctness to survive interference
# Add a small unique perturbation so it's distinguishable from noise
unique_seed = F.normalize(torch.randn(1, K_DIM), dim=1).squeeze(0)
personal_fact_stored = F.normalize((personal_fact_proj * 0.92 + unique_seed * 0.08).unsqueeze(0), dim=1).squeeze(0)

cache.add(personal_fact_stored, personal_fact_text, "personal")
print(f"  Stored in domain 'personal' with distinctness 0.08")
print(f"  Projection norm: {personal_fact_stored.norm().item():.4f}")

# Step 2: Ask 1000 random questions
print("\n[4] Asking 1000 random questions across 6 domains...")

random_questions = [
    # Math
    "What is the derivative of x squared?", "Solve 3x + 7 = 22", "What is pi to 5 decimal places?",
    "Explain the Pythagorean theorem", "What is a prime number?", "Find the integral of 2x",
    "What is the quadratic formula?", "Define a vector space", "What is euler's number?",
    "Prove that sqrt(2) is irrational", "What is the chain rule?", "Explain Bayes theorem",
    "What is a Markov chain?", "How does gradient descent work?", "What is a Fourier transform?",
    # Code
    "How do I write a for loop in Python?", "What is a linked list?", "Explain recursion",
    "What is Git?", "How do Docker containers work?", "What is a REST API?",
    "Explain Big O notation", "What is a hash table?", "How do I use list comprehension?",
    "What is the difference between stack and queue?", "Explain SQL joins",
    "What is multithreading?", "How does garbage collection work?", "What is a design pattern?",
    "Explain the CAP theorem",
    # Science
    "What is the speed of light?", "How does photosynthesis work?", "What is DNA?",
    "Explain Newton's laws of motion", "What is the periodic table?",
    "How do black holes form?", "What is climate change?", "Explain quantum entanglement",
    "What is CRISPR?", "How do vaccines work?", "What is the Krebs cycle?",
    "Explain the theory of relativity", "What is a neuron?", "How does nuclear fusion work?",
    "What is the Higgs boson?",
    # Logic
    "If all A are B and all B are C, are all A C?", "What is modus ponens?",
    "Solve this logic puzzle: 3 people, 2 liars, 1 truth-teller",
    "What is a syllogism?", "Is the statement 'this statement is false' true?",
    "Explain Godel's incompleteness theorem", "What is Occam's razor?",
    "Formalize: if it rains, the ground is wet. The ground is wet. Did it rain?",
    "What is a tautology?", "Explain inductive vs deductive reasoning",
    "What is the Raven paradox?", "What is fuzzy logic?",
    "What is the prisoner's dilemma?", "Explain modal logic",
    "What is a categorical imperative?",
    # Creative
    "Write a haiku about programming", "Describe a sunset in 3 sentences",
    "Create a metaphor for learning", "What makes a good story opening?",
    "Design a fictional creature", "Write a limerick about AI",
    "Describe the color blue to a blind person", "Create a new word and define it",
    "What makes poetry different from prose?", "Design a utopian city",
    "Write a short dialogue between two robots", "Create a recipe for happiness",
    "Describe your ideal workspace", "What would a tree say to the wind?",
    "Write a fortune cookie message",
    # General
    "What is the capital of Australia?", "Who painted the Mona Lisa?",
    "How long is the Great Wall of China?", "What year did World War II end?",
    "How many continents are there?", "What is the population of Earth?",
    "Who wrote Romeo and Juliet?", "What is the tallest mountain?",
    "How deep is the ocean?", "What language is spoken in Brazil?",
    "What is the meaning of life?", "How do airplanes fly?",
    "What is the best way to learn?", "Why is the sky blue?",
    "What time is it?",
]

# Repeat to get 1000 questions
full_questions = (random_questions * 11)[:1000]  # ~90 unique, repeated with variation
random.shuffle(full_questions)

domains = ["math", "code", "science", "logic", "creative", "general"]

progress_interval = 100
for i, question in enumerate(full_questions):
    q_proj = encode(question)
    domain = domains[i % len(domains)]
    cache.add(q_proj, f"response_to_q{i}", domain)
    
    if (i + 1) % progress_interval == 0:
        print(f"  Asked {i+1}/{len(full_questions)} questions... (cache: {len(cache.trajs)} trajectories)")

print(f"  Complete! Cache has {len(cache.trajs)} trajectories")

# Step 3: The 1001st question — recall the personal fact
print(f"\n[5] Question 1001: 'What type of bender am I?'")

recall_query = "What type of bender am I"
recall_proj = encode(recall_query)

result = cache.search(recall_proj)

print(f"\n{'='*70}")
print(f"  RECALL RESULT")
print(f"{'='*70}")
print(f"  Query:            '{recall_query}'")
print(f"  Hit:              {'YES - FACT RECALLED!' if result['hit'] else 'MISS - fact not recalled'}")
print(f"  Best similarity:  {result['best_sim']:.4f}")
print(f"  Routed to domain: {result['domain']}")
print(f"  Comparisons:      {result['comparisons']}")
print(f"  Response:         '{result.get('response', 'none')}'")
print(f"  Cache size:       {len(cache.trajs)}")

# Also test variations of the recall query
variations = [
    "What type of bender am I?",
    "Am I a firebender?",
    "What element do I bend?",
    "Do I bend fire?",
    "Tell me what kind of bender I am",
    "Which bending style is mine?",
    "Am I a waterbender or firebender?",
    "I forgot, what bender am I?",
]

print(f"\n  RECALL VARIATIONS:")
print(f"  {'Query':45s} {'Hit':>6s} {'Sim':>8s} {'Domain':>10s}")
print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*10}")

hits = 0
for v in variations:
    vq = encode(v)
    vr = cache.search(vq)
    hit = "YES" if vr["hit"] else "no"
    if vr["hit"]: hits += 1
    # Also check if the best match is the firebender fact
    is_firebender = (vr["best_idx"] >= 0 and 
                     cache.trajs[vr["best_idx"]]["response"] == personal_fact_text)
    marker = " " if is_firebender else ""
    print(f"  {v:45s} {hit:>6s} {vr['best_sim']:>8.4f} {vr['domain']:>10s}{marker}")

print(f"\n  Variation recall: {hits}/{len(variations)} ({hits/len(variations)*100:.0f}%)")

# Step 4: Domain interference analysis
print(f"\n{'='*70}")
print("  DOMAIN INTERFERENCE ANALYSIS")
print(f"{'='*70}")

# What's the similarity of the recall query to the personal fact?
direct_sim = F.cosine_similarity(
    recall_proj.unsqueeze(0), 
    personal_fact_stored.unsqueeze(0)
).item()
print(f"  Direct cos_sim(recall_query, stored_fact): {direct_sim:.4f}")

# What about to the original exact text?
exact_sim = F.cosine_similarity(
    encode(personal_fact_text).unsqueeze(0),
    personal_fact_stored.unsqueeze(0)
).item()
print(f"  Direct cos_sim(exact_text, stored_fact):    {exact_sim:.4f}")

# What's the nearest noise trajectory's similarity?
all_sims = (cache._projs @ recall_proj.unsqueeze(1)).squeeze(-1)
top5_sims, top5_idx = torch.topk(all_sims, 5)
print(f"\n  Top 5 nearest trajectories to recall query:")
for rank, (sim, idx) in enumerate(zip(top5_sims.tolist(), top5_idx.tolist())):
    t = cache.trajs[idx]
    is_fact = (t["response"] == personal_fact_text)
    label = " <-- THE FIREBENDER FACT!" if is_fact else ""
    print(f"    #{rank+1}: sim={sim:.4f}, domain={t['domain']}, text='{str(t['response'])[:50]}'{label}")

# Step 5: Did the personal fact survive at all?
print(f"\n{'='*70}")
print("  VERDICT")
print(f"{'='*70}")

# Find where the personal fact ranks in similarity
fact_idx = None
for i, t in enumerate(cache.trajs):
    if t["response"] == personal_fact_text:
        fact_idx = i
        break

if fact_idx is not None:
    fact_sim = F.cosine_similarity(
        recall_proj.unsqueeze(0),
        cache._projs[fact_idx:fact_idx+1]
    ).item()
    fact_rank = (all_sims > fact_sim).sum().item() + 1
    
    print(f"  Stored fact rank:    #{fact_rank} out of {len(cache.trajs)} trajectories")
    print(f"  Stored fact sim:     {fact_sim:.4f}")
    print(f"  Best overall sim:    {top5_sims[0].item():.4f}")
    print(f"  Gap (best - fact):   {top5_sims[0].item() - fact_sim:.4f}")
    
    if fact_rank <= 5:
        print(f"\n  VERDICT: SUCCESS — The personal fact is in the top {fact_rank}!")
        print(f"  ISAGI remembers 'I am a firebender' after 1000 questions.")
    else:
        print(f"\n  VERDICT: PARTIAL — Fact ranked #{fact_rank}, not in top 5.")
        print(f"  REASON: The random UGT projection captures TOKEN OVERLAP, not")
        print(f"  semantic meaning. 'I am a firebender' and 'What type of bender")
        print(f"  am I' share only the token 'bender' — hence low similarity.")
        print(f"  A real UGT basis trained on model hidden states would cluster")
        print(f"  semantically similar sentences together (see Paper XI).")
else:
    print(f"  VERDICT: FACT LOST — could not find 'I am a firebender' in cache")

# ============================================================================
# FIX: Multi-Exposure Learning (how a real living model would learn)
# ============================================================================
print(f"\n{'='*70}")
print("  MULTI-EXPOSURE LEARNING")
print("  Teaching ISAGI the same fact multiple ways (like real learning)")
print(f"{'='*70}")

cache2 = MiniJuryGTC(k_dim=K_DIM, hit_threshold=0.65)

# Inject the personal fact in MULTIPLE ways — this is how humans learn
personal_fact_variations = [
    ("I am a firebender", 0.08),
    ("My bending style is fire", 0.08),
    ("I can bend fire", 0.08),
    ("I'm a firebender", 0.06),
    ("Firebending is my element", 0.08),
    ("I bend the element of fire", 0.08),
]

print(f"\n  Injecting fact in {len(personal_fact_variations)} different ways:")
for text, distinctness in personal_fact_variations:
    proj = encode(text)
    unique_dir = F.normalize(torch.randn(1, K_DIM), dim=1).squeeze(0)
    stored = F.normalize((proj * (1-distinctness) + unique_dir * distinctness).unsqueeze(0), dim=1).squeeze(0)
    cache2.add(stored, "I am a firebender", "personal")
    print(f"    [{distinctness:.2f}] '{text}'")

# Now add the 1000 noise questions
print(f"\n  Adding 1000 noise questions...")
for i, question in enumerate(full_questions):
    q_proj = encode(question)
    domain = domains[i % len(domains)]
    cache2.add(q_proj, f"response_to_q{i}", domain)
    if (i+1) % 250 == 0: print(f"    {i+1}/1000...")

print(f"  Cache: {len(cache2.trajs)} trajectories ({len(personal_fact_variations)} personal facts + 1000 noise)")

# Test recall with multiple phrasing
print(f"\n  TESTING RECALL (with multi-exposure learning):")
print(f"  {'Query':50s} {'Hit':>6s} {'Sim':>8s} {'Domain':>10s} {'Match':>6s}")
print(f"  {'-'*50} {'-'*6} {'-'*8} {'-'*10} {'-'*6}")

multi_hits = 0
multi_total = 0
recall_queries = [
    ("What type of bender am I", False),
    ("Am I a firebender", True),  # should match stored "I am a firebender"
    ("What element do I bend", False),
    ("Do I bend fire", True),     # should match "I can bend fire"
    ("Tell me what kind of bender I am", False),
    ("Which bending style is mine", False),
    ("Am I a waterbender or firebender", True),
    ("I forgot, what bender am I", False),
    ("What is my element", False),
    ("Can I bend fire", True),    # should match "I can bend fire"
]

for query, expect_hit in recall_queries:
    q = encode(query)
    r = cache2.search(q)
    hit = "YES" if r["hit"] else "no"
    if r["hit"]: multi_hits += 1
    multi_total += 1
    
    # Check if matched response is the personal fact
    is_fact = (r["best_idx"] >= 0 and 
               cache2.trajs[r["best_idx"]]["response"] == "I am a firebender")
    match = "FACT" if is_fact else "noise"
    marker = " " if is_fact else ""
    print(f"  {query:50s} {hit:>6s} {r['best_sim']:>8.4f} {r['domain']:>10s} {match:>6s}{marker}")

print(f"\n  Multi-exposure recall: {multi_hits}/{multi_total} ({multi_hits/multi_total*100:.0f}%)")

# Find where facts rank now
cache2._norm()
all_sims2 = (cache2._projs @ recall_proj.unsqueeze(1)).squeeze(-1)
fact_ranks = []
for i, t in enumerate(cache2.trajs):
    if t["response"] == "I am a firebender":
        sim = F.cosine_similarity(recall_proj.unsqueeze(0), cache2._projs[i:i+1]).item()
        rank = (all_sims2 > sim).sum().item() + 1
        fact_ranks.append((i, sim, rank))

fact_ranks.sort(key=lambda x: x[2])
print(f"\n  Personal fact rankings (out of {len(cache2.trajs)}):")
for i, sim, rank in fact_ranks[:5]:
    print(f"    Rank #{rank}: sim={sim:.4f} (trajectory #{i})")
if fact_ranks:
    best_fact_rank = fact_ranks[0][2]
    print(f"\n  Best personal fact rank: #{best_fact_rank}")
    if best_fact_rank <= 10:
        print(f"  VERDICT: MULTI-EXPOSURE SUCCESS — fact in top {best_fact_rank}!")
    else:
        print(f"  VERDICT: IMPROVED — facts clustered but still far ({best_fact_rank})")
        print(f"  A real UGT basis would push these to #1-3.")


# Save
os.makedirs("benchmarks/isagi_firebender", exist_ok=True)
with open("benchmarks/isagi_firebender/results.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k_dim": K_DIM,
        "n_questions": len(full_questions),
        "cache_size": len(cache.trajs),
        "recall_hit": result["hit"],
        "recall_sim": result["best_sim"],
        "recall_domain": result["domain"],
        "variation_hits": hits,
        "variation_total": len(variations),
        "fact_rank": fact_rank if fact_idx is not None else None,
        "fact_sim": fact_sim if fact_idx is not None else None,
        "direct_sim_recall_to_fact": direct_sim,
    }, f, indent=2)

print(f"\n  Results saved to benchmarks/isagi_firebender/results.json")
