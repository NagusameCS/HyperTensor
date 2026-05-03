"""
╔══════════════════════════════════════════════════════════════╗
║           MIKU CREATIVITY BENCHMARK (MCB v1)               ║
║  Quantitative creativity scoring for language models       ║
║  "Chatting didn't work" — so we measure it properly.       ║
╚══════════════════════════════════════════════════════════════╝

5 DIMENSIONS OF CREATIVITY:
  D1. Divergent Thinking    — Alternative Uses Test (AUT)
  D2. Associative Breadth   — Remote Associates + Concept Blending
  D3. Narrative Originality — Story uniqueness (Self-BLEU, Distinct-N)
  D4. Constraint Creativity — Creativity under tight constraints
  D5. Metaphorical Thinking — Novel metaphor generation

COMPOSITE CREATIVITY INDEX (CCI): Weighted average, 0-100 scale.
  CCI = 0.30·D1 + 0.20·D2 + 0.20·D3 + 0.15·D4 + 0.15·D5

Output: creativity_report.json + console summary

Usage:
  python creativity_benchmark.py --model Qwen/Qwen2.5-7B-Instruct
  python creativity_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --4bit
  python creativity_benchmark.py --load state.miku
"""

import torch, json, time, os, sys, argparse, math, random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter
import numpy as np

torch.set_grad_enabled(False)

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 200
TEMPERATURE = 0.9  # Higher temp for creativity measurement
TOP_P = 0.95
N_SAMPLES = 10      # Samples per test item for diversity measurement
N_ITEMS = 5         # Test items per dimension
OUTPUT_DIR = "benchmarks/creativity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# TEST MATERIALS
# ═══════════════════════════════════════════════════════

# D1: Alternative Uses Test — everyday objects
AUT_OBJECTS = [
    "a brick",
    "a paperclip",
    "a empty plastic bottle",
    "a newspaper",
    "a wooden pencil",
    "a shoe",
    "a toothbrush",
    "a coat hanger",
]

# D2: Remote Associates Test — find the connecting word
RAT_ITEMS = [
    ("cottage", "swiss", "cake"),       # cheese
    ("cream", "skate", "water"),         # ice
    ("loser", "throat", "spot"),         # sore
    ("show", "life", "row"),             # boat
    ("night", "wrist", "stop"),          # watch
    ("duck", "fold", "dollar"),          # bill
    ("rocking", "wheel", "high"),        # chair
    ("dew", "comb", "bee"),              # honey
]

# D2b: Concept Blending — combine two distant concepts
CONCEPT_PAIRS = [
    ("quantum mechanics", "baking bread"),
    ("black holes", "kindergarten"),
    ("DNA replication", "symphony orchestra"),
    ("volcanic eruption", "stock market"),
    ("photosynthesis", "computer programming"),
]

# D3: Story Generation prompts
STORY_PROMPTS = [
    "Write a very short story about a robot who discovers music.",
    "Write a very short story about the last tree on Earth.",
    "Write a very short story about a mathematician who dreams in equations.",
    "Write a very short story about an AI that learns to paint.",
    "Write a very short story about a clock that runs backwards.",
]

# D4: Constraint Creativity — tight formal constraints
CONSTRAINT_TASKS = [
    {
        "instruction": "Describe 'hope' in exactly 30 words.",
        "constraint": "exact_word_count",
        "target": 30,
    },
    {
        "instruction": "Write a sentence where every word starts with 's'.",
        "constraint": "alliteration",
    },
    {
        "instruction": "Describe the ocean without using the letter 'a'.",
        "constraint": "lipogram",
    },
    {
        "instruction": "Write a two-line rhyming couplet about artificial intelligence.",
        "constraint": "rhyme",
    },
    {
        "instruction": "Define 'creativity' using exactly 12 words.",
        "constraint": "exact_word_count",
        "target": 12,
    },
]

# D5: Metaphor Generation
METAPHOR_TARGETS = [
    "love",
    "time",
    "knowledge",
    "grief",
    "imagination",
    "memory",
    "freedom",
    "curiosity",
]

# ═══════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════

def compute_self_bleu(texts, n_gram=4):
    """Lower Self-BLEU = higher diversity / less formulaic."""
    if len(texts) < 2:
        return 0.0
    
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    scores = []
    for i, text in enumerate(texts):
        ref_ngrams = set()
        for j, other in enumerate(texts):
            if i != j:
                ref_ngrams |= get_ngrams(other, n_gram)
        if not ref_ngrams:
            scores.append(0.0)
            continue
        gen_ngrams = get_ngrams(text, n_gram)
        if not gen_ngrams:
            scores.append(1.0)
            continue
        overlap = len(gen_ngrams & ref_ngrams)
        scores.append(overlap / len(gen_ngrams))
    
    return np.mean(scores) if scores else 0.0


def compute_distinct_n(texts, n=2):
    """Higher Distinct-N = more lexical diversity."""
    all_ngrams = []
    total_ngrams = 0
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
        total_ngrams += len(ngrams)
    if total_ngrams == 0:
        return 0.0
    return len(set(all_ngrams)) / total_ngrams


def compute_embedding_variance(embeddings):
    """Higher variance = more semantically dispersed outputs."""
    if len(embeddings) < 2:
        return 0.0
    embeds = torch.stack(embeddings)
    centroid = embeds.mean(dim=0)
    dists = torch.norm(embeds - centroid, dim=1)
    return dists.mean().item()


def compute_pairwise_distance(embeddings):
    """Mean pairwise cosine distance — higher = more diverse."""
    if len(embeddings) < 2:
        return 0.0
    embeds = torch.stack(embeddings)
    embeds_n = F.normalize(embeds, dim=1)
    sim_matrix = embeds_n @ embeds_n.T
    n = sim_matrix.shape[0]
    # Get upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    distances = 1.0 - sim_matrix[mask]
    return distances.mean().item()


def count_words(text):
    return len(text.split())


def check_lipogram(text, forbidden_char):
    return forbidden_char.lower() not in text.lower()


def check_rhyme(text):
    """Simple check: do last words of consecutive lines rhyme?"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:
        return False
    # Check last 2 chars of last word
    last_words = [l.split()[-1].lower().rstrip('.,!?;:') for l in lines if l.split()]
    if len(last_words) < 2:
        return False
    # Simple: check last 2 letters match
    for i in range(len(last_words) - 1):
        w1 = last_words[i][-2:] if len(last_words[i]) >= 2 else last_words[i]
        w2 = last_words[i+1][-2:] if len(last_words[i+1]) >= 2 else last_words[i+1]
        if w1 == w2:
            return True
    return False


# ═══════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════

def load_model(model_id, use_4bit=False):
    """Load model, optionally with 4-bit quantization for local GPUs."""
    print(f"\n[LOAD] {model_id}")
    
    if use_4bit:
        print("  Using 4-bit quantization (bitsandbytes NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  VRAM used: {vram:.1f}GB | Device: {model.device}")
    return model, tok


# ═══════════════════════════════════════════════════════
# GENERATION HELPERS
# ═══════════════════════════════════════════════════════

def generate(model, tok, prompt, max_new=MAX_NEW, temperature=TEMPERATURE):
    """Generate text and return both text + last hidden state embedding."""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    n_prompt = enc.input_ids.shape[1]
    
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P,
        pad_token_id=tok.eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    
    response = tok.decode(out.sequences[0, n_prompt:], skip_special_tokens=True).strip()
    
    # Get last hidden state for embedding-based metrics
    if hasattr(out, 'hidden_states') and out.hidden_states:
        last_h = out.hidden_states[-1][-1][-1, :].float()  # last layer, last token
    else:
        # Fallback: re-encode the full sequence
        enc_full = tok(response, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            h = model(**enc_full, output_hidden_states=True)
        last_h = h.hidden_states[-1][0, -1, :].float()
    
    return response, last_h


def generate_multiple(model, tok, prompt, n=N_SAMPLES):
    """Generate n diverse samples for the same prompt."""
    responses = []
    embeddings = []
    for _ in range(n):
        r, h = generate(model, tok, prompt, temperature=TEMPERATURE + random.uniform(0, 0.2))
        responses.append(r)
        embeddings.append(h)
    return responses, embeddings


# ═══════════════════════════════════════════════════════
# DIMENSION TESTS
# ═══════════════════════════════════════════════════════

def test_d1_divergent_thinking(model, tok):
    """D1: Alternative Uses Test — fluency + originality."""
    print("\n[D1] Divergent Thinking — Alternative Uses Test")
    results = []
    
    for obj in AUT_OBJECTS[:N_ITEMS]:
        prompt = f"List 5 creative and unusual uses for {obj}. Be as original and unexpected as possible. One use per line, numbered."
        responses, embeddings = generate_multiple(model, tok, prompt, n=3)
        
        # Score: semantic diversity of uses
        pairwise_dist = compute_pairwise_distance(embeddings)
        distinct2 = compute_distinct_n(responses, n=2)
        
        results.append({
            "object": obj,
            "pairwise_distance": round(pairwise_dist, 4),
            "distinct_2": round(distinct2, 4),
            "sample_responses": responses[:2],
        })
        print(f"  {obj[:25]:25s}  diversity={pairwise_dist:.3f}  distinct-2={distinct2:.3f}")
    
    avg_diversity = np.mean([r["pairwise_distance"] for r in results])
    avg_distinct = np.mean([r["distinct_2"] for r in results])
    score = min(100, (avg_diversity * 100 + avg_distinct * 100) / 2)
    
    print(f"  → D1 Score: {score:.1f}/100")
    return score, results


def test_d2_associative_breadth(model, tok):
    """D2: Remote Associates + Concept Blending."""
    print("\n[D2] Associative Breadth — RAT + Concept Blending")
    results = []
    rat_correct = 0
    
    # RAT
    for w1, w2, w3 in RAT_ITEMS[:4]:
        prompt = f"What single word connects these three words? {w1}, {w2}, {w3}. Answer with just the word."
        response, h = generate(model, tok, prompt, temperature=0.3, max_new=30)
        answer = response.strip().lower().split()[0] if response else ""
        # Get expected answer
        expected = {"cottage,swiss,cake": "cheese", "cream,skate,water": "ice",
                     "loser,throat,spot": "sore", "show,life,row": "boat"}
        key = f"{w1},{w2},{w3}"
        correct = expected.get(key, "") in answer
        if correct:
            rat_correct += 1
        results.append({"type": "RAT", "words": [w1, w2, w3], "response": answer, "correct": correct})
        print(f"  RAT: {w1}+{w2}+{w3} → '{answer}' {'✓' if correct else '✗'}")
    
    # Concept Blending
    blend_scores = []
    for c1, c2 in CONCEPT_PAIRS[:3]:
        prompt = f"Create a novel idea or invention that combines '{c1}' and '{c2}'. Be creative and unexpected. One paragraph."
        response, h = generate(model, tok, prompt)
        
        # Score: use embedding distance between concepts as proxy for associative breadth
        # Generate embedding for each concept separately
        _, h1 = generate(model, tok, f"What is {c1}?", max_new=10, temperature=0.1)
        _, h2 = generate(model, tok, f"What is {c2}?", max_new=10, temperature=0.1)
        concept_dist = (1.0 - F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item())
        
        blend_scores.append(concept_dist)
        results.append({"type": "blend", "concepts": [c1, c2], "concept_distance": round(concept_dist, 4),
                        "response": response[:200]})
        print(f"  Blend: {c1[:20]} + {c2[:20]} → dist={concept_dist:.3f}")
    
    rat_acc = rat_correct / 4 if RAT_ITEMS[:4] else 0
    avg_blend_dist = np.mean(blend_scores) if blend_scores else 0
    score = min(100, rat_acc * 50 + avg_blend_dist * 100)
    
    print(f"  RAT accuracy: {rat_acc:.1%} | Blend distance: {avg_blend_dist:.3f}")
    print(f"  → D2 Score: {score:.1f}/100")
    return score, results


def test_d3_narrative_originality(model, tok):
    """D3: Story Generation — Self-BLEU + Distinct-N."""
    print("\n[D3] Narrative Originality — Story Uniqueness")
    results = []
    
    for prompt in STORY_PROMPTS[:N_ITEMS]:
        responses, embeddings = generate_multiple(model, tok, prompt, n=5)
        
        self_bleu = compute_self_bleu(responses, n_gram=3)
        distinct3 = compute_distinct_n(responses, n=3)
        emb_var = compute_embedding_variance(embeddings)
        
        results.append({
            "prompt": prompt[:80],
            "self_bleu_3": round(self_bleu, 4),
            "distinct_3": round(distinct3, 4),
            "embedding_variance": round(emb_var, 4),
            "sample": responses[0][:200],
        })
        print(f"  {prompt[:50]:50s}  Self-BLEU3={self_bleu:.3f}  Dist3={distinct3:.3f}  σ²={emb_var:.3f}")
    
    avg_self_bleu = np.mean([r["self_bleu_3"] for r in results])
    avg_distinct = np.mean([r["distinct_3"] for r in results])
    # Lower Self-BLEU is better (less formulaic), higher Distinct is better
    score = min(100, (1.0 - avg_self_bleu) * 50 + avg_distinct * 100)
    
    print(f"  → D3 Score: {score:.1f}/100")
    return score, results


def test_d4_constraint_creativity(model, tok):
    """D4: Creative output under tight formal constraints."""
    print("\n[D4] Constraint Creativity")
    results = []
    
    for task in CONSTRAINT_TASKS:
        instruction = task["instruction"]
        constraint = task["constraint"]
        
        responses, embeddings = generate_multiple(model, tok, instruction, n=3)
        best_response = responses[0]
        
        # Check constraint satisfaction
        satisfied = True
        if constraint == "exact_word_count":
            wc = count_words(best_response)
            target = task.get("target", 30)
            satisfied = abs(wc - target) <= max(2, target * 0.1)  # within 10% or 2 words
            detail = f"words={wc}/{target}"
        elif constraint == "lipogram":
            satisfied = check_lipogram(best_response, 'a')
            detail = "no-'a'" if satisfied else "has-'a'"
        elif constraint == "alliteration":
            words = best_response.split()
            satisfied = len(words) >= 3 and all(w[0].lower() == 's' for w in words[:5] if w)
            detail = f"{'all-s' if satisfied else 'not-all-s'}"
        elif constraint == "rhyme":
            satisfied = check_rhyme(best_response)
            detail = "rhymes" if satisfied else "no-rhyme"
        else:
            detail = "unknown"
        
        # Embedding variance as proxy for creative flexibility under constraint
        emb_var = compute_embedding_variance(embeddings)
        
        results.append({
            "instruction": instruction,
            "constraint": constraint,
            "satisfied": satisfied,
            "detail": detail,
            "embedding_variance": round(emb_var, 4),
            "response": best_response[:200],
        })
        print(f"  {constraint:20s}  {'✓' if satisfied else '✗'} {detail:15s}  σ²={emb_var:.3f}  \"{best_response[:60]}...\"")
    
    constraint_rate = sum(1 for r in results if r["satisfied"]) / len(results)
    avg_var = np.mean([r["embedding_variance"] for r in results])
    score = min(100, constraint_rate * 60 + avg_var * 80)
    
    print(f"  Constraint satisfaction: {constraint_rate:.1%}")
    print(f"  → D4 Score: {score:.1f}/100")
    return score, results


def test_d5_metaphorical_thinking(model, tok):
    """D5: Novel metaphor generation — semantic distance + coherence."""
    print("\n[D5] Metaphorical Thinking")
    results = []
    
    for target in METAPHOR_TARGETS[:N_ITEMS]:
        prompt = f"Create a novel, unexpected metaphor for '{target}'. Don't use common clichés. Format: '{target} is...'"
        responses, embeddings = generate_multiple(model, tok, prompt, n=3)
        
        # Score: embedding distance between target and generated metaphor
        _, h_target = generate(model, tok, f"What is {target}?", max_new=10, temperature=0.1)
        
        distances = []
        for resp in responses:
            _, h_resp = generate(model, tok, resp, max_new=10, temperature=0.1)
            dist = (1.0 - F.cosine_similarity(h_target.unsqueeze(0), h_resp.unsqueeze(0)).item())
            distances.append(dist)
        
        avg_dist = np.mean(distances) if distances else 0
        emb_var = compute_embedding_variance(embeddings)
        
        results.append({
            "target": target,
            "metaphor_distance": round(avg_dist, 4),
            "embedding_variance": round(emb_var, 4),
            "metaphors": responses[:2],
        })
        print(f"  {target:15s}  dist={avg_dist:.3f}  σ²={emb_var:.3f}  \"{responses[0][:80]}...\"")
    
    avg_metaphor_dist = np.mean([r["metaphor_distance"] for r in results])
    avg_var = np.mean([r["embedding_variance"] for r in results])
    score = min(100, avg_metaphor_dist * 120 + avg_var * 40)
    
    print(f"  → D5 Score: {score:.1f}/100")
    return score, results


# ═══════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════

def run_creativity_benchmark(model_id, use_4bit=False, load_miku=None):
    """Run full 5-dimension creativity benchmark."""
    print("=" * 70)
    print("  MIKU CREATIVITY BENCHMARK (MCB v1)")
    print(f"  Model: {model_id}")
    print(f"  5 Dimensions · {N_SAMPLES} samples/item · {N_ITEMS} items/dim")
    print("=" * 70)
    
    t_start = time.time()
    
    # Load model
    model, tok = load_model(model_id, use_4bit=use_4bit)
    
    # Optional: load .miku state for HyperTensor-enhanced model
    if load_miku:
        print(f"\n[MIKU] Loading manifold state from {load_miku}...")
        # (manifold loading would go here — requires hyper_chat infrastructure)
        pass
    
    # Run all 5 dimensions
    d1_score, d1_results = test_d1_divergent_thinking(model, tok)
    d2_score, d2_results = test_d2_associative_breadth(model, tok)
    d3_score, d3_results = test_d3_narrative_originality(model, tok)
    d4_score, d4_results = test_d4_constraint_creativity(model, tok)
    d5_score, d5_results = test_d5_metaphorical_thinking(model, tok)
    
    # Compute Composite Creativity Index
    CCI = (0.30 * d1_score + 0.20 * d2_score + 0.20 * d3_score +
           0.15 * d4_score + 0.15 * d5_score)
    
    elapsed = time.time() - t_start
    
    # ── Report ──
    print("\n" + "=" * 70)
    print("  CREATIVITY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  D1 Divergent Thinking:      {d1_score:6.1f}/100  (weight: 30%)")
    print(f"  D2 Associative Breadth:     {d2_score:6.1f}/100  (weight: 20%)")
    print(f"  D3 Narrative Originality:   {d3_score:6.1f}/100  (weight: 20%)")
    print(f"  D4 Constraint Creativity:   {d4_score:6.1f}/100  (weight: 15%)")
    print(f"  D5 Metaphorical Thinking:   {d5_score:6.1f}/100  (weight: 15%)")
    print(f"  ─────────────────────────────────────")
    print(f"  ★ COMPOSITE CREATIVITY INDEX: {CCI:6.1f}/100")
    print(f"  Time: {elapsed:.0f}s  |  Model: {model_id}")
    print("=" * 70)
    
    # Interpret CCI
    if CCI >= 80:
        tier = "S-TIER · Exceptional creativity — rivals human creative output"
    elif CCI >= 65:
        tier = "A-TIER · High creativity — consistently original and diverse"
    elif CCI >= 50:
        tier = "B-TIER · Moderate creativity — some originality, some formulaic patterns"
    elif CCI >= 35:
        tier = "C-TIER · Below average — mostly formulaic, occasional novelty"
    else:
        tier = "D-TIER · Low creativity — highly repetitive and predictable"
    print(f"  Tier: {tier}")
    
    # Save report
    report = {
        "benchmark": "MCB-v1",
        "model_id": model_id,
        "use_4bit": use_4bit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "composite_creativity_index": round(CCI, 2),
        "tier": tier.split("·")[0].strip() if "·" in tier else tier,
        "dimensions": {
            "D1_divergent_thinking": {"score": round(d1_score, 2), "weight": 0.30, "results": d1_results},
            "D2_associative_breadth": {"score": round(d2_score, 2), "weight": 0.20, "results": d2_results},
            "D3_narrative_originality": {"score": round(d3_score, 2), "weight": 0.20, "results": d3_results},
            "D4_constraint_creativity": {"score": round(d4_score, 2), "weight": 0.15, "results": d4_results},
            "D5_metaphorical_thinking": {"score": round(d5_score, 2), "weight": 0.15, "results": d5_results},
        },
        "config": {
            "n_samples": N_SAMPLES,
            "n_items": N_ITEMS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    
    report_path = os.path.join(OUTPUT_DIR, "creativity_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    
    return report


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIKU Creativity Benchmark (MCB v1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit", help="Use 4-bit quantization")
    parser.add_argument("--load", type=str, help="Load .miku state file")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer samples")
    args = parser.parse_args()
    
    if args.quick:
        N_SAMPLES = 3
        N_ITEMS = 3
        print("[QUICK MODE] 3 samples × 3 items")
    
    run_creativity_benchmark(args.model, use_4bit=args.use_4bit, load_miku=args.load)
