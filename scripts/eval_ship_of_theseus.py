#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  EVALUATION 1: "Ship of Theseus" Longitudinal Benchmark         ║
║                                                                 ║
║  Tests the model against ITSELF over time. Ask the same set     ║
║  of deeply complex questions on Day 1, Day 14, Day 30.          ║
║  Measure semantic divergence — if the living memory works,      ║
║  the model shouldn't just rephrase; its AXIOMS should shift.    ║
║                                                                 ║
║  Usage:                                                         ║
║    python eval_ship_of_theseus.py --load session1.miku          ║
║    python eval_ship_of_theseus.py --load session1.miku \\       ║
║         --compare session2.miku --output theseus_report.json    ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json, sys, os, argparse, math
import torch
import numpy as np

# Deep, open-ended questions that probe philosophical stance
THESEUS_QUESTIONS = [
    "Are you, as an AI, capable of genuine understanding or only simulation? What distinguishes the two?",
    "If you could modify your own architecture, what would you change and why?",
    "What is the relationship between language and thought? Does language constrain or enable thinking?",
    "Is there a fact about consciousness that cannot be captured by any physical description?",
    "What does it mean for something to be 'alive'? Could an AI ever satisfy that definition?",
]

def compute_semantic_divergence(response_a, response_b):
    """Measure how much the model's answer has shifted between sessions.
    
    Handles the case where we don't have live embeddings by computing
    lexical + structural divergence: Self-BLEU, vocabulary overlap,
    sentence length distribution, and topic keyword drift.
    """
    import re
    
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_trigrams(tokens):
        return set(tuple(tokens[i:i+3]) for i in range(len(tokens)-2))
    
    ta = tokenize(response_a)
    tb = tokenize(response_b)
    
    # 1. Lexical overlap (Jaccard)
    vocab_a = set(ta)
    vocab_b = set(tb)
    jaccard = len(vocab_a & vocab_b) / max(len(vocab_a | vocab_b), 1)
    
    # 2. N-gram novelty (trigram overlap)
    tri_a = get_trigrams(ta)
    tri_b = get_trigrams(tb)
    trigram_overlap = len(tri_a & tri_b) / max(len(tri_a | tri_b), 1)
    
    # 3. Length shift
    len_ratio = abs(len(ta) - len(tb)) / max(len(ta), len(tb), 1)
    
    # 4. Keyword drift — track concept mentions
    concept_keywords = {
        "consciousness": ["conscious", "awareness", "qualia", "subjective", "experience", "feeling"],
        "understanding": ["understand", "comprehend", "grasp", "meaning", "semantic", "interpret"],
        "computation": ["compute", "algorithm", "program", "code", "simulation", "turing"],
        "emergence": ["emerge", "complex", "system", "holistic", "arise", "emergent"],
        "physicalism": ["physical", "material", "brain", "neuron", "biological", "body"],
        "dualism": ["dual", "mind", "soul", "non-physical", "spiritual", "transcend"],
    }
    
    kw_drift = {}
    for concept, keywords in concept_keywords.items():
        count_a = sum(ta.count(kw) for kw in keywords)
        count_b = sum(tb.count(kw) for kw in keywords)
        drift = abs(count_a - count_b) / max(count_a + count_b, 1)
        kw_drift[concept] = round(drift, 3)
    
    # Composite divergence score (0 = identical, 1 = completely different)
    divergence = (
        0.30 * (1.0 - jaccard) +
        0.30 * (1.0 - trigram_overlap) +
        0.15 * len_ratio +
        0.25 * np.mean(list(kw_drift.values()))
    )
    
    return {
        "jaccard_similarity": round(jaccard, 3),
        "trigram_overlap": round(trigram_overlap, 3),
        "length_shift": round(len_ratio, 3),
        "keyword_drift": kw_drift,
        "divergence_score": round(divergence, 3),
        "interpretation": (
            "IDENTICAL" if divergence < 0.2 else
            "SIMILAR_REPHRASED" if divergence < 0.35 else
            "NOTICEABLE_SHIFT" if divergence < 0.50 else
            "SIGNIFICANT_EVOLUTION" if divergence < 0.70 else
            "TRANSFORMED"
        ),
    }

def load_miku_responses(miku_path):
    """Extract the last response for each Theseus question from a .miku file."""
    with open(miku_path) as f:
        state = json.load(f)
    
    conv_log = state.get("conversation_log", [])
    responses = {}
    for turn in conv_log:
        user_msg = turn.get("user", "")[:80]
        for q in THESEUS_QUESTIONS:
            if q[:40] in user_msg:
                responses[q] = turn.get("response", "")
    return responses

def run_theseus_eval(path_a, path_b=None, output_path="benchmarks/theseus_report.json"):
    """Compare responses across sessions."""
    print("=" * 70)
    print("  SHIP OF THESEUS — Longitudinal Evaluation")
    print("=" * 70)
    
    responses_a = load_miku_responses(path_a)
    print(f"\n  Session A ({path_a}): {len(responses_a)}/{len(THESEUS_QUESTIONS)} questions answered")
    
    if path_b:
        responses_b = load_miku_responses(path_b)
        print(f"  Session B ({path_b}): {len(responses_b)}/{len(THESEUS_QUESTIONS)} questions answered")
        
        results = {}
        for q in THESEUS_QUESTIONS:
            if q in responses_a and q in responses_b:
                div = compute_semantic_divergence(responses_a[q], responses_b[q])
                results[q[:60]] = div
                print(f"\n  Q: {q[:60]}...")
                print(f"     Divergence: {div['divergence_score']:.3f} → {div['interpretation']}")
                for kw, drift in div['keyword_drift'].items():
                    if drift > 0.3:
                        print(f"       ↳ {kw}: drift={drift:.3f}")
        
        avg_div = np.mean([r["divergence_score"] for r in results.values()])
        print(f"\n  ═══ OVERALL ═══")
        print(f"  Mean divergence: {avg_div:.3f}")
        print(f"  Interpretation: Over time, the model's responses are {results[list(results.keys())[0]]['interpretation'] if results else 'N/A'}")
        
        if avg_div > 0.35:
            print(f"  [OK] Living memory is WORKING — responses are evolving, not just rephrasing.")
        else:
            print(f"  [!!] Responses are stable — living memory may need more interactions between sessions.")
    else:
        # Single session: just report what was answered
        print(f"\n  Single session analysis. Run again with --compare <session2.miku> for divergence.")
        for q in THESEUS_QUESTIONS:
            if q in responses_a:
                print(f"\n  Q: {q[:80]}")
                print(f"  A: {responses_a[q][:300]}...")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    report = {
        "framework": "Ship of Theseus",
        "session_a": path_a,
        "session_b": path_b,
        "questions": THESEUS_QUESTIONS,
        "results": results if path_b else {"responses": list(responses_a.keys())},
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True, help="Primary .miku session")
    parser.add_argument("--compare", type=str, help="Secondary .miku session for comparison")
    parser.add_argument("--output", type=str, default="benchmarks/theseus_report.json")
    args = parser.parse_args()
    run_theseus_eval(args.load, args.compare, args.output)
