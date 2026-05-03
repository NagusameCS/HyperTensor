#!/usr/bin/env python3
"""
Phase 2: Brutal Math Data Segregation for Pure Model M.

Constructs a zero-contamination math/logic corpus with aggressive filtering:
- Strips ALL conversational wrappers ("Sure!", "Let me help", etc.)
- Keeps ONLY: equations, proofs, formal reasoning, code blocks
- Sources: OpenWebMath, GSM8K, MATH, AMPS, Lean/Coq proofs
- Output: plain .txt files with one math segment per line (tokenized-ready)

The goal: Force Attention matrices to learn multi-step structural routing
while keeping FFN memory banks starved of natural language.
"""

import json
import os
import re
import sys
from pathlib import Path

# === CONFIGURATION ===
OUTPUT_DIR = Path("data/pure_math")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOTAL_TOKENS = 1_000_000_000  # ~1B tokens target
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 4096

# === AGGRESSIVE FILTERS ===

# Patterns that indicate conversational/instructional wrappers — MUST BE STRIPPED
CONVERSATIONAL_STRIP = [
    r"(?i)^(sure|absolutely|certainly|of course)[,!.].*",
    r"(?i)^(let me|i will|i can|i'll|we'll|we will)\b.*",
    r"(?i)^(here is|here's|here are)\b.*",
    r"(?i)^(the answer is|the solution is|the result is)\b.*",
    r"(?i)^(i hope|hopefully|good luck)\b.*",
    r"(?i)^(question[:]?\s*).*",
    r"(?i)^(problem[:]?\s*).*",
    r"(?i)\b(step\s*\d+[:.]?\s*).*",  # "Step 1:" — keep the math after, strip the label
    r"(?i)^(explanation[:]?\s*).*",
    r"(?i)^(solution[:]?\s*).*",
    r"(?i)^(let's|let us)\b.*",
]

# Patterns that indicate natural language / prose — EXCLUDE ENTIRELY
NATURAL_LANGUAGE_EXCLUDE = [
    r"(?i)\b(once upon a time)\b",
    r"(?i)\b(chapter\s+\d+)\b",
    r"(?i)\b(the\s+\w+\s+said)\b",
    r"(?i)\b(she\s+said|he\s+said|they\s+said)\b",
    r"(?i)\b(went to|walked to|arrived at)\b",
    r"(?i)\b(feeling|felt|emotion|happy|sad|angry)\b",
    r"(?i)\b(beautiful|gorgeous|wonderful|amazing)\b",
    r"(?i)\b(story|novel|poem|verse|rhyme)\b",
    r"(?i)\b(character|protagonist|narrator)\b",
]

# Math-positive patterns — at least ONE must match
MATH_POSITIVE = [
    r"\d+",                    # Contains numbers
    r"[+\-*/=<>]",            # Contains math operators
    r"\\\w+",                  # Contains LaTeX commands
    r"\b(proof|theorem|lemma|corollary|axiom|definition)\b",
    r"\b(implies|therefore|hence|thus|consequently)\b",
    r"\b(let|assume|suppose|given|consider)\b.*\b(then|hence|therefore)\b",
    r"\b(forall|exists|in|subseteq|cup|cap|setminus)\b",
    r"\b(f(x)|g(x)|h(x)|lim|int|sum|prod)\b",
    r"\b(induction|contradiction|contrapositive)\b",
    r"\{.*\}.*\{.*\}",         # Set notation
    r"\b(:=|::=|\\equiv)\b",   # Definition operators
    r"```",                     # Code blocks
    r"\b(def |fn |function |class |import |from )\b",  # Code
]


def is_fundamentally_math(text: str) -> bool:
    """Returns True only if text is fundamentally mathematical/logical content."""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    
    # Must NOT contain natural language markers
    for pat in NATURAL_LANGUAGE_EXCLUDE:
        if re.search(pat, text):
            return False
    
    # Must contain at least one math-positive pattern
    has_math = False
    for pat in MATH_POSITIVE:
        if re.search(pat, text):
            has_math = True
            break
    if not has_math:
        return False
    
    # Math content should have high symbol-to-word ratio
    words = len(re.findall(r'[a-zA-Z]+', text))
    symbols = len(re.findall(r'[^a-zA-Z\s]', text))
    if words > 0 and symbols / max(words, 1) < 0.05:
        return False  # Too wordy, not enough math
    
    return True


def strip_conversational(text: str) -> str:
    """Aggressively strip conversational wrappers, keeping only the core math."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check if this line is purely conversational
        is_conversational = False
        for pat in CONVERSATIONAL_STRIP:
            if re.match(pat, stripped):
                # Try to extract the math part after the conversational prefix
                # e.g., "Step 1: Compute x + y = z" -> keep after colon
                if ':' in stripped:
                    after = stripped.split(':', 1)[1].strip()
                    if after and is_fundamentally_math(after):
                        cleaned.append(after)
                is_conversational = True
                break
        
        if not is_conversational:
            cleaned.append(stripped)
    
    return '\n'.join(cleaned)


def format_math_sample(text: str, source: str) -> str:
    """Format a math sample in a clean, proof-oriented style."""
    # Strip all conversational fluff
    text = strip_conversational(text)
    if len(text) < MIN_TEXT_LENGTH:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    return text


def load_gsm8k() -> list[str]:
    """Load GSM8K training data, aggressively stripped of instruction wrappers."""
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        for row in ds:
            question = row.get("question", "")
            answer = row.get("answer", "")
            # GSM8K answers often have "#### final_answer" format
            # Strip the "####" marker — keep the reasoning chain
            answer = re.sub(r'####\s*\d+.*$', '', answer).strip()
            
            combined = f"Q: {question}\nA: {answer}"
            cleaned = format_math_sample(combined, "gsm8k")
            if cleaned:
                samples.append(cleaned)
            if len(samples) >= 8000:
                break
        print(f"  GSM8K: {len(samples)} samples")
    except Exception as e:
        print(f"  GSM8K: SKIP ({e})")
    return samples


def load_math_dataset() -> list[str]:
    """Load MATH dataset problems (competition math)."""
    samples = []
    try:
        from datasets import load_dataset
        # Try multiple MATH dataset sources
        for ds_path in ["HuggingFaceH4/MATH-500", "hendrycks/math", "math_dataset"]:
            try:
                ds = load_dataset(ds_path, split="train")
                for row in ds:
                    problem = row.get("problem", "") or row.get("question", "") or row.get("text", "")
                    solution = row.get("solution", "") or row.get("answer", "") or ""
                    combined = f"{problem}\n{solution}"
                    cleaned = format_math_sample(combined, ds_path)
                    if cleaned:
                        samples.append(cleaned)
                    if len(samples) >= 5000:
                        break
                if samples:
                    break
            except Exception:
                continue
        print(f"  MATH: {len(samples)} samples")
    except Exception as e:
        print(f"  MATH: SKIP ({e})")
    return samples


def load_openwebmath() -> list[str]:
    """Load OpenWebMath — raw mathematical web content."""
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("open-web-math/open-web-math", split="train", 
                          streaming=True)
        count = 0
        for row in ds:
            text = row.get("text", "")
            cleaned = format_math_sample(text, "openwebmath")
            if cleaned:
                samples.append(cleaned)
            count += 1
            if len(samples) >= 20000 or count >= 100000:
                break
        print(f"  OpenWebMath: {len(samples)} samples (from {count} raw)")
    except Exception as e:
        print(f"  OpenWebMath: SKIP ({e})")
    return samples


def load_proof_data() -> list[str]:
    """Load formal proof data (Lean/Coq/Isabelle)."""
    samples = []
    try:
        from datasets import load_dataset
        # Try proof-focused datasets
        for ds_path in ["hoskinson-center/proof-pile", "EleutherAI/proof-pile-2"]:
            try:
                ds = load_dataset(ds_path, split="train", streaming=True)
                count = 0
                for row in ds:
                    text = row.get("text", "") or row.get("content", "")
                    # Check for formal proof syntax
                    if any(kw in text.lower() for kw in ["theorem", "lemma", "proof", "qed", 
                           "induction", "forall", "exists", "λ", "→", "∀", "∃"]):
                        cleaned = format_math_sample(text, ds_path)
                        if cleaned:
                            samples.append(cleaned)
                    count += 1
                    if len(samples) >= 3000 or count >= 50000:
                        break
                if samples:
                    break
            except Exception:
                continue
        print(f"  Proof data: {len(samples)} samples")
    except Exception as e:
        print(f"  Proof data: SKIP ({e})")
    return samples


def main():
    print("=" * 70)
    print("PURE MATH CORPUS BUILDER")
    print("Goal: Zero-contamination math/logic data for Model M")
    print("=" * 70)
    
    all_samples = []
    
    # Load from all sources
    print("\n[1/4] Loading GSM8K...")
    all_samples.extend(load_gsm8k())
    
    print("\n[2/4] Loading MATH dataset...")
    all_samples.extend(load_math_dataset())
    
    print("\n[3/4] Loading OpenWebMath...")
    all_samples.extend(load_openwebmath())
    
    print("\n[4/4] Loading formal proof data...")
    all_samples.extend(load_proof_data())
    
    # Deduplicate
    seen = set()
    unique = []
    for s in all_samples:
        h = hash(s[:200])  # Quick hash on first 200 chars
        if h not in seen:
            seen.add(h)
            unique.append(s)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(all_samples)} raw -> {len(unique)} unique samples")
    est_tokens = len(unique) * 200
    print(f"Estimated tokens: ~{est_tokens:,} ({est_tokens/1e6:.1f}M)")
    
    # Save as plain text (one sample per line, JSON-escaped)
    output_path = OUTPUT_DIR / "math_pure_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in unique:
            f.write(json.dumps({"text": sample}, ensure_ascii=False) + "\n")
    
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Also save a raw text version for inspection
    raw_path = OUTPUT_DIR / "math_pure_train.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(unique[:100]):  # First 100 for inspection
            f.write(f"=== SAMPLE {i+1} ===\n")
            f.write(sample[:500])
            f.write("\n...\n\n")
    
    print(f"Inspection samples: {raw_path}")
    print("\nDONE. Ready for training.")


if __name__ == "__main__":
    main()
