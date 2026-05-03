#!/usr/bin/env python3
"""
Phase 2: Brutal Language Data Segregation for Pure Model L.

Constructs a zero-contamination natural language corpus with aggressive filtering:
- DELETES any document containing equations, LaTeX, numbers, code blocks
- Keeps ONLY: pure prose, literature, dialogue, conversational text
- Sources: Project Gutenberg, Wikipedia prose (no STEM), OpenWebText, dialogue
- Output: plain .txt files with one prose segment per line

The goal: Force FFN memory banks to learn massive vocabulary, grammar, syntax
while keeping Attention matrices starved of strict procedural logic.
"""

import json
import os
import re
import sys
from pathlib import Path

# === CONFIGURATION ===
OUTPUT_DIR = Path("data/pure_language")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOTAL_TOKENS = 1_000_000_000  # ~1B tokens target
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 4096

# === AGGRESSIVE EXCLUSION FILTERS ===
# ANY match = document is REJECTED entirely

MATH_EXCLUDE_PATTERNS = [
    # LaTeX
    r"\\begin\{", r"\\end\{", r"\\frac\{", r"\\sqrt\{", r"\\sum", r"\\int",
    r"\\alpha", r"\\beta", r"\\gamma", r"\\delta", r"\\epsilon",
    r"\\infty", r"\\partial", r"\\nabla", r"\\otimes", r"\\oplus",
    r"\\mathbb\{", r"\\mathbf\{", r"\\mathcal\{", r"\\mathfrak\{",
    r"\\left\(", r"\\right\)", r"\\left\[", r"\\right\]",
    
    # Math delimiters
    r"\$\$", r"\$[^$]+\$", r"\\\[", r"\\\]", r"\\\(\s", r"\\\)",
    
    # Equations / formulas
    r"[+\-*/=<>]{2,}",         # Multiple operators in sequence
    r"\b\d+\s*[+\-*/]\s*\d+\s*=\s*\d+\b",  # "2 + 2 = 4"
    r"\b\d{4,}\b",              # Large numbers (years are OK but risk math)
    r"[a-z]_\{\w+\}",           # LaTeX subscripts like x_{i}
    r"[a-z]\^\{\w+\}",          # LaTeX superscripts like x^{2}
    r"\\times|\\div|\\pm|\\mp",
    
    # Code blocks
    r"```",                     # Markdown code blocks
    r"\b(def |fn |function |class |import |from |return |yield )\b",
    r"\b(public |private |static |void |int |float |double |bool )\b",
    r"\b(console\.log|print\(|System\.out|printf\()",
    
    # Variable/function notation
    r"\b(f\(x\)|g\(x\)|h\(x\))\b",
    r"\b(x_\d|y_\d|z_\d)\b",
    
    # Math keywords
    r"\b(theorem|lemma|corollary|proof|qed|axiom|definition)\b",
    r"\b(derivative|integral|limit|converge|diverge)\b",
    
    # Chemical formulas
    r"\b(H\dO|CO\d|NaCl|Fe\d|CaCO\d)\b",
    
    # STEM heavy content markers
    r"\b(equation|formula|algorithm|computation|polynomial)\b",
]

STEM_PAGE_PATTERNS = [
    # Wikipedia STEM categories
    r"(?i)\b(mathematics|physics|chemistry|biology|engineering)\b",
    r"(?i)\b(computer science|programming|software|coding)\b",
    r"(?i)\b(statistics|probability|calculus|algebra|geometry)\b",
    r"(?i)\b(electron|circuit|voltage|current|resistor|capacitor)\b",
    r"(?i)\b(molecule|atom|electron|proton|neutron|quantum)\b",
]

# Language-positive patterns
LANGUAGE_POSITIVE = [
    r"[a-zA-Z]{5,}",           # Has real words (5+ letters)
    r"[.!?]['\")]?\s",         # Has sentence structure
    r"\b(the|and|that|with|have|this|from|they|what|when|were|them)\b",
]


def is_pure_language(text: str) -> bool:
    """Returns True only if text contains NO math/stem contamination."""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    
    # CRITICAL: Reject ANY document with math contamination
    for pat in MATH_EXCLUDE_PATTERNS:
        if re.search(pat, text):
            return False
    
    # Reject STEM-heavy pages
    stem_count = 0
    for pat in STEM_PAGE_PATTERNS:
        if re.search(pat, text):
            stem_count += 1
    if stem_count >= 3:  # Too many STEM keywords
        return False
    
    # Must have natural language structure
    has_language = False
    for pat in LANGUAGE_POSITIVE:
        if re.search(pat, text):
            has_language = True
            break
    if not has_language:
        return False
    
    # Reject if too many special characters (likely code/math)
    special_ratio = len(re.findall(r'[^a-zA-Z\s.,!?;:\'"-]', text)) / max(len(text), 1)
    if special_ratio > 0.15:
        return False
    
    # Reject if too many numbers
    digit_ratio = len(re.findall(r'\d', text)) / max(len(text), 1)
    if digit_ratio > 0.05:
        return False
    
    return True


def clean_prose(text: str) -> str:
    """Clean prose text while preserving natural language flow."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove any lingering math artifacts
    text = re.sub(r'\$\$?[^$]+\$\$?', '', text)  # Remove inline math
    text = re.sub(r'\\\w+\{[^}]*\}', '', text)    # Remove LaTeX commands
    
    # Truncate
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    return text.strip()


def load_wikitext() -> list[str]:
    """Load WikiText-103, filtering out STEM pages aggressively."""
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for row in ds:
            text = row.get("text", "")
            if not text or len(text.strip()) < 100:
                continue
            if is_pure_language(text):
                cleaned = clean_prose(text)
                # Split very long articles into segments
                if len(cleaned) > 200:
                    samples.append(cleaned)
            if len(samples) >= 30000:
                break
        print(f"  WikiText-103: {len(samples)} samples")
    except Exception as e:
        print(f"  WikiText-103: SKIP ({e})")
    return samples


def load_gutenberg() -> list[str]:
    """Load Project Gutenberg books, filtering out technical works."""
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("pg19", split="train")
        for row in ds:
            text = row.get("text", "")
            if is_pure_language(text):
                cleaned = clean_prose(text)
                if cleaned:
                    samples.append(cleaned)
            if len(samples) >= 10000:
                break
        print(f"  Project Gutenberg (PG-19): {len(samples)} samples")
    except Exception as e:
        print(f"  PG-19: SKIP ({e})")
    
    # Try alternative Gutenberg source
    if len(samples) < 5000:
        try:
            from datasets import load_dataset
            ds = load_dataset("bookcorpus", split="train")
            for row in ds:
                text = row.get("text", "")
                if is_pure_language(text):
                    cleaned = clean_prose(text)
                    if cleaned:
                        samples.append(cleaned)
                if len(samples) >= 15000:
                    break
            print(f"  BookCorpus: {len(samples)} total samples")
        except Exception as e:
            print(f"  BookCorpus: SKIP ({e})")
    
    return samples


def load_dialogue() -> list[str]:
    """Load conversational/dialogue data (no math contamination)."""
    samples = []
    try:
        from datasets import load_dataset
        # Try dialogue datasets
        for ds_path in ["daily_dialog", "blended_skill_talk", "conv_ai"]:
            try:
                ds = load_dataset(ds_path, split="train")
                for row in ds:
                    # Extract dialogue text
                    text = ""
                    if "dialog" in row:
                        text = " ".join(str(t) for t in row["dialog"])
                    elif "utterances" in row:
                        text = " ".join(str(t) for t in row["utterances"])
                    elif "text" in row:
                        text = row["text"]
                    
                    if is_pure_language(text):
                        cleaned = clean_prose(text)
                        if cleaned:
                            samples.append(cleaned)
                    if len(samples) >= 5000:
                        break
                if samples:
                    break
            except Exception:
                continue
        print(f"  Dialogue: {len(samples)} samples")
    except Exception as e:
        print(f"  Dialogue: SKIP ({e})")
    return samples


def load_wikipedia_prose() -> list[str]:
    """Load Wikipedia articles, filtering to only non-STEM content."""
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikipedia", "20220301.en", split="train", 
                          streaming=True)
        for row in ds:
            text = row.get("text", "")
            title = row.get("title", "")
            
            # Skip STEM articles by title
            if any(re.search(p, title, re.IGNORECASE) for p in STEM_PAGE_PATTERNS):
                continue
            
            if is_pure_language(text):
                cleaned = clean_prose(text)
                if cleaned:
                    samples.append(cleaned)
            if len(samples) >= 15000:
                break
        print(f"  Wikipedia (non-STEM): {len(samples)} samples")
    except Exception as e:
        print(f"  Wikipedia: SKIP ({e})")
    return samples


def main():
    print("=" * 70)
    print("PURE LANGUAGE CORPUS BUILDER")
    print("Goal: Zero-contamination natural language data for Model L")
    print("=" * 70)
    
    all_samples = []
    
    print("\n[1/4] Loading WikiText (filtered)...")
    all_samples.extend(load_wikitext())
    
    print("\n[2/4] Loading Project Gutenberg/BookCorpus...")
    all_samples.extend(load_gutenberg())
    
    print("\n[3/4] Loading conversational/dialogue data...")
    all_samples.extend(load_dialogue())
    
    print("\n[4/4] Loading Wikipedia (non-STEM only)...")
    all_samples.extend(load_wikipedia_prose())
    
    # Deduplicate
    seen = set()
    unique = []
    for s in all_samples:
        h = hash(s[:200])
        if h not in seen:
            seen.add(h)
            unique.append(s)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(all_samples)} raw -> {len(unique)} unique samples")
    est_tokens = len(unique) * 300
    print(f"Estimated tokens: ~{est_tokens:,} ({est_tokens/1e6:.1f}M)")
    
    # Save
    output_path = OUTPUT_DIR / "language_pure_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in unique:
            f.write(json.dumps({"text": sample}, ensure_ascii=False) + "\n")
    
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Inspection samples
    raw_path = OUTPUT_DIR / "language_pure_train.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(unique[:50]):
            f.write(f"=== SAMPLE {i+1} ===\n")
            f.write(sample[:500])
            f.write("\n...\n\n")
    
    print(f"Inspection samples: {raw_path}")
    print("\nDONE. Ready for training.")


if __name__ == "__main__":
    main()
