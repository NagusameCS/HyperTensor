#!/usr/bin/env python3
"""
STREAMLINED PURE DATA BUILDER --- Single script for both Math and Language.

Strategy:
  - Use CONTINUED PRETRAINING from SmolLM2-135M base (shared init guaranteed)
  - Need ~100M tokens per model for domain specialization
  - Aggressive filtering removes cross-domain contamination
  - Downloads from accessible HuggingFace datasets

Usage:
  python scripts/build_pure_data.py --skill math
  python scripts/build_pure_data.py --skill language
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ===========================================================================
# AGGRESSIVE MATH FILTERS
# ===========================================================================

MATH_EXCLUDE = [
    # Natural language markers --- MUST NOT appear in math corpus
    r"(?i)\b(once upon a time)\b",
    r"(?i)\b(chapter\s+\d+)\b",
    r"(?i)\b(she\s+said|he\s+said|they\s+said)\b",
    r"(?i)\b(feeling|felt|emotion|happy|sad|angry)\b",
    r"(?i)\b(story|novel|poem|verse|rhyme)\b",
    r"(?i)\b(character|protagonist)\b",
    r"(?i)\b(sure,? !|absolutely!|certainly!|of course!)\b",
]

MATH_INCLUDE = [
    r"\d",                          # Any digit
    r"[+\-*/=<>]",                 # Math operators
    r"\\\w+",                       # LaTeX
    r"\b(proof|theorem|lemma|axiom|corollary|definition)\b",
    r"\b(implies|therefore|hence|thus|let|assume|suppose)\b",
    r"\b(forall|exists|in\b|sqrt|sum|int|lim|inf)\b",
    r"```",                         # Code blocks
    r"\b(:=|::=)\b",               # Definition operators
]

# ===========================================================================
# AGGRESSIVE LANGUAGE FILTERS
# ===========================================================================

LANG_EXCLUDE = [
    # Math/STEM markers --- MUST NOT appear in language corpus
    r"\\begin\{", r"\\end\{", r"\\frac\{", r"\\sqrt\{",
    r"\\sum", r"\\int", r"\\alpha", r"\\beta", r"\\infty",
    r"\$\$", r"\$[^$]+\$", r"\\\[", r"\\\]",
    r"\b\d{4,}\b",                  # Large numbers
    r"[+\-*/=<>]{2,}",             # Multiple operators
    r"```",                         # Code blocks
    r"\b(def |fn |function |class |import |from |return )\b",
    r"\b(theorem|lemma|corollary|proof|qed|axiom)\b",
    r"\b(derivative|integral|equation|formula|algorithm)\b",
    r"[a-z]_\{\w+\}",               # LaTeX subscripts
    r"[a-z]\^\{\w+\}",              # LaTeX superscripts
    r"\b(f\(x\)|g\(x\)|h\(x\))\b",
]

LANG_INCLUDE = [
    r"[a-zA-Z]{5,}",               # Real words
    r"[.!?]['\")]?\s",             # Sentence structure
    r"\b(the|and|that|with|have|this|from|they|what|when)\b",
]


def filter_math(text: str) -> bool:
    """Keep only pure math/logic content."""
    if not text or len(text.strip()) < 50:
        return False
    for pat in MATH_EXCLUDE:
        if re.search(pat, text):
            return False
    for pat in MATH_INCLUDE:
        if re.search(pat, text):
            return True
    return False


def filter_language(text: str) -> bool:
    """Keep only pure natural language content."""
    if not text or len(text.strip()) < 100:
        return False
    for pat in LANG_EXCLUDE:
        if re.search(pat, text):
            return False
    # Check digit ratio --- too many digits = math/STEM
    digits = len(re.findall(r'\d', text))
    if digits / max(len(text), 1) > 0.03:
        return False
    for pat in LANG_INCLUDE:
        if re.search(pat, text):
            return True
    return False


def clean_text(text: str, max_len: int = 4096) -> str:
    """Normalize and truncate text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    if len(text) > max_len:
        text = text[:max_len]
    return text.strip()


# ===========================================================================
# DATA SOURCES
# ===========================================================================

def load_hf_dataset(ds_path: str, ds_config: str = None, split: str = "train",
                    streaming: bool = True, max_samples: int = 100000,
                    text_fields: list[str] = None) -> list[str]:
    """Generic HF dataset loader with error handling."""
    from datasets import load_dataset
    
    if text_fields is None:
        text_fields = ["text", "content", "question", "answer", "problem", "solution"]
    
    samples = []
    try:
        kwargs = {"split": split}
        if ds_config:
            kwargs["name"] = ds_config
        if streaming:
            ds = load_dataset(ds_path, streaming=True, **kwargs)
        else:
            ds = load_dataset(ds_path, **kwargs)
        
        count = 0
        for row in ds:
            # Try all text fields
            text = ""
            for field in text_fields:
                if field in row and row[field]:
                    text = str(row[field])
                    break
            if not text:
                # Try concatenating known fields
                parts = []
                for k, v in row.items():
                    if isinstance(v, str) and len(v) > 10:
                        parts.append(v)
                text = " ".join(parts)
            
            if text:
                samples.append(clean_text(text))
            
            count += 1
            if len(samples) >= max_samples:
                break
            if count % 50000 == 0:
                print(f"  ... processed {count:,} rows, kept {len(samples):,}")
        
        print(f"  {ds_path}: {len(samples):,} samples (from {count:,} rows)")
    except Exception as e:
        print(f"  {ds_path}: SKIP ({e})")
    
    return samples


def build_math_corpus() -> list[str]:
    """Build pure math corpus from multiple sources."""
    print("\n" + "="*60)
    print("BUILDING PURE MATH CORPUS")
    print("="*60)
    
    all_samples = []
    
    # 1. GSM8K
    print("\n[1] GSM8K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        for row in ds:
            q = row.get("question", "")
            a = row.get("answer", "")
            a = re.sub(r'####\s*\d+.*$', '', a).strip()
            text = f"Q: {q}\nA: {a}"
            if filter_math(text):
                all_samples.append(clean_text(text))
            if len(all_samples) >= 8000:
                break
        print(f"  GSM8K: {min(8000, len(all_samples)):,} samples")
    except Exception as e:
        print(f"  GSM8K: SKIP ({e})")
    
    # 2. OpenWebMath
    print("\n[2] OpenWebMath...")
    owm = load_hf_dataset("open-web-math/open-web-math", max_samples=50000)
    math_owm = [s for s in owm if filter_math(s)]
    all_samples.extend(math_owm)
    
    # 3. Math StackExchange (via math_dataset)
    print("\n[3] Math reasoning datasets...")
    for ds_path, ds_split in [("HuggingFaceH4/MATH-500", "test")]:
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_path, split=ds_split)
            for row in ds:
                problem = row.get("problem", "") or row.get("question", "")
                solution = row.get("solution", "") or row.get("answer", "")
                text = f"{problem}\n{solution}"
                if filter_math(text):
                    all_samples.append(clean_text(text))
                if len(all_samples) >= 70000:
                    break
        except Exception as e:
            print(f"  {ds_path}: SKIP ({e})")
    
    # 4. ArXiv / proof data
    print("\n[4] ArXiv/proof...")
    # Try proof-pile-2 subsets
    for ds_path in ["EleutherAI/proof-pile-2"]:
        proof = load_hf_dataset(ds_path, max_samples=20000)
        math_proof = [s for s in proof if filter_math(s)]
        all_samples.extend(math_proof)
    
    # Deduplicate
    seen = set()
    unique = []
    for s in all_samples:
        h = hash(s[:200])
        if h not in seen:
            seen.add(h)
            unique.append(s)
    
    # Estimate tokens
    total_chars = sum(len(s) for s in unique)
    est_tokens = total_chars // 4  # Rough: 4 chars per token
    
    print(f"\n{'='*60}")
    print(f"MATH CORPUS: {len(unique):,} samples, ~{est_tokens:,} tokens")
    print(f"{'='*60}")
    
    return unique


def build_language_corpus() -> list[str]:
    """Build pure language corpus from multiple sources."""
    print("\n" + "="*60)
    print("BUILDING PURE LANGUAGE CORPUS")
    print("="*60)
    
    all_samples = []
    
    # 1. WikiText-103 --- reliable, works with new datasets
    print("\n[1] WikiText-103...")
    wiki = load_hf_dataset("wikitext", "wikitext-103-raw-v1", 
                           streaming=False, max_samples=80000)
    lang_wiki = [s for s in wiki if filter_language(s)]
    all_samples.extend(lang_wiki)
    print(f"  WikiText: {len(lang_wiki):,} kept after filtering")
    
    # 2. FineWeb --- modern clean web text in Parquet format
    print("\n[2] FineWeb (CC-MAIN-2024-10 sample)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", 
                          split="train", streaming=True)
        count = 0
        for row in ds:
            text = row.get("text", "")
            if filter_language(text):
                all_samples.append(clean_text(text))
            count += 1
            if len(all_samples) >= 60000:
                break
            if count >= 200000:
                break
        print(f"  FineWeb: {min(60000, len(all_samples)):,} kept (from {count:,} rows)")
    except Exception as e:
        print(f"  FineWeb: SKIP ({e})")
        # Fallback: use more WikiText
        print("  Falling back to additional WikiText...")
        more_wiki = load_hf_dataset("wikitext", "wikitext-103-raw-v1",
                                     streaming=False, max_samples=50000)
        lang_more = [s for s in more_wiki if filter_language(s) and s not in set(all_samples)]
        all_samples.extend(lang_more[:30000])
    
    # 3. OpenWebText2 --- if available
    print("\n[3] OpenWebText2...")
    try:
        owt = load_hf_dataset("the_pile_openwebtext2", max_samples=30000)
        lang_owt = [s for s in owt if filter_language(s)]
        all_samples.extend(lang_owt)
        print(f"  OpenWebText2: {len(lang_owt):,} kept")
    except Exception as e:
        print(f"  OpenWebText2: SKIP ({e})")
        # Use more WikiText as fallback
        extra = load_hf_dataset("wikitext", "wikitext-103-raw-v1",
                                streaming=False, max_samples=50000)
        lang_extra = [s for s in extra if filter_language(s)]
        existing = set(hash(s[:200]) for s in all_samples)
        new_extra = [s for s in lang_extra if hash(s[:200]) not in existing]
        all_samples.extend(new_extra[:30000])
    
    # Deduplicate
    seen = set()
    unique = []
    for s in all_samples:
        h = hash(s[:200])
        if h not in seen:
            seen.add(h)
            unique.append(s)
    
    total_chars = sum(len(s) for s in unique)
    est_tokens = total_chars // 4
    
    print(f"\n{'='*60}")
    print(f"LANGUAGE CORPUS: {len(unique):,} samples, ~{est_tokens:,} tokens")
    print(f"{'='*60}")
    
    return unique


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Build pure single-skill training data")
    parser.add_argument("--skill", required=True, choices=["math", "language"])
    args = parser.parse_args()
    
    out_dir = Path(f"data/pure_{args.skill}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.skill == "math":
        samples = build_math_corpus()
        out_file = out_dir / "math_pure_train.jsonl"
    else:
        samples = build_language_corpus()
        out_file = out_dir / "language_pure_train.jsonl"
    
    # Save
    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
    
    size_mb = out_file.stat().st_size / 1e6
    est_tokens = sum(len(s) for s in samples) // 4
    
    print(f"\nSaved: {out_file} ({size_mb:.1f} MB)")
    print(f"Estimated tokens: ~{est_tokens:,} ({est_tokens/1e6:.1f}M)")
    print(f"\nReady for training:")
    print(f"  python scripts/train_pure_model.py --skill {args.skill} --seed 42")
    
    # Quick quality check
    if samples:
        print(f"\nSample check (first 200 chars):")
        print(f"  {samples[0][:200]}...")


if __name__ == "__main__":
    main()
