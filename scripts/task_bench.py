#!/usr/bin/env python3
"""
Task-Level Benchmark Harness (Tier 2 / Paper F).

Evaluates GRC-compressed models on standard benchmarks beyond perplexity:
MMLU (multiple-choice knowledge), GSM8K (math reasoning), HumanEval (code).

Runs each benchmark at multiple compression ranks and produces a
per-rank accuracy table.  This is the critical adoption evidence:
practitioners care about task performance, not just PPL.

Usage:
  # MMLU (needs data/mmlu/ downloaded)
  python scripts/task_bench.py --model models/smollm2-135m-instruct-q8_0.gguf \
    --benchmark mmlu --ranks 64,128,256,512,1024 \
    --out benchmarks/task_bench

  # GSM8K
  python scripts/task_bench.py --model models/smollm2-135m-instruct-q8_0.gguf \
    --benchmark gsm8k --ranks 64,128,256,512,1024

  # All benchmarks
  python scripts/task_bench.py --model models/smollm2-135m-instruct-q8_0.gguf \
    --benchmark all --ranks 256,512,1024 --n-shot 5

Dataset setup:
  data/mmlu/          ← MMLU .csv files by subject (from Hendrycks et al.)
  data/gsm8k_test.jsonl ← GSM8K test set
  data/humaneval.jsonl  ← HumanEval problems
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


# ===========================================================================
# Binary + model helpers
# ===========================================================================

def detect_exe() -> Optional[Path]:
    for p in [
        ROOT / "build_host" / "geodessical2.exe",
        ROOT / "build_host" / "geodessical.exe",
    ]:
        if p.exists():
            return p
    return None


def run_prompt(exe: Path, model: Path, prompt: str, k: Optional[int],
               n_tokens: int = 256, temp: float = 0.0,
               ctx_size: int = 2048) -> str:
    """Run a single prompt through the model and return generated text.
    Uses plain prompt format — the instruct model handles plain
    Question/Answer formatting correctly without ChatML tokens."""
    args = [
        str(exe), str(model),
        "--ctx-size", str(ctx_size),
        "-p", prompt, "-n", str(n_tokens),
        "--temp", str(temp),
    ]
    if k is not None:
        args += [
            "--ott-full", "--no-verifier",
            "--axex-compress", "--axex-compress-rank", str(k),
        ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=300,
                              encoding='utf-8', errors='replace')
        return proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


# ===========================================================================
# MMLU
# ===========================================================================

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def load_mmlu(data_dir: Path, subject: str, split: str = "test") -> list[dict]:
    """Load MMLU questions for a subject. Returns list of {question, choices, answer}."""
    path = data_dir / split / f"{subject}_{split}.csv"
    if not path.exists():
        # Try alternative path structures
        alt = data_dir / f"{subject}_{split}.csv"
        if alt.exists():
            path = alt
        else:
            return []

    import csv as csv_mod
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv_mod.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 6:
                questions.append({
                    "question": row[0],
                    "A": row[1], "B": row[2], "C": row[3], "D": row[4],
                    "answer": row[5].strip(),
                })
    return questions


def format_mmlu_prompt(few_shot: list[dict], question: dict) -> str:
    """Format an MMLU question as a prompt with optional few-shot examples."""
    prompt = ""
    for ex in few_shot:
        prompt += f"Question: {ex['question']}\n"
        for opt in ["A", "B", "C", "D"]:
            prompt += f"{opt}. {ex[opt]}\n"
        prompt += f"Answer: {ex['answer']}\n\n"

    prompt += f"Question: {question['question']}\n"
    for opt in ["A", "B", "C", "D"]:
        prompt += f"{opt}. {question[opt]}\n"
    prompt += "Answer:"
    return prompt


def extract_mmlu_answer(output: str) -> Optional[str]:
    """Extract A/B/C/D answer from model output."""
    # Look for "Answer: X" pattern
    m = re.search(r"Answer:\s*([A-D])", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Look for standalone A/B/C/D after the prompt
    lines = output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line in "ABCD" and len(line) == 1:
            return line
        m = re.match(r"^([A-D])[\.\)\s]", line)
        if m:
            return m.group(1)
    # Last resort: find last A/B/C/D in output
    matches = re.findall(r"\b([A-D])\b", output)
    if matches:
        return matches[-1]
    return None


def run_mmlu(exe: Path, model: Path, k: Optional[int], data_dir: Path,
             n_shot: int = 5, max_q_per_subject: int = 50,
             subjects: Optional[list[str]] = None) -> dict:
    """Run MMLU evaluation. Returns {subject: accuracy}."""
    if subjects is None:
        subjects = MMLU_SUBJECTS[:10]  # Default: first 10 subjects

    results = {}
    total_correct = 0
    total_questions = 0

    # Build few-shot examples from dev set
    dev_examples = []
    for subj in subjects[:2]:  # Use first 2 subjects for dev examples
        dev_qs = load_mmlu(data_dir, subj, "dev")
        if dev_qs:
            dev_examples.extend(dev_qs[:n_shot])
            break
    if len(dev_examples) < n_shot:
        dev_examples = []  # Fall back to zero-shot
        print("  (zero-shot: no dev examples found)")

    for subject in subjects:
        questions = load_mmlu(data_dir, subject)
        if not questions:
            continue
        questions = questions[:max_q_per_subject]

        correct = 0
        for i, q in enumerate(questions):
            prompt = format_mmlu_prompt(dev_examples, q)
            output = run_prompt(exe, model, prompt, k, n_tokens=8, temp=0.0)
            extracted = extract_mmlu_answer(output)
            if extracted == q["answer"]:
                correct += 1

            if (i + 1) % 20 == 0:
                print(f"    {subject}: {i+1}/{len(questions)}  "
                      f"acc={correct/(i+1):.2%}", flush=True)

        acc = correct / len(questions) if questions else 0.0
        results[subject] = round(acc, 4)
        total_correct += correct
        total_questions += len(questions)
        print(f"  {subject:>35s}: {correct:3d}/{len(questions):3d}  {acc:.2%}")

    results["_overall"] = round(total_correct / total_questions, 4) if total_questions else 0.0
    return results


# ===========================================================================
# GSM8K
# ===========================================================================

def load_gsm8k(path: Path) -> list[dict]:
    """Load GSM8K test set from JSONL."""
    if not path.exists():
        return []
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def extract_gsm8k_answer(output: str) -> Optional[float]:
    """Extract the final numeric answer from GSM8K output.
    GSM8K answers end with '#### N' where N is the answer."""
    m = re.search(r"####\s*([\d,.]+)", output)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # Fallback: find last number in output
    numbers = re.findall(r"[\d,.]+", output)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def run_gsm8k(exe: Path, model: Path, k: Optional[int], data_path: Path,
              n_shot: int = 5, max_q: int = 100) -> dict:
    """Run GSM8K evaluation."""
    questions = load_gsm8k(data_path)
    if not questions:
        return {"error": "no gsm8k data found", "_overall": 0.0}

    questions = questions[:max_q]
    correct = 0

    # Few-shot examples from the first n_shot questions
    few_shot = ""
    for ex in questions[:n_shot]:
        few_shot += f"Question: {ex['question']}\n"
        few_shot += f"Answer: {ex['answer']}\n\n"

    for i, q in enumerate(questions):
        if i < n_shot:
            continue  # Skip few-shot examples from evaluation

        prompt = few_shot + f"Question: {q['question']}\nAnswer: Let's think step by step.\n"
        output = run_prompt(exe, model, prompt, k, n_tokens=512, temp=0.0)
        extracted = extract_gsm8k_answer(output)
        target_str = re.search(r"####\s*([\d,.]+)", q["answer"])
        if target_str and extracted is not None:
            target = float(target_str.group(1).replace(",", ""))
            if abs(extracted - target) < 0.01:
                correct += 1

        if (i + 1) % 20 == 0:
            n_eval = i + 1 - n_shot
            acc = correct / n_eval if n_eval > 0 else 0.0
            print(f"    gsm8k: {i+1}/{len(questions)}  acc={acc:.2%}", flush=True)

    n_eval = len(questions) - n_shot
    acc = correct / n_eval if n_eval > 0 else 0.0
    return {"gsm8k": round(acc, 4), "_overall": round(acc, 4)}


# ===========================================================================
# HumanEval
# ===========================================================================

def load_humaneval(path: Path) -> list[dict]:
    """Load HumanEval problems."""
    if not path.exists():
        return []
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def check_humaneval_pass(code: str, test: str, entry_point: str,
                         timeout: float = 5.0) -> bool:
    """Run HumanEval test cases against generated code. Returns pass/fail."""
    import signal as _signal

    # Combine generated code with test
    full_code = code + "\n\n" + test + f"\n\ncheck({entry_point})"

    try:
        # Run in a subprocess for safety
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def run_humaneval(exe: Path, model: Path, k: Optional[int], data_path: Path,
                  max_q: int = 50) -> dict:
    """Run HumanEval evaluation."""
    problems = load_humaneval(data_path)
    if not problems:
        return {"error": "no humaneval data found", "_overall": 0.0}

    problems = problems[:max_q]
    passed = 0

    for i, p in enumerate(problems):
        prompt = p["prompt"]
        output = run_prompt(exe, model, prompt, k, n_tokens=512, temp=0.0)

        # Extract just the function body (remove everything after the function)
        code = prompt + output
        # Try to find the end of the function
        func_end = code.find("\n\n", len(prompt))
        if func_end > 0:
            code = code[:func_end]

        result = check_humaneval_pass(
            code, p["test"], p.get("entry_point", "candidate"),
        )
        if result:
            passed += 1

        if (i + 1) % 10 == 0:
            acc = passed / (i + 1)
            print(f"    humaneval: {i+1}/{len(problems)}  pass@{i+1}={acc:.2%}",
                  flush=True)

    acc = passed / len(problems) if problems else 0.0
    return {"humaneval": round(acc, 4), "_overall": round(acc, 4)}


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Task-Level Benchmark Harness (Tier 2 / Paper F)")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--out", default="benchmarks/task_bench")
    ap.add_argument("--benchmark", default="mmlu",
                    choices=["mmlu", "gsm8k", "humaneval", "all"])
    ap.add_argument("--ranks", default="256,512,1024")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--max-q", type=int, default=100,
                    help="Max questions per benchmark")
    ap.add_argument("--subjects", default="",
                    help="Comma-separated MMLU subjects (empty=first 10)")
    args = ap.parse_args()

    exe = Path(args.exe) if args.exe else detect_exe()
    if not exe or not exe.exists():
        print("ERROR: geodessical binary not found.", file=sys.stderr)
        sys.exit(2)
    model = Path(args.model)
    ranks = [int(x) for x in args.ranks.split(",")]
    data_dir = Path(args.data_dir)

    benchmarks = ["mmlu", "gsm8k", "humaneval"] if args.benchmark == "all" else [args.benchmark]
    all_ranks = [None] + ranks  # None = baseline

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"model": str(model), "benchmarks": {}, "ranks": ranks}

    for bm in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bm}")
        print(f"{'='*60}")

        summary["benchmarks"][bm] = {}

        for k in all_ranks:
            k_label = "baseline" if k is None else str(k)
            print(f"\n  --- k={k_label} ---")
            t0 = time.time()

            if bm == "mmlu":
                subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
                if not subjects:
                    subjects = None
                result = run_mmlu(exe, model, k, data_dir / "mmlu",
                                  args.n_shot, args.max_q, subjects)
            elif bm == "gsm8k":
                gsm8k_path = data_dir / "gsm8k_test.jsonl"
                result = run_gsm8k(exe, model, k, gsm8k_path,
                                   args.n_shot, args.max_q)
            elif bm == "humaneval":
                he_path = data_dir / "humaneval.jsonl"
                result = run_humaneval(exe, model, k, he_path, args.max_q)

            elapsed = time.time() - t0
            summary["benchmarks"][bm][k_label] = result
            print(f"  k={k_label:>8s}  overall={result.get('_overall', 0):.4f}  "
                  f"({elapsed:.0f}s)")

    # Write summary
    with open(out_dir / "task_bench_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print results
    print(f"\n{'='*60}")
    print("Task Benchmark Summary")
    print(f"{'='*60}")
    for bm in benchmarks:
        print(f"\n{bm}:")
        header = f"{'k':>10s}"
        for k_label in ["baseline"] + [str(r) for r in ranks]:
            header += f"  {k_label:>10s}"
        print(header)
        print("-" * len(header))

        if bm == "mmlu":
            # Per-subject and overall
            all_subjects = set()
            for k_label in ["baseline"] + [str(r) for r in ranks]:
                if k_label in summary["benchmarks"][bm]:
                    all_subjects.update(
                        s for s in summary["benchmarks"][bm][k_label]
                        if s != "_overall"
                    )
            sorted_subjects = sorted(all_subjects)[:15]
            for subj in sorted_subjects + ["_overall"]:
                line = f"{subj:>10s}"
                for k_label in ["baseline"] + [str(r) for r in ranks]:
                    val = summary["benchmarks"][bm].get(k_label, {}).get(subj, "-")
                    if isinstance(val, float):
                        line += f"  {val:10.4f}"
                    else:
                        line += f"  {str(val):>10s}"
                print(line)
        else:
            line = f"{'overall':>10s}"
            for k_label in ["baseline"] + [str(r) for r in ranks]:
                val = summary["benchmarks"][bm].get(k_label, {}).get("_overall", "-")
                if isinstance(val, float):
                    line += f"  {val:10.4f}"
                else:
                    line += f"  {str(val):>10s}"
            print(line)

    print(f"\n[done] {out_dir / 'task_bench_summary.json'}")


if __name__ == "__main__":
    main()
