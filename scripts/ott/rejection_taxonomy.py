#!/usr/bin/env python3
"""
D2 --- Rejection Taxonomy Analyser
=================================
Parses geodessical rejection_log.tsv (written by --ott-rejection-log) and
classifies each rejected draft into one of three types from the paper:

  Type I  --- Vocabulary / Tokenisation Mismatch
             The draft token decodes to a piece with non-printable bytes,
             multi-byte boundary artefacts, or a token ID outside the model
             vocab.  Root cause: geodesic operates in embedding space and
             can land on tokens that form syntactically invalid pieces.

  Type II --- Manifold Divergence
             The draft token is a valid piece but the verifier chose a
             *different* token.  Root cause: the geodesic trajectory has
             drifted off the linguistic manifold (curvature / GRC has not
             corrected the geodesic back onto the verifier's probability
             simplex).

  Type III --- Early-Turn Context Collapse
             The rejection occurred within the first WARMUP_STEPS (default 4)
             of a new generation turn.  Root cause: geodesic embeddings are
             position-independent, so predictions made before sufficient
             causal context is in the window diverge structurally.

Usage:
    python scripts/ott/rejection_taxonomy.py --log rejection_log.tsv [--out taxonomy_report.json]

Outputs:
    - Per-rejection labelled records
    - Aggregate count matrix
    - taxonomy_report.json (if --out given)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

WARMUP_STEPS = 4          # steps still in warmup window -> Type III candidate
MIN_PIECE_LEN = 1         # valid piece must have at least 1 printable char


def classify_rejection(row: dict) -> str:
    """Return Type I / II / III for one rejection event."""
    draft_tok  = int(row["draft_tok"])
    draft_piece = row["draft_piece"]
    verifier_tok = int(row["verifier_tok"])
    warmup     = int(row["warmup"])

    # Type III --- early-turn context collapse (checked first so warmup rejections
    # are not mis-attributed to manifold divergence even if piece is valid)
    if warmup == 1:
        return "Type III"

    # Type I --- vocabulary / tokenisation mismatch
    # Conditions: empty piece, piece with only non-printable chars, or draft == verifier
    # (the verifier would have accepted if piece were the right token; if draft_tok < 0
    # or decode produced nothing, it is definitely a vocab boundary issue)
    if draft_tok < 0:
        return "Type I"
    stripped = draft_piece.strip()
    if len(stripped) == 0:
        return "Type I"
    # Check for non-printable / control characters in piece
    non_printable = sum(1 for c in draft_piece if ord(c) < 32 and c not in ("\t", "\n", "\r"))
    if non_printable > len(draft_piece) * 0.5:
        return "Type I"
    # Check for lone UTF-8 continuation bytes (byte pattern ?\x80--\xBF alone)
    if re.search(r'[\x80-\xbf]', draft_piece) and not re.search(r'[\xc0-\xff]', draft_piece):
        return "Type I"

    # Type II --- manifold divergence: valid piece, verifier chose differently
    return "Type II"


def analyse(log_path: Path):
    """Parse rejection log, classify, and return report dict."""
    records = []
    header_found = False
    columns = []

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if not header_found:
                # First non-comment line is the column header
                columns = parts
                header_found = True
                continue
            if len(parts) != len(columns):
                continue  # skip malformed rows
            row = dict(zip(columns, parts))
            records.append(row)

    if not records:
        print(f"[D2] No rejection records found in {log_path}", file=sys.stderr)
        return None

    counts = defaultdict(int)
    per_type = {"Type I": [], "Type II": [], "Type III": []}
    annotated = []

    for row in records:
        rtype = classify_rejection(row)
        counts[rtype] += 1
        per_type[rtype].append(row)
        annotated.append({**row, "rejection_type": rtype})

    total = len(records)
    summary = {
        "total_rejections": total,
        "Type I":  {"count": counts["Type I"],  "pct": 100 * counts["Type I"]  / total},
        "Type II": {"count": counts["Type II"], "pct": 100 * counts["Type II"] / total},
        "Type III":{"count": counts["Type III"],"pct": 100 * counts["Type III"]/ total},
    }

    # Gather Type II first-rejection positions (draft_pos distribution)
    if per_type["Type II"]:
        positions = [int(r["draft_pos"]) for r in per_type["Type II"]]
        summary["Type II"]["pos_p50"] = sorted(positions)[len(positions)//2]
        summary["Type II"]["pos_p90"] = sorted(positions)[int(len(positions)*0.9)]

    return {"summary": summary, "records": annotated}


def print_report(report: dict):
    s = report["summary"]
    total = s["total_rejections"]
    print(f"\n{'='*60}")
    print(f"  D2 Rejection Taxonomy Report")
    print(f"  Total rejections analysed: {total}")
    print(f"{'='*60}")
    for t in ("Type I", "Type II", "Type III"):
        c = s[t]["count"]
        p = s[t]["pct"]
        label = {
            "Type I":   "Vocabulary / Tokenisation Mismatch",
            "Type II":  "Manifold Divergence",
            "Type III": "Early-Turn Context Collapse",
        }[t]
        print(f"  {t}: {c:4d}  ({p:5.1f}%)  --- {label}")
        if t == "Type II" and "pos_p50" in s[t]:
            print(f"           Rejection position p50={s[t]['pos_p50']}  p90={s[t]['pos_p90']}")
    print(f"{'='*60}\n")

    # Root-cause priority guidance
    dominant = max(("Type I", "Type II", "Type III"), key=lambda t: s[t]["count"])
    print(f"  Primary driver: {dominant}")
    guidance = {
        "Type I":  ("  -> Harden geodesic piece-quality filter (geodesic_piece_quality_ok).\n"
                    "    Enforce vocab boundary checks before draft is proposed."),
        "Type II": ("  -> Increase GRC correction budget or tighten verifier margin.\n"
                    "    Consider raising --ott-spec-thresh or running C3 calibration sweep."),
        "Type III":("  -> Increase SPEC_WARMUP_N in host/main.c (currently 4).\n"
                    "    Delay geodesic drafting until more causal context is available."),
    }
    print(guidance[dominant])
    print()


def main():
    parser = argparse.ArgumentParser(description="D2 Rejection Taxonomy Analyser")
    parser.add_argument("--log", required=True, help="Path to rejection_log.tsv")
    parser.add_argument("--out", default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[D2] Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    report = analyse(log_path)
    if report is None:
        sys.exit(1)

    print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[D2] JSON report written to {out_path}")


if __name__ == "__main__":
    main()
