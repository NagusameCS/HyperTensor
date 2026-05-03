#!/usr/bin/env python3
"""
scripts/paperA_proof/update_paperE_rho.py

Read docs/figures/paper-e/rho_sweep/rho_summary.json (output of
grc_distill.py --print-rho) and update the rho placeholder values in
ARXIV_SUBMISSIONS/paper-E/grc-light-distillation.tex.

Replacements:
  1. Line matching 'gives $\\rho \\approx ...$;' in sec:gapbound
  2. Line matching 'roughly $...\\%$ of the projection-discarded'
  3. Line matching 'rho{\\approx}...' in sec:status

Also updates the 'Phase 1 manifest' entry for rho in the Done list.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RHO_JSON  = ROOT / "docs" / "figures" / "paper-e" / "rho_sweep" / "rho_summary.json"
TEX_PATH  = ROOT / "ARXIV_SUBMISSIONS" / "paper-E" / "grc-light-distillation.tex"


def main() -> None:
    if not RHO_JSON.exists():
        print(f"[update_rho] ERROR: {RHO_JSON} not found. Run --print-rho first.",
              file=sys.stderr)
        sys.exit(1)

    data = json.loads(RHO_JSON.read_text(encoding="utf-8-sig"))
    rho = data["mean_rho"]
    k   = data["rank"]
    r   = data["lora_rank"]
    n   = data["n_layers"]
    pct = rho * 100

    print(f"[update_rho] measured rho={rho:.4f} ({pct:.1f}%)  k={k} r={r} n_layers={n}")

    tex = TEX_PATH.read_text(encoding="utf-8")
    changes = 0

    # Use direct string replacement to avoid regex backslash complexity in LaTeX
    replacements = [
        (
            r"$\rho \approx 0.42$;",
            f"$\\rho \\approx {rho:.2f}$;",
        ),
        (
            r"roughly $42\%$ of the",
            f"roughly ${pct:.1f}\\%$ of the",
        ),
        (
            r"$\rho{\approx}0.42$ at",
            f"$\\rho{{\\approx}}{rho:.2f}$ at",
        ),
        (
            r"$\rho{\approx}0.42$ and",
            f"$\\rho{{\\approx}}{rho:.2f}$ and",
        ),
    ]

    for old, new in replacements:
        if old in tex:
            tex = tex.replace(old, new, 1)
            print(f"[update_rho] replaced: {old!r}")
            changes += 1
        else:
            print(f"[update_rho] WARNING: not found: {old!r}", file=sys.stderr)

    if changes == 0:
        print("[update_rho] ERROR: no replacements made. Check tex patterns.", file=sys.stderr)
        sys.exit(1)

    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"[update_rho] wrote {TEX_PATH} ({changes} replacements)")
    print(f"[update_rho] rho placeholder 0.42 -> {rho:.2f} ({pct}%)")


if __name__ == "__main__":
    main()
