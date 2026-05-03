# HyperTensor — Fifteen-Part Framework for Geometric Neural Compression

## Papers (I-XV)

| Part | Folder | Title | Status |
|------|--------|-------|--------|
| I | paper-I | GRC Attention Compression |  arXiv-ready |
| II | paper-II | Geodesic Projection Pipeline |  arXiv-ready |
| III | paper-III | Geodesic Speculative Decoding |  arXiv-ready |
| IV | paper-IV | OTT/GTC Manifold Runtime |  arXiv-ready |
| V | paper-V | GRC Light Distillation |  arXiv-ready |
| VI | paper-VI | Task-Level Asymmetric Degradation |  Structural |
| VII | paper-VII | FFN Cluster Compression |  PPL pending |
| VIII | paper-VIII | GTC as Vector Database |  Deployment pending |
| IX | paper-IX | Super-Baseline Universality |  Cross-GPU pending |
| X | paper-X | CECI Chimeric Model Vector Bridging |  Data collected |
| XI | paper-XI | Universal Geodesic Taxonomy |  Blueprint |
| XII | paper-XII | The Geodesic Compiler |  Blueprint |
| XIII | paper-XIII | Orthogonal Geodesic Deviation |  Blueprint |
| XIV | paper-XIV | Geodesic Sniping |  Blueprint |
| XV | paper-XV | Completely Organic Generation |  Blueprint |

## Experiment Registry

30 experiments in `../benchmarks/experiment_registry.json` — 8 verified, 12 scripts built, 10 blueprints.

## Build

```bash
make all      # All 15 papers
make X        # Single paper (I-XV)
make clean    # Remove aux files
```

## Folder Layout

```
paper-{I..XV}/
 *.tex             # The submission
 hypertensor.sty   # Shared preamble
 refs.bib          # Bibliography
 figures/          # (some papers)
```

## Building

Requires: `pdflatex`, `biber`, `latexmk` (MiKTeX or TeX Live).

```bash
cd paper-X
pdflatex -interaction=nonstopmode paper.tex
biber paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Or use `make X` for a single paper.

   ```bash
   tar czf paper-A.tar.gz \
       -C paper-A main.tex main.bbl hypertensor.sty
   ```

   Note: with `main.bbl` shipped, arXiv does not need `refs.bib` and does
   not need to run biber. Including `refs.bib` is harmless but unnecessary.

5. Upload `paper-A.tar.gz` at <https://arxiv.org/submit>.

The convenience target `make submit-A` (etc.) in this folder's `Makefile`
performs steps 3–4 automatically.

## arXiv checklist (apply to each paper before upload)

- [ ] `main.pdf` builds cleanly with no overfull-hbox or undefined-reference
      warnings.
- [ ] `main.bbl` is included in the tarball; `refs.bib` is **not** required.
- [ ] No `\todo{}` or `\note{}` macros remain.
- [ ] No TeX comments containing private remarks (run `strip_comments.sh`).
- [ ] Title, author, abstract match the arXiv submission form.
- [ ] All figure files (if any) are in the tarball with relative paths.
- [ ] No absolute paths (`/Users/...`) anywhere in `main.tex`.
- [ ] License selected on submission form (the project licence is in the
      repository root `LICENSE`).
- [ ] Subject category and cross-lists match the suggestions above.

## Reproduction packages

Each paper's reproduction commands point to scripts and expected outputs in
the main repository:

- Paper A: `repro/REPRODUCE.md`, `repro/expected_outputs/`,
  `scripts/benchmark_whitepaper_finalize.ps1`,
  `scripts/validation_cycle.ps1`.
- Paper B: `legacy/axiom_vis/`, `docs/figures/spectra_summary.json`,
  `scripts/inventory_models.py`.
- Paper C: `host/main.c`, `ott_readiness_report.json`,
  `scripts/benchmark_decode_nopipe.ps1`.
- Paper D: `scripts/gtc/{manifold,jacobi,gtc_benchmark,record_store}.py`,
  `docs/figures/gtc/`.

## Versioning

These are **v1** of the arXiv preparation. Each paper's title page records
its source-paper revision date. After arXiv assigns identifiers, link them
back into the project README.

## Pre-submission empirical-data checklist

Four runnable scripts in `scripts/` produce CSV + `\input`-able `.tex`
snippets that drop directly into the papers. They are honest-or-fail: each
errors out if a prereq is missing rather than fabricating any number.

| # | Script | Targets | Outputs (under `docs/data/`) | Used in |
|---|--------|---------|------------------------------|---------|
| 1 | [`scripts/benchmark_ncu_l2_profile.ps1`](../scripts/benchmark_ncu_l2_profile.ps1) | NCU L2 hit rate + DRAM bytes, baseline vs. GRC k=1024 | `ncu_l2_profile.csv`, `ncu_l2_profile.tex` | Paper A § Cache-Fit Hypothesis |
| 2 | [`scripts/run_lm_eval_suite.ps1`](../scripts/run_lm_eval_suite.ps1) | GSM8K, HumanEval, MBPP via lm-eval, baseline vs. GRC k=1536 | `lm_eval_results.json`, `lm_eval_results.tex` | Paper A § Results, Paper C § End-to-End |
| 3 | [`scripts/context_length_sweep.ps1`](../scripts/context_length_sweep.ps1) | Decode tok/s at ctx ∈ {128, 512, 1024, 2048, 4096}, baseline vs. GRC k=1024 | `context_length_sweep.csv`, `context_length_sweep.tex` | Paper A § Results |
| 4 | [`scripts/benchmark_rank_pareto.ps1`](../scripts/benchmark_rank_pareto.ps1) | Granular rank Pareto at k ∈ {512, 768, 1024, 1280, 1536} | `rank_pareto.csv`, `rank_pareto.tex` | Paper A § Results |

Each generated `.tex` is a single `\begin{tabular}…\end{tabular}` and is
designed to be wrapped in a `\begin{table}…\end{table}` with a paper-side
caption, e.g.:

```latex
\begin{table}[t]
  \centering
  \caption{Granular rank--Pareto sweep on Llama-3.1-8B / RTX 4070 Laptop.
           Ratios $>1$ exceed the uncompressed baseline.}
  \label{tab:rank-pareto}
  \input{../../docs/data/rank_pareto.tex}
\end{table}
```

These four runs are gated on local hardware time, not on this checkout.
Run them before tagging an arXiv-ready PDF; until then the corresponding
table cells should be left blank (preferred) or marked `\pending{}` rather
than filled with placeholder numbers.
