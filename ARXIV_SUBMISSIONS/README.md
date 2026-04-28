# HyperTensor — arXiv Submissions Master Folder

This folder is structured per arXiv's recommended workflow: one self-contained
subfolder per paper, every file the LaTeX build needs sitting next to
`main.tex`. There are **no shared parents** to traverse — each subfolder is
ready to be tarred and uploaded.

## The four submissions

| Folder | Title | Source web papers | arXiv category (suggested) |
|--------|-------|-------------------|----------------------------|
| [paper-A/](paper-A/) | **Calibration-Free Low-Rank Attention Compression for Bandwidth-Bound LLM Decode** | web Paper 1 (+ Paper 0 background) | `cs.LG` (cross-list `cs.PF`) |
| [paper-B/](paper-B/) | **Geodesic Projection: A Multi-Slot Compression Pipeline with Adaptive Phase-, Gauge-, Thermal-, and Drift-Aware Extensions** | web Papers 2 + 6 | `cs.LG` (cross-list `cs.AR`) |
| [paper-C/](paper-C/) | **Geodesic Speculative Decoding under Compression** | web Paper 3 | `cs.LG` (cross-list `cs.CL`) |
| [paper-D/](paper-D/) | **Organic Training Theory and Geodesic Trajectory Caching: Theory and Empirical Anchor on Three Open-Weight Models** | web Papers 4 + 5 | `cs.LG` (cross-list `math.DG`) |

Web Paper 0 (introduction) is intentionally not submitted: it is a
navigational/lexicon page, not standalone arXiv material. Its concepts
(Roofline, intrinsic dim, Eckart–Young, manifold hypothesis) are absorbed
into Paper A's introduction and background.

## Folder layout per paper

```
paper-X/
├── main.tex          # The submission. Cites refs.bib.
├── hypertensor.sty   # Shared preamble, copied locally (arXiv builds flat).
├── refs.bib          # Full bibliography, copied locally.
└── README.md         # Per-paper notes + build commands.
```

## Build instructions (local)

You need TeX Live or MacTeX. The minimum package set is `latexmk`, `biber`,
`tcolorbox`, `newtxtext`, `newtxmath`, `biblatex`, `cleveref`, `siunitx`,
`tabularx`, `booktabs`, `mathtools`, `enumitem`, `microtype`, `listings`.

```bash
# Install on macOS:
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk biber tcolorbox newtx biblatex cleveref siunitx \
                   tabularx booktabs mathtools enumitem microtype listings \
                   collection-fontsrecommended

# Build any paper:
cd paper-A
latexmk -pdf -bibtex- -interaction=nonstopmode main.tex
# (latexmk auto-runs biber for biblatex)
```

Output: `paper-A/main.pdf`.

## Preparing for arXiv upload

arXiv requires the **TeX source**, not the PDF. The recommended workflow:

1. Build locally and verify `main.pdf` is correct.
2. Strip TeX comments before upload (arXiv archives all source forever and
   any private notes in comments would be permanent). A helper script is
   included:

   ```bash
   cd paper-A
   ../strip_comments.sh main.tex > main.cleaned.tex
   mv main.cleaned.tex main.tex
   ```

3. Pre-build the bibliography so arXiv does not need biber/biblatex
   resolution (arXiv's autoTeX runs `bibtex` only):

   ```bash
   latexmk -pdf -interaction=nonstopmode main.tex
   # Verify main.bbl is generated next to main.tex
   ```

4. Tar **only** the four files needed:

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
