# Paper A — Calibration-Free Low-Rank Attention Compression

**Source:** web Paper 1 (Geodessical Runtime Compression v0.6.0 technical
report) plus the foundational lexicon material from web Paper 0 (Roofline,
intrinsic dim, Eckart–Young, manifold hypothesis), absorbed into §2 and §6.

**Headline result.** $106.27\%$ relative decode throughput at $k{=}1024$ on
Llama-3.1-8B-Instruct (Q4\_K\_M) / RTX 4070 Laptop, with $t=53.88$,
$p<10^{-10}$, bootstrap CI $[1.0607,1.0650]$. At $k{=}1536$, throughput is
statistically indistinguishable from baseline (97.55%) at +13.30% WikiText-2
PPL.

## Coverage map (for reviewers)

| Web-paper item | Where it lives in `main.tex` |
|---|---|
| Hardware spec table (web §1.1) | Appendix, "Hardware, model, and runtime configuration" |
| Model spec (web §1.2) | Same appendix table |
| Runtime spec (web §1.3) | Same appendix table |
| Compression scope / construction (web §2.1–2.3) | §3 Method |
| Architectural constraints (web §2.4) | §9 Limitations + appendix |
| Storage footprint (web §3) | Appendix, "Storage footprint" |
| VRAM profile (web §4) | Appendix, "VRAM profile" |
| Power draw (web §5) | Appendix, "Power draw" |
| Quality / PPL (web §6) | §5 Results, headline table |
| Throughput (web §7) | §5 Results, headline table |
| Spectral analysis 7.A | Appendix, "Full spectral analysis" |
| Statistical tests 7.B | §7 Statistical Significance |
| Eckart–Young 7.C | §8 Theoretical bound |
| Validation gates (web §8) | Appendix, "Validation-gate summary" |
| Benefits / downsides (web §9) | Appendix, "Benefits and limitations catalogue" |
| What is/is not demonstrated (web §10) | §9 Limitations + appendix gap table |
| Methodological gaps (web §10.3) | Appendix, "Methodological gaps" |
| Reproducibility (web §11) | §11 + appendix Phase status |
| Phase status (web §12) | Appendix, "Phase status" |
| Cache-fit hypothesis | §6 (extends web §7 with a falsifiable cross-GPU prediction table) |

## Build

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

## arXiv prep (from the parent folder)

```bash
make submit-A   # produces ../dist/paper-A.tar.gz with main.bbl and .sty
```
