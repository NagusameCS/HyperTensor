# HyperTensor — arXiv LaTeX Sources

This directory contains the LaTeX sources for arXiv submissions derived from
the seven HTML papers in `docs/papers/`. Reorganization (author judgment):

| arXiv | Title | Source HTML |
|-------|-------|-------------|
| **A** | Calibration-Free Low-Rank Attention Compression for Bandwidth-Bound LLM Decode | `docs/papers/01-attention-compression.html` |
| **B** | Geodesic Projection: A Multi-Slot Compression Pipeline with Adaptive Extensions | `docs/papers/02-geodesic-projection.html` + `docs/papers/06-adaptive-compression.html` |
| **C** | Geodesic Speculative Decoding under Compression | `docs/papers/03-speculative-decoding.html` |
| **D** | Organic Training Theory and Geodesic Trajectory Caching: Theory and Empirical Anchor | `docs/papers/04-organic-training-theory.html` + `docs/papers/05-gtc-ott-runtime.html` |

`docs/papers/00-introduction.html` is a pedagogical web-only article; it is
**not** part of the arXiv set (does not meet the originality / novelty
expectation of arXiv, but remains useful for site visitors).

## Layout

```
arxiv/
  README.md                        ← this file
  Makefile                         ← top-level: `make`, `make A`, `make clean`
  common/
    hypertensor.sty                ← shared preamble
    refs.bib                       ← shared bibliography
  paperA-attention-compression/
    main.tex
  paperB-geodesic-projection/
    main.tex
  paperC-speculative-decoding/
    main.tex
  paperD-ott-gtc/
    main.tex
```

Each `main.tex` includes `\usepackage{../common/hypertensor}` and uses
`\addbibresource{../common/refs.bib}` (biber + biblatex).

## Building

Requires a recent TeX Live (2022+) with `latexmk`, `biber`, and the
`biblatex`, `booktabs`, `microtype`, `hyperref`, `cleveref`, `amsmath`,
`amssymb`, `amsthm`, `mathtools`, `xcolor`, `tcolorbox`, `tikz`,
`pgfplots`, `listings`, `siunitx` packages.

```sh
cd arxiv
make            # builds all four PDFs
make A          # paper A only
make clean      # remove intermediate files
make distclean  # remove PDFs too
```

The compiled PDFs land at:

* `arxiv/paperA-attention-compression/main.pdf`
* `arxiv/paperB-geodesic-projection/main.pdf`
* `arxiv/paperC-speculative-decoding/main.pdf`
* `arxiv/paperD-ott-gtc/main.pdf`

After approval, those PDFs are copied to `docs/papers/pdf/paper-{A,B,C,D}.pdf`
and linked from PDF download buttons in each paper's HTML hero.

## arXiv submission notes

Each `main.tex` is self-contained except for the shared `common/`
files. To submit to arXiv, run `make submit-A` (etc.) which creates a
flattened tarball at `arxiv/dist/paperA.tar.gz` containing `main.tex`,
the inlined `hypertensor.sty`, the `.bbl` file, and any local figures.
arXiv does not run biber/bibtex on submitted sources, so the `.bbl`
must be included.

## Citation policy

References were upgraded from web-style inline mentions to a proper
BibTeX file at `common/refs.bib`. Where the web papers cited only
informal links (GitHub, Wikipedia), we substituted the canonical
peer-reviewed or arXiv preprint citation. New citations were added
where appropriate to situate the work in the literature
(Vaswani 2017, LLaMA 3, GPTQ, AWQ, SliceGPT, ASVD, FWSVD, LoRA,
Roofline, StreamingLLM, Leviathan/Chen speculative, Medusa, EAGLE,
Oja 1982, do Carmo Riemannian geometry, Williams Roofline, etc.).
