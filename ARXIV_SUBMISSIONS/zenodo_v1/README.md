# HyperTensor Volume — Zenodo v1

**Title:** HyperTensor: A Geometric Framework for Transformer Acceleration, Safety, and Number-Theoretic Application (Volume Edition)

**Author:** William Ken Ohara Stewart  
**ORCID:** [0009-0006-2398-0162](https://orcid.org/0009-0006-2398-0162)  
**Date:** May 9, 2026  
**License:** MIT  
**DOI:** [10.5281/zenodo.20077378](https://doi.org/10.5281/zenodo.20077378)  
**Repository:** https://github.com/NagusameCS/HyperTensor

## Contents

- `volume_extended.tex` — Single-source volume manuscript (18 papers + foundation + appendix)
- `volume_extended.pdf` — Compiled PDF (210 pages)
- `refs.bib` — Bibliography (biblatex)
- `hypertensor.sty` — Custom LaTeX style
- `figures/` — All figures referenced in the volume
- `LICENSE` — MIT license

## Build

```bash
pdflatex volume_extended.tex
biber    volume_extended
pdflatex volume_extended.tex
pdflatex volume_extended.tex
```

Tested with MiKTeX (pdflatex) + biber 2.21.

## Build Status

- 210 pages, ~2.7 MB PDF
- 0 undefined citations
- 0 undefined references
- 0 hard errors
- 71 multiply-defined-label warnings — cosmetic only, inherent to stitching 18 self-contained papers (per-paper `\bibitem` keys and generic `sec:intro`/`sec:method` labels collide); does not affect the rendered PDF. See "Volume Limitations and Open Items" section in the manuscript.

## Citation

```bibtex
@misc{stewart2026hypertensor,
  author       = {Stewart, William Ken Ohara},
  title        = {HyperTensor: A Geometric Framework for Transformer Acceleration, Safety, and Number-Theoretic Application (Volume Edition)},
  year         = {2026},
  month        = {may},
  howpublished = {Zenodo},
  note         = {Volume v1, 18 papers + foundation, 210 pp.}
}
```

## Scope

This is a working preprint: a coherent research notebook with eighteen chapters, released as a single document so a reader can see the full geometric framework at once. It is not eighteen standalone journal submissions. The "Volume Limitations and Open Items" section near the end of the manuscript enumerates the open measurement items and honest-reporting caveats that hold across the volume.
