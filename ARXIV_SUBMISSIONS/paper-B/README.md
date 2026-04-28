# Paper B — Geodesic Projection + Adaptive Extensions

**Source:** web Paper 2 (Geodesic Projection: production pipeline) merged
with web Paper 6 (Adaptive Compression). Two-part structure: Part I covers
the static GP pipeline; Part II covers the four adaptive layers (MCR Phase-
aware, Axiom Gauge, Thermal Rank, Online Basis).

## Why merge

- Web Paper 2 establishes the seven-slot pipeline and cross-architecture
  manifold evidence (intrinsic dim 11/17/25 across SmolLM2-135M /
  Gemma-4-E2B / Phi-3.5-mini, parameter range $33\times$).
- Web Paper 6 layers on dynamic mechanisms that all sit on top of the same
  per-layer projection $P_\ell$ that Paper 2 builds. Splitting them across
  two arXiv submissions would force every reader of Paper 6 to chase Paper 2
  for definitions of $P_\ell$, MCR, the geometry cache, and the depth-sink.
  Merged, both fit cleanly in one paper at $\sim 25$ pages.

## Coverage map

| Web-paper item | Section in `main.tex` |
|---|---|
| Cross-architecture intrinsic dim (web Paper 2) | §2 Cross-architecture manifold evidence |
| Seven-slot SVD spectra | §3 |
| MCR per-layer rank | §4 |
| Persistent geometry cache + depth-sink | §5 |
| Static-GP empirical anchor | §6 |
| Invariance-failure table (web Paper 6 §1) | §7 MCR phase-aware allocation |
| MCR Mix/Compress/Refine + sink bypass | §7 |
| Axiom Gauge: GL(d) closed form, bakeable inference | §8 |
| Thermal Rank: NVML linear-interp, tokens-per-joule | §9 |
| Online Basis: rejection-driven Oja, version counter | §10 |
| How the four mechanisms compose | §11 |
| Limitations (all four design-validated, not measured) | §12 |
| Reproduction commands | §13 |

## Build

```bash
latexmk -pdf -bibtex -interaction=nonstopmode main.tex
```
