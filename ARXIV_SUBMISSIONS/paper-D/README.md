# Paper D — Organic Training Theory and Geodesic Trajectory Caching

**Source:** web Paper 4 (Organic Training Theory + GTC theory) merged with
web Paper 5 (GTC/OTT runtime empirical anchor). Two-part structure: Part I
formalises OTT/GTC; Part II reports measurements on three open-weight models.

## Why merge

Web Paper 5 was, by its own self-description, "the document Paper 4
promised but never wrote": every result in Paper 5 is anchored by a
specific claim in Paper 4. Splitting them across two arXiv submissions
would force the reader to chase definitions across papers for every
empirical number. The merged paper preserves both viewpoints: the theory
chapter (Part I) is honest about what is universal vs deployment-scoped,
and the measurement chapter (Part II) reports 12 of 17 testable claims as
measured-pass with a complete gap analysis for the remainder.

## Coverage map

| Web-paper item | Section in `ott-gtc-manifold-runtime.tex` |
|---|---|
| OTT manifold view, Fisher metric, geodesic equation | §2 |
| Fisher-metric / heat-kernel bridge (Varadhan) | §2.1 |
| GTC record format and resonance | §3 |
| Connection to AttnRes / block-summary geodesic selection | §4 |
| Formal addendum (assumption-explicit theorem templates) | §5 |
| Five open problems (universal vs deployment-scoped) | §6 |
| Manifold-fit setup from Phase-1 exports | §7 |
| Coverage table (90.4–91.5%, scale-invariant within $\pm 0.5\%$) | §8 |
| Batch Jacobi resonance (97$\times$ at $B{=}10$) | §9 |
| Compressed record store (5.96 KB/record, 30.9 µs/q) | §10 |
| Density caveat for live decode-step substitution | §10 |
| OTT runtime anchor ($\alpha=0.385$, 76.5 tok/s) | §11 |
| Instruct-greedy-EOS pathology + fix | §11.1 |
| Geometry-cache consistency-equivalence rule | §11.2 |
| Gap analysis vs OTT 17-claim list (12 measured, 1 negative, 2 partial, 2 deployment-resolved) | §12 |
| What is genuinely new here (3 small primitives) | §13 |
| Reproduction | §14 |

## Build

```bash
latexmk -pdf -interaction=nonstopmode ott-gtc-manifold-runtime.tex
```
