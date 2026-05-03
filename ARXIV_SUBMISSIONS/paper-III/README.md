# Paper C — Geodesic Speculative Decoding under Compression

**Source:** web Paper 3. Composes Paper A (compression) with speculative
decoding. Headline: $76.5$ tok/s with $\alpha = 38.5\%$ acceptance and
$1.53\times$ speedup on SmolLM2-135M-Instruct Q8\_0 (ChatML), within the
closed-form throughput-model prediction.

## Coverage map

| Web-paper item | Section in `geodesic-speculative-decoding.tex` |
|---|---|
| Closed-form throughput formula with $\Delta\alpha$ penalty | §3 |
| Three-$\alpha$ prediction table at $\gamma=4$ | §3 |
| AttnRes interaction analysis (helps via magnitude, hurts via subspace collapse) | §4 |
| KV-cache compression composition | §5 |
| First end-to-end measurement (76.5 tok/s, $\alpha=0.385$, 1.53$\times$) | §6 |
| Instruct-greedy-EOS pathology and `llm_topk_excluding` fix | §6 |
| `SPEC_MIN_RESP_N=4` guard | §6 |
| OneDecode / OTT-OD / OTT-SWARM drafter modes | §8 |
| Related work | §9 |
| Status (`--ott-swarm-k 8` crashes, `--ott-perfect` hangs) | §10 |

## Build

```bash
latexmk -pdf -interaction=nonstopmode geodesic-speculative-decoding.tex
```
