## Paper C --- Decode Throughput Under GRC Compression

Summary: preliminary decode measurements indicate usable throughput with GRC and provide early OTT/AttnRes interaction evidence.

No farm decode log rows yet; showing OTT/AttnRes empirical sections below.


---

# OTT Speculative Decode --- Empirical Speedup Table

Paper C empirical anchor data  
Rows show mean tok/s ± 95% CI and acceptance rate α across 10 locked prompts.
Speedup = mean_tok_s / baseline_tok_s for the same model.

| Model | Mode | thresh | batch | tok/s | ±CI | α (%) | ±CI | geo_frac | Speedup |
|-------|------|--------|-------|-------|-----|-------|-----|----------|---------|
| SmolLM2-135M | baseline | 0.0 | 0 | 71.92 | ±21.96 | --- | ±--- | --- | 1.0 |
| SmolLM2-135M | spec | 0.45 | 4 | 33.66 | ±19.22 | --- | ±--- | --- | 0.468 |
| SmolLM2-135M | spec_grc | 0.45 | 4 | 81.31 | ±12.57 | 46.9 | ±--- | 46.9% | 1.131 |

## Key observations

- Geodesic hit rate (`geo_frac`) shows what fraction of accepted tokens came
  from the Riemannian geodesic draft vs. the transformer verifier correction path.
- Speedup > 1.0 confirms the speculative path outperforms autoregressive decode
  on this hardware. Speedup < 1.0 means the verifier overhead dominates.
- spec_grc rows test whether GRC compression at k=1024 affects α.
  Significant α drop would indicate the compressed attention manifold diverges
  from the uncompressed verifier's predicted distribution.
