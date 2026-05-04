# Paper G: Structure-Aware FFN Compression

Status: Analysis complete (ffn_cluster_compress.py). Needs PPL measurement.
Target venue: arXiv cs.LG
Estimated completion: 2--3 weeks

## Abstract (draft)

Paper A compresses only attention (Q/K/V), leaving FFN at full rank because
global SVD of FFN weights is unacceptably lossy --- FFN spectra are nearly flat,
with k95/d ≈ 0.85 vs 0.41 for attention. This paper demonstrates that
structure-aware compression --- clustering FFN columns by activation pattern and
applying per-cluster SVD --- recovers 21--25% of the global-SVD error at the same
total rank budget on SmolLM2-135M (4 clusters, k=0.25n). We extend the analysis
to Llama-3.1-8B, measure end-to-end PPL, and establish the first viable
compression path for FFN weights, potentially doubling the total bytes saved
by GRC (FFN is ~65% of weight bytes).

## Key Claims

1. Column-cluster compression beats global SVD by 21--25% at the same total
   rank budget. The gain comes from preserving local structure: FFN columns that
   co-activate share a low-rank subspace, while columns that activate independently
   need separate subspaces.

2. L2-magnitude clustering is a strong baseline --- it captures the
   massive-activation phenomenon: a few columns have outlier L2 norms and need
   their own high-rank subspace.

3. Activation-guided clustering (Phase 2) should beat L2 clustering ---
   columns that fire together compress together. A few hundred forward passes
   on a calibration corpus enable co-activation clustering.

4. At 4 clusters and k=0.50n, per-cluster SVD achieves near-zero error
   on SmolLM2-135M --- the cluster structure fully captures FFN geometry at
   moderate compression ratios.

5. FFN compression + GRC attention compression = 2--3 total byte reduction
   vs uncompressed, potentially unlocking larger models on consumer GPUs.

## Data Already Collected

- SmolLM2-135M: 3 layers  3 cluster counts  3 k-fractions = 27 measurements
- Key result: 20.9--25.0% error reduction with 4 clusters at k=0.25n
- Full results in `benchmarks/ffn_cluster/ffn_cluster_summary.json`

## Next Steps

1. Extend to Llama-3.1-8B (FFN intermediate dim 14336 vs 1536)
2. Implement activation-guided clustering (requires forward passes)
3. Measure end-to-end PPL with cluster-compressed FFN + GRC attention
4. Compare against block-diagonal factorization baselines
5. Write-up: ~10 pages, 5 figures, 4 tables

## Key Figure Ideas

1. Fig 1: Global SVD vs cluster SVD error, SmolLM2 + Llama-8B side-by-side
2. Fig 2: Per-cluster singular value spectra --- showing that clusters have
   more concentrated spectra than the full FFN
3. Fig 3: PPL vs total compression ratio (attention k + FFN cluster config)
4. Fig 4: Activation co-occurrence matrix -> cluster assignments
5. Fig 5: Bytes saved breakdown: attention vs FFN vs total
