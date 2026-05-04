# Paper I: The 106% Anomaly as a General GPU Phenomenon

Status: Framework complete (super_baseline_general.py). Needs cross-GPU measurements.
Target venue: arXiv cs.AR / cs.PF (Architecture / Performance)
Estimated completion: 4--6 weeks (requires multi-GPU access)

## Abstract (draft)

Paper A reported a surprising result: GRC-compressed attention at k=1024 runs
6.27% faster than the uncompressed baseline on an RTX 4070 Laptop. The mechanism
is kernel fusion replacing three independent GEMV launches with a fused trio
operating on k-dimensional intermediates, plus partial L2 cache residency.
This paper generalizes the "super-baseline" phenomenon: for any bandwidth-bound
GPU kernel with a dimensionality-reduction step, there exists a critical rank k*
below which the compressed path is faster than the uncompressed path. We derive
the condition T(k)/T(∞) > 1 in terms of byte ratio, bandwidth regime, and FLOP
overhead, predict k* for 10 GPU types, and identify 6 kernel classes beyond
attention where the effect should appear --- including LoRA-augmented FFN,
KV-cache projection, MoE routing, and speculative draft verification.

## Key Claims

1. Super-baseline is a universal phenomenon: Any bandwidth-bound kernel with
   dimensionality reduction has a k* where compressed > uncompressed. The effect
   requires (a) L2-resident working set, (b) structural kernel fusion, and
   (c) byte savings exceeding FLOP overhead.

2. **k* scales with L2 capacity**: RTX 4070 Laptop (32 MB) -> k=1024,
   RTX 4090 (72 MB) -> k=1536, A100 (40 MB) -> k=1024, H100 (50 MB) -> k=1280.

3. Six kernel classes identified: Attention QKV projection (CONFIRMED),
   LoRA-augmented FFN (PREDICTED), KV-cache projection (PREDICTED), MoE routing
   (PREDICTED), embedding table lookup (UNLIKELY), speculative draft verification
   (CONFIRMED via Paper C).

4. The effect is NOT L2-cache residency: Paper A Exp B ruled out L2 as the
   causal driver. The mechanism is structural kernel fusion --- fewer independent
   GEMV launches reduce scheduler pressure and DRAM round-trips.

5. Beyond ML: The same analysis applies to any bandwidth-bound kernel with a
   dimensionality-reduction step --- database compression, signal processing, etc.

## Data Already Collected

- RTX 4070 Laptop: k*=1024, 1.0627 speedup (Paper A, CONFIRMED)
- 10-GPU prediction table in `benchmarks/super_baseline/super_baseline_analysis.json`
- 6 kernel class analysis in `scripts/super_baseline_general.py --kernels`

## Cross-GPU Validation Needed

| GPU | L2 | Predicted k* | Status |
|-----|----|-------------|--------|
| RTX 4070 Laptop | 32 MB | 1024 | CONFIRMED |
| RTX 4090 | 72 MB | 1536 | NEEDS MEASUREMENT |
| A100 | 40 MB | 1024 | NEEDS MEASUREMENT |
| H100 | 50 MB | 1280 | NEEDS MEASUREMENT |
| L40S | 96 MB | 1536 | NEEDS MEASUREMENT |

## Next Steps

1. Run p3_cross_gpu.py on 2--3 additional GPU types (EC2)
2. Verify k* shifts with L2 as predicted
3. Measure the LoRA and KV-cache kernel classes
4. Generalize the byte-ratio formula to arbitrary kernel shapes
5. Write-up: ~12 pages, 7 figures, 5 tables

## Key Figure Ideas

1. Fig 1: T(k)/T(∞) vs k for 5 GPU types, overlaid
2. Fig 2: Phase diagram --- (L2 capacity, BW) -> k* heatmap
3. Fig 3: Kernel fusion breakdown --- DRAM bytes per GEMV launch
4. Fig 4: Cache-fit model vs measurement --- parity plot
5. Fig 5: Cross-GPU k* vs L2 --- does the predicted scaling hold?
6. Fig 6: Generalization to non-ML kernels --- byte-ratio formula
7. Fig 7: 6 kernel class applicability matrix
