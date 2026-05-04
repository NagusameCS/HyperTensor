# Paper H: GTC as a RAG Replacement

Status: Analysis complete (gtc_vs_rag.py). Needs live deployment measurement.
Target venue: arXiv cs.IR / cs.LG
Estimated completion: 3--4 weeks (requires GTC runtime deployment)

## Abstract (draft)

Retrieval-Augmented Generation (RAG) improves LLM outputs by retrieving relevant
text chunks from a vector database and feeding them into the model's context window.
Paper D introduced Geodesic Trajectory Caching (GTC): a compressed record store
(5.96 KB/record, 30.9 µs lookup) that caches completed token-prediction trajectories
along with their Magnus-3 Jacobi propagators. This paper argues that GTC IS a
vector database specialized for token prediction --- and that it outperforms
general-purpose RAG by 15.5 in throughput while providing exact (not approximate)
predictions within the validity radius. We propose a hybrid GTC+RAG architecture
where ANN retrieval narrows the candidate set and geodesic refinement delivers
the final prediction, combining RAG's scalability with GTC's precision.

## Key Claims

1. GTC outperforms vector-DB RAG by 15.5 for 1M queries (2.8s vs 44s per
   query-equivalent). The win comes from avoiding full attention on cache hits.

2. GTC predictions are exact to first order within validity radius ρ̂
   (Jacobi error < 0.1%), while RAG introduces LLM hallucination risk on
   retrieved context.

3. GTC coverage is 90.4--91.5% at 25% cache fraction, scale-invariant
   across a 33 parameter range. RAG recall degrades with corpus size.

4. Hybrid GTC+RAG dominates both: two-stage ANN -> geodesic refinement
   gives RAG's scalability plus GTC's precision. On GTC hit (90%+), zero
   attention needed; on miss, fall back to RAG.

5. GTC records are domain-specific prediction caches, not general text
   chunks. A GTC library trained on code is a code-prediction accelerator;
   trained on medical text, a medical-prediction accelerator.

## Data Already Collected

- GTC vs RAG head-to-head: 7-metric comparison in `benchmarks/gtc_vs_rag/`
- GTC record store: 5.96 KB/record, 30.9 µs lookup (Paper D §store)
- GTC coverage: 90.4--91.5% at 25% cache (Paper D §coverage)
- Batch Jacobi resonance: 97 speedup at B=10 (Paper D §resonance)

## Next Steps

1. Deploy GTC runtime with a real RAG pipeline (e.g., LlamaIndex + ChromaDB)
2. Measure end-to-end: query -> retrieve -> predict latency for both systems
3. Measure prediction quality: exact-match rate for GTC vs RAG on standard QA
4. Measure domain transfer: train GTC on Wikipedia, test on StackOverflow
5. Write-up: ~10 pages, 6 figures, 4 tables

## Key Figure Ideas

1. Fig 1: Architecture comparison --- RAG pipeline vs GTC pipeline side-by-side
2. Fig 2: Latency CDF --- 10K queries, GTC vs RAG vs Hybrid
3. Fig 3: Coverage vs cache size --- GTC scale-invariance vs RAG degradation
4. Fig 4: Prediction quality --- exact-match rate by domain
5. Fig 5: Storage efficiency --- records per GB for GTC vs text chunks
6. Fig 6: Domain transfer heatmap --- train domain  test domain
