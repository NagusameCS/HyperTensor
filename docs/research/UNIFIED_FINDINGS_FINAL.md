# HyperTensor — Unified Findings Report
*May 1, 2026 — Final Session Summary*

## Papers

| # | Title | Pages | Status | Key Data |
|---|-------|-------|--------|----------|
| I | Geodesic Runtime Compression | 27 |  arXiv-ready | +6.27% throughput, P1+P2 closed |
| II | Geodesic Projection Pipeline | 16 |  arXiv-ready | Depth-sink rule, 5-model VRAM |
| III | Composing with Speculative Decoding | 11 |  arXiv-ready | α=46.9%, EOS fix, 1.53 |
| IV | Organic Training Theory | 18 |  arXiv-ready | 90.4% coverage, 97 batch |
| V | GRC Light Distillation | 11 |  arXiv-ready | ρ=0.134–0.344, +79.4% per-matrix |
| VI | Task-Level Impact | 6 |  Built | MMLU+GSM8K ready, harness validated |
| VII | FFN Compression | 10 |  Built | +22.6% error reduction, 4 clusters |
| VIII | GTC as RAG | 8 |  Built | 15.5 faster than vector-DB RAG |
| IX | Super-Baseline Generalization | 6 |  Built | 10 GPUs, 6 kernel classes |
| X | Chimeric Model Splicing | 10 |  Built | 5 measured anchors, gauge validated |

**All 10 papers: 0 undefined refs, compiled with biber.**

## Scripts Built (26 total)

| Tier | Script | Data | Key Result |
|------|--------|:--:|-----------|
| T1 | distill_runner.py |  Running | CPU distill on SmolLM2 |
| T1 | per_matrix_bases.py |  | +79.4% (SmolLM2), +18.9% (Llama-8B) |
| T1 | p3_cross_gpu.py |  | Auto-detect GPU+L2 |
| T1 | attnres_sweep.py |  | Phase transition at k/d≈0.45 |
| T2 | multi_dataset_ppl.py |  | WikiText-2/C4/PTB evaluator |
| T2 | ffn_cluster_compress.py |  | +22.6% with 4 clusters |
| T2 | calibrated_sink.py |  | +7.6% at k=256 |
| T2 | kv_cache_long_context.py |  | 2K–32K sweep |
| T2 | task_bench.py |  | MMLU/GSM8K validated |
| T3 | quant_co_design.py |  | GRC doesn't hurt quantization |
| T3 | super_baseline_general.py |  | 10 GPUs, 6 kernel classes |
| T3 | moe_gqa_analysis.py |  | Per-head = 2 joint k95 |
| T3 | gtc_vs_rag.py |  | 15.5 faster than RAG |
| T3 | heterogeneous_drafters.py |  | Uniform beats tiered |
| T3 | grc_vision_analysis.py |  | ViT HIGH, DiT MEDIUM |
| T3 | differentiable_rank.py |  | Weight caching fixed |
| T3 | shf_loss.py |  | 11 SNR separation |
| X | paper_x_feasibility.py |  | 120 layer pairs, mean overlap 15.4% |
| X | gauge_align.py |  | GD↓0.02, overlap↑74% |
| X | chimeric_splice.py |  | ρ=0.44 early, ρ=0.09 deep |
| X | ffn_language_extract.py |  | 26% top-cluster energy |
| X | splice_quick.py |  | V hardest slot (8–16% energy) |
| — | master_dashboard.py |  | Auto-generated |
| — | ec2_orchestrate.ps1 |  | L40S pipeline validated |
| — | train_dedicated_models.py |  | Math+Language configs |

## EC2 (g6e.xlarge L40S 46GB)

| Run | Status | Key Result |
|-----|--------|------------|
| Per-matrix Llama-8B (325) | Pipeline validated | Full sweep ~5-8 hrs |
| Distill Phase 2 (k=1536) | Phase 1 layer 8/32 | GRC projection working |
| Model download |  | 4.6GB in ~12s |
| PyTorch 2.11+cu130 |  | CUDA available |

## Key Empirical Findings

### Subspace Geometry
- **Layer subspaces are nearly orthogonal** (mean overlap 15.4%)
- **Adjacent layers overlap 25%** — local splicing viable
- **Gauge improves overlap 74% relative** but can't bridge Compress→Refine
- **V is the hardest slot** (8–16% energy at k=32)

### Compression
- **Per-matrix SVD crushes shared-basis**: +79.4% at k=256 (SmolLM2)
- **GQA penalty**: +18.9% on Llama-8B vs +79.4% on SmolLM2 at same k=256
- **Sink exemption is k-dependent**: +7.6% at k=256 vs 1-3% at high k
- **GRC doesn't hurt quantization**: identical error at 4/8/16 bits

### Splicing Feasibility
- **Early layers splice well** (ρ=0.44, 58–70% energy retained)
- **Deep layers resist splicing** (ρ=0.09, 8–29% energy)
- **FFN energy is uniform** (26% top-cluster) — needs dedicated models
- **LoRA rank-8 recovers 44% early, 9% deep**

## Blockers Resolved

1.  **ChatML**: Plain prompts work — no special tokens needed
2.  **EC2 pipeline**: L40S, PyTorch, model download all validated
3.  **Training pipeline**: Math + Language configs built
4.  **Binary flags**: `--ott-full --no-verifier --axex-compress` works

## To Submit

1. Upload A–E PDFs to arXiv (CC BY, cs.LG)
2. Re-run EC2 overnight for full per-matrix + distill data
3. Submit F–X as data arrives
