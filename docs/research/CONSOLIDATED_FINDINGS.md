# HyperTensor — Consolidated Findings Report
*May 1, 2026*

## Papers A–D: Open Items and Closure Status

### Paper A — GRC Compression

| Open Item | Status | Resolution |
|-----------|--------|------------|
| P1: Multi-k NCU sweep |  CLOSED | n=2 replicates, flat at 11.46%±0.013. CONFIRMED |
| P2: L2 capacity manipulation |  CLOSED | Exp B: DRAM proxy flat at 1.194 across Δ. CONSISTENT-OR-FUSION |
| P3: Cross-GPU validation |  OPEN | Script built (p3_cross_gpu.py). Needs RTX 4090/A100 access |
| Per-matrix bases (EY optimum) |  MEASURED | 44.8–94.3% error reduction vs shared-basis on SmolLM2-135M |
| Sink-channel exemption k-dependence |  MEASURED | +7.6% at k=256 vs 1–3% at high k. Lever is k-dependent |
| FFN compression |  ANALYZED | 21–25% error reduction with 4 L2-clusters at k=0.25n |
| Task-level benchmarks |  OPEN | Harness built (task_bench.py). MMLU+GSM8K downloaded. Binary blocked |

### Paper B — GP Pipeline

| Open Item | Status | Resolution |
|-----------|--------|------------|
| MCR A/B measurement |  CLOSED | Clean: -0.13% vs shared-rank (within noise). Contaminated run identified and retaken |
| GL(d) gauge optimisation |  OPEN | Design spec complete. Awaiting v0.2 implementation |
| Thermal rank controller |  OPEN | Telemetry trace captured. Awaiting closed-loop A/B |
| Rejection-driven Oja |  OPEN | Hooks implemented. Awaiting drift ablation |
| Cross-model load/VRAM table |  MEASURED | 5 models, 457MB–6,444MB VRAM. Table embedded in paper |
| Depth-sink rule (ℓ*≈2L/3) |  MEASURED | 4 models confirmed. Awaiting Llama-70B validation |

### Paper C — Speculative Decoding

| Open Item | Status | Resolution |
|-----------|--------|------------|
| AttnRes  GRC sweep |  BLOCKED | Binary does not support --axex-compress without --ott-full |
| α-vs-k sweep (attres) |  STALLED | Script built. Last run hung at k=256 |
| Accept-collapse at low k |  MEASURED | k=128: all 0%. k=256: one outlier at 56.2%. Sharp collapse confirmed |
| OTT empirical sweep (SmolLM2) |  COMPLETE | 10-prompt, 3-mode. SPEC+GRC: 1.131, α=46.9% |
| Llama-8B α measurement |  MEASURED | α=46.9%, σ=0%. 8-prompt sweep. Geodesic acceptance is geometry-determined |
| Instruct-greedy-EOS fix |  SHIPPED | llm_topk_excluding + min-response guard. Novel contribution |

### Paper D — OTT/GTC Theory

| Open Item | Status | Resolution |
|-----------|--------|------------|
| φ diffeomorphism |  OPEN | Deployment-scoped resolved. Universal closure open |
| v₀ initial velocity |  OPEN | Surrogate deployed. Universal closed-form open |
| Curvature-warp knowledge injection |  NEGATIVE | 0/32 single-model + 0/12 cross-model. Documented as negative result |
| HJB-regularised training |  OPEN | SHF loss module built (shf_loss.py). 11 SNR demo validated. Awaiting training integration |
| Live decode-step substitution |  OPEN | Density-gated. Jacobi quality confirmed. Cloud density is blocker |
| OTT runtime anchor |  MEASURED | geodesic_ready, α=38.5%, 76.5 tok/s, 1.53 |

---

## Papers E–I: New Data and Status

### Paper E — Light Distillation

| Data Point | Value | Source |
|-----------|-------|--------|
| ρ (Llama-8B, k=1024, r=8) | 0.1340 | grc_distill.py --print-rho |
| ρ (Llama-8B, k=1536, r=8) | 0.1355 | grc_distill.py --print-rho |
| ρ (SmolLM2-135M, k=256, r=8) | 0.3443 | grc_distill.py --print-rho |
| Per-matrix error reduction (k=256) | +79.4% | per_matrix_bases.py |
| Sink improvement (k=256) | +7.6% | calibrated_sink.py |
| Distill runner status | Built, import-verified | distill_runner.py |
| EC2 distill run |  Pending | Needs g5.xlarge+ |

### Paper F — Task-Level Impact

| Data Point | Value | Source |
|-----------|-------|--------|
| MMLU dataset | 57 subjects, 14,042 questions | Downloaded to data/mmlu/ |
| GSM8K dataset | 1,319 questions | Downloaded to data/gsm8k_test.jsonl |
| Task harness | Built | task_bench.py |
| Baseline execution |  Binary blocked | --axex-compress doesn't produce output without --ott-full |

### Paper G — FFN Compression

| Data Point | Value | Source |
|-----------|-------|--------|
| 4-cluster improvement (k=0.25n) | +22.6% | ffn_cluster_compress.py |
| 8-cluster improvement (k=0.25n) | +15.9% | ffn_cluster_compress.py |
| Per-layer consistency | 20.9–25.0% across layers | ffn_cluster_compress.py |
| PPL measurement |  Pending | Needs GGUF export of cluster-compressed weights |

### Paper H — GTC as RAG

| Data Point | Value | Source |
|-----------|-------|--------|
| GTC throughput (1M queries) | 2,828s | gtc_vs_rag.py |
| RAG throughput (1M queries) | 43,900s | gtc_vs_rag.py |
| Speedup | 15.5 | gtc_vs_rag.py |
| Per-hit latency (GTC) | 30.9 µs | Paper D |
| Per-hit latency (RAG) | 50.5 ms | Literature estimate |
| Live A/B deployment |  Pending | Needs GTC runtime + RAG pipeline |

### Paper I — Super-Baseline Generalization

| Data Point | Value | Source |
|-----------|-------|--------|
| GPUs analyzed | 10 | super_baseline_general.py |
| Kernel classes identified | 6 | super_baseline_general.py --kernels |
| Confirmed super-baselines | 1 (RTX 4070 Laptop, k*=1024) | Paper A |
| Cross-GPU validation |  Pending | Needs 4090/A100/H100 access |

---

## Binary Limitations (Root Cause of Blocked Scripts)

The geodessical2.exe binary has a flag interaction issue:
- `--axex-compress` requires `--ott-full` to produce parseable output
- Basic decode mode with compression produces no stdout tok/s metrics
- This blocks: attnres_sweep.py, kv_cache_long_context.py, task_bench.py, p3_cross_gpu.py

**Workaround**: Use `--ott-full --no-verifier` for compressed basic decode.
**Need to test**: `--ott-full --no-verifier --axex-compress --axex-compress-rank K`

---

## Remaining Execution Items (Priority Order)

1. **Test binary workaround**: `--ott-full --no-verifier --axex-compress` → unblocks 4 scripts
2. **EC2 orchestrated run**: $2–4 on g6e.xlarge → P3 + distill + per-matrix on Llama-8B
3. **Task benchmarks**: Run task_bench.py on SmolLM2 with workaround → Paper F data
4. **Cross-GPU validation**: p3_cross_gpu.py on 2+ GPU types → Paper I data
5. **Biber passes**: Resolve cross-refs for all 9 papers
