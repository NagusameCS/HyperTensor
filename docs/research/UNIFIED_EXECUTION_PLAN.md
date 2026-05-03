# HyperTensor — Unified Execution Plan

*Last updated: 2026-05-01*

Every research item across all three tiers has a script built. This document
lists every remaining action item in execution order, with estimated time and
blockers.

---

## Immediate (This Week)

### 1. Submit Papers A–D to arXiv
**Time**: 2 hours | **Blocker**: None

- [ ] Final read-through of all 4 PDFs
- [ ] Upload to arXiv: cs.LG primary, CC BY license
- [ ] Paper E held until distill data available

### 2. Fix AttnRes Binary Path
**Time**: 2–4 hours | **Blocker**: C debugging

- [ ] `--axex-compress` without `--ott-full` must emit parseable tok/s
- [ ] Or: add `--ott-full --ott-spec-batch 0` to suppress speculative path
- [ ] Re-run `scripts/attnres_sweep.py` → data for Paper C §attnres

### 3. Download Benchmark Datasets
**Time**: 1 hour | **Blocker**: Disk space (~2 GB)

- [ ] MMLU: `data/mmlu/test/*.csv` (57 subjects)
- [ ] GSM8K: `data/gsm8k_test.jsonl`
- [ ] HumanEval: `data/humaneval.jsonl`
- [ ] WikiText-2 test: `data/wikitext2_test.txt` (for multi_dataset_ppl.py)
- [ ] Run smoke test: `python scripts/task_bench.py --benchmark mmlu --max-q 10`

---

## Short-Term (1–2 Weeks)

### 4. EC2 Orchestrated Run
**Time**: 3–4 hours wall, ~$2–4 | **Blocker**: AWS credentials

```powershell
./scripts/ec2_orchestrate.ps1 -InstanceType g6e.xlarge -KeyName hypertensor-key
```

This single command produces:
- **P3 cross-GPU data** → validates cache-fit model on L40S (96MB L2)
- **Distill Phase 2 data** → real numbers for all 4 Paper E predictions
- **Per-matrix bases on Llama-8B** → extends SmolLM2 results to frontier scale

### 5. Task Benchmark Sweep
**Time**: 4–6 hours (mostly compute) | **Blocker**: Dataset download (#3)

```bash
python scripts/task_bench.py --model models/llama3.1-8b-instruct-q4_k_m.gguf \
  --benchmark all --ranks 256,512,768,1024,1536 --n-shot 5 --max-q 200
```

Produces all data for Paper F. Estimated: 6–8 hours on RTX 4070 Laptop.

### 6. Multi-Dataset PPL Sweep
**Time**: 1–2 hours | **Blocker**: Dataset files

```bash
python scripts/multi_dataset_ppl.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --ranks 64,128,256,512,1024 --datasets wikitext2,c4,ptb
```

Establishes whether PPL-vs-rank trade-off is dataset-dependent.

---

## Medium-Term (2–4 Weeks)

### 7. Complete Paper E with Distill Data
**Time**: 1 day | **Blocker**: EC2 run (#4)

- [ ] Replace all "predictions, not measurements" text with real numbers
- [ ] Recompile PDF, verify all claims
- [ ] Submit to arXiv as Part V

### 8. Write Paper F (Task-Level Impact)
**Time**: 1 week | **Blocker**: Task benchmark data (#5)

- [ ] Use outline in `docs/research/paper_f_task_impact.md`
- [ ] Create `ARXIV_SUBMISSIONS/paper-F/` with .tex skeleton
- [ ] Fill in benchmark results
- [ ] Submit to arXiv

### 9. FFN Compression PPL Measurement
**Time**: 2–3 days | **Blocker**: Binary support for FFN cluster compression

- [ ] Implement per-cluster FFN compression in C runtime (or Python → GGUF path)
- [ ] Measure PPL with cluster-compressed FFN + GRC attention
- [ ] Data for Paper G

### 10. KV-Cache Long-Context Test
**Time**: 1 day | **Blocker**: 32K-token test prompt

```bash
python scripts/kv_cache_long_context.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --contexts 2048,4096,8192,16384,32768 --rank 256
```

Verifies whether `--axex-kv` cuts VRAM by predicted ~75% at 32K context.

---

## Longer-Term (1–3 Months)

### 11. Cross-GPU P3 Validation (Full)
**Time**: 1 week | **Blocker**: Access to RTX 4090 + A100 + H100

- [ ] Run `p3_cross_gpu.py` on 3+ GPU types
- [ ] Verify k* shifts with L2 capacity
- [ ] Data for Paper I

### 12. Fix Differentiable Rank Performance
**Time**: 1 day | **Blocker**: Python refactor

- [ ] Add weight caching to `differentiable_rank.py` (load GGUF once)
- [ ] Run full 30-layer optimization
- [ ] Compare learned allocation vs MCR phases

### 13. Write Remaining Papers
**Time**: Variable | **Blocker**: Respective data

| Paper | Topic | Data Ready? | Est. Writing Time |
|-------|-------|:-----------:|-------------------|
| E | Distillation | After EC2 | 1 day |
| F | Task-Level Impact | After benchmarks | 1 week |
| G | FFN Compression | After PPL measurement | 1 week |
| H | GTC as RAG | After deployment | 2 weeks |
| I | Super-Baseline Generalization | After cross-GPU | 2 weeks |
| J | Heterogeneous Drafters | Simulation only | 3 days |
| K | MoE  GRC | Partial | 1 week |
| L | Differentiable Rank | After optimization | 1 week |

---

## Quick-Reference: All Scripts and Their Entry Points

| Script | Run Command |
|--------|------------|
| `p3_cross_gpu.py` | `python scripts/p3_cross_gpu.py --model <gguf> --ranks 512,768,1024,1280,1536,2048` |
| `per_matrix_bases.py` | `python scripts/per_matrix_bases.py --model <gguf> --ranks 128,256,512,768,1024 --all-layers` |
| `distill_runner.py` | `python scripts/distill_runner.py --teacher <hf> --gguf <gguf> --corpus <txt> --rank 1536 --lora-rank 8 --steps 500` |
| `attnres_sweep.py` | `python scripts/attnres_sweep.py --model <gguf> --d-model 576` |
| `task_bench.py` | `python scripts/task_bench.py --model <gguf> --benchmark all --ranks 256,512,1024` |
| `multi_dataset_ppl.py` | `python scripts/multi_dataset_ppl.py --model <gguf> --ranks 64,128,256,512,1024` |
| `ffn_cluster_compress.py` | `python scripts/ffn_cluster_compress.py --model <gguf> --n-clusters 4,8,16` |
| `calibrated_sink.py` | `python scripts/calibrated_sink.py --model <gguf> --rank 256 --sink-T 32` |
| `kv_cache_long_context.py` | `python scripts/kv_cache_long_context.py --model <gguf> --contexts 2048,4096,8192,16384,32768` |
| `shf_loss.py` | `python scripts/shf_loss.py --demo` |
| `quant_co_design.py` | `python scripts/quant_co_design.py --model <gguf> --bits 2,3,4,8,16` |
| `moe_gqa_analysis.py` | `python scripts/moe_gqa_analysis.py --model <gguf>` |
| `gtc_vs_rag.py` | `python scripts/gtc_vs_rag.py` |
| `grc_vision_analysis.py` | `python scripts/grc_vision_analysis.py` |
| `super_baseline_general.py` | `python scripts/super_baseline_general.py --all-gpus` |
| `heterogeneous_drafters.py` | `python scripts/heterogeneous_drafters.py --gamma 4` |
| `differentiable_rank.py` | `python scripts/differentiable_rank.py --model <gguf> --iterations 500` |
| `master_dashboard.py` | `python scripts/master_dashboard.py` |
| EC2 orchestration | `./scripts/ec2_orchestrate.ps1 -InstanceType g6e.xlarge -KeyName hypertensor-key` |

---

## Status Summary

| Category | Ready | Pending | Blocked |
|----------|:-----:|:-------:|:-------:|
| arXiv papers | 4 (A–D) | 1 (E) | EC2 |
| Derivative papers | 0 | 8 (F–L + outlines) | Data collection |
| Scripts with data | 14 | 5 | EC2 / dataset download |
| EC2-dependent | 0 | 3 | AWS credentials |
| Binary fixes needed | 0 | 1 | AttnRes path |

**Total remaining person-weeks**: ~8–12 (mostly data collection + writing)
**Total remaining EC2 cost**: ~$5–10
