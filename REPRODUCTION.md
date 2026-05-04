# HyperTensor --- Complete Reproduction Guide

## May 2, 2026 | All experiments, exact commands, expected outputs

---

## Prerequisites

```bash
# Python 3.10+ with venv
python -m venv .venv
source .venv/bin/activate  # Linux
.\.venv\Scripts\Activate.ps1  # Windows

# Core dependencies
pip install torch transformers peft datasets accelerate safetensors numpy
pip install bitsandbytes  # For QLoRA training
pip install sentencepiece protobuf  # Tokenizers

# Ollama (for model deployment)
# Install from https://ollama.com

# EC2 (optional, for GPU experiments)
# AWS CLI + SSH key at ~/.ssh/hypertensor-key.pem
# Instance: g6e.xlarge (L40S, 46GB VRAM)
```

---

## Paper I: GRC Attention Compression

### Experiment A1: PPL vs k (Safe Frontier)

What it measures: WikiText-2 perplexity at compression ranks k={32..1536} on SmolLM2-135M.

```bash
python scripts/experiment_a1_ppl_vs_k.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --ranks 32,64,128,256,512,576,1024,1536 \
  --out benchmarks/experiment_a1_ppl_vs_k
```

Expected output:
```
k=32:  PPL=inf (token collapse)
k=64:  PPL=1423.5
k=128: PPL=847.2
k=256: PPL=98.1   (+1486% --- CATASTROPHIC)
k=512: PPL=7.1    (+14.4% --- SAFE)
k=576: PPL=6.4    (+3.2%)
k=1024: PPL=6.2   (baseline)
k=1536: PPL=6.2   (baseline)
```
Paper I reference: Section "SmolLM2-135M safety validation (Exp A1)"

### Experiment A2: Multi-k NCU Sweep (Kernel Fusion Evidence)

What it measures: L2 cache hit rates across k to test cache-fit vs fusion hypotheses.

```bash
# Requires NVIDIA Nsight Compute + RTX 4070 Laptop
# Run from geodessical runtime directory
python scripts/paperA_proof/parse_multi_k.py \
  --ncu-dir docs/figures/paper-a/ncu/ \
  --out benchmarks/paper_a_multi_k
```

Expected output:
```
k          attn_L2_hit_rate
baseline   7.537 +/- 0.013
384        11.464 +/- 0.009
512        11.469 +/- 0.033
768        11.463 +/- 0.023
1024       11.462 +/- 0.001
1280       11.475 +/- 0.003
1536       11.462 +/- 0.028
```
Interpretation: Flat L2 hit-rate plateau confirms kernel fusion, not cache residency.

### Experiment A3: Headline Throughput (Super-Baseline)

What it measures: Decode throughput at k=1024 and k=1536 vs baseline on Llama-3.1-8B.

```bash
# Requires: geodessical runtime binary, RTX 4070 Laptop, 8GB VRAM
# Run from geodessical directory
./geodessical --model models/llama3.1-8b-instruct-q4_k_m.gguf \
  --axex-compress --axex-rank 1024 \
  --prompt "Explain how a transformer works" -n 256 --no-verifier
```

Expected output (Paper I Table 1):
```
k=1024: decode 106.27% baseline, PPL +61.4%
k=1536: decode 97.55% baseline, PPL +13.30%
```

---

## Paper IV: OTT/GTC Manifold Runtime

### Experiment GTC-Coverage: Cache Coverage Measurement

What it measures: Fraction of queries served by GTC cache at various cache sizes.

```bash
# Requires SmolLM2-135M model
python scripts/gtc/decode_substitution_dense.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --n-queries 10000 --cache-fractions 0.10,0.25,0.50,0.75
```

Expected output (Paper IV Section II):
```
Cache 10%: coverage 72.3%
Cache 25%: coverage 90.4-91.5%  <- scale-invariant ±0.5% across 33× params
Cache 50%: coverage 99.6%        <- measured on Gemma-4-E2B
```

### Experiment GTC-Lookup: Query Latency

What it measures: Single GTC record lookup time.

```bash
python scripts/gtc/batch_jacobi.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --n-records 100000 --n-queries 10000
```

Expected output:
```
Avg lookup: 30.9 µs/query
Record size: 5.96 KB
Magnus-3 integration: ~6s for 24 records (CPU)
```

### Experiment GTC-Jacobi: Batch Jacobi Correction

What it measures: Reconstruction error scaling with batch size.

```bash
python scripts/gtc/jacobi.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --batch-sizes 1,10,100,1000,10000
```

Expected output:
```
B=1:       baseline
B=10:      97× speedup
B=10,000:  60× speedup
Error:     pinned to float64 roundoff floor
```

---

## Paper V: GRC Light Distillation

### Experiment E1: Per-Matrix SVD vs Shared GRC

What it measures: Frobenius error reduction from using per-matrix (Q,K,V separate) bases vs shared GRC basis.

```bash
python scripts/per_matrix_bases.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --ranks 128,256,512
```

Expected output:
```
k=128: shared err=0.602 -> per-matrix err=0.332 (+44.8%)
k=256: shared err=0.359 -> per-matrix err=0.074 (+79.4%)
k=512: shared err=0.081 -> per-matrix err=0.005 (+94.3%)
```
Llama-8B variant: `--model models/llama3.1-8b-instruct-q4_k_m.gguf`
Expected: +18.9% improvement (GQA reduces shared-basis penalty).

### Experiment E2: Distillation PPL Recovery

What it measures: WikiText-2 PPL after LoRA distillation on GRC-compressed model.

```bash
# EC2 L40S: SmolLM2-135M
python scripts/experiment_e2_distill_ppl.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --k 512 --lora-r 8 --distill-steps 1000 \
  --out benchmarks/experiment_e2_distill_ppl
```

Expected output:
```
k=256:  GRC PPL=98.1  -> Distilled=54.3  (47.7% recovery)
k=512:  GRC PPL=7.1   -> Distilled=6.0   (107.1% OVER-recovery)
k=1024: GRC PPL=6.5   -> Distilled=6.2   (98.3% recovery)
Baseline PPL: 6.2
```

---

## Paper VI: Task-Level Impact

### Experiment F1: MMLU + GSM8K on SmolLM2-135M

What it measures: Benchmark accuracy at GRC compression ranks.

```bash
# EC2 L40S
python scripts/experiment_f1_task_benchmarks.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --benchmarks mmlu,gsm8k \
  --ranks 256,512,full \
  --n-shot 5 \
  --out benchmarks/experiment_f1_task_benchmarks
```

Expected output:
```
k=256:  MMLU=28.1%  GSM8K=0.0%
k=512:  MMLU=29.3%  GSM8K=0.0%
k=full: MMLU=27.8%  GSM8K=0.0%
```
Interpretation: MMLU survives; GSM8K at 0% because SmolLM2-135M too small for math.

### Experiment F2/F3: GSM8K + HumanEval on Llama-8B (PENDING)

```bash
# Needs EC2 L40S or ≥24GB GPU --- QUEUED
python scripts/experiment_f1_task_benchmarks.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --benchmarks gsm8k,humaneval \
  --ranks 256,512,1024,1536,full \
  --n-shot 5 \
  --out benchmarks/experiment_f2_f3_llama8b
```

---

## Paper VII: FFN Cluster Compression

### Experiment G1: Per-Cluster SVD vs Global SVD

What it measures: Local Frobenius reconstruction error improvement from clustering.

```bash
python scripts/experiment_g2_ffn_cluster_ppl.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --clusters 4,8,16 --k-fracs 0.25,0.50,0.75 \
  --measure error-only
```

Expected output (Paper VII Table 1):
```
C=4,  k_frac=0.25: global_err=0.553 -> cluster_err=0.428 (+22.6%)
C=8,  k_frac=0.25: global_err=0.553 -> cluster_err=0.465 (+15.9%)
C=16, k_frac=0.25: global_err=0.553 -> cluster_err=0.489 (+11.4%)
C=4,  k_frac=0.50: near-zero error for both (>99.9% improvement)
```

### Experiment G2: FFN Cluster PPL (NEGATIVE RESULT)

What it measures: End-to-end WikiText-2 PPL with cluster-compressed FFN.

```bash
# EC2 L40S
python scripts/experiment_g2_ffn_cluster_ppl.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --clusters 4 --k-fracs 0.25,0.50,0.75 \
  --measure ppl \
  --out benchmarks/experiment_g2_ffn_cluster_ppl
```

Expected output (Paper VII G2):
```
k_frac=0.25: PPL=98.3  (15.9× baseline) --- CATASTROPHIC
k_frac=0.50: PPL=22.7  (3.7× baseline)   --- FAILED
k_frac=0.75: PPL=8.9   (+43.5%)          --- worse than attention GRC
Baseline PPL: 6.2
```
Key finding: Local reconstruction improvement does NOT guarantee end-to-end PPL.

### Experiment G3: Combined Attn+FFN Byte Savings

What it measures: Total weight bytes saved from combined attention + FFN compression.

```bash
python scripts/experiment_i4_h2_g3.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --k-attn 512 --k-ffn-frac 0.75
```

Expected output:
```
Total bytes: 239 MB -> 177 MB
Savings: 1.35× (corrected from 2.5× prediction)
```

### Experiment P2: Activation-Weighted FFN (NEGATIVE RESULT)

```bash
# Phase 2: Collect weight column norms (instant)
python scripts/p2_ffn_actweighted.py --phase 2

# Phase 3: Apply weighted compression + measure PPL
python scripts/p2_ffn_actweighted.py --phase 3 --k-frac 0.50
```

Expected output (Phase 3):
```
Baseline PPL: 27.24
Compressed PPL: 1230.58 (45.2× baseline --- FAILED)
```
Key finding: Weight column norms are NOT a valid proxy for activation importance.

---

## Paper VIII: GTC as Vector Database

### Experiment H1: GTC vs RAG Throughput (CORRECTED)

What it measures: 1M-query throughput comparison with corrected miss-fallback model.

```bash
python scripts/pivot_p3_gtc_hitrate_correction.py
```

Expected output:
```
GTC Hit Rate | GTC/query | RAG/query | Speedup
0.0%         | 28.000ms  | 50.500ms  | 1.8×
50.0%        | 14.015ms  | 50.500ms  | 3.6×
90.0%        | 2.828ms   | 50.500ms  | 17.9×
91.5%        | 2.408ms   | 50.500ms  | 21.0×  <- Paper IV coverage
99.6%        | 0.143ms   | 50.500ms  | 353.7× <- 50% cache
```

### Experiment H2: GTC 50% Cache Library Builder

What it measures: Storage requirements and pre-computation cost for 200K-record GTC library.

```bash
python scripts/gtc_50pct_cache_scaler.py
```

Expected output:
```
Records at 50% cache: 200,000
Storage: 1,164 MB (1.137 GB)
Fits in: 8GB VRAM 
Speedup: 353.7× over RAG
Pre-compute cost: ~$0.65 on EC2 spot (16-core, 0.9 hrs)
```

### Experiment H3: Build GTC Library (CPU)

```bash
python scripts/build_gtc_library.py \
  --fraction 0.50 --full-library 400000 \
  --out benchmarks/gtc_library_50pct
```

---

## Paper IX: Super-Baseline General

### Experiment I4: KV-Cache Projection

What it measures: VRAM savings from projecting KV cache to k dimensions.

```bash
python scripts/experiment_i4_h2_g3.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --context 32768 --kv-rank 64
```

Expected output:
```
k=64:   15.8× VRAM savings
k=128:  7.9× VRAM savings
k=256:  3.9× VRAM savings
```

### Experiment I5: LoRA FFN Fusion

What it measures: Throughput improvement from LoRA-augmented FFN at k=256.

```bash
python scripts/experiment_i5_lora_fusion.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --lora-rank 256 \
  --out benchmarks/experiment_i5_lora_ffn_fusion
```

Expected output:
```
LoRA FFN fusion: +76.6% throughput at k=256
k* for LoRA FFN: 256 (NOT 1024-1536 as Paper IX originally predicted)
k* is kernel-specific, not L2-cache-specific
```

### Experiment I1-I3,I6: Cross-GPU Validation (PENDING)

```bash
# Requires: A100 (I2), H100 (I3), RTX 4090 (I1), L40S (I6) hardware
python scripts/p3_cross_gpu.py \
  --model models/llama3.1-8b-instruct-q4_k_m.gguf \
  --ranks 512,768,1024,1280,1536,2048 \
  --out benchmarks/p3_cross_gpu
```

---

## Paper X: CECI Chimeric Model Vector Bridging

### Experiment J1: Within-Model CECI (FALSIFIED)

What it measures: Subspace overlap between layers within a single model at k=32.

```bash
python scripts/paper_x_feasibility.py \
  --model models/smollm2-135m-instruct-q8_0.gguf \
  --k 32
```

Expected output:
```
120 layer pairs measured
Mean GD: 0.919
Mean overlap: 15.4%
Viable pairs: 0/120 (0.0%)
CECI within-model: FALSIFIED
```

### Experiment J2: Cross-Model CECI (CONFIRMED)

Prerequisites: Train pure math and language models first.

```bash
# Step 1: Train pure math model (local GPU, ~2.5 hrs)
python scripts/train_pure_lora.py \
  --skill math --steps 10000 --batch-size 2 \
  --output outputs/pure_models/smollm2-135m-math-pure

# Step 2: Train pure language model (local GPU, ~53 min)
python scripts/train_pure_lora.py \
  --skill language --steps 10000 --batch-size 2 \
  --output outputs/pure_models/smollm2-135m-language-pure

# Step 3: CECI splice at k=512
python scripts/ceci_cross_model.py \
  --math outputs/pure_models/smollm2-135m-math-pure/final \
  --language outputs/pure_models/smollm2-135m-language-pure/final \
  --k 512 --out benchmarks/ceci_cross_model_j2_k512

# Step 4: CECI at full rank (trivial check)
python scripts/ceci_cross_model.py \
  --math outputs/pure_models/smollm2-135m-math-pure/final \
  --language outputs/pure_models/smollm2-135m-language-pure/final \
  --k 576 --out benchmarks/ceci_cross_model_j2_k576
```

Expected output (k=512):
```
GD: μ=0.014, σ=0.008
Overlap: μ=99.98%
ρ (LoRA r=8): μ=0.304
Q rel error: μ=33.4%
VIABLE: 13/30 (43.3%)
SHARED SCAFFOLD CONFIRMED: cross-model 67× better than within-model
```

Expected output (k=576, full rank):
```
GD: μ=0.000, Q_err: 0.000
VIABLE: 30/30 (100%) --- trivial (no compression)
```

### Experiment J3: Build Chimeric Model

```bash
# Build HORIMIYA (k=512, 13/30 spliced)
python scripts/create_horimiya_splice.py \
  --math-model outputs/pure_models/smollm2-135m-math-pure/final \
  --language-model outputs/pure_models/smollm2-135m-language-pure/final \
  --k 512 --out outputs/chimeric/HORIMIYA

# Build HORIMIYA-MP (k=576, 30/30 spliced)
python scripts/create_horimiya_splice.py \
  --math-model outputs/pure_models/smollm2-135m-math-pure/final \
  --language-model outputs/pure_models/smollm2-135m-language-pure/final \
  --k 576 --out outputs/chimeric/HORIMIYA-MP

# Benchmark all 4 models
python scripts/benchmark_chimeric.py --model all
```

Expected output:
```
Model         PPL    tok/s
MIYA          33.44  15.6
HORI          25.59  13.1
HORIMIYA      28.66  14.3
HORIMIYA-MP   25.94  17.3
```

### Experiment J4: Qwen2.5-1.5B CECI (RUNNING on EC2)

What it tests: Does shared scaffold hold at 3× larger scale with REAL compression (k/d=0.50)?

```bash
# EC2 L40S (46GB VRAM)
# Step 1: Math LoRA
/home/ubuntu/venv/bin/python3 -u ~/train_qwen_pure.py --skill math --steps 4000

# Step 2: Language LoRA
/home/ubuntu/venv/bin/python3 -u ~/train_qwen_pure.py --skill language --steps 4000

# Step 3: CECI splice
/home/ubuntu/venv/bin/python3 -u ~/ceci_qwen.py
```

Status: Math DONE (43:52). Language running (~86 min ETA). CECI queued.

---

## Ollama Deployment

```bash
# After building models, publish to Ollama:

# MIYA (pure math)
ollama create MIYA -f modelfiles/MIYA.modelfile
ollama cp MIYA Nagusamecs/MIYA
ollama push Nagusamecs/MIYA

# HORI (pure language)
ollama create HORI -f modelfiles/HORI_publish.modelfile
ollama cp HORI Nagusamecs/HORI
ollama push Nagusamecs/HORI

# HORIMIYA (k=512, 13/30)
ollama create HORIMIYA -f modelfiles/HORIMIYA.modelfile
ollama cp HORIMIYA Nagusamecs/HORIMIYA
ollama push Nagusamecs/HORIMIYA

# HORIMIYA-MP (k=576, 30/30)
ollama create HORIMIYA-MP -f modelfiles/HORIMIYA-MP.modelfile
ollama cp HORIMIYA-MP Nagusamecs/HORIMIYA-MP
ollama push Nagusamecs/HORIMIYA-MP

# Pull from anywhere:
ollama pull Nagusamecs/MIYA
ollama pull Nagusamecs/HORI
ollama pull Nagusamecs/HORIMIYA
ollama pull Nagusamecs/HORIMIYA-MP
```

---

## Pivot Experiments (CPU-only)

```bash
# P1: Safe k/d ratio calibration across 10 model families
python scripts/pivot_p1_safe_kd_calibration.py

# P2: FFN activation-weighted analysis (weight norm proxy FAILED)
python scripts/pivot_p2_ffn_activation_weighted.py

# P3: GTC hit rate correction (1121.8× -> 21×/354×)
python scripts/pivot_p3_gtc_hitrate_correction.py
```

---

## Experiment Status Matrix

| ID | Paper | Status | Hardware | Runtime |
|----|-------|--------|----------|---------|
| A1 | I |  DONE | Local CPU | ~10 min |
| A2 | I |  DONE | RTX 4070 Laptop | ~4.5 hrs |
| A3 | I |  DONE | RTX 4070 Laptop | ~30 min |
| E1 | V |  DONE | Local CPU | ~2 min |
| E2 | V |  DONE | EC2 L40S | ~1 hr |
| F1 | VI |  DONE | EC2 L40S | ~2 hrs |
| F2 | VI |  PENDING | EC2 L40S | ~6 hrs |
| F3 | VI |  PENDING | EC2 L40S | ~8 hrs |
| G1 | VII |  DONE | Local CPU | ~5 min |
| G2 | VII |  DONE | EC2 L40S | ~1 hr |
| G3 | VII |  DONE | EC2 L40S | ~30 min |
| H1 | VIII |  DONE | Local CPU | ~1 sec |
| H2 | VIII |  DONE | Local CPU | ~1 sec |
| I4 | IX |  DONE | EC2 L40S | ~30 min |
| I5 | IX |  DONE | EC2 L40S | ~1 hr |
| I1-I3,I6 | IX |  PENDING | A100/H100/RTX4090/L40S | ~30 min each |
| J1 | X |  DONE | Local CPU | ~2 min |
| J2 | X |  DONE | Local CPU | ~15 sec |
| J4 | X |  RUNNING | EC2 L40S | ~2.5 hrs total |
| P1 | --- |  DONE | Local CPU | ~1 sec |
| P2 | VII |  DONE | Local CPU | ~5 min (negative) |
| P3 | VIII |  DONE | Local CPU | ~1 sec |

---

## Quick Start (SmolLM2-135M only, no GPU needed)

```bash
# 1. Clone and setup
git clone https://github.com/NagusameCS/HyperTensor
cd HyperTensor
python -m venv .venv && source .venv/bin/activate
pip install torch transformers peft datasets safetensors numpy

# 2. Download base model
# (automatic on first use via HuggingFace)

# 3. Run the core experiments (CPU-only)
python scripts/pivot_p1_safe_kd_calibration.py       # Safe k/d ratios
python scripts/pivot_p3_gtc_hitrate_correction.py     # GTC speedup
python scripts/per_matrix_bases.py --model HuggingFaceTB/SmolLM2-135M  # Per-matrix SVD
python scripts/ceci_cross_model.py --math outputs/pure_models/smollm2-135m-math-pure/final --language outputs/pure_models/smollm2-135m-language-pure/final --k 512  # CECI

# 4. View results
ls benchmarks/*/
```
