# Negative Results --- Next Steps

## May 2, 2026 | Honest failures and the path forward

---

## 1. CECI Within-Model Splicing (J1): FALSIFIED

Result: 0/120 layer pairs viable at k=32 (GD=0.919, overlap=15.4%).
Why: Layers within a single model have intrinsically incompatible subspaces.
Next step: Cross-model CECI with shared initialization (J2).  CONFIRMED --- GD=0.014, 13/30 viable at k=512. CECI requires shared birth.

---

## 2. CECI k=256: Only 1/30 Viable

Result: At k=256, ρ=0.177, Q_err=74.6%. Rank too aggressive for d=576.
Why: k/d=0.44 discards 56% of dimensions. Paper I showed k≥512 needed for PPL safety.
Next step: Use k≥512 (Paper I safe frontier). At k=512: 13/30 viable (43.3%).
Further: Per-layer adaptive rank (Paper II MCR) could improve deeper-layer ρ.
Further: LoRA light-distillation (Paper V) to recover the 33.4% Q residual.

---

## 3. CECI Full-Rank 30/30: TRIVIAL

Result: At k=576 (full dimension), GD=0, Q_err=0, 30/30 viable.
Why: No compression. All information preserved. Not a meaningful result.
Next step: This is the upper bound. The challenge is finding the minimum k where viability holds. For SmolLM2-135M: k≈512.
Scaling question: For Llama-8B (d=4096), what k gives >50% viable? Requires EC2.

---

## 4. FFN Cluster Compression (G2): PPL PROXY FAILED

Result: PPL 98.3 at k_frac=0.25 (15.9× baseline), PPL 22.7 at k_frac=0.50 (3.7× baseline).
Why: Local Frobenius reconstruction improvement (+22.6%) does NOT predict end-to-end PPL. FFN layers are key-value memory banks --- compressing columns discards facts, not routing fidelity.
Next step (A): Activation-weighted SVD using actual forward-pass activation statistics (NOT weight column norms --- proven misleading by P2).
Next step (B): LoRA FFN distillation (Paper V protocol applied to FFN). Expected to recover significant PPL gap based on attention distillation success (107% over-recovery).
Next step (C): Hybrid attention-only compression. Accept that FFN is fragile and compress only attention (safe at k≥512). Combined savings: 1.35×, still meaningful.

---

## 5. P2 Weight-Norm Proxy: FAILED

Result: Activation-weighted FFN using L2 column norms gave PPL 1230 (45.2× baseline) --- WORSE than uniform clustering (22.7 PPL, 3.7× baseline).
Why: L2 column norms do NOT correlate with activation frequency. Massive-activation phenomenon operates on runtime hidden states, not static weight columns.
Next step: Real activation collection via forward passes on WikiText-2 (500 samples). Requires GPU. Script: `scripts/p2_ffn_actweighted.py --phase 2` with hook-based collection.
Hardware: Local RTX 4070 sufficient (SmolLM2-135M fits in 8GB). ~30 min runtime.

---

## 6. GTC H1: 1,121.8× RETRACTED

Result: Flawed simulation compared GTC lookup-only against RAG full pipeline without modeling miss fallback.
Why: On a GTC miss, you MUST fall back to full generation (28ms). The original comparison ignored this.
Corrected: 21.0× at 91.5% hit (25% cache), 353.7× at 99.6% hit (50% cache).
Next step: Build 200K-record GTC library (1.14 GB, $0.65 EC2 spot). Script exists: `scripts/build_gtc_library.py`.
Further: Side-by-side GTC vs RAG with real FAISS + LLM on EC2 L40S.

---

## 7. Safe Frontier k≥256 Over-Prediction: CORRECTED

Result: Paper I originally predicted k≥256 safe. Measured: k=256 gives +1486% PPL (98.1 vs 6.2).
Why: Signal preservation (88.9%) ≠ PPL preservation. Small energy losses amplify massively in PPL.
Correction: Safe frontier is k≥512 for SmolLM2-135M (k/d≥0.89). Admitted in Paper I.
Next step: Calibrate safe k/d ratio across model families. P1 pivot provides predictions for 10 model sizes. Needs validation on 2 more models.

---

## 8. Cross-GPU Super-Baseline (I1-I3, I6): PENDING

Result: Only RTX 4070 Laptop measured (1 of 10 GPU types). All other rows are analytic predictions.
Why: No physical access to A100, H100, RTX 4090, L40S hardware.
Next step: Acquire cloud GPU instances (EC2 p4d for A100, Lambda Labs for H100, local or cloud RTX 4090). Each experiment: ~30 min. Total cost: ~$20.
Alternative: Publish as predictions with clear "UNMEASURED" labels. Already done in Paper IX.

---

## 9. Llama-8B Task Benchmarks (F2, F3): PENDING

Result: All Llama-8B numbers in Paper VI are structural predictions, not measurements.
Why: SmolLM2-135M too small --- GSM8K baseline 0% (not informative). Need ≥24GB GPU for Llama-8B.
Next step: EC2 g6e.xlarge (L40S, 46GB). F2 GSM8K: ~6 hrs. F3 HumanEval: ~8 hrs. Total: ~$22.
Scripts ready: `scripts/experiment_f1_task_benchmarks.py --model meta-llama/Meta-Llama-3.1-8B-Instruct`

---

## 10. Papers XI-XV: BLUEPRINTS

Result: Zero measurements. All claims are structural predictions.
Why: Require UGT training infrastructure (Paper XI) which hasn't been built.
Next step: Build UGT training loop with TOP loss. Requires new model training from scratch. Estimated: 1-2 months of engineering. EC2 L40S: ~$200 for training runs.

---

## Summary Matrix

| ID | Result | Severity | Next Step | Cost |
|----|--------|----------|-----------|------|
| J1 | CECI within-model FALSIFIED | Low (expected) | Already pivoted to cross-model (J2) | $0 |
| J2-k256 | Only 1/30 viable | Low | Use k≥512 (already done) | $0 |
| G2 | FFN PPL catastrophic | High | LoRA FFN distillation (EC2) | ~$5 |
| P2 | Weight norms misleading | Medium | Real activation collection (local GPU) | $0 |
| H1 | 1,121.8× RETRACTED | High (fixed) | Already corrected to 21×/354× | $0 |
| I1-I3 | Cross-GPU pending | Medium | Acquire GPU hardware | ~$20 |
| F2/F3 | Llama-8B pending | Medium | EC2 L40S benchmarks | ~$22 |
| XI-XV | Blueprints only | Low (expected) | Build UGT infrastructure | ~$200 |
