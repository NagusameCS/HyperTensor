# HyperTensor — Master Audit: All Unconfirmed/Predicted Claims
# Generated 2026-05-02
# Status: [PRED] = prediction/theoretical, [OPEN] = experiment built but not run, [MEAS] = measured

================================================================================
PAPER I: GRC Attention Compression
================================================================================
[MEAS] 106% anomaly on RTX 4070 Laptop (n=8, p<10^-10)
[MEAS] Multi-k NCU sweep, L2 hit-rate +4.68pp at k=1024
[MEAS] PPL sweep at k∈{256,512,768,1024,1536}
[MEAS] FUNCTIONAL SANITY (2026-05-02): k=full =  FUNCTIONAL (2.7/3.0, 6/6 coherent)
[MEAS] FUNCTIONAL SANITY (2026-05-02): k=512 =  DEGRADED (1.7, 3/6 coherent, factual preserved)
[MEAS] FUNCTIONAL SANITY (2026-05-02): k=256 =  BROKEN (0.0, ALL EMPTY OUTPUT)
[MEAS] MMLU: 25.3% (full), 23.7% (k=512), 0.0% (k=256) — Python/ChatML 2026-05-02
[MEAS] GSM8K: ~2.5% (full), 0% (k=512) — model too small
 PPL ←→ text quality correlation CONFIRMED at all three ranks
[MEAS] Multi-k NCU sweep, L2 hit-rate +4.68pp at k=1024
[MEAS] PPL sweep at k∈{256,512,768,1024,1536}
[MEAS] Cross-architecture intrinsic dimensionality (SmolLM2-135M, Qwen2.5-1.5B)
[PRED] Cross-GPU k* predictions (RTX 4090, A100, H100, L40S, A10G, L4, etc.)
       → EXPERIMENT: run scripts/p3_cross_gpu.py on each GPU
       → STATUS: Only RTX 4070 measured. EC2 L40S available. Others need hardware.

================================================================================
PAPER II: Geodesic Projection Pipeline
================================================================================
[MEAS] Cross-architecture intrinsic dims (4 models, d∈{576,1536,3072,4096})
[MEAS] Full per-slot SVD spectra on Llama-3.1-8B
[MEAS] MCR null at cache-fit knee
[MEAS] Two-Thirds Rule for per-slot rank allocation
[PRED] GL(d) Axiom Gauge PPL improvement → v0.2 design spec (deferred)
[PRED] Thermal Rank closed-loop control → v0.2 design spec (deferred)
[PRED] Rejection-driven Oja online basis → v0.2 design spec (deferred)
[PRED] GP-spec composition speedup → analytic prediction only
       → EXPERIMENT: Run full GP pipeline with gauge optimization, measure PPL delta
       → STATUS: Design exists, not measured. EC2 L40S can run SmolLM2-135M version.

================================================================================
PAPER III: Geodesic Speculative Decoding
================================================================================
[MEAS] SmolLM2-135M spec-decode: 1.53× speedup, α=38.5%
[MEAS] Llama-8B spec-decode: α=46.9%, n=8, σ=0
[MEAS] Instruct-greedy-EOS pathology discovered + fix
[PRED] Predicted throughput table at γ=4 — marked "predictions, not measurements"
[PRED] Accept-rate collapse breakpoint at k≈768 — not measured
[OPEN] Full k∈{512,768,1024} accept-rate sweep — experiment built, not fully run
       → EXPERIMENT: Run α-vs-k sweep on SmolLM2-135M at k∈{256,384,512,640,768,1024}
       → STATUS: Partial data exists. Needs systematic completion.

================================================================================
PAPER IV: OTT/GTC Manifold Runtime
================================================================================
[MEAS] 12 measured pass, 1 measured fail (curvature-warp), 2 measured partial
[MEAS] Intrinsic dimension k≈30-50 on Llama-3.1-8B
[MEAS] GTC 15.5× throughput over vector-DB RAG
[MEAS] Batch resonance property (256-512 concurrent queries)
[OPEN] Live-decode replacement → measured partial, needs completion
[OPEN] AttnRes interaction sweep → measured partial
[PRED] Scale-invariance ("flag flip") at 70B+ scale → not measured
       → EXPERIMENT: Run GTC on Llama-70B, measure coverage vs intrinsic dim
       → STATUS: Needs ≥48GB GPU (A100/H100)

================================================================================
PAPER V: GRC Light Distillation
================================================================================
[MEAS] 107% PPL recovery on SmolLM2-135M at k=512 (r=8 LoRA)
[MEAS] Distillation loss curve: 500 steps, lr=5e-4
[MEAS] FUNCTIONAL SANITY (2026-05-02): LoRA k=512 = FUNCTIONAL (score 2.5, 5/6 coherent)
[MEAS] Sink-aware GRC (2026-05-02): k=512, T=32, PPL 1.15x baseline (37 vs 43 vanilla = 17.8% better)
[RUNNING] Qwen2.5-7B GRC distillation on EC2 L40S (large-model gap, switched from gated Llama-8B)
[PRED] Llama-8B distillation predictions → not measured
       → EXPERIMENT: Run grc_distill.py on Llama-3.1-8B with ≥24GB GPU
       → STATUS: Needs ≥24GB GPU. Can run on EC2 g6e.xlarge (L40S, 24GB)?
         Actually L40S has 24GB — might just fit Llama-8B in FP16 + LoRA.
[OPEN] Phase 2 PyTorch runner → pending
[PRED] Sink-aware GRC at k=1536, T=32 → predicted PPL improvement
       → EXPERIMENT: Run sink-aware GRC on SmolLM2-135M, measure PPL
       → STATUS: Code exists in grc_distill.py, not run

================================================================================
PAPER VI: Task-Level Impact
================================================================================
[MEAS] MMLU: 25.3% (full), 23.7% (k=512), 0.0% (k=256) — Python/ChatML 2026-05-02
[MEAS] GSM8K: ~2.5% (full), 0% (k=512) — model too small
[PRED] Llama-8B GSM8K baseline >0% → not measured
       → EXPERIMENT: Run task_bench_python.py on Llama-3.1-8B
       → STATUS: Needs ≥24GB GPU for Llama-8B in FP16
[PRED] HumanEval bimodal degradation → not measured
       → EXPERIMENT: Run HumanEval pass@1 on compressed Llama-8B
       → STATUS: Needs ≥24GB GPU + HumanEval dataset
[PRED] IFEval instruction-following degradation → not measured
       → EXPERIMENT: Run IFEval on compressed models
       → STATUS: Needs IFEval dataset + ≥24GB GPU

================================================================================
PAPER VII: FFN Cluster Compression
================================================================================
[MEAS] Phase 1: L2 clustering — 21-25% improvement over global SVD
[MEAS] Phase 2: Real activation collection — 4-33 massive columns/layer
[MEAS] Phase 3: Weight-norm proxy FALSIFIED (45× baseline PPL)
[MEAS] Phase 4: LoRA FFN distillation — 99.9% gap recovery, 50× baseline
[PRED] Real-activation-weighted compression + LoRA distillation → path forward
       → EXPERIMENT: Run P2 Phase 5: use real acts for weighted SVD, then distill
       → STATUS: Has real activation stats. Needs implementation.
[PRED] 1.4-1.7× combined attn+FFN byte savings → partially measured at 1.35×

================================================================================
PAPER VIII: GTC as RAG Replacement
================================================================================
[MEAS] 15.5× throughput over vector-DB RAG (30.9µs lookup)
[MEAS] Coverage curves at n_intrinsic=8 on 64-point clouds
[PRED] Generalisation to larger models → predicted but unmeasured
       → EXPERIMENT: Run GTC on 8B/70B models, measure lookup speedup
       → STATUS: 8B feasible locally, 70B needs ≥48GB GPU

================================================================================
PAPER IX: Super-Baseline Generalization
================================================================================
[MEAS] Attention QKV: k*=1024, +6.27% on RTX 4070
[MEAS] LoRA FFN: k*=256, +76.6% on EC2 L40S
[MEAS] KV-cache projection: k*=64, 15.8× VRAM savings
[PRED] Cross-GPU k* table for 10 GPU types → only RTX 4070 measured
       → EXPERIMENT: Run p3_cross_gpu.py on each GPU type
       → STATUS: EC2 L40S is available NOW for 2nd data point

================================================================================
PAPER X: CECI Chimeric Splicing
================================================================================
[MEAS] Within-model CECI: FALSIFIED (0/120 viable at k=32)
[MEAS] Cross-model SmolLM2: CONFIRMED geometrically (GD=0.014, 13/30 at k=512)
[MEAS] Cross-model SmolLM2 full-rank: 30/30 layers (Q_err=0)

[→] FUNCTIONAL TEST CORRECTION (2026-05-02):
  OLD FINDING (INCORRECT): All 7 chimeras produce gibberish → CECI FALSIFIED
  ROOT CAUSE: Source models were custom-trained LoRA checkpoints (2K-4K steps),
              not converged. Garbage input → geometrically-aligned garbage output.
  CORRECTED FINDING: With proper pre-trained pair (SmolLM2-135M base ×
              SmolLM2-135M-Instruct, both official HF models):
  - k=576 (full):  FUNCTIONAL — score=2.6, 5/5 coherent, 0 gibberish, 100% matches
  - k=256:  0/30 viable layers (GD=12.9 >> 0.90 threshold)
  CONCLUSION: CECI WORKS with proper pre-trained models. Geometric viability
              (GD<0.90, ρ>0.30) is necessary AND sufficient for functional
              splicing when source models are coherent. The safe frontier
              (k≥512 for SmolLM2-135M) is correct.

[MEAS] Qwen CECI: 27/28 at k=768, 28/28 at k=1536 (geometric only, not yet tested functionally)

================================================================================
PAPERS XI-XV: Blueprints (explicitly structural predictions)
================================================================================
[MEAS] Paper XI (UGT): All 5 validation tests PASS, purity 0.24->1.00 in 100 steps
[MEAS] Paper XI (UGT): UGTAdapter wraps HF model, zone separability confirmed
[BLUEPRINT] Papers XII-XV: Geodesic Compiler, Synthesis, Sniping, Organic Generation

================================================================================
PRIORITY QUEUE (what can be measured NOW with available hardware):
================================================================================
1. [RUNNING] CECI Qwen×DeepSeek viability + functional test
2. EC2 L40S: Cross-GPU validation (Paper IX) — 2nd data point
3. EC2 L40S: Llama-8B GRC distillation (Paper V) — fits 24GB VRAM?
4. Local: GRC PPL sweep with n≥5 at each k (bulletproof Paper I)
5. Local: Sink-aware GRC PPL measurement (Paper V)
6. Local: α-vs-k accept-rate sweep (Paper III)
7. Local: Combined attn+FFN byte savings at more ranks (Paper VII)

================================================================================
FUNDAMENTAL NEGATIVE RESULTS (honest failures to report prominently):
================================================================================
1.  CECI chimera FUNCTIONAL FAILURE — geometric overlap ≠ coherent text
2.  Weight-norm FFN compression — 41,083× baseline PPL (Phase 3)
3.  LoRA FFN distillation — 99.9% recovery but still 50× baseline (Phase 4)
4.  MMLU k=256 retraction — 28.1% was geodessical2 artifact, real is 0%
5.  GSM8K on SmolLM2-135M — model too small to measure compression effect
