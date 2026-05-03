# HyperTensor — Derivative Research Papers (J–L)

Quick-reference outlines for the remaining Tier 3 derivative papers.

---

## Paper J: Heterogeneous Drafters for Speculative Decoding

**Tier**: 3 | **Script**: `heterogeneous_drafters.py` | **Data**:  Done

### Status
Simulation complete. Key negative result: heterogeneous drafting (different k
per draft slot) does NOT beat uniform k=1024 at γ=4 on Llama-8B. The acceptance-vs-rank
curve is too sharp — α collapses from 46.9% to 0% between k=768 and k=256,
leaving no room for tiered compression.

### Key Claims
1. **Acceptance collapse is catastrophic, not gradual**: α(k) follows a sigmoid
   centered at k≈768 (k/d≈0.19). Below this, α→0. Above this, α≈α(∞).
2. **The sharp transition rules out heterogeneous drafting** at current
   implementation quality. To make tiered drafting viable, the acceptance curve
   must be smoothed — either via better compression (per-matrix bases) or
   via calibration-aware projection.
3. **Future path**: If per-matrix bases or calibrated sink channels can push
   k_critical lower (e.g., from 768 to 384), heterogeneous drafting becomes
   viable with k₁₂=768, k₃₄=256.
4. **Token-type-adaptive drafting**: Different token types (code vs prose)
   may have different acceptance curves. Code tokens might tolerate more
   aggressive compression because syntax is more predictable.

### Data
- Simulation grid: 8 rank options  heterogeneous pairs
- Best uniform: k=1024, 0.0058 tok/ms
- Best heterogeneous: k₁₂=1024, k₃₄=256, 0.0051 tok/ms
- See `benchmarks/heterogeneous_drafters/heterogeneous_summary.json`

### Next
1. Measure actual α(k) curve with per-matrix bases (should be smoother)
2. Implement token-type detection (code vs prose classifier)
3. Re-simulate with smoothed α curve
4. Write-up: ~6 pages, 4 figures

---

## Paper K: MoE  GRC — Why Grouped-Query Attention Resists Compression

**Tier**: 3 | **Script**: `moe_gqa_analysis.py` | **Data**:  Done

### Status
Analysis complete on SmolLM2-135M (detected as GQA, 3:1 Q:KV ratio).
Key finding: per-head k95 sum ≈ 2 joint k95. Joint compression averages
out per-head specialization, losing routing fidelity.

### Key Claims
1. **GQA inflates joint Gram rank**: With n_q > n_kv, W_Q dominates the joint
   Gram, giving k_int/d ≈ 0.70 vs 0.52 for MHA. This is geometry, not a bug.
2. **Per-head compression is the fix**: Compress each KV head independently
   (k_total ≈ 2 k_joint, but preserves per-head specialization).
3. **GQA models need different compression strategy**: The tall-rectangular
   W_Q matrix means Q-heads project into different subspaces. A shared basis
   can't capture all of them.
4. **MoE routing may also be compressible**: The gating network (W_gate)
   routes tokens to experts. If the routing subspace is low-rank (likely —
   routing is routing), GRC could compress the gating path too.

### Data
- SmolLM2-135M: joint k95=282 (0.49d), per-head sum=558 (2.0)
- Per-head k95: 26–89 per head (0.05–0.15d)
- See `benchmarks/moe_gqa/moe_gqa_summary.json`

### Next
1. Run on a true large-scale GQA model (Llama-8B, Gemma4-27B)
2. Implement per-head compression in the C runtime
3. Measure PPL with per-head vs joint compression
4. Analyze MoE gating network compressibility
5. Write-up: ~8 pages, 5 figures

---

## Paper L: Differentiable Compression Rank via Gumbel-Softmax

**Tier**: 3 | **Script**: `differentiable_rank.py` | **Data**:  Prototype

### Status
Prototype built. REINFORCE-based gradient-free optimization of per-layer rank
logits. Currently too slow (re-loads GGUF per iteration). Needs weight caching
optimization before full execution.

### Key Claims
1. **Rank is a discrete choice that can be relaxed**: Gumbel-Softmax over
   k ∈ {64, 128, 256, ..., 1536} gives differentiable rank selection.
2. **L1 penalty on E[k] encourages sparse rank allocation**: Layers that
   need high rank get it; layers that don't, don't.
3. **The learned allocation should recover MCR's Mix/Compress/Refine phases**
   from first principles — no hand-crafted heuristic needed.
4. **Extension to full training**: In a real training loop, the rank logits
   become learnable parameters optimized jointly with the LoRA/LM objective.

### Data
- Prototype: 6 layers, 200 iterations, λ=0.005
- Iter 0: E[k]=351, sampled k=768 (bias toward middle ranks)
- See `scripts/differentiable_rank.py`

### Next
1. Add weight caching (load GGUF once, cache all layers in memory)
2. Run full optimization (500 iterations, all 30 layers)
3. Compare learned allocation vs MCR phases
4. Build PyTorch version with autograd for real training integration
5. Write-up: ~8 pages, 5 figures
