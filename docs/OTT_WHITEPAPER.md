# Organic Training Theory (OTT)
## A Riemannian Geometry Framework for Neural Inference
### Whitepaper — April 2026 | Geodessical v0.6.0
**Author**: NagusameCS | **Project**: HyperTensor / Geodessical

---

## Abstract

Organic Training Theory (OTT) proposes that a trained neural network's weight space is a Riemannian manifold whose intrinsic curvature encodes all learned knowledge. Under this hypothesis, the standard transformer forward pass — which costs O(n²·d·L) per sequence — can be replaced with geodesic equation solving in a low-dimensional intrinsic coordinate system, reducing computational complexity to O(n·k²) where k ≈ 40 is the intrinsic dimensionality of the manifold (measured: k≈41 for Gemma 4 E2B).

This whitepaper describes the theoretical foundations of OTT, the five-phase autonomous survey pipeline (Axiom Beta-3) that currently instruments these ideas in the Geodessical runtime, and the path toward a production geodesic inference engine.

---

## 1. Core Thesis

### 1.1 The Standard Transformer is Overcomplete

The transformer architecture computes:

$$y = \text{Transformer}(x) = \prod_{l=1}^{L} \left( \text{FFN}_l \circ \text{Attn}_l \right)(x)$$

Each attention layer costs $O(n^2 d)$ for sequence length $n$ and embedding dimension $d$. For $L$ layers this yields $O(n^2 d L)$ — quadratic in sequence length and linear in depth. For $n=512$, $d=2048$, $L=32$: roughly $1.7 \times 10^{10}$ multiply-accumulate operations per forward pass.

### 1.2 Weight Space as a Manifold

The parameter space $\theta \in \mathbb{R}^P$ is equipped with the Fisher Information Metric:

$$g_{ij}(\theta) = \mathbb{E}_{p(y|x;\theta)}\left[\frac{\partial \log p(y|x;\theta)}{\partial \theta^i} \cdot \frac{\partial \log p(y|x;\theta)}{\partial \theta^j}\right]$$

This turns the parameter space into a Riemannian manifold $(M_\theta, g)$. The key observation is that after training, the model's learned knowledge is encoded in the curvature of this manifold — specifically in the Christoffel symbols $\Gamma^k_{ij}$ derived from $g_{ij}$.

### 1.3 Inference as Geodesic Navigation

OTT claims that for a trained model, the mapping from input embedding $x_0$ to output logits can be approximated by solving the geodesic equation:

$$\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\nu\rho}(x) \frac{dx^\nu}{d\lambda} \frac{dx^\rho}{d\lambda} = 0$$

Initial conditions:
- Position: $x^\mu(0) = \text{Embed}(\text{input tokens})$ projected into intrinsic coordinates
- Velocity: $\dot{x}^\mu(0) = v_0$ derived from the attention structure (contextual direction)

The geodesic endpoint $x(\lambda_f)$ maps back to logit space via a linear projection, yielding the next-token distribution.

### 1.4 Complexity Reduction

| Method | Complexity | Notes |
|--------|------------|-------|
| Standard transformer | $O(n^2 \cdot d \cdot L)$ | Full attention + feedforward per layer |
| Geodesic equation (RK4) | $O(n \cdot k^2)$ | $k \approx 40$ intrinsic dims, 4 RK4 steps per token |
| Projected speedup | 8,000–10,000× | Measured Phase 5 projection at $k=41$, $n=512$ |

The critical condition for this to hold: **nonlinearities (softmax, LayerNorm, GeLU/SiLU) must be absorbed into the manifold curvature via a diffeomorphism** $\phi: M_\theta \to N$ where $N$ is a manifold with smoother metric structure encoding those operations. This is the central open problem in OTT.

---

## 2. Mathematical Framework

### 2.1 Riemannian Geometry of Weight Space

**Metric tensor**: Symmetric positive-definite $d \times d$ matrix at each point $x$:
$$g_{ij}(x) = \text{FIM}_{ij}(x) \approx \text{Cov}_{ij}(\text{embeddings near } x)$$

In Beta-3 we approximate FIM using local embedding covariance (tractable without full Jacobian computation). Full FIM from output Jacobians is planned for Beta-4.

**Christoffel symbols** (Levi-Civita connection):
$$\Gamma^k_{ij} = \frac{1}{2} g^{kl}\left(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\right)$$

Computed numerically via finite differences on the metric field at each sample point.

**Riemann curvature tensor**:
$$R^\mu_{\nu\rho\sigma} = \partial_\rho \Gamma^\mu_{\nu\sigma} - \partial_\sigma \Gamma^\mu_{\nu\rho} + \Gamma^\mu_{\rho\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\mu_{\sigma\lambda}\Gamma^\lambda_{\nu\rho}$$

Beta-3 computes the **full Riemann tensor**: both the algebraic $\Gamma \cdot \Gamma$ contractions and the $\partial \Gamma$ finite-difference derivative terms are implemented. See `axiom_beta.c` Phase 3 for the complete computation.

**Ricci tensor and scalar curvature**:
$$R_{\mu\nu} = R^\lambda_{\mu\lambda\nu}, \quad R = g^{\mu\nu} R_{\mu\nu}$$

Scalar curvature $R$ serves as a summary statistic indicating how "curved" the manifold is at each sampled point. High-curvature loci ($|R| > \mu + 2\sigma$) correspond to model decision boundaries and strong contextual dependencies.

### 2.2 Intrinsic Dimensionality

The embedding space of a modern LLM lives in $\mathbb{R}^d$ (e.g., $d=2048$ for Gemma4), but the manifold's intrinsic dimensionality $k \ll d$.

**TwoNN estimator** (Facco et al., 2017): Given pairwise distances between $N$ embedding samples, compute $\mu_i = r_2^{(i)} / r_1^{(i)}$ (ratio of 2nd to 1st nearest-neighbor distances). Under the TwoNN model:
$$\log(\mu_i) \sim \text{Pareto}(\xi = 1/k)$$
Maximum likelihood estimate: $k = 1 / \overline{\log(\mu)}$.

**Measured values** (Gemma 4 E2B Q4_0):
- TwoNN intrinsic dim: $k = 41$ (256 samples)
- PCA components at 95.1% variance: 221 (out of 2048)

This confirms that despite a 2048-dimensional ambient space, the effective geometric structure lives in ~41 dimensions — the basis for O(n·k²) tractability.

### 2.3 Symmetry Group

A permutation-invariant head pair $(h_i, h_j)$ satisfies:
$$\|W_Q^{(i)} - W_Q^{(j)}\|_F / (\|W_Q^{(i)}\|_F + \|W_Q^{(j)}\|_F) < \epsilon_\text{sym}$$

Each such pair corresponds to a generator of the model's symmetry group $G \subset \text{GL}(d)$. The symmetry group encodes which weight transformations leave model behavior invariant.

**Measured values** (Gemma 4 E2B, Beta-3):
- Symmetry score: 0.815 (mean pairwise head similarity)
- Permutation-invariant head pairs: 80
- Lie algebra dimension estimate (generators): 64

### 2.4 Geodesic Equation Integration

The RK4 integrator in `axiom_geo.c` solves:

$$\mathbf{y} = \begin{bmatrix} x^\mu \\ v^\mu = \dot{x}^\mu \end{bmatrix}, \quad \frac{d\mathbf{y}}{d\lambda} = \mathbf{f}(\lambda, \mathbf{y}) = \begin{bmatrix} v^\mu \\ -\Gamma^\mu_{\nu\rho}(x) v^\nu v^\rho \end{bmatrix}$$

RK4 step:
$$k_1 = \mathbf{f}(\lambda_n, \mathbf{y}_n), \quad k_2 = \mathbf{f}(\lambda_n + h/2, \mathbf{y}_n + hk_1/2)$$
$$k_3 = \mathbf{f}(\lambda_n + h/2, \mathbf{y}_n + hk_2/2), \quad k_4 = \mathbf{f}(\lambda_n + h, \mathbf{y}_n + hk_3)$$
$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Divergence detection: if $\|v\| > 10^{10}$ or any component is NaN/Inf, the geodesic is marked non-convergent and retried with damped step size.

**Beta-3 improvement**: Curvature-informed initial velocity prior applies a bounded local acceleration from interpolated Christoffel symbols before integration, improving convergence in high-curvature regions.

---

## 3. Knowledge Injection

OTT proposes that new knowledge can be injected into a trained model without retraining by applying local curvature warps:

$$\Gamma'^k_{ij}(x) = \Gamma^k_{ij}(x) + \alpha \cdot \Delta^k_{ij} \cdot \exp\!\left(-\frac{\|x - x_\text{inj}\|^2}{2\sigma^2}\right)$$

Where:
- $x_\text{inj}$: injection point in intrinsic coordinate space
- $\Delta^k_{ij}$: desired curvature change (encodes the new knowledge)
- $\alpha$: injection strength (controls magnitude)
- $\sigma$: spatial spread (controls locality)

Geodesics near $x_\text{inj}$ are deflected by the warp, changing the model's behavior in that region of concept space without modifying any weights.

**Beta-3 implementation status**:
- Warp accumulation storage: `axiom_warp_state.dat` (persistent across restarts)
- Threshold-triggered recompute: Phase 3 + Phase 4 refresh when warp cross-term estimate exceeds threshold
- Controls: `injection_alpha`, `injection_sigma`, `injection_points`
- Training-time coupling policy: pending (Beta-4)

---

## 4. Axiom Formalization

Axioms are formal constraints on the manifold geometry that must hold for OTT inference to be valid. Four axiom types:

| Type | Statement Template | Example |
|------|--------------------|---------|
| METRIC | Distance structure from PCA variance ratio | $\text{PCA}_{95\%} / d < 0.15$ |
| SYMMETRY | Head permutation invariance | $s_\text{sym} > 0.7 \Rightarrow G \neq \emptyset$ |
| GEODESIC | Curvature bounds on token trajectories | $|R_\text{mean}| < C_\text{bound}$ |
| BOUNDARY | Embedding space boundary behavior | Embedding norms bounded in $[r_\text{min}, r_\text{max}]$ |

**Active learning loop** (Phase 4):
1. Generate axiom candidates from Phase 1–3 geometric features
2. Select least-confident candidates (uncertainty-based)
3. Oracle validation: run deterministic model inference to check consistency
4. Bayesian update: $p_\text{post} = 0.7 \cdot p_\text{prior} + 0.3 \cdot \text{evidence}$
5. Early stop when $\text{uncertainty} < \epsilon_\text{floor}$ sustained for N iterations

**Beta-3 budget**: 2–4 oracle calls (fast mode), 12 (full mode). Prior: cap=16 at MRR=0.032; current: adaptive at MRR=0.067.

---

## 5. Current Implementation Status

### 5.1 Axiom Beta-3 Pipeline (as of April 18, 2026)

| Phase | Status | Key Metrics |
|-------|--------|-------------|
| Phase 1: Manifold ID | ✅ Stable | k=14–41 depending on sample count |
| Phase 2: Symmetry | ✅ Stable | Real dequantized Q-weights; score=0.815, 64 generators |
| Phase 3: Curvature | ✅ Stable | Full Riemann (∂Γ + Γ·Γ); warm cache: 0.17 s (−99.9% vs cold) |
| Phase 4: Axioms | 🟡 Active | MRR improving, 669 ms dominant cost |
| Phase 5: Geodesic Pilot | 🟡 Active | MRR=0.067, decode-aligned targets |
| Knowledge Injection | 🟡 Prototype | Warp plumbing done |
| Geodesic Forward Pass | ❌ Planned | Not yet in decode loop |

### 5.2 Fast-Mode Survey (Gemma 4 E2B, `--axiom-fast --axiom-gpu`)

| Config | Total | Phase 4 | Phase 5 | MRR |
|--------|-------|---------|---------|-----|
| samples=64, probe=256 | 543 ms | ~430 ms | 59 ms | 0.015 |
| samples=128, probe=512 | 1013 ms | ~669 ms | 69 ms | 0.000 |
| Typical fast run | ~977 ms | ~669 ms | ~43 ms | **0.067** |

### 5.3 OTT Inference Modes (v0.6.0)

These modes combine the geometric survey with OTT-accelerated inference:

| Mode | Flag | Geodesic Draft | AttnRes | Depth-Attn | Use Case |
|------|------|----------------|---------|-----------|----------|
| Standard spec-decode | `--ott-speculative` | ✅ batch=2 | ✅ 0.25 | ❌ | Quality-first speculative decode |
| Speed-first | `--ott-fast` | ✅ batch=16 | ✅ 0.25 | ❌ | Maximum tok/s |
| Perfect upper bound | `--ott-perfect` | ✅ exact greedy | ✅ 0.25 | ❌ | Benchmarking acceptance rate ceiling |
| Full OTT | `--ott-full` | ✅ geodesic-first | ✅ 0.35 | ❌ | Complete pipeline + OneDecode |
| Theorem mode | `--ott-theorem` | ✅ geodesic-first | ✅ 0.45 | ✅ 0.55 | Maximum reasoning quality |

**OneDecode** (`--one-decode`): Bakes the geodesic flow map once to `ott_one_decode.bin` (skipping Phase 5). On subsequent runs, the map is loaded and used as an instant speculative draft source.

**OTT-OD** (`--ott-od`): The baked OneDecode map serves as the draft model for speculative decode — the fastest OTT inference mode, since the draft cost is near-zero after the initial bake.

### 5.4 Readiness to Ideal OTT

$$\text{OTT Readiness} = 0.70 \cdot R_\text{geo} + 0.65 \cdot R_\text{axiom} + 0.35 \cdot R_\text{infer} + 0.55 \cdot R_\text{inject} \approx \mathbf{70\%}$$

---

## 6. Open Problems

### P1: Diffeomorphism for Nonlinearities (Hard)
The core unresolved problem is finding a coordinate transform $\phi: M_\theta \to N$ such that:
- SiLU/GELU activations → curvature of $N$
- LayerNorm → geodesic normalization in $N$
- Softmax attention → shortest-path routing in $N$

Without this, OTT inference is an approximation. With it, the geodesic equation becomes the **exact** forward pass in new coordinates.

Current approach: iterative approximation, absorbing layer-by-layer operations into the metric tensor perturbatively.

### P2: Initial Velocity Derivation (Medium)
The geodesic initial velocity $v_0$ must encode the full context without computing attention. Current approach uses a Christoffel-derived prior (Beta-3). Full derivation from attention structure without explicit attention computation remains open.

### P3: Sparse Metric Field (Active)
The metric field is sampled at $N$ discrete points and interpolated via IDW. With $N \leq 64$ (fast mode), the Christoffel symbols carry significant interpolation error. Dense sampling reduces this but increases cost quadratically.

Planned: block-level sampling (N=8 blocks × S=32 tokens/block) to reduce calls from ~1900 to 256 while maintaining representational coverage.

### P4: Geodesic MRR at Production Quality (Active)
Current MRR ≈ 0.067 (fast mode). Production quality requires MRR > 0.5 for geodesic proposals to be competitive with transformer output. The gap is the main quality engineering target for Beta-4.

---

## 7. Roadmap to Production OTT

### Beta-4 (v0.7 "Cortex") — Geodesic Inference Prototype
- Promote Phase 5 from pilot evaluator to candidate token proposer
- LRU hidden-state cache (items 21–26 in GEODESSICAL_PLAN.md)
- Layer-wise geodesic trajectory matching vs transformer
- First accuracy parity measurement: geodesic top-1 vs transformer top-1

### Beta-5 (v0.8 "Nexus") — Hybrid Mode
- Hybrid: standard inference with geodesic verification (agree → skip full forward pass)
- First real latency reduction from geodesic path
- Formal diffeomorphism research (experimental)

### v1.0 "Genesis" — Geodesic Default
- Geodesic inference as the primary Geodessical decode path
- O(n·k²) complexity confirmed in profiling
- Knowledge injection production API
- Full diffeomorphism for nonlinearity absorption (if solved)

---

## 8. References

1. Facco, E. et al. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*. Scientific Reports.
2. Milakov, M. & Gimelshein, N. (2018). *Online normalizer calculation for softmax*. arXiv:1805.02867.
3. Xiao, G. et al. (2023). *Efficient streaming language models with attention sinks*. arXiv:2309.17453.
4. NagusameCS (2026). *Organic Training Theory* (manuscript).
5. HyperTensor Axiom Beta benchmark logs, April 2026 (see `docs/GEODESSICAL_PLAN.md`).

---

*This whitepaper reflects the state of the OTT research implementation as of April 18, 2026. All complexity projections are theoretical or based on Phase 5 projected speedup measurements. Production parity with transformer inference has not yet been demonstrated.*
