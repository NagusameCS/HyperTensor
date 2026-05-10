<img width="1280" height="640" alt="IRIS-MD (3)" src="https://github.com/user-attachments/assets/a746745d-121c-444d-8183-352e36fc0d23" />

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20077378.svg)](https://doi.org/10.5281/zenodo.20077378)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A geometric framework for understanding, compressing, and extending transformer language models.
Eighteen papers spanning compression theory, living-model adaptation, and a geometric attack
on the Riemann Hypothesis.

| | |
|---|---|
| Papers | [Volume (I-XV)](ARXIVSUBMISSIONS/volume.pdf) · [Volume Extended (I-XVIII)](ARXIVSUBMISSIONS/volumeextended.pdf) · [Jury Proof](ARXIVSUBMISSIONS/jury_proof.pdf) |
| Website | [nagusamecs.github.io/HyperTensor](https://nagusamecs.github.io/HyperTensor) |
| DOI (Volume v1) | [10.5281/zenodo.20077378](https://doi.org/10.5281/zenodo.20077378) |
| Reproduction | [REPRODUCTION.md](REPRODUCTION.md) |
| Status | 15/18 papers at 100% · External verification 14/14 · Jury 53× at N=1M |

---

## What This Project Contains

Papers I-X: Empirical Kernel. GRC attention compression (106% throughput at k=1024 via L2 cache residency), cross-GPU transfer (3 GPU architectures validated), speculative geodesic decoding, OTT manifold runtime, CECI model grafting (7 published chimeras, 5/7 improve MMLU), and cluster compression.

Papers XI-XV: Living-Model Stack. Universal Geodesic Taxonomy for shared coordinate systems across models (bilateral subspace overlap 0.968 at 1.5B, 7B demonstrated). Native geodesic training directly in compressed manifolds. Safe OGD with mathematical safety guarantee (0% TEH by construction). Behavioral sniping at 7.4× specificity. COG+TEH for lifelong learning (10,000 interaction convergence confirmed by Mann-Kendall test).

Papers XVI-XVIII: Riemann Hypothesis. A geometric attack using Z₂-symmetry from the functional equation. AGT detects critical-line zeros at 100% accuracy across 50,000 primes and 1,030 zeta zeros (1D critical subspace at all scales, 800× separation). ACM learns the involution in latent space. The Bridge protocol composes all three papers into a unified proof-search pipeline validated on 105 known zeros (jury confidence J ≈ 1 − 10⁻³¹⁵).

The Geometric Jury. Eight proven theorems provide the mathematical foundation: aggregation uniqueness (J = 1 − ∏(1−cᵢ)), instinct horizon derivation, convergence rate, centroid entanglement, contrastive separation, ensemble optimality, regression superiority, and sample complexity. The jury is the unifying principle across all 18 papers.

## Key Results (May 5, 2026)

| Result | Measurement |
|---|---|
| AGT at 50K primes + 1K zeros | 100% detection, k90=k95=1, 800× separation |
| COG 10K interactions | 14 trajectories, metric saturates, MK p=0.015 |
| Bilateral UGT 1.5B | Subspace overlap 0.968 |
| 7B bilateral UGT | Principal angles 0.01–0.11° (L40S, 4-bit + checkpointing) |
| Jury scaling at N=1M | 53× faster than O(N) full scan |
| External verification | 14/14 (100%) on real 1.5B model (sklearn/scipy/numpy) |
| Performance optimizations | 9/12 verified (randomized SVD 9×, svd_lowrank 10.6×, batch cosine 220×) |

## Quick Start

```bash
# External verification (requires Qwen2.5-1.5B, ~3.1GB VRAM)
python scripts/verifyexternal15b.py

# AGT at scale (fits 8GB VRAM)
python scripts/agt_10k.py --scale 50000

# COG lifelong learning
python scripts/cog_10k.py --n 10000

# Universal domain brain mapper (any HuggingFace model)
python scripts/ugtdomainmapper.py --model Qwen/Qwen2.5-0.5B-Instruct

# Jury bridge (Riemann meta-jury)
python scripts/jury_bridge.py
```

## Repository Layout

| Directory | Contents |
|---|---|
| `ARXIVSUBMISSIONS/` | 18 paper TeX sources, compiled PDFs, juryproof.tex |
| `scripts/` | Python research scripts, benchmarks, verification tools |
| `benchmarks/` | All measurement data (AGT, COG, verification, optimization results) |
| `docs/` | Handoff documents, Riemann proof, comprehensive state, papers |
| `runtime/` | C inference runtime, JIT kernels, tensor operations |
| `host/` | Geodessical host, MCP server, GPU daemon |
| `legacy/` | TensorOS freestanding-OS code (boot, kernel, virt) — preserved, not required |

## Build (C Runtime)

```powershell
.\build_host.ps1
```

## License

HyperTensor is **dual-licensed**:

- **Source code** (scripts, runtime, tooling, build files) — [MIT License](LICENSE).
- **Manuscripts and figures** in `ARXIV_SUBMISSIONS/` and `docs/` (Volume, Jury Proof, individual papers) — [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0).

The Zenodo deposit ([10.5281/zenodo.20077378](https://doi.org/10.5281/zenodo.20077378)) releases the manuscript under CC BY 4.0. Reuse of either layer requires attribution; see the respective license texts for full terms.
