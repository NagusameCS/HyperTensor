<img width="1280" height="640" alt="IRIS-MD (3)" src="https://github.com/user-attachments/assets/a746745d-121c-444d-8183-352e36fc0d23" />

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20077378.svg)](https://doi.org/10.5281/zenodo.20077378)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![views](https://repoviews.wkohara.workers.dev/gh/NagusameCS/HyperTensor?v=3)

A geometric framework for understanding, compressing, and extending transformer language models.
Eighteen papers spanning compression theory, living-model adaptation, and a geometric attack
on the Riemann Hypothesis.

| | |
|---|---|
| Papers | [Volume (I-XV)](ARXIVSUBMISSIONS/volume.pdf) · [Volume Extended (I-XVIII)](ARXIVSUBMISSIONS/volumeextended.pdf) · [Jury Proof](ARXIVSUBMISSIONS/jury_proof.pdf) |
| Website | [nagusamecs.github.io/HyperTensor](https://nagusamecs.github.io/HyperTensor) |
| Civilized | [github.com/NagusameCS/civilized-HyperTensor](https://github.com/NagusameCS/civilized-HyperTensor) |
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

## HyperRetro — Integrated Hybrid Mode

Alongside the standalone HyperTensor runtime, the [`hyperretro/`](hyperretro/)
sub-project retrofits the same geometric primitives into the standard
PyTorch / HuggingFace / vLLM stack — FlashAttention-style: just `import
hyperretro` and the kernels run under the hood.

- **PyTorch extension**: `hyperretro.gemv_dual_q8_0(x, Wa, Wb)` wraps the
  fused dual-Q8 GEMV from `runtime/nn/cuda_kernels.cu` as a JIT-built
  `torch.utils.cpp_extension`, with torch/numpy fallbacks. Smoke bench
  on this machine shows ~2.3× over two separate Q8 GEMVs
  (`benchmarks/hyperretro_kernel_smoke.json`).
- **HuggingFace compression**: `hyperretro-compress --model … --rank 1024
  --sink 4 --out …` loads a vanilla HF model, applies GRC / sink-aware
  projection, and writes the result as standard `.safetensors` — fully
  loadable by `AutoModelForCausalLM.from_pretrained` with no
  HyperTensor runtime required.
- **vLLM draft adapter**: `hyperretro.vllm.GeodesicDraft` is a vLLM-shaped
  proposer that swaps in the geodesic k-space step as the draft
  algorithm in speculative decoding.

See [`hyperretro/README.md`](hyperretro/README.md) for details and the
3-way (baseline / HyperRetro / standalone HyperTensor) benchmark harness
under `hyperretro/bench/`.

## Quickstart — Reproduction Toolkit (ht-repro)

The `ht-repro` toolkit packages the framework as an installable Python product:
auto-downloads HuggingFace models, exposes a local REST API, persists run history
and GTC trajectories to SQLite, and ships with a native runtime binary
(`geodessical`).

### Install (development)

```bash
git clone https://github.com/NagusameCS/HyperTensor.git
cd HyperTensor
pip install -e .                              # ht-repro + ht-graft CLIs
pip install hypertensor_runtime/dist/*.whl    # bundled native binary

# Windows: also fetch the OpenBLAS DLL the native binary links against
pwsh scripts/fetch_openblas_windows.ps1
```

### Three-command demo

```bash
# 1. Start the web UI + REST API on http://localhost:8772
ht-repro serve

# 2. Pre-fetch a model into ~/.ht-repro/models  (any HuggingFace repo)
ht-repro models pull Qwen/Qwen2.5-0.5B-Instruct

# 3. Run a graft between two models — both auto-downloaded on demand
ht-graft --donor Qwen/Qwen2.5-0.5B-Instruct \
         --recipient gpt2 --layers 0,1,2
```

### REST API

When `ht-repro serve` is running, the same operations are available over HTTP:

```bash
curl http://localhost:8772/api/v1/health
curl http://localhost:8772/api/v1/gpu
curl -X POST http://localhost:8772/api/v1/sort \
     -H 'Content-Type: application/json' \
     -d '{"data":[3,1,2]}'
curl -X POST http://localhost:8772/api/v1/infer \
     -H 'Content-Type: application/json' \
     -d '{"model":"gpt2","prompt":"Hello"}'
```

Set `HT_REPRO_TOKEN=<secret>` to require `Authorization: Bearer <secret>` on every
request. Jobs are persisted to `~/.ht-repro/store.db` and survive a server
restart — query `GET /api/v1/jobs/<id>` to resume tracking after a reboot.

### Python API

```python
from ht_repro import gpu, models, storage
import hypertensor_runtime as hr

gpu.summary()                           # 'CUDA — NVIDIA RTX 4070 Laptop (8.0 GB, 1 device(s))'
path  = models.resolve("gpt2")          # auto-downloads if missing
runs  = storage.recent_runs(limit=10)   # last 10 runs from SQLite

# Invoke the bundled native binary directly
hr.run_geodessical("--help")
```

### Run the test suite

```bash
pip install pytest
pytest tests/unit_ht_repro -v   # 44 unit tests, no GPU/network required
python tests/audit_commercial.py   # 33-check commercial-viability audit
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
