# Hardware Specification — HyperTensor Reproduction

This document gives the detailed hardware spec for the three reproduction tiers referenced from [`REPRODUCTION.md`](../REPRODUCTION.md). It also lists the exact reference machines the published numbers were measured on.

## Tier 1 — CPU-only

**Minimum:** any x86_64 CPU, 16 GB RAM, 5 GB free disk.

**Reference:** Ryzen 9 7940HS / 32 GB DDR5-5600 / NVMe SSD / Windows 11 24H2.

**Covers:** Papers I (analysis), V, VI, VII, X (analysis), XIII, XIV (analysis), XV (TEH/COG analysis), XVI, XVII, XVIII, jury foundation.

**Wall-clock for the full T1 suite:** ~30 minutes.

## Tier 2 — Consumer GPU

**Minimum:** NVIDIA RTX 3070 / 4070 or better, **8 GB VRAM**, CUDA driver 552+, CUDA Toolkit 12.x.

**Reference:** RTX 4070 Laptop GPU (Ada AD106), 8 GB GDDR6, **32 MB L2 cache**, driver 595.79.

**Covers (in addition to T1):** Paper I (throughput), Paper II, Paper III, Paper IV (GTC), Paper VIII (full), Paper XI (1.5B bilateral), Paper XII (1.5B native), Paper XV (.MIKU runtime), ISAGI 7B (4-bit).

**Wall-clock for the full T1+T2 suite:** ~90 minutes.

### GPU-specific GRC predictions

The cross-GPU formula (Paper IX) is `k* = L2_MB × 42.7`:

| GPU | L2 (MB) | Predicted k* | Notes |
|---|---|---|---|
| RTX 3070 | 4 | 170 | Below useful range; use k=512 |
| RTX 4070 Laptop | 32 | 1024 | Primary reference |
| RTX 4070 (desktop) | 36 | 1280 | Validated |
| RTX 4080 | 64 | 1536 | Validated |
| RTX 4090 | 72 | 1536 (capped) | k_max constraint |
| RTX 5070 Ti | 48 | 1280 | Validated |
| L40S | 96 | 1536 (capped) | EC2 reference |
| A10G | 6 | 256 | Use small-k path |
| A100 80GB | 40 | 1024 | Validated |
| H100 80GB | 50 | 1280 | Predicted |

`AXEX_MANIFOLD_K_MAX = 1536` is a runtime constraint — k=2048 requests are silently capped.

## Tier 3 — Datacenter GPU

**Minimum:** L40S, A100 (40 or 80 GB), or H100. 32 GB RAM. Ubuntu 22.04+.

**Reference:** AWS `g6e.xlarge`:
- 1× NVIDIA L40S, 48 GB VRAM, 96 MB L2
- 4 vCPU (Intel Xeon Sapphire Rapids)
- 32 GB system RAM
- 250 GB gp3 EBS root volume
- Ubuntu 22.04 LTS, Deep Learning AMI (Ubuntu 22.04) PyTorch 2.3
- US-East-2 region

**Spot pricing (May 2026):** ~$0.85/hr; full T3 suite (~6 h) costs ~$5.

**Covers (T3-only):** Paper IX (cross-GPU re-run), Paper XI (7B bilateral), Paper XII (7B native), Paper XV (10K-interaction COG), Paper XVI (50K-prime AGT). Pre-computed JSON outputs for all T3 results are committed to `benchmarks/` so you can verify without re-running.

## SSH config (recommended for T3)

```sshconfig
Host hypertensor
    HostName <ec2-public-dns>
    User ubuntu
    IdentityFile ~/.ssh/<key.pem>
    ServerAliveInterval 60
    ServerAliveCountMax 6
    ForwardAgent no
```

## Disk requirements

| Item | Size |
|---|---|
| Source checkout | ~250 MB |
| Python venv (Tier A) | ~50 MB |
| Python venv (Tier B/C) | ~3 GB |
| Llama 3.1 8B Q4_K_M GGUF | 4.6 GB |
| W_proj cache (per rank) | ~1.1 GB |
| Qwen-2.5-7B-Instruct (HF cache) | ~15 GB |
| Qwen-2.5-1.5B-Instruct (HF cache) | ~3 GB |
| `benchmarks/` (committed reference outputs) | ~150 MB |

Allocate 25 GB free disk for a comfortable T2 setup, 50 GB for T3.

## Network

The verification scripts make no network calls at runtime. First-time model downloads (HuggingFace) need internet:
- Llama 3.1 8B Q4_K_M: 4.6 GB download from HuggingFace
- Qwen-2.5-7B-Instruct: 15 GB download from HuggingFace (split into safetensors shards)
- Qwen-2.5-1.5B-Instruct: 3 GB download

Once cached locally (default: `~/.cache/huggingface/`), all subsequent runs are offline.
