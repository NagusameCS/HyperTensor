# Session logs — 2026-04-28/29 actaware-PCA fix campaign

All logs collected during the cross-device validation of the actaware-PCA
default-flip fix (commit `122b8c3`).

## Layout

```
session_logs/
├── README.md                      <-- this file
├── local/                         RTX 4070 Laptop (Win11)
│   ├── baseline_4070_q4km.log     Llama-3.1-8B Q4_K_M, no compression  → 34.6 tok/s
│   ├── fix_test.log               Llama-3.1-8B Q4_K_M, --axex-compress
│   │                              (default → weight-PCA, k=1024)       → 38.4 tok/s (+11.0%)
│   └── smollm_local.log           Smollm2-135M Q8_0 + --axex-compress  → reaches PCA OK,
│                                  decode fails with [error generating response]
│                                  (UNRELATED pre-existing bug, also reproduces remote)
└── remote/                        Arch Linux + RTX 3050 6 GB
    ├── environment.txt            uname / nvidia-smi / nvcc / zig versions
    ├── wproj_cache_8b_k1024.log   First 8B run, k=1024 — killed at layer 8/32
    │                              by 30-min timeout (cold weight-PCA on 3050
    │                              is ~3 min/layer at k=1024 = ~96 min total).
    ├── grc_8b_k256.log            Second 8B run, k=256 — IN PROGRESS at time
    │                              of capture (layer 5/32). Detached, 45-min
    │                              budget. Output cache will persist to disk.
    └── smollm_run.log             Smollm2 run on 3050 — same decode error
                                   as local, confirming bug is not remote-specific.
```

The CUDA-built remote binaries (`geodessical` 4.1 MB, `cuda_kernels.so` 1.5 MB)
are in `../binaries_remote_arch_rtx3050/` (excluded from git via .gitignore size).

## Headline numbers

| Machine          | OS      | GPU       | Model                | Quant  | Mode                              | Decode tok/s |
|------------------|---------|-----------|----------------------|--------|-----------------------------------|--------------|
| RTX 4070 Laptop  | Win11   | 8 GB      | Llama-3.1-8B-Instruct| Q4_K_M | baseline                          | 34.6         |
| RTX 4070 Laptop  | Win11   | 8 GB      | Llama-3.1-8B-Instruct| Q4_K_M | --axex-compress (k=1024)          | 38.4 (+11.0%)|
| RTX 3050 6GB     | Arch    | 6 GB      | Llama-3.1-8B-Instruct| Q8_0   | baseline (overflows VRAM)         | 2.7          |
| RTX 3050 6GB     | Arch    | 6 GB      | Llama-3.1-8B-Instruct| Q8_0   | --axex-compress k=256             | (in progress)|

## Build provenance (remote)

```
Linux archlinux 6.19.9-arch1-1
NVIDIA GeForce RTX 3050, 595.58.03, 6144 MiB
nvcc release 13.1, V13.1.115 (sm_86 target)
zig 0.14.0 (drop-in C compiler for x86_64-linux-gnu)
openblas 0.3.32-2 (/usr/lib/libopenblas.so)
geodessical: 4 220 912 bytes (CUDA-enabled, ENABLE_CUDA=1)
cuda_kernels.so: 1 537 528 bytes
ott_wproj_cache_915B16D6.bin: 52 113 872 bytes (smollm2 weight-PCA k=512 cache)
```

Built via `scripts/campaign/build_remote_arch_cuda.sh` from same source tree
as commit `122b8c3` (host/main.c includes the actaware-PCA default-flip fix).
