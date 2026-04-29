# Cross-device validation — 2026-04-29

Validation that the actaware-PCA fix (commit `122b8c3`) executes correctly on
a second physical machine, second GPU vendor tier, second OS.

## Hardware

| Machine | OS | CPU | GPU | VRAM | HBM peak (real) |
|---|---|---|---|---|---|
| Local laptop | Windows 11 | Ryzen 9 7940HS (16 thr) | RTX 4070 Laptop | 8 GB | ~256 GB/s |
| Remote box   | Arch Linux 6.19 | unknown (32 thr) | RTX 3050 6 GB | 6 GB | ~168 GB/s |

Code constants assume desktop-4070 (336 GB/s, 40 TFLOPS) so HBM-% column is
informational only — raw tok/s is the apples-to-apples number.

## Llama-3.1-8B-Instruct decode @ -n 16, --temp 0, prompt="hello"

| Machine | Quant | Mode | Decode tok/s | Notes |
|---|---|---|---|---|
| RTX 4070 Laptop | Q4_K_M (4.2 GB) | baseline | 34.6 | fits in VRAM |
| RTX 4070 Laptop | Q4_K_M | --axex-compress (default→weight-PCA, k=1024, attn-only, skip-O) | 38.4 | +11.0% |
| RTX 3050 6 GB | Q8_0 (8.1 GB) | baseline | 2.7 | overflows VRAM |
| RTX 3050 6 GB | Q8_0 | --axex-compress k=256 | (running, populates wproj cache) | |

Cross-quant numbers for the 3050 are not directly comparable to the 4070 row
above because Q8_0 is 2× larger and spills out of the 6 GB GPU. Once the
k=256 compressed run completes the cache, the same model also fits in VRAM
on the 3050 and we get a fair on-GPU number.

## What the fix proves

* `--axex-compress` (the default user-facing flag) on the 4070 went from
  silently exiting -1 after ~10 min in `axpca_compute_topk_weighted` to
  finishing in 3.3 s and yielding +11% decode throughput.
* Same source rebuilt with `scripts/campaign/build_remote_arch_cuda.sh`
  on Arch + zig 0.14 + nvcc sm_86 produces a CUDA-enabled binary that
  reaches the post-PCA decode stage on the 3050 (verified with smollm2 and
  with 8B partial-cache run; 32-layer cold weight-PCA at k=1024 is ~96 min,
  longer than the supervisor timeout, so this campaign uses k=256/512 on
  the 3050).
