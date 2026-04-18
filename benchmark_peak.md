# Peak Benchmark: Geodessical GPU vs Ollama GPU

This file is a raw benchmark summary. Read it alongside `../BENCHMARK_ANALYSIS.md` if you want the interpretation and follow-up fixes.

**Date:** 2026-04-15 13:47
**CPU:** AMD Ryzen 9 7940HS w/ Radeon 780M Graphics     
**GPU:** NVIDIA GeForce RTX 4070 Laptop GPU  |  VRAM total: 8188 MB
**Model:** gemma4-2b (google_gemma-4-E2B-it Q4_0, 3.2 GB)
**Trials:** 5 measured per condition + 1 warmup discarded
**Conditions:** GPU backend only -- optimal settings for each runtime

**Metrics:**
- Decode t/s = decode-phase tokens per second
- Prefill t/s = prompt processing throughput
- TTFT ms = Time To First Token
- E2E ms = total wall time (TTFT + decode)
- Avg GPU% = mean GPU utilization during inference
- Peak VRAM = max VRAM used during inference (MB)
- Avg Watt = mean GPU power draw (W)
- Avg CPU% = mean system CPU load during inference
- Peak RAM = peak process working set (MB)

---

## Performance: Head-to-Head

### Prompt: long

| Runtime | N | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | ms/tok |
|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:------:|
| Geo-GPU | 128 | 128 | 71 | 114.4 | 684 | 2487.4 | 19.4 |
| Geo-GPU | 512 | 512 | 88.4 | 100.3 | 689.6 | 6482.2 | 12.7 |
| Oll-GPU | 128 | 128 | 116.9 | 4105 | 11.5 | 1106.2 | 8.6 |
| Oll-GPU | 512 | 512 | 114.8 | 3487.1 | 12.6 | 4472.6 | 8.7 |
| **Geo/Oll ratio** | **128** | | **0.61x** | | | | |
| **Geo/Oll ratio** | **512** | | **0.77x** | | | | |

### Prompt: short

| Runtime | N | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | ms/tok |
|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:------:|
| Geo-GPU | 128 | 14 | 35.9 | 115.1 | 259.2 | 649.4 | 46.4 |
| Geo-GPU | 512 | 14 | 35.8 | 115 | 260.2 | 651 | 46.5 |
| Oll-GPU | 128 | 128 | 118 | 1976.5 | 10.9 | 1095.4 | 8.6 |
| Oll-GPU | 512 | 321 | 115.5 | 1549.5 | 13.1 | 2794.4 | 8.7 |
| **Geo/Oll ratio** | **128** | | **0.3x** | | | | |
| **Geo/Oll ratio** | **512** | | **0.31x** | | | | |

---

## Resource Usage: Head-to-Head

### Prompt: long

| Runtime | N | Avg GPU% | Peak VRAM MB | Avg Watt | Avg CPU% | Peak RAM MB |
|---------|--:|:--------:|:------------:|:--------:|:--------:|:-----------:|
| Geo-GPU | 128 | 1 | 3617 | 20.3 | 23.4 | 1173 |
| Geo-GPU | 512 | 74.3 | 5602 | 62.8 | 22.7 | 2626 |
| Oll-GPU | 128 | 85.7 | 3617 | 60.1 | 24 | 1045 |
| Oll-GPU | 512 | 73.4 | 3617 | 62.9 | 28.1 | 1047 |

### Prompt: short

| Runtime | N | Avg GPU% | Peak VRAM MB | Avg Watt | Avg CPU% | Peak RAM MB |
|---------|--:|:--------:|:------------:|:--------:|:--------:|:-----------:|
| Geo-GPU | 128 | 22.5 | 3617 | 35 | 21.8 | 1211 |
| Geo-GPU | 512 | 33.8 | 3617 | 32.6 | 29 | 1167 |
| Oll-GPU | 128 | 87.7 | 3617 | 70 | 28 | 1047 |
| Oll-GPU | 512 | 83.2 | 3617 | 67.1 | 27.2 | 1047 |

---

## Efficiency: Decode t/s per Watt

| Runtime | Prompt | N | Decode t/s | Avg Watt | t/s per W |
|---------|--------|--:|:----------:|:--------:|:---------:|
| Geo-GPU | long | 128 | 71 | 20.3 | 3.498 |
| Geo-GPU | long | 512 | 88.4 | 62.8 | 1.408 |
| Geo-GPU | short | 128 | 35.9 | 35 | 1.026 |
| Geo-GPU | short | 512 | 35.8 | 32.6 | 1.098 |
| Oll-GPU | long | 128 | 116.9 | 60.1 | 1.945 |
| Oll-GPU | long | 512 | 114.8 | 62.9 | 1.825 |
| Oll-GPU | short | 128 | 118 | 70 | 1.686 |
| Oll-GPU | short | 512 | 115.5 | 67.1 | 1.721 |

---

## Raw Results (all trials)

| Runtime | Prompt | N | Trial | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | GPU% | VRAM MB | Watt | CPU% | RAM MB |
|---------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:----:|:-------:|:----:|:----:|:------:|
| Geo-GPU | long | 128 | 1 | 128 | 70.4 | 114.7 | 701 | 2518 | 0 | 3617 | 16.9 | 24 | 1160 |
| Geo-GPU | long | 128 | 2 | 128 | 70.2 | 113.3 | 692 | 2515 | 1 | 3617 | 17.6 | 23 | 1131 |
| Geo-GPU | long | 128 | 3 | 128 | 71 | 114.3 | 682 | 2484 | 0 | 3617 | 31.1 | 22 | 1173 |
| Geo-GPU | long | 128 | 4 | 128 | 71.2 | 114.8 | 680 | 2477 | 1 | 3617 | 18.2 | 19 | 1150 |
| Geo-GPU | long | 128 | 5 | 128 | 72 | 115 | 665 | 2443 | 0 | 3617 | 17.7 | 29 | 1160 |
| Geo-GPU | long | 512 | 1 | 512 | 88.4 | 100.4 | 689 | 6480 | 83.5 | 5602 | 70.3 | 19.5 | 2626 |
| Geo-GPU | long | 512 | 2 | 512 | 88.3 | 100.3 | 695 | 6495 | 70 | 5602 | 57.9 | 27.8 | 2626 |
| Geo-GPU | long | 512 | 3 | 512 | 88.8 | 100.4 | 671 | 6438 | 74.2 | 5602 | 59.6 | 18 | 2626 |
| Geo-GPU | long | 512 | 4 | 512 | 88.2 | 100.2 | 698 | 6504 | 70 | 5602 | 62.5 | 31.2 | 2626 |
| Geo-GPU | long | 512 | 5 | 512 | 88.3 | 100.3 | 695 | 6494 | 74 | 5602 | 63.7 | 17 | 2626 |
| Geo-GPU | short | 128 | 1 | 14 | 37.3 | 122.3 | 252 | 627 | 0 | 3617 | 18.1 | 22 | 1160 |
| Geo-GPU | short | 128 | 2 | 14 | 35.3 | 111.1 | 262 | 659 | 1 | 3617 | 18.1 | 21 | 1150 |
| Geo-GPU | short | 128 | 3 | 14 | 37.4 | 122.9 | 252 | 626 | 44 | 3617 | 56.3 | 19 | 1160 |
| Geo-GPU | short | 128 | 4 | 14 | 34.1 | 108.6 | 271 | 681 | 0 | 3617 | 34 | 26 | 1160 |
| Geo-GPU | short | 128 | 5 | 14 | 35.4 | 110.6 | 259 | 654 | 0 | 3617 | 48.4 | 21 | 1211 |
| Geo-GPU | short | 512 | 1 | 14 | 35.4 | 111.1 | 260 | 655 | 0 | 3617 | 16.8 | 27 | 1160 |
| Geo-GPU | short | 512 | 2 | 14 | 35.1 | 111.2 | 264 | 663 | 73 | 3617 | 56.5 | 27 | 1160 |
| Geo-GPU | short | 512 | 3 | 14 | 36.6 | 119.7 | 257 | 639 | 1 | 3617 | 16.9 | 33 | 1160 |
| Geo-GPU | short | 512 | 4 | 14 | 37.1 | 121.3 | 254 | 631 | 60 | 3617 | 54.9 | 21 | 1167 |
| Geo-GPU | short | 512 | 5 | 14 | 34.9 | 111.5 | 266 | 667 | 1 | 3617 | 17.8 | 37 | 1160 |
| Oll-GPU | long | 128 | 1 | 128 | 116.6 | 2123.5 | 20.2 | 1118 | 88 | 3617 | 69.6 | 26 | 1041 |
| Oll-GPU | long | 128 | 2 | 128 | 117.3 | 4690.1 | 9.2 | 1100 | 83 | 3617 | 65.7 | 30 | 1044 |
| Oll-GPU | long | 128 | 3 | 128 | 118 | 4464.7 | 9.6 | 1094 | 86 | 3617 | 67.8 | 21 | 1045 |
| Oll-GPU | long | 128 | 4 | 128 | 115.9 | 4766 | 9 | 1114 | 0 | 3617 | 46.7 | 20 | 1045 |
| Oll-GPU | long | 128 | 5 | 128 | 116.8 | 4480.8 | 9.6 | 1105 | 0 | 3617 | 50.6 | 23 | 1045 |
| Oll-GPU | long | 512 | 1 | 512 | 114.1 | 4561.2 | 9.4 | 4496 | 57 | 3617 | 60.9 | 37.3 | 1047 |
| Oll-GPU | long | 512 | 2 | 512 | 114 | 3270.9 | 13.1 | 4506 | 86 | 3617 | 67.6 | 27.3 | 1047 |
| Oll-GPU | long | 512 | 3 | 512 | 116 | 3338.3 | 12.9 | 4427 | 87.3 | 3617 | 68.8 | 24.3 | 1047 |
| Oll-GPU | long | 512 | 4 | 512 | 114.6 | 3130 | 13.7 | 4483 | 57 | 3617 | 51.8 | 22.7 | 1047 |
| Oll-GPU | long | 512 | 5 | 512 | 115.4 | 3135 | 13.7 | 4451 | 79.7 | 3617 | 65.5 | 29 | 1047 |
| Oll-GPU | short | 128 | 1 | 128 | 119 | 1108.2 | 18 | 1094 | 0 | 0 | 0 | 0 | 0 |
| Oll-GPU | short | 128 | 2 | 128 | 118.5 | 2206.2 | 9.1 | 1089 | 87 | 3617 | 70.1 | 24 | 1047 |
| Oll-GPU | short | 128 | 3 | 128 | 117.8 | 2137.6 | 9.4 | 1096 | 88 | 3617 | 70.4 | 26 | 1045 |
| Oll-GPU | short | 128 | 4 | 128 | 117.2 | 2251.2 | 8.9 | 1101 | 88 | 3617 | 69.6 | 34 | 1045 |
| Oll-GPU | short | 128 | 5 | 128 | 117.7 | 2179.2 | 9.2 | 1097 | 0 | 0 | 0 | 0 | 0 |
| Oll-GPU | short | 512 | 1 | 342 | 116.4 | 1966.7 | 10.2 | 2949 | 85.5 | 3617 | 69.4 | 29.5 | 1047 |
| Oll-GPU | short | 512 | 2 | 330 | 113.8 | 1443.2 | 13.9 | 2914 | 86.5 | 3617 | 69.9 | 21.5 | 1047 |
| Oll-GPU | short | 512 | 3 | 304 | 114.7 | 1348.4 | 14.8 | 2666 | 85 | 3617 | 69 | 34 | 1045 |
| Oll-GPU | short | 512 | 4 | 358 | 115.5 | 1478.8 | 13.5 | 3114 | 71 | 3617 | 56.9 | 26 | 1047 |
| Oll-GPU | short | 512 | 5 | 271 | 117 | 1510.2 | 13.2 | 2329 | 88 | 3617 | 70.2 | 25 | 1045 |

