# Three-Way Benchmark: Geodessical vs HyperTensor vs Ollama

## System

| Parameter | Value |
|-----------|-------|
| Date | 2026-04-15 23:00:28 |
| CPU | AMD Ryzen 9 7940HS w/ Radeon 780M Graphics      |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 591.86, 8188 MiB |
| System RAM | 31 GB |
| Model | google_gemma-4-E2B-it-Q4_0.gguf (Gemma4 2B Q4_0, 3.2 GB) |
| Geodessical | v (GPU/CUDA, batch prefill) |
| HyperTensor | v (GPU/CUDA, batched causal attention) |
| Ollama | ollama version is 0.20.4 (GPU, num_gpu=999) |
| Trials | **30** measured per condition + 2 warmups discarded |
| Trial order | Randomised per trial to reduce ordering bias |
| Monitor cadence | 500 ms (nvidia-smi) |

## Metrics Glossary

| Metric | Description |
|--------|-------------|
| Decode t/s | Tokens generated per second (decode phase only) |
| Prefill t/s | Prompt tokens processed per second |
| TTFT ms | Time To First Token (prefill latency) |
| E2E ms | Total wall time: TTFT + decode |
| ms/tok | Milliseconds per output token |
| Avg GPU% | Mean GPU SM utilisation during inference |
| Peak VRAM | Maximum VRAM used (MB) |
| Avg Watt | Mean GPU power draw (W) |
| Avg CPU% | Mean system CPU load |
| Peak RAM | Peak process working set (MB) |
| t/s/W | Decode tokens per joule (efficiency) |
| CI 95% | 95% confidence interval half-width (1.96*σ/√n) |

---

## Decode Throughput (tok/s) — All Conditions

*Higher = better. Mean ± 95%CI over 30 trials.*

| Runtime | Prompt | N | Mean (t/s) | ±95%CI | Median | σ | Min | Max | n |
|---------|--------|--:|:----------:|:------:|:------:|:-:|:---:|:---:|:-:|
| Geodessical | short | 32 | **55.63** | 0.98 | 55.6 | 2.74 | 49.4 | 59.4 | 30 |
| Geodessical | short | 128 | **56.87** | 0.65 | 57.3 | 1.82 | 53.9 | 59.4 | 30 |
| Geodessical | short | 512 | **55.52** | 0.85 | 55.8 | 2.39 | 49.4 | 59.4 | 30 |
| Geodessical | medium | 32 | **1.71** | 0.04 | 1.8 | 0.11 | 1.5 | 1.9 | 30 |
| Geodessical | medium | 128 | **0.48** | 0.03 | 0.5 | 0.09 | 0.3 | 0.6 | 29 |
| Geodessical | medium | 512 | **0.1** | 0 | 0.1 | 0 | 0.1 | 0.1 | 30 |
| Geodessical | code | 32 | **65.55** | 1.26 | 66.3 | 3.51 | 59.4 | 71.6 | 30 |
| Geodessical | code | 128 | **85.72** | 1.49 | 86.2 | 4.16 | 71.2 | 90 | 30 |
| Geodessical | code | 512 | **82.38** | 1.07 | 81.5 | 3 | 77.5 | 88.6 | 30 |
| HyperTensor | short | 32 | **39.11** | 0.68 | 39.3 | 1.89 | 34.7 | 41.3 | 30 |
| HyperTensor | short | 128 | **40.02** | 0.35 | 40.3 | 0.99 | 37.8 | 41.7 | 30 |
| HyperTensor | short | 512 | **39.29** | 0.59 | 39.8 | 1.64 | 33.2 | 41.3 | 30 |
| HyperTensor | medium | 32 | **2.97** | 0.06 | 3 | 0.16 | 2.5 | 3.2 | 30 |
| HyperTensor | medium | 128 | **2.59** | 0.16 | 2.7 | 0.45 | 1.1 | 3.2 | 30 |
| HyperTensor | medium | 512 | **2.1** | 0.19 | 2.1 | 0.52 | 0.2 | 3 | 30 |
| HyperTensor | code | 32 | **51.78** | 0.82 | 52.8 | 2.3 | 46.9 | 54.6 | 30 |
| HyperTensor | code | 128 | **87.52** | 1.21 | 87.6 | 3.37 | 75.3 | 90.9 | 30 |
| HyperTensor | code | 512 | **90.55** | 0.9 | 90.5 | 2.5 | 87 | 95.7 | 30 |
| Ollama | short | 32 | **117.22** | 1.74 | 117.8 | 4.87 | 103.7 | 126.1 | 30 |
| Ollama | short | 128 | **116.96** | 0.71 | 116.9 | 1.99 | 113 | 122.4 | 30 |
| Ollama | short | 512 | **115.14** | 0.77 | 115.6 | 2.16 | 109.2 | 119.3 | 30 |
| Ollama | medium | 32 | **159.89** | 19.25 | 122.5 | 53.8 | 96.5 | 238.4 | 30 |
| Ollama | medium | 128 | **118.18** | 17.62 | 111 | 49.23 | 52.5 | 237 | 30 |
| Ollama | medium | 512 | **102.45** | 17.69 | 74.3 | 49.43 | 23 | 207.1 | 30 |
| Ollama | long | 32 | **72.73** | 1.05 | 71.8 | 2.93 | 67.1 | 78.6 | 30 |
| Ollama | long | 128 | **109.42** | 2.64 | 109.7 | 7.37 | 76.6 | 118.7 | 30 |
| Ollama | long | 512 | **110.99** | 1.08 | 110.9 | 3.01 | 106.9 | 116.7 | 30 |
| Ollama | code | 32 | **116.76** | 2 | 118.2 | 5.59 | 99.8 | 124.8 | 30 |
| Ollama | code | 128 | **115.34** | 1.65 | 117.2 | 4.6 | 102.2 | 121 | 30 |
| Ollama | code | 512 | **111.14** | 1.09 | 110.9 | 3.05 | 107 | 117.5 | 30 |

---

## Time To First Token (ms) — All Conditions

*Lower = better. Mean ± 95%CI over 30 trials.*

| Runtime | Prompt | N | Mean (ms) | ±95%CI | Median | σ | Min | Max | n |
|---------|--------|--:|:---------:|:------:|:------:|:-:|:---:|:---:|:-:|
| Geodessical | short | 32 | **113.23** | 2.38 | 112 | 6.66 | 104 | 127 | 30 |
| Geodessical | short | 128 | **110** | 1.39 | 109 | 3.88 | 105 | 119 | 30 |
| Geodessical | short | 512 | **113.2** | 2.05 | 112 | 5.73 | 105 | 126 | 30 |
| Geodessical | medium | 32 | **176.37** | 3.72 | 177 | 10.38 | 162 | 201 | 30 |
| Geodessical | medium | 128 | **230.27** | 39.99 | 195 | 111.75 | 170 | 738 | 30 |
| Geodessical | medium | 512 | **239.07** | 14.64 | 259 | 40.9 | 175 | 289 | 30 |
| Geodessical | code | 32 | **186.83** | 4.48 | 184 | 12.52 | 170 | 209 | 30 |
| Geodessical | code | 128 | **177.53** | 6.13 | 173 | 17.14 | 166 | 250 | 30 |
| Geodessical | code | 512 | **187.7** | 4.08 | 188 | 11.39 | 169 | 207 | 30 |
| HyperTensor | short | 32 | **159.17** | 2.79 | 158 | 7.8 | 150 | 181 | 30 |
| HyperTensor | short | 128 | **154.3** | 1.5 | 154 | 4.19 | 148 | 164 | 30 |
| HyperTensor | short | 512 | **157.07** | 2.41 | 156 | 6.74 | 147 | 179 | 30 |
| HyperTensor | medium | 32 | **250** | 5.42 | 246 | 15.14 | 231 | 289 | 30 |
| HyperTensor | medium | 128 | **295.47** | 25.75 | 269 | 71.95 | 234 | 582 | 30 |
| HyperTensor | medium | 512 | **498.93** | 292.58 | 355 | 817.62 | 243 | 4790 | 30 |
| HyperTensor | code | 32 | **256.8** | 4.37 | 255 | 12.22 | 243 | 284 | 30 |
| HyperTensor | code | 128 | **255.8** | 5.87 | 254 | 16.4 | 239 | 314 | 30 |
| HyperTensor | code | 512 | **263.83** | 4.71 | 267 | 13.16 | 241 | 289 | 30 |
| Ollama | short | 32 | **9.77** | 0.69 | 9.4 | 1.94 | 8.9 | 19.6 | 30 |
| Ollama | short | 128 | **9.69** | 0.57 | 9.4 | 1.6 | 8.9 | 17.5 | 30 |
| Ollama | short | 512 | **14.47** | 1.43 | 13.2 | 3.98 | 9.5 | 28.3 | 30 |
| Ollama | medium | 32 | **10.02** | 0.75 | 9.4 | 2.09 | 8.7 | 20.4 | 30 |
| Ollama | medium | 128 | **196.22** | 303.9 | 10.9 | 849.25 | 8.9 | 4676 | 30 |
| Ollama | medium | 512 | **235.16** | 242.57 | 99.6 | 677.86 | 12.9 | 3793.8 | 30 |
| Ollama | long | 32 | **90.14** | 13.89 | 89.7 | 38.82 | 13.4 | 214.9 | 30 |
| Ollama | long | 128 | **9.73** | 0.24 | 9.6 | 0.66 | 8.9 | 11.5 | 30 |
| Ollama | long | 512 | **14.14** | 0.67 | 13.9 | 1.87 | 9 | 18.6 | 30 |
| Ollama | code | 32 | **10.52** | 1.8 | 9.5 | 5.02 | 8.7 | 36.9 | 30 |
| Ollama | code | 128 | **10.32** | 1.21 | 9.5 | 3.37 | 8.9 | 25.7 | 30 |
| Ollama | code | 512 | **13.95** | 0.48 | 13.9 | 1.34 | 9.5 | 17.5 | 30 |

---

## Prefill Throughput (tok/s) — All Conditions

*Higher = better.*

| Runtime | Prompt | N | Mean (t/s) | ±95%CI | Median | σ | Min | Max |
|---------|--------|--:|:----------:|:------:|:------:|:-:|:---:|:---:|
| Geodessical | short | 32 | **116.1** | 2.31 | 117.2 | 6.45 | 103.1 | 123.9 |
| Geodessical | short | 128 | **118.31** | 2.13 | 121.9 | 5.94 | 107.4 | 124.1 |
| Geodessical | short | 512 | **115.91** | 2.18 | 119 | 6.1 | 102 | 123.6 |
| Geodessical | medium | 32 | **79.21** | 1.85 | 80.8 | 5.18 | 69.3 | 86.3 |
| Geodessical | medium | 128 | **67.74** | 6.35 | 75.5 | 17.75 | 0.5 | 82.9 |
| Geodessical | medium | 512 | **53.06** | 0.6 | 52.6 | 1.68 | 50.2 | 56.1 |
| Geodessical | code | 32 | **106.06** | 2.09 | 105.3 | 5.83 | 95.8 | 115.8 |
| Geodessical | code | 128 | **110.1** | 1.8 | 110.4 | 5.03 | 95.5 | 115.4 |
| Geodessical | code | 512 | **106.29** | 1.41 | 106.1 | 3.93 | 99.5 | 114.9 |
| HyperTensor | short | 32 | **116.88** | 2.33 | 120.6 | 6.52 | 102.5 | 123.6 |
| HyperTensor | short | 128 | **118.65** | 1.85 | 121.1 | 5.16 | 109.7 | 123.7 |
| HyperTensor | short | 512 | **116.69** | 2.33 | 119.9 | 6.51 | 99.9 | 123.2 |
| HyperTensor | medium | 32 | **138.87** | 3.92 | 142.9 | 10.96 | 106.1 | 154.9 |
| HyperTensor | medium | 128 | **115.07** | 8.32 | 119.9 | 23.26 | 51.8 | 150.4 |
| HyperTensor | medium | 512 | **85.88** | 7.31 | 90.3 | 20.43 | 6.5 | 121.5 |
| HyperTensor | code | 32 | **110.92** | 1.84 | 113.6 | 5.15 | 97.6 | 116.3 |
| HyperTensor | code | 128 | **112.94** | 0.96 | 112.2 | 2.67 | 107.6 | 116 |
| HyperTensor | code | 512 | **96.28** | 0.95 | 96.7 | 2.65 | 92.4 | 101.6 |
| Ollama | short | 32 | **2090.71** | 83.17 | 2141.4 | 232.43 | 1021.5 | 2254.7 |
| Ollama | short | 128 | **2097.23** | 77.67 | 2128.5 | 217.04 | 1141 | 2247.2 |
| Ollama | short | 512 | **1447.11** | 93.12 | 1521.8 | 260.23 | 707.1 | 2107.7 |
| Ollama | medium | 32 | **2757.47** | 118.18 | 2861.3 | 330.26 | 1325.1 | 3092.7 |
| Ollama | medium | 128 | **2007.12** | 396.64 | 2606.4 | 1108.42 | 5.8 | 3021.5 |
| Ollama | medium | 512 | **397.53** | 151.5 | 293.1 | 423.36 | 7.1 | 2088.8 |
| Ollama | long | 32 | **830.1** | 286.85 | 611 | 801.61 | 251.3 | 4025.3 |
| Ollama | long | 128 | **5580.54** | 127.08 | 5651.6 | 355.13 | 4707.4 | 6073.7 |
| Ollama | long | 512 | **3884.55** | 196.47 | 3969.4 | 549.04 | 2906.4 | 6030.8 |
| Ollama | code | 32 | **2851.71** | 154.48 | 2953.5 | 431.68 | 759.4 | 3206.2 |
| Ollama | code | 128 | **2845.67** | 157.16 | 2955.2 | 439.19 | 1088.5 | 3145.7 |
| Ollama | code | 512 | **2027.77** | 79.25 | 2020.1 | 221.46 | 1604.1 | 2947.4 |

---

## Head-to-Head Ratios vs Ollama

*Ratio = Runtime mean / Ollama mean. >1.0 = faster than Ollama.*

### Decode t/s ratio

| Runtime | Prompt | N | Ratio vs Ollama | Runtime t/s | Ollama t/s |
|---------|--------|--:|:---------------:|:-----------:|:----------:|
| Geodessical | short | 32 | **0.475x** ✗ | 55.63 | 117.22 |
| Geodessical | short | 128 | **0.486x** ✗ | 56.87 | 116.96 |
| Geodessical | short | 512 | **0.482x** ✗ | 55.52 | 115.14 |
| Geodessical | medium | 32 | **0.011x** ✗ | 1.71 | 159.89 |
| Geodessical | medium | 128 | **0.004x** ✗ | 0.48 | 118.18 |
| Geodessical | medium | 512 | **0.001x** ✗ | 0.1 | 102.45 |
| Geodessical | code | 32 | **0.561x** ✗ | 65.55 | 116.76 |
| Geodessical | code | 128 | **0.743x** ✗ | 85.72 | 115.34 |
| Geodessical | code | 512 | **0.741x** ✗ | 82.38 | 111.14 |
| HyperTensor | short | 32 | **0.334x** ✗ | 39.11 | 117.22 |
| HyperTensor | short | 128 | **0.342x** ✗ | 40.02 | 116.96 |
| HyperTensor | short | 512 | **0.341x** ✗ | 39.29 | 115.14 |
| HyperTensor | medium | 32 | **0.019x** ✗ | 2.97 | 159.89 |
| HyperTensor | medium | 128 | **0.022x** ✗ | 2.59 | 118.18 |
| HyperTensor | medium | 512 | **0.02x** ✗ | 2.1 | 102.45 |
| HyperTensor | code | 32 | **0.443x** ✗ | 51.78 | 116.76 |
| HyperTensor | code | 128 | **0.759x** ✗ | 87.52 | 115.34 |
| HyperTensor | code | 512 | **0.815x** ✗ | 90.55 | 111.14 |

### TTFT ratio (lower ratio = faster)

| Runtime | Prompt | N | Ratio vs Ollama | Runtime ms | Ollama ms |
|---------|--------|--:|:---------------:|:----------:|:---------:|
| Geodessical | short | 32 | **11.59x** | 113.23 | 9.77 |
| Geodessical | short | 128 | **11.352x** | 110 | 9.69 |
| Geodessical | short | 512 | **7.823x** | 113.2 | 14.47 |
| Geodessical | medium | 32 | **17.602x** | 176.37 | 10.02 |
| Geodessical | medium | 128 | **1.174x** | 230.27 | 196.22 |
| Geodessical | medium | 512 | **1.017x** | 239.07 | 235.16 |
| Geodessical | code | 32 | **17.76x** | 186.83 | 10.52 |
| Geodessical | code | 128 | **17.203x** | 177.53 | 10.32 |
| Geodessical | code | 512 | **13.455x** | 187.7 | 13.95 |
| HyperTensor | short | 32 | **16.292x** | 159.17 | 9.77 |
| HyperTensor | short | 128 | **15.924x** | 154.3 | 9.69 |
| HyperTensor | short | 512 | **10.855x** | 157.07 | 14.47 |
| HyperTensor | medium | 32 | **24.95x** | 250 | 10.02 |
| HyperTensor | medium | 128 | **1.506x** | 295.47 | 196.22 |
| HyperTensor | medium | 512 | **2.122x** | 498.93 | 235.16 |
| HyperTensor | code | 32 | **24.411x** | 256.8 | 10.52 |
| HyperTensor | code | 128 | **24.787x** | 255.8 | 10.32 |
| HyperTensor | code | 512 | **18.913x** | 263.83 | 13.95 |

---

## Resource Usage Summary

| Runtime | Prompt | N | Avg GPU% | Peak VRAM MB | Avg Watt | Max Watt | Avg CPU% | Peak RAM MB |
|---------|--------|--:|:--------:|:------------:|:--------:|:--------:|:--------:|:-----------:|
| Geodessical | short | 32 | 44.57 | 4063 | 34.07 | 41.4 | 29.57 | 1211 |
| Geodessical | short | 128 | 86.5 | 4063 | 57.43 | 59.1 | 26 | 1160 |
| Geodessical | short | 512 | 54 | 4074 | 37.05 | 63.7 | 32.33 | 1211 |
| Geodessical | medium | 32 | 22.27 | 4080 | 27.4 | 55.6 | 48.87 | 1232 |
| Geodessical | medium | 128 | 46.62 | 7927 | 37.89 | 66.2 | 51.36 | 3825 |
| Geodessical | medium | 512 | 83.49 | 7550 | 47.69 | 59 | 49.22 | 2689 |
| Geodessical | code | 32 | 34.43 | 5517 | 37.34 | 65.3 | 30.33 | 1251 |
| Geodessical | code | 128 | 56.91 | 7504 | 57.12 | 78.9 | 29.2 | 2162 |
| Geodessical | code | 512 | 42.42 | 5543 | 56.07 | 69.5 | 33.63 | 1292 |
| HyperTensor | short | 32 | 42.29 | 4063 | 28.82 | 40.6 | 27.4 | 1227 |
| HyperTensor | short | 128 | 57.19 | 4063 | 40.52 | 63.9 | 24.9 | 1278 |
| HyperTensor | short | 512 | 46.25 | 4074 | 45.52 | 65.4 | 27.97 | 1250 |
| HyperTensor | medium | 32 | 22.3 | 4080 | 27.85 | 37.4 | 35.57 | 1254 |
| HyperTensor | medium | 128 | 29.94 | 7916 | 32.14 | 64.3 | 50.42 | 2271 |
| HyperTensor | medium | 512 | 25.62 | 7496 | 27.21 | 47.7 | 52.79 | 2683 |
| HyperTensor | code | 32 | 37.83 | 5517 | 36.36 | 62.7 | 32.63 | 1230 |
| HyperTensor | code | 128 | 62.5 | 7500 | 56.93 | 74 | 34.95 | 2625 |
| HyperTensor | code | 512 | 81.37 | 7517 | 63.28 | 80.6 | 28.98 | 2625 |
| Ollama | short | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 128 | 45.45 | 4063 | 31.41 | 63.5 | 25.57 | 1043 |
| Ollama | short | 512 | 38.34 | 4074 | 35.65 | 65 | 28.68 | 1042 |
| Ollama | medium | 32 | 1 | 4070 | 27.8 | 27.8 | 2 | 1040 |
| Ollama | medium | 128 | 27.9 | 7848 | 31.75 | 57.6 | 43.73 | 3717 |
| Ollama | medium | 512 | 24.43 | 5724 | 25.52 | 51.5 | 44.84 | 3285 |
| Ollama | long | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 128 | 18.08 | 5770 | 26.17 | 62.7 | 33.13 | 818 |
| Ollama | long | 512 | 63.18 | 5522 | 52.93 | 66.4 | 31.59 | 254 |
| Ollama | code | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 128 | 44.89 | 5526 | 57.4 | 78.2 | 27.03 | 242 |
| Ollama | code | 512 | 68.05 | 5543 | 61.74 | 69.4 | 32.18 | 187 |

---

## Efficiency: Decode tok/s per Watt

| Runtime | Prompt | N | Decode t/s | Avg Watt | t/s per W |
|---------|--------|--:|:----------:|:--------:|:---------:|
| Geodessical | short | 32 | 55.63 | 34.07 | 1.6328 |
| Geodessical | short | 128 | 56.87 | 57.43 | 0.9902 |
| Geodessical | short | 512 | 55.52 | 37.05 | 1.4985 |
| Geodessical | medium | 32 | 1.71 | 27.4 | 0.0624 |
| Geodessical | medium | 128 | 0.48 | 37.89 | 0.0127 |
| Geodessical | medium | 512 | 0.1 | 47.69 | 0.0021 |
| Geodessical | code | 32 | 65.55 | 37.34 | 1.7555 |
| Geodessical | code | 128 | 85.72 | 57.12 | 1.5007 |
| Geodessical | code | 512 | 82.38 | 56.07 | 1.4692 |
| HyperTensor | short | 32 | 39.11 | 28.82 | 1.357 |
| HyperTensor | short | 128 | 40.02 | 40.52 | 0.9877 |
| HyperTensor | short | 512 | 39.29 | 45.52 | 0.8631 |
| HyperTensor | medium | 32 | 2.97 | 27.85 | 0.1066 |
| HyperTensor | medium | 128 | 2.59 | 32.14 | 0.0806 |
| HyperTensor | medium | 512 | 2.1 | 27.21 | 0.0772 |
| HyperTensor | code | 32 | 51.78 | 36.36 | 1.4241 |
| HyperTensor | code | 128 | 87.52 | 56.93 | 1.5373 |
| HyperTensor | code | 512 | 90.55 | 63.28 | 1.4309 |
| Ollama | short | 32 | 117.22 | 0 | 0 |
| Ollama | short | 128 | 116.96 | 31.41 | 3.7237 |
| Ollama | short | 512 | 115.14 | 35.65 | 3.2297 |
| Ollama | medium | 32 | 159.89 | 27.8 | 5.7514 |
| Ollama | medium | 128 | 118.18 | 31.75 | 3.7222 |
| Ollama | medium | 512 | 102.45 | 25.52 | 4.0145 |
| Ollama | long | 32 | 72.73 | 0 | 0 |
| Ollama | long | 128 | 109.42 | 26.17 | 4.1811 |
| Ollama | long | 512 | 110.99 | 52.93 | 2.0969 |
| Ollama | code | 32 | 116.76 | 0 | 0 |
| Ollama | code | 128 | 115.34 | 57.4 | 2.0094 |
| Ollama | code | 512 | 111.14 | 61.74 | 1.8001 |

---

## Raw Results (all 30 trials per condition)

| Runtime | Prompt | N | Trial | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | Wall ms | GPU% | VRAM MB | Watt | CPU% | RAM MB |
|---------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:-------:|:----:|:-------:|:----:|:----:|:------:|
| Geodessical | code | 32 | 1 | 32 | 62.5 | 104.7 | 206 | 718 | 1789 | 88 | 5500 | 62.8 | 24 | 1160 |
| Geodessical | code | 32 | 2 | 32 | 59.4 | 95.8 | 205 | 744 | 1833 | 7 | 5500 | 25.2 | 26 | 1160 |
| Geodessical | code | 32 | 3 | 32 | 59.7 | 95.9 | 204 | 740 | 1873 | 6 | 5500 | 17.2 | 29 | 1251 |
| Geodessical | code | 32 | 4 | 32 | 62.9 | 105.3 | 205 | 714 | 1801 | 7 | 5500 | 17.6 | 28 | 1250 |
| Geodessical | code | 32 | 5 | 32 | 61.4 | 102.6 | 209 | 730 | 1787 | 6 | 5500 | 17.2 | 26 | 1158 |
| Geodessical | code | 32 | 6 | 32 | 63 | 101.9 | 194 | 702 | 1680 | 0 | 5500 | 40.8 | 34 | 1228 |
| Geodessical | code | 32 | 7 | 32 | 66 | 104.6 | 179 | 664 | 1608 | 86 | 5500 | 65.3 | 39 | 1234 |
| Geodessical | code | 32 | 8 | 32 | 67.4 | 111.6 | 188 | 663 | 1664 | 1 | 5500 | 37.8 | 19 | 1222 |
| Geodessical | code | 32 | 9 | 32 | 63 | 101.3 | 191 | 699 | 1742 | 0 | 5500 | 44.1 | 31 | 1218 |
| Geodessical | code | 32 | 10 | 32 | 66.4 | 106.7 | 182 | 664 | 1683 | 0 | 5500 | 16.8 | 26 | 1173 |
| Geodessical | code | 32 | 11 | 32 | 70.8 | 115.6 | 175 | 627 | 1618 | 86 | 5500 | 25.3 | 31 | 1220 |
| Geodessical | code | 32 | 12 | 32 | 64.1 | 104.9 | 194 | 693 | 1771 | 1 | 5500 | 16.9 | 36 | 1222 |
| Geodessical | code | 32 | 13 | 32 | 61.4 | 100 | 201 | 722 | 2168 | 1 | 5500 | 16.8 | 52 | 933 |
| Geodessical | code | 32 | 14 | 32 | 59.8 | 97 | 204 | 739 | 1711 | 17 | 5500 | 60.8 | 37 | 1211 |
| Geodessical | code | 32 | 15 | 32 | 63.1 | 101.6 | 192 | 699 | 1643 | 6 | 5500 | 43.1 | 31 | 1216 |
| Geodessical | code | 32 | 16 | 32 | 69.4 | 115.6 | 184 | 645 | 1526 | 2 | 5500 | 43 | 26 | 1199 |
| Geodessical | code | 32 | 17 | 32 | 66.7 | 105.9 | 177 | 657 | 1552 | 0 | 5500 | 29.8 | 26 | 1196 |
| Geodessical | code | 32 | 18 | 32 | 68.8 | 114.1 | 185 | 650 | 1545 | 16 | 5500 | 36 | 31 | 1160 |
| Geodessical | code | 32 | 19 | 32 | 66.7 | 104.9 | 175 | 655 | 1569 | 87 | 5500 | 38.9 | 27 | 1160 |
| Geodessical | code | 32 | 20 | 32 | 67.4 | 106.6 | 175 | 650 | 1535 | 9 | 5500 | 43.2 | 37 | 1160 |
| Geodessical | code | 32 | 21 | 32 | 70.5 | 115.3 | 176 | 630 | 1579 | 0 | 5500 | 35.5 | 32 | 1182 |
| Geodessical | code | 32 | 22 | 32 | 62.5 | 102.8 | 201 | 713 | 1580 | 88 | 5500 | 42.4 | 35 | 1189 |
| Geodessical | code | 32 | 23 | 32 | 66.8 | 106.2 | 177 | 656 | 1575 | 0 | 5517 | 43.2 | 23 | 1216 |
| Geodessical | code | 32 | 24 | 32 | 66.3 | 105.5 | 179 | 662 | 1586 | 1 | 5517 | 43.5 | 40 | 1167 |
| Geodessical | code | 32 | 25 | 32 | 71.3 | 115.8 | 172 | 621 | 1522 | 87 | 5517 | 43.2 | 23 | 1228 |
| Geodessical | code | 32 | 26 | 32 | 65.8 | 104.7 | 180 | 666 | 1594 | 1 | 5517 | 42.9 | 37 | 1230 |
| Geodessical | code | 32 | 27 | 32 | 68.1 | 109.3 | 178 | 648 | 1531 | 6 | 5517 | 53.7 | 30 | 1157 |
| Geodessical | code | 32 | 28 | 32 | 66.1 | 104 | 176 | 660 | 1577 | 96 | 5517 | 40 | 29 | 1211 |
| Geodessical | code | 32 | 29 | 32 | 71.6 | 115.8 | 171 | 618 | 1526 | 87 | 5517 | 36.9 | 25 | 1222 |
| Geodessical | code | 32 | 30 | 32 | 67.7 | 105.7 | 170 | 643 | 1541 | 0 | 5517 | 40.3 | 20 | 1160 |
| Geodessical | code | 128 | 1 | 79 | 89.7 | 115 | 168 | 1049 | 1963 | 86 | 5517 | 37.3 | 17 | 1252 |
| Geodessical | code | 128 | 2 | 79 | 86.2 | 110.4 | 174 | 1090 | 2000 | 0 | 5517 | 69.3 | 27 | 1211 |
| Geodessical | code | 128 | 3 | 79 | 86.2 | 110 | 171 | 1088 | 2012 | 89 | 5517 | 59.1 | 31 | 1156 |
| Geodessical | code | 128 | 4 | 79 | 89.2 | 114.9 | 172 | 1058 | 1960 | 0 | 5517 | 58.4 | 32 | 1217 |
| Geodessical | code | 128 | 5 | 79 | 87.9 | 113.5 | 176 | 1075 | 2384 | 88 | 5517 | 60.2 | 34 | 1211 |
| Geodessical | code | 128 | 6 | 79 | 86.4 | 112.3 | 183 | 1097 | 2061 | 3 | 5517 | 59 | 34 | 1171 |
| Geodessical | code | 128 | 7 | 79 | 81.1 | 105.6 | 198 | 1172 | 2414 | 0 | 5518 | 16.8 | 33 | 1222 |
| Geodessical | code | 128 | 8 | 79 | 82.3 | 110.4 | 217 | 1177 | 2321 | 1 | 5526 | 16.8 | 62 | 1211 |
| Geodessical | code | 128 | 9 | 79 | 71.2 | 95.5 | 250 | 1359 | 3748 | 42.5 | 7504 | 48.1 | 56 | 2162 |
| Geodessical | code | 128 | 10 | 79 | 80.8 | 103.4 | 186 | 1164 | 2646 | 88 | 5526 | 57.9 | 23 | 1211 |
| Geodessical | code | 128 | 11 | 79 | 78.8 | 100.2 | 184 | 1187 | 2677 | 1 | 5526 | 62.3 | 20 | 1160 |
| Geodessical | code | 128 | 12 | 79 | 84.6 | 107 | 167 | 1101 | 2017 | 0 | 5526 | 67.6 | 31 | 1160 |
| Geodessical | code | 128 | 13 | 79 | 84.5 | 107.6 | 173 | 1108 | 2014 | 99 | 5526 | 74.6 | 19 | 1175 |
| Geodessical | code | 128 | 14 | 79 | 84 | 107 | 173 | 1113 | 2040 | 0 | 5526 | 54.7 | 26 | 1228 |
| Geodessical | code | 128 | 15 | 79 | 89.4 | 115.1 | 171 | 1055 | 1971 | 0 | 5526 | 59.5 | 29 | 1217 |
| Geodessical | code | 128 | 16 | 79 | 84.8 | 108 | 173 | 1105 | 2006 | 0 | 5526 | 58.2 | 24 | 1155 |
| Geodessical | code | 128 | 17 | 79 | 84.9 | 108.4 | 174 | 1105 | 2020 | 0 | 5526 | 74.1 | 29 | 1160 |
| Geodessical | code | 128 | 18 | 79 | 89.9 | 115.4 | 168 | 1047 | 1951 | 88 | 5526 | 51.5 | 25 | 1160 |
| Geodessical | code | 128 | 19 | 79 | 84.9 | 108.1 | 171 | 1101 | 2076 | 1 | 5526 | 65.6 | 27 | 1160 |
| Geodessical | code | 128 | 20 | 79 | 84.6 | 107.7 | 172 | 1106 | 2023 | 2 | 5526 | 57.4 | 31 | 1211 |
| Geodessical | code | 128 | 21 | 79 | 84.4 | 107.7 | 174 | 1110 | 2022 | 0 | 5526 | 64.4 | 37 | 1156 |
| Geodessical | code | 128 | 22 | 79 | 89.6 | 115.1 | 169 | 1051 | 1951 | 0 | 5526 | 52.7 | 20 | 1175 |
| Geodessical | code | 128 | 23 | 79 | 89.8 | 115.1 | 167 | 1047 | 1964 | 88 | 5526 | 66 | 25 | 1215 |
| Geodessical | code | 128 | 24 | 79 | 89.3 | 115 | 172 | 1057 | 2060 | 89 | 5526 | 64.1 | 24 | 1157 |
| Geodessical | code | 128 | 25 | 79 | 89.3 | 115.3 | 174 | 1059 | 2016 | 0 | 5526 | 58.8 | 27 | 1166 |
| Geodessical | code | 128 | 26 | 79 | 90 | 115.2 | 166 | 1044 | 1965 | 88 | 5526 | 62.6 | 23 | 1160 |
| Geodessical | code | 128 | 27 | 79 | 89.6 | 114.9 | 168 | 1050 | 1976 | 0 | 5526 | 64.9 | 40 | 1219 |
| Geodessical | code | 128 | 28 | 79 | 83.9 | 106.7 | 174 | 1116 | 2032 | 0 | 5526 | 53 | 23 | 1156 |
| Geodessical | code | 128 | 29 | 79 | 89.6 | 115 | 169 | 1051 | 1946 | 57 | 5526 | 64.6 | 20 | 1169 |
| Geodessical | code | 128 | 30 | 79 | 84.6 | 107.6 | 172 | 1106 | 2022 | 0 | 5526 | 54 | 27 | 1160 |
| Geodessical | code | 512 | 1 | 79 | 84.8 | 107.5 | 169 | 1101 | 2026 | 0 | 5526 | 53.9 | 23 | 1175 |
| Geodessical | code | 512 | 2 | 79 | 84.1 | 107.9 | 179 | 1118 | 2034 | 33 | 5526 | 63.7 | 28 | 1224 |
| Geodessical | code | 512 | 3 | 79 | 84.3 | 106.8 | 169 | 1106 | 2047 | 85 | 5526 | 66.7 | 26 | 1279 |
| Geodessical | code | 512 | 4 | 79 | 84 | 107.4 | 177 | 1118 | 2075 | 0 | 5526 | 61.8 | 20 | 1157 |
| Geodessical | code | 512 | 5 | 79 | 79.7 | 103.8 | 201 | 1192 | 2268 | 0 | 5526 | 60 | 39 | 1226 |
| Geodessical | code | 512 | 6 | 79 | 81.1 | 105.2 | 193 | 1167 | 2200 | 52 | 5526 | 63.6 | 45 | 1160 |
| Geodessical | code | 512 | 7 | 79 | 87.7 | 114.1 | 182 | 1083 | 2098 | 7 | 5526 | 46.3 | 32 | 1160 |
| Geodessical | code | 512 | 8 | 79 | 82.3 | 105 | 179 | 1139 | 2098 | 87 | 5526 | 65.7 | 30 | 1204 |
| Geodessical | code | 512 | 9 | 79 | 80.9 | 104.8 | 195 | 1172 | 2144 | 54 | 5526 | 64.6 | 45 | 1292 |
| Geodessical | code | 512 | 10 | 79 | 83 | 105.9 | 177 | 1129 | 2102 | 99 | 5526 | 69.5 | 33 | 1166 |
| Geodessical | code | 512 | 11 | 79 | 77.5 | 99.5 | 195 | 1214 | 2223 | 2 | 5526 | 58.2 | 47 | 1217 |
| Geodessical | code | 512 | 12 | 79 | 81.5 | 106.1 | 196 | 1165 | 2143 | 99 | 5526 | 66 | 35 | 1211 |
| Geodessical | code | 512 | 13 | 79 | 87.3 | 111.8 | 172 | 1077 | 2184 | 6 | 5526 | 50.8 | 32 | 1175 |
| Geodessical | code | 512 | 14 | 79 | 83.6 | 106.9 | 178 | 1123 | 2067 | 0 | 5526 | 56.6 | 26 | 1230 |
| Geodessical | code | 512 | 15 | 79 | 79.2 | 102.1 | 196 | 1194 | 2253 | 6 | 5526 | 20.5 | 34 | 1156 |
| Geodessical | code | 512 | 16 | 79 | 82.9 | 107.2 | 188 | 1141 | 2157 | 24 | 5529 | 61.7 | 39 | 1194 |
| Geodessical | code | 512 | 17 | 79 | 79.8 | 102.3 | 189 | 1179 | 2141 | 4 | 5529 | 46 | 33 | 1160 |
| Geodessical | code | 512 | 18 | 79 | 80.9 | 105.5 | 198 | 1174 | 2163 | 3 | 5529 | 18.2 | 37 | 1160 |
| Geodessical | code | 512 | 19 | 79 | 81.3 | 106.4 | 201 | 1173 | 2136 | 2 | 5529 | 57 | 36 | 1160 |
| Geodessical | code | 512 | 20 | 79 | 81.3 | 106.5 | 202 | 1174 | 2186 | 0 | 5543 | 58.1 | 45 | 1160 |
| Geodessical | code | 512 | 21 | 79 | 85.9 | 111.7 | 186 | 1106 | 2070 | 89 | 5543 | 64.1 | 37 | 1160 |
| Geodessical | code | 512 | 22 | 79 | 88.6 | 114.6 | 175 | 1067 | 1989 | 89 | 5543 | 67.1 | 34 | 1160 |
| Geodessical | code | 512 | 23 | 79 | 78.2 | 99.7 | 187 | 1197 | 2077 | 0 | 5543 | 56.2 | 28 | 1211 |
| Geodessical | code | 512 | 24 | 79 | 80.9 | 101.8 | 171 | 1148 | 2104 | 43 | 5543 | 66.2 | 35 | 1210 |
| Geodessical | code | 512 | 25 | 79 | 88.4 | 114.9 | 180 | 1074 | 1967 | 7 | 5543 | 62.5 | 28 | 1224 |
| Geodessical | code | 512 | 26 | 79 | 80.3 | 104.9 | 202 | 1186 | 2096 | 65 | 5543 | 62.4 | 26 | 1189 |
| Geodessical | code | 512 | 27 | 79 | 79.7 | 104.6 | 207 | 1198 | 2255 | 7 | 5543 | 18.2 | 23 | 1232 |
| Geodessical | code | 512 | 28 | 79 | 78.7 | 102.2 | 202 | 1206 | 2168 | 6 | 5543 | 49.4 | 43 | 1215 |
| Geodessical | code | 512 | 29 | 79 | 81.2 | 105.6 | 197 | 1170 | 2174 | 89 | 5543 | 63.8 | 43 | 1211 |
| Geodessical | code | 512 | 30 | 79 | 82.2 | 106.1 | 188 | 1149 | 2109 | 60 | 5543 | 63.3 | 27 | 1175 |
| Geodessical | long | 32 | 1 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 2 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 3 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 4 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 5 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 6 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 7 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 8 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 9 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 10 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 11 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 12 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 13 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 14 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 15 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 16 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 17 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 18 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 19 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 20 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 21 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 22 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 23 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 24 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 25 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 26 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 27 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 28 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 29 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 32 | 30 | ERR | no GD output: on and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 32 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 1 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 2 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 3 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 4 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 5 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 6 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 7 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 8 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 9 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 10 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 11 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 12 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 13 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 14 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 15 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 16 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 17 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 18 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 19 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 20 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 21 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 22 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 23 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 24 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 25 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 26 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 27 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 28 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 29 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 128 | 30 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 128 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 1 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 2 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 3 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 4 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 5 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 6 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 7 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 8 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 9 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 10 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 11 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 12 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 13 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 14 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 15 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 16 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 17 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 18 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 19 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 20 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 21 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 22 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 23 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 24 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 25 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 26 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 27 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 28 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 29 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | long | 512 | 30 | ERR | no GD output: n and its role in training stability. Compare self-attention to cross-attention."
[GD] Generating 512 tokens...

[GPU] batch-prefill scratch: n=46 dim=1536 ff=12288 (~9 MB)
[error generating response] | | | | | | | | | |
| Geodessical | medium | 32 | 1 | 1 | 1.8 | 81.6 | 177 | 747 | 1656 | 1 | 4070 | 29.3 | 47 | 1203 |
| Geodessical | medium | 32 | 2 | 1 | 1.6 | 74.4 | 191 | 812 | 1734 | 8 | 4070 | 26.8 | 49 | 1219 |
| Geodessical | medium | 32 | 3 | 1 | 1.8 | 85.4 | 170 | 715 | 1636 | 3 | 4070 | 26.6 | 51 | 1160 |
| Geodessical | medium | 32 | 4 | 1 | 1.7 | 80.8 | 177 | 750 | 1643 | 0 | 4070 | 36 | 48 | 1159 |
| Geodessical | medium | 32 | 5 | 1 | 1.8 | 84.3 | 167 | 714 | 1630 | 1 | 4070 | 26.1 | 48 | 1160 |
| Geodessical | medium | 32 | 6 | 1 | 1.8 | 83.5 | 162 | 707 | 1619 | 20 | 4070 | 33 | 54 | 1211 |
| Geodessical | medium | 32 | 7 | 1 | 1.8 | 83 | 178 | 741 | 1782 | 0 | 4070 | 27.2 | 51 | 1150 |
| Geodessical | medium | 32 | 8 | 1 | 1.8 | 84.6 | 167 | 712 | 1650 | 7 | 4070 | 50.5 | 39 | 1232 |
| Geodessical | medium | 32 | 9 | 1 | 1.7 | 79.4 | 176 | 755 | 1701 | 82 | 4070 | 26.4 | 45 | 1215 |
| Geodessical | medium | 32 | 10 | 1 | 1.7 | 80.8 | 178 | 752 | 1645 | 2 | 4070 | 35.2 | 34 | 1215 |
| Geodessical | medium | 32 | 11 | 1 | 1.7 | 73.5 | 170 | 775 | 1692 | 0 | 4070 | 27.4 | 48 | 1232 |
| Geodessical | medium | 32 | 12 | 1 | 1.8 | 83.1 | 172 | 729 | 1653 | 2 | 4070 | 27.9 | 37 | 1150 |
| Geodessical | medium | 32 | 13 | 1 | 1.8 | 84.4 | 177 | 733 | 1622 | 0 | 4070 | 27.6 | 36 | 1157 |
| Geodessical | medium | 32 | 14 | 1 | 1.9 | 86.3 | 165 | 701 | 1629 | 1 | 4070 | 27 | 41 | 1222 |
| Geodessical | medium | 32 | 15 | 1 | 1.8 | 78.8 | 165 | 736 | 1654 | 74 | 4070 | 36.8 | 61 | 1201 |
| Geodessical | medium | 32 | 16 | 1 | 1.8 | 83.5 | 167 | 717 | 1614 | 88 | 4070 | 18.7 | 46 | 1160 |
| Geodessical | medium | 32 | 17 | 1 | 1.8 | 84.5 | 167 | 712 | 1628 | 0 | 4070 | 26.1 | 42 | 1211 |
| Geodessical | medium | 32 | 18 | 1 | 1.8 | 80 | 164 | 727 | 1651 | 0 | 4070 | 36.3 | 44 | 1211 |
| Geodessical | medium | 32 | 19 | 1 | 1.8 | 82.8 | 170 | 727 | 1657 | 2 | 4070 | 26.3 | 37 | 1211 |
| Geodessical | medium | 32 | 20 | 1 | 1.7 | 79 | 168 | 741 | 1644 | 83 | 4070 | 30.6 | 46 | 1215 |
| Geodessical | medium | 32 | 21 | 1 | 1.8 | 82.9 | 182 | 750 | 1736 | 4 | 4070 | 26.5 | 46 | 1177 |
| Geodessical | medium | 32 | 22 | 1 | 1.6 | 72.5 | 192 | 825 | 2057 | 4 | 4071 | 16.7 | 74 | 1211 |
| Geodessical | medium | 32 | 23 | 1 | 1.6 | 72.7 | 171 | 783 | 2100 | 5 | 4071 | 14.4 | 47 | 1232 |
| Geodessical | medium | 32 | 24 | 1 | 1.5 | 69.9 | 201 | 860 | 1921 | 6 | 4071 | 16.7 | 66 | 1217 |
| Geodessical | medium | 32 | 25 | 1 | 1.7 | 80.6 | 179 | 755 | 1724 | 0 | 4071 | 16 | 60 | 1160 |
| Geodessical | medium | 32 | 26 | 1 | 1.6 | 74.1 | 178 | 788 | 1813 | 5 | 4071 | 16.4 | 49 | 1160 |
| Geodessical | medium | 32 | 27 | 1 | 1.6 | 75.1 | 187 | 800 | 1791 | 1 | 4071 | 14 | 54 | 1224 |
| Geodessical | medium | 32 | 28 | 1 | 1.6 | 75.8 | 196 | 814 | 1904 | 0 | 4071 | 27 | 50 | 1221 |
| Geodessical | medium | 32 | 29 | 1 | 1.5 | 69.7 | 191 | 840 | 2049 | 87 | 4079 | 55.6 | 62 | 1166 |
| Geodessical | medium | 32 | 30 | 1 | 1.5 | 69.3 | 186 | 833 | 1822 | 4 | 4080 | 16.8 | 54 | 1215 |
| Geodessical | medium | 128 | 1 | 1 | 0.6 | 82.9 | 170 | 1883 | 3002 | 61 | 4080 | 50.9 | 51 | 1149 |
| Geodessical | medium | 128 | 2 | 1 | 0.6 | 81 | 182 | 1942 | 3181 | 0 | 4080 | 26.6 | 47 | 1195 |
| Geodessical | medium | 128 | 3 | 1 | 0.6 | 82.8 | 207 | 1960 | 3084 | 6 | 4080 | 16.4 | 55 | 1149 |
| Geodessical | medium | 128 | 4 | 1 | 0.5 | 75.8 | 178 | 2045 | 3042 | 4 | 4080 | 28.1 | 51 | 1211 |
| Geodessical | medium | 128 | 5 | 1 | 0.5 | 76.6 | 195 | 2060 | 3215 | 39.5 | 6059 | 39.8 | 69.5 | 2625 |
| Geodessical | medium | 128 | 6 | 1 | 0.5 | 76.9 | 196 | 2057 | 3137 | 88 | 4081 | 63.4 | 71 | 1212 |
| Geodessical | medium | 128 | 7 | 1 | 0.6 | 80.7 | 176 | 1938 | 3007 | 2 | 4081 | 16.6 | 46 | 1219 |
| Geodessical | medium | 128 | 8 | 1 | 0.5 | 76.5 | 194 | 2060 | 3107 | 7 | 4081 | 40.8 | 42 | 1167 |
| Geodessical | medium | 128 | 9 | 1 | 0.5 | 78.1 | 189 | 2017 | 3161 | 44.5 | 6060 | 43.1 | 60.5 | 2625 |
| Geodessical | medium | 128 | 10 | 1 | 0.5 | 77.1 | 199 | 2058 | 3115 | 89 | 4082 | 61.6 | 51 | 1250 |
| Geodessical | medium | 128 | 11 | 1 | 0.5 | 73.8 | 194 | 2122 | 3130 | 8 | 4082 | 16.9 | 54 | 1182 |
| Geodessical | medium | 128 | 12 | 1 | 0.5 | 75.5 | 188 | 2071 | 3128 | 7 | 4082 | 44.5 | 57 | 1160 |
| Geodessical | medium | 128 | 13 | 1 | 0.5 | 74.1 | 192 | 2110 | 3161 | 39 | 4082 | 47.6 | 65 | 1265 |
| Geodessical | medium | 128 | 14 | 1 | 0.5 | 76.2 | 202 | 2082 | 3163 | 40.5 | 6094 | 40.8 | 43 | 2625 |
| Geodessical | medium | 128 | 15 | 1 | 0.5 | 72 | 201 | 2178 | 3344 | 84.5 | 6095 | 56.7 | 54 | 2625 |
| Geodessical | medium | 128 | 16 | 1 | 0.5 | 75.5 | 203 | 2102 | 3184 | 7 | 4116 | 17.8 | 54 | 1250 |
| Geodessical | medium | 128 | 17 | 1 | 0.6 | 80.3 | 180 | 1953 | 2922 | 87 | 4116 | 66.2 | 40 | 1182 |
| Geodessical | medium | 128 | 18 | 1 | 0.6 | 80.6 | 179 | 1946 | 2934 | 4 | 4116 | 56.8 | 50 | 1247 |
| Geodessical | medium | 128 | 19 | 1 | 0.6 | 81.7 | 184 | 1933 | 2853 | 83 | 4116 | 26.5 | 43 | 1160 |
| Geodessical | medium | 128 | 20 | 1 | 0.5 | 71.6 | 178 | 2143 | 3082 | 1 | 4116 | 28.3 | 49 | 1167 |
| Geodessical | medium | 128 | 21 | 1 | 0.4 | 58.6 | 247 | 2678 | 3659 | 59 | 7858 | 40 | 67 | 2625 |
| Geodessical | medium | 128 | 22 | 1 | 0 | 0.5 | 463 | 235018 | 286270 | 87.2 | 7927 | 22.5 | 80.2 | 3825 |
| Geodessical | medium | 128 | 23 | 1 | 0.4 | 54.1 | 289 | 2944 | 4590 | 64.5 | 6938 | 28.9 | 31 | 2625 |
| Geodessical | medium | 128 | 24 | 1 | 0.4 | 60.1 | 291 | 2711 | 4072 | 64 | 7258 | 39.6 | 39 | 2625 |
| Geodessical | medium | 128 | 25 | 1 | 0.4 | 59.6 | 255 | 2657 | 4194 | 57 | 7002 | 39.7 | 26.5 | 2625 |
| Geodessical | medium | 128 | 26 | 1 | 0.3 | 59.3 | 738 | 3634 | 4576 | 54.5 | 7119 | 33.9 | 46.5 | 2625 |
| Geodessical | medium | 128 | 27 | 1 | 0.3 | 34.7 | 184 | 4053 | 5514 | 94.3 | 7387 | 42.8 | 35.7 | 2689 |
| Geodessical | medium | 128 | 28 | 1 | 0.4 | 56.1 | 269 | 2820 | 4222 | 59 | 7322 | 34 | 51 | 2625 |
| Geodessical | medium | 128 | 29 | 1 | 0.4 | 58.5 | 198 | 2580 | 4013 | 50.5 | 7450 | 38.5 | 51.5 | 2654 |
| Geodessical | medium | 128 | 30 | 1 | 0.3 | 40.9 | 187 | 3506 | 4819 | 60 | 7475 | 27.3 | 59.5 | 2689 |
| Geodessical | medium | 512 | 1 | 1 | 0.1 | 51.7 | 287 | 10467 | 11884 | 83.8 | 7363 | 46.5 | 49.2 | 2625 |
| Geodessical | medium | 512 | 2 | 1 | 0.1 | 56 | 279 | 9696 | 11346 | 79.2 | 7235 | 48.8 | 44.3 | 2625 |
| Geodessical | medium | 512 | 3 | 1 | 0.1 | 53.5 | 281 | 10119 | 11742 | 78.5 | 7363 | 44.2 | 51.8 | 2625 |
| Geodessical | medium | 512 | 4 | 1 | 0.1 | 53.7 | 194 | 9903 | 11694 | 79.8 | 7491 | 47.8 | 53.5 | 2689 |
| Geodessical | medium | 512 | 5 | 1 | 0.1 | 54.2 | 289 | 10030 | 11586 | 85 | 7378 | 49.5 | 69 | 2625 |
| Geodessical | medium | 512 | 6 | 1 | 0.1 | 52 | 199 | 10240 | 11804 | 80.5 | 7506 | 46.2 | 52.5 | 2689 |
| Geodessical | medium | 512 | 7 | 1 | 0.1 | 52.4 | 194 | 10152 | 11830 | 84.2 | 7497 | 48.4 | 45 | 2689 |
| Geodessical | medium | 512 | 8 | 1 | 0.1 | 54.4 | 257 | 9911 | 11539 | 82.8 | 7371 | 49.2 | 63.7 | 2625 |
| Geodessical | medium | 512 | 9 | 1 | 0.1 | 55.9 | 175 | 9509 | 11318 | 81.8 | 7533 | 49.7 | 53.8 | 2689 |
| Geodessical | medium | 512 | 10 | 1 | 0.1 | 56 | 188 | 9518 | 11050 | 95.3 | 7533 | 53.8 | 73.8 | 2689 |
| Geodessical | medium | 512 | 11 | 1 | 0.1 | 51.3 | 197 | 10369 | 12137 | 84.7 | 7533 | 46.2 | 75.1 | 2689 |
| Geodessical | medium | 512 | 12 | 1 | 0.1 | 54.6 | 259 | 9885 | 11496 | 80.2 | 7405 | 48.1 | 68.2 | 2625 |
| Geodessical | medium | 512 | 13 | 1 | 0.1 | 55.9 | 242 | 9642 | 11157 | 85 | 7149 | 48.6 | 55.8 | 2625 |
| Geodessical | medium | 512 | 14 | 1 | 0.1 | 56.1 | 176 | 9468 | 11188 | 85 | 7533 | 52.5 | 61 | 2689 |
| Geodessical | medium | 512 | 15 | 1 | 0.1 | 54.1 | 192 | 9851 | 11329 | 82.8 | 7533 | 48 | 60.3 | 2689 |
| Geodessical | medium | 512 | 16 | 1 | 0.1 | 51.6 | 285 | 10482 | 12283 | 80.5 | 7277 | 42.9 | 51.2 | 2625 |
| Geodessical | medium | 512 | 17 | 1 | 0.1 | 50.2 | 277 | 10717 | 12704 | 85.2 | 7533 | 43 | 48.7 | 2654 |
| Geodessical | medium | 512 | 18 | 1 | 0.1 | 52 | 199 | 10246 | 11981 | 88.2 | 7533 | 49.3 | 46.7 | 2689 |
| Geodessical | medium | 512 | 19 | 1 | 0.1 | 51.6 | 266 | 10448 | 12119 | 82.8 | 7277 | 47.4 | 55.3 | 2625 |
| Geodessical | medium | 512 | 20 | 1 | 0.1 | 51.9 | 279 | 10413 | 12041 | 79.2 | 7405 | 46.9 | 34.3 | 2625 |
| Geodessical | medium | 512 | 21 | 1 | 0.1 | 52.7 | 188 | 10088 | 11650 | 92.3 | 7533 | 53 | 27.5 | 2689 |
| Geodessical | medium | 512 | 22 | 1 | 0.1 | 53.1 | 196 | 10035 | 11704 | 79.2 | 7533 | 49.7 | 32.7 | 2689 |
| Geodessical | medium | 512 | 23 | 1 | 0.1 | 50.4 | 228 | 10615 | 12192 | 85 | 7533 | 46.7 | 30.1 | 2689 |
| Geodessical | medium | 512 | 24 | 1 | 0.1 | 52.6 | 267 | 10265 | 11776 | 78.2 | 7277 | 46.8 | 32.8 | 2625 |
| Geodessical | medium | 512 | 25 | 1 | 0.1 | 52.3 | 263 | 10307 | 11936 | 81.3 | 7405 | 48.8 | 29.7 | 2625 |
| Geodessical | medium | 512 | 26 | 1 | 0.1 | 52 | 268 | 10384 | 11877 | 84.3 | 7476 | 46.3 | 48 | 2625 |
| Geodessical | medium | 512 | 27 | 1 | 0.1 | 52.8 | 278 | 10260 | 11722 | 84.2 | 7358 | 46.5 | 37.7 | 2625 |
| Geodessical | medium | 512 | 28 | 1 | 0.1 | 52.4 | 271 | 10307 | 11835 | 84.7 | 7422 | 45.4 | 63.7 | 2625 |
| Geodessical | medium | 512 | 29 | 1 | 0.1 | 52.2 | 215 | 10235 | 11775 | 85.3 | 7550 | 46.6 | 27.8 | 2689 |
| Geodessical | medium | 512 | 30 | 1 | 0.1 | 52.2 | 283 | 10365 | 11727 | 85.8 | 7422 | 43.8 | 33.3 | 2625 |
| Geodessical | short | 32 | 1 | 13 | 49.4 | 103.1 | 127 | 390 | 1525 | 7 | 4063 | 30.2 | 24 | 1163 |
| Geodessical | short | 32 | 2 | 13 | 55.1 | 116.6 | 116 | 352 | 1428 | 85 | 4063 | 33.4 | 29 | 1159 |
| Geodessical | short | 32 | 3 | 13 | 50.6 | 106.6 | 126 | 383 | 1395 | 33 | 4063 | 29.3 | 32 | 1163 |
| Geodessical | short | 32 | 4 | 13 | 52 | 107.7 | 120 | 370 | 1411 | 7 | 4063 | 28.9 | 35 | 1160 |
| Geodessical | short | 32 | 5 | 13 | 54.9 | 117.2 | 117 | 354 | 1344 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 6 | 13 | 54.4 | 118.7 | 121 | 360 | 1357 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 7 | 13 | 50.6 | 105.3 | 124 | 381 | 1386 | 89 | 4063 | 36.1 | 37 | 1160 |
| Geodessical | short | 32 | 8 | 13 | 58.6 | 122 | 107 | 329 | 1356 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 9 | 13 | 58 | 119.7 | 107 | 331 | 1310 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 10 | 13 | 58 | 123.7 | 111 | 335 | 1312 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 11 | 13 | 58 | 123.6 | 110 | 334 | 1324 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 12 | 13 | 58 | 123.7 | 110 | 334 | 1284 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 13 | 13 | 56 | 112 | 107 | 339 | 1323 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 14 | 13 | 54.6 | 112 | 112 | 350 | 1313 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 15 | 13 | 54.6 | 111.7 | 113 | 351 | 1360 | 88 | 4063 | 39.2 | 32 | 1211 |
| Geodessical | short | 32 | 16 | 13 | 55.6 | 112.6 | 109 | 343 | 1382 | 3 | 4063 | 41.4 | 18 | 1160 |
| Geodessical | short | 32 | 17 | 13 | 54.2 | 112.2 | 115 | 355 | 1314 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 18 | 13 | 53.9 | 120.6 | 125 | 366 | 1379 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 19 | 13 | 59.4 | 121.6 | 104 | 323 | 1266 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 20 | 13 | 53.5 | 112 | 118 | 361 | 1320 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 21 | 13 | 53.5 | 111.6 | 118 | 361 | 1310 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 22 | 13 | 56.8 | 123.1 | 115 | 344 | 1351 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 23 | 13 | 58.8 | 122.2 | 106 | 327 | 1301 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 24 | 13 | 59.4 | 123.9 | 106 | 325 | 1310 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 25 | 13 | 58 | 123.5 | 111 | 335 | 1295 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 26 | 13 | 55.6 | 110.8 | 108 | 342 | 1304 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 27 | 13 | 55.6 | 111.1 | 108 | 342 | 1310 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 28 | 13 | 58.8 | 120.7 | 105 | 326 | 1292 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 29 | 13 | 54.4 | 110.2 | 112 | 351 | 1311 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 32 | 30 | 13 | 58.6 | 123.3 | 109 | 331 | 1309 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 1 | 13 | 58.3 | 123.4 | 109 | 332 | 1314 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 2 | 13 | 53.9 | 112.4 | 116 | 357 | 1398 | 85 | 4063 | 55.8 | 21 | 1160 |
| Geodessical | short | 128 | 3 | 13 | 54.6 | 112.2 | 113 | 351 | 1305 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 4 | 13 | 55.6 | 122.1 | 119 | 353 | 1327 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 5 | 13 | 58.8 | 123.7 | 108 | 329 | 1315 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 6 | 13 | 56.5 | 123 | 116 | 346 | 1313 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 7 | 13 | 57.5 | 118.8 | 108 | 334 | 1293 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 8 | 13 | 54.9 | 108.5 | 108 | 345 | 1310 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 9 | 13 | 58.6 | 122.1 | 107 | 329 | 1348 | 88 | 4063 | 59.1 | 28 | 1160 |
| Geodessical | short | 128 | 10 | 13 | 57 | 121 | 112 | 340 | 1327 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 11 | 13 | 58.6 | 123.7 | 109 | 331 | 1316 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 12 | 13 | 58.8 | 122 | 107 | 328 | 1316 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 13 | 13 | 55.6 | 112.6 | 109 | 343 | 1328 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 14 | 13 | 54.9 | 111.6 | 111 | 348 | 1356 | 0 | 4063 | 57.4 | 29 | 1160 |
| Geodessical | short | 128 | 15 | 13 | 57.3 | 123 | 113 | 340 | 1348 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 16 | 13 | 59.1 | 121.9 | 105 | 325 | 1337 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 17 | 13 | 59.1 | 123.4 | 106 | 326 | 1298 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 18 | 13 | 59.1 | 124.1 | 107 | 327 | 1306 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 19 | 13 | 57.5 | 123.7 | 112 | 338 | 1301 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 20 | 13 | 57.5 | 121.7 | 111 | 337 | 1309 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 21 | 13 | 55.1 | 109 | 107 | 343 | 1298 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 22 | 13 | 55.6 | 121.8 | 119 | 353 | 1345 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 23 | 13 | 59.1 | 123.2 | 106 | 326 | 1292 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 24 | 13 | 59.4 | 123.9 | 106 | 325 | 1315 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 25 | 13 | 58.6 | 123.3 | 108 | 330 | 1292 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 26 | 13 | 55.3 | 112.6 | 110 | 345 | 1315 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 27 | 13 | 53.9 | 111.4 | 115 | 356 | 1321 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 28 | 13 | 56 | 110.9 | 106 | 338 | 1305 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 29 | 13 | 54.4 | 107.4 | 109 | 348 | 1350 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 128 | 30 | 13 | 55.6 | 111 | 108 | 342 | 1310 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 1 | 13 | 58.3 | 120.5 | 106 | 329 | 1374 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 2 | 13 | 56 | 112.6 | 107 | 339 | 1320 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 3 | 13 | 56 | 112.3 | 107 | 339 | 1306 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 4 | 13 | 52.8 | 111.5 | 120 | 366 | 1324 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 5 | 13 | 57.8 | 122.5 | 110 | 335 | 1285 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 6 | 13 | 55.8 | 113 | 108 | 341 | 1594 | 91 | 4063 | 29.7 | 20 | 1160 |
| Geodessical | short | 512 | 7 | 13 | 57.8 | 123.4 | 111 | 336 | 1385 | 83 | 4074 | 30.5 | 41 | 1211 |
| Geodessical | short | 512 | 8 | 13 | 59.4 | 123.6 | 106 | 325 | 1297 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 9 | 13 | 56.3 | 119.8 | 114 | 345 | 1454 | 1 | 4063 | 25.9 | 27 | 1074 |
| Geodessical | short | 512 | 10 | 13 | 53.9 | 108.2 | 112 | 353 | 1478 | 0 | 4063 | 26.9 | 48 | 1151 |
| Geodessical | short | 512 | 11 | 13 | 57.8 | 122.7 | 111 | 336 | 1370 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 12 | 13 | 55.1 | 119 | 118 | 354 | 1331 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 13 | 13 | 57.3 | 121.4 | 112 | 339 | 1357 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 14 | 13 | 52.4 | 113.5 | 124 | 372 | 1454 | 0 | 4070 | 56.9 | 43 | 1160 |
| Geodessical | short | 512 | 15 | 13 | 52.4 | 110 | 121 | 369 | 1423 | 0 | 4070 | 26.6 | 35 | 1175 |
| Geodessical | short | 512 | 16 | 13 | 53.9 | 109.1 | 112 | 353 | 1370 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 17 | 13 | 57 | 122.6 | 114 | 342 | 1419 | 81 | 4070 | 63.7 | 28 | 1160 |
| Geodessical | short | 512 | 18 | 13 | 54.6 | 119.3 | 120 | 358 | 1335 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 19 | 13 | 57.8 | 120.3 | 109 | 334 | 1299 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 20 | 13 | 53.3 | 111.5 | 118 | 362 | 1546 | 22 | 4070 | 61.2 | 39 | 872 |
| Geodessical | short | 512 | 21 | 13 | 58.8 | 121.4 | 105 | 326 | 1348 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 22 | 13 | 57.8 | 122.3 | 110 | 335 | 1366 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 23 | 13 | 54.6 | 112.2 | 113 | 351 | 1373 | 99 | 4070 | 31.8 | 20 | 1160 |
| Geodessical | short | 512 | 24 | 13 | 54.6 | 107.6 | 108 | 346 | 1387 | 52 | 4070 | 30.8 | 21 | 1149 |
| Geodessical | short | 512 | 25 | 13 | 54.4 | 112 | 114 | 353 | 1335 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 26 | 13 | 55.3 | 120.4 | 119 | 354 | 1389 | 3 | 4070 | 30.2 | 29 | 1148 |
| Geodessical | short | 512 | 27 | 13 | 52 | 108.6 | 121 | 371 | 1533 | 0 | 4070 | 30.4 | 37 | 1188 |
| Geodessical | short | 512 | 28 | 13 | 54.6 | 111.1 | 111 | 349 | 1342 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 29 | 13 | 58.3 | 122.8 | 109 | 332 | 1318 | 0 | 0 | 0 | 0 | 0 |
| Geodessical | short | 512 | 30 | 13 | 49.4 | 102 | 126 | 389 | 1386 | 0 | 0 | 0 | 0 | 0 |
| HyperTensor | code | 32 | 1 | 32 | 48.6 | 106.2 | 266 | 924 | 1935 | 6 | 5500 | 40.8 | 29 | 1219 |
| HyperTensor | code | 32 | 2 | 32 | 48.5 | 107.6 | 269 | 929 | 1974 | 5 | 5500 | 17.5 | 47 | 1228 |
| HyperTensor | code | 32 | 3 | 32 | 48.1 | 106.1 | 278 | 943 | 1965 | 7 | 5500 | 17 | 37 | 1160 |
| HyperTensor | code | 32 | 4 | 32 | 48.6 | 105.8 | 269 | 928 | 1957 | 87 | 5500 | 48.5 | 36 | 1184 |
| HyperTensor | code | 32 | 5 | 32 | 52.9 | 115.5 | 251 | 856 | 1782 | 0 | 5500 | 16.8 | 23 | 1160 |
| HyperTensor | code | 32 | 6 | 32 | 53.7 | 115.5 | 244 | 840 | 1781 | 84 | 5500 | 62.7 | 21 | 1227 |
| HyperTensor | code | 32 | 7 | 32 | 53.2 | 115 | 247 | 849 | 1769 | 0 | 5500 | 36.9 | 23 | 1168 |
| HyperTensor | code | 32 | 8 | 32 | 53.7 | 115.3 | 244 | 840 | 1745 | 57 | 5500 | 49.9 | 32 | 1160 |
| HyperTensor | code | 32 | 9 | 32 | 52.4 | 108.7 | 245 | 856 | 1767 | 2 | 5500 | 39.8 | 19 | 1221 |
| HyperTensor | code | 32 | 10 | 32 | 53.9 | 115.5 | 243 | 837 | 1733 | 1 | 5500 | 50.2 | 24 | 1160 |
| HyperTensor | code | 32 | 11 | 32 | 53.5 | 115.6 | 246 | 844 | 1743 | 1 | 5500 | 37.8 | 21 | 1214 |
| HyperTensor | code | 32 | 12 | 32 | 52.8 | 115.3 | 253 | 859 | 1797 | 1 | 5500 | 44 | 42 | 1211 |
| HyperTensor | code | 32 | 13 | 32 | 48.1 | 105.8 | 283 | 948 | 1983 | 6 | 5500 | 17.1 | 45 | 1160 |
| HyperTensor | code | 32 | 14 | 32 | 46.9 | 97.6 | 284 | 967 | 1820 | 8 | 5500 | 15.8 | 45 | 1228 |
| HyperTensor | code | 32 | 15 | 32 | 49.4 | 106.5 | 271 | 919 | 1799 | 65 | 5500 | 58.6 | 34 | 1159 |
| HyperTensor | code | 32 | 16 | 32 | 53.2 | 115.4 | 258 | 859 | 1702 | 2 | 5500 | 37.9 | 29 | 1160 |
| HyperTensor | code | 32 | 17 | 32 | 54.2 | 116.3 | 247 | 837 | 1658 | 1 | 5500 | 36 | 34 | 1211 |
| HyperTensor | code | 32 | 18 | 32 | 51.2 | 109.1 | 265 | 890 | 1700 | 0 | 5500 | 37.6 | 34 | 1203 |
| HyperTensor | code | 32 | 19 | 32 | 49.6 | 105.5 | 269 | 914 | 1792 | 88 | 5500 | 40.6 | 35 | 1160 |
| HyperTensor | code | 32 | 20 | 32 | 53.8 | 115.5 | 249 | 844 | 1656 | 16 | 5500 | 34.5 | 35 | 1167 |
| HyperTensor | code | 32 | 21 | 32 | 54.6 | 116.1 | 244 | 830 | 1679 | 88 | 5500 | 39.4 | 32 | 1230 |
| HyperTensor | code | 32 | 22 | 32 | 52.6 | 113.6 | 258 | 866 | 1748 | 99 | 5500 | 36.1 | 50 | 1167 |
| HyperTensor | code | 32 | 23 | 32 | 54 | 116 | 249 | 842 | 1649 | 0 | 5517 | 35.4 | 31 | 1223 |
| HyperTensor | code | 32 | 24 | 32 | 51.7 | 107.7 | 255 | 874 | 1737 | 61 | 5517 | 41.5 | 34 | 1160 |
| HyperTensor | code | 32 | 25 | 32 | 54 | 115.3 | 249 | 842 | 1724 | 3 | 5517 | 37.5 | 38 | 1211 |
| HyperTensor | code | 32 | 26 | 32 | 53.6 | 115.7 | 249 | 846 | 1744 | 0 | 5517 | 17.6 | 37 | 1160 |
| HyperTensor | code | 32 | 27 | 32 | 50.6 | 104.7 | 262 | 894 | 1762 | 34 | 5517 | 27 | 32 | 1182 |
| HyperTensor | code | 32 | 28 | 32 | 54 | 113.5 | 243 | 836 | 1676 | 87 | 5517 | 39.9 | 25 | 1175 |
| HyperTensor | code | 32 | 29 | 32 | 51.1 | 105.7 | 255 | 881 | 1697 | 99 | 5517 | 35.8 | 26 | 1211 |
| HyperTensor | code | 32 | 30 | 32 | 50.8 | 105.4 | 259 | 889 | 1707 | 0 | 5517 | 40.5 | 29 | 1160 |
| HyperTensor | code | 128 | 1 | 128 | 87.1 | 112.2 | 257 | 1727 | 2535 | 81 | 5517 | 61.2 | 30 | 1217 |
| HyperTensor | code | 128 | 2 | 128 | 86.6 | 111.9 | 259 | 1737 | 2562 | 0 | 5517 | 58.5 | 32 | 1211 |
| HyperTensor | code | 128 | 3 | 128 | 90.4 | 115.6 | 244 | 1660 | 2489 | 97 | 5517 | 58.6 | 22 | 1248 |
| HyperTensor | code | 128 | 4 | 128 | 90.5 | 115.6 | 243 | 1658 | 2483 | 0 | 5517 | 58 | 35 | 1160 |
| HyperTensor | code | 128 | 5 | 128 | 89 | 114.6 | 255 | 1693 | 2525 | 89 | 5517 | 64 | 34 | 1228 |
| HyperTensor | code | 128 | 6 | 128 | 89.7 | 115 | 250 | 1677 | 2576 | 88 | 5517 | 55.3 | 42 | 1160 |
| HyperTensor | code | 128 | 7 | 128 | 83.1 | 108.5 | 281 | 1821 | 2725 | 1 | 5518 | 60.5 | 48 | 1160 |
| HyperTensor | code | 128 | 8 | 128 | 83.3 | 108.8 | 281 | 1818 | 2959 | 0 | 5518 | 16.9 | 70 | 1189 |
| HyperTensor | code | 128 | 9 | 128 | 81.8 | 107.6 | 283 | 1847 | 2873 | 89 | 5526 | 62.8 | 57 | 1217 |
| HyperTensor | code | 128 | 10 | 128 | 75.3 | 110.8 | 314 | 2013 | 12327 | 2 | 5526 | 31.3 | 92 | 1237 |
| HyperTensor | code | 128 | 11 | 128 | 85.4 | 109.3 | 263 | 1762 | 3167 | 65 | 7500 | 41.1 | 16 | 2625 |
| HyperTensor | code | 128 | 12 | 128 | 84.3 | 108.4 | 273 | 1792 | 3129 | 34.5 | 7500 | 41 | 26.5 | 2625 |
| HyperTensor | code | 128 | 13 | 128 | 87.6 | 112.1 | 255 | 1716 | 2513 | 88 | 5526 | 57.9 | 33 | 1211 |
| HyperTensor | code | 128 | 14 | 128 | 87.4 | 111.9 | 252 | 1716 | 2528 | 88 | 5526 | 60.9 | 33 | 1248 |
| HyperTensor | code | 128 | 15 | 128 | 89.7 | 115.3 | 247 | 1674 | 2544 | 57 | 5526 | 74 | 36 | 1160 |
| HyperTensor | code | 128 | 16 | 128 | 90.6 | 115.9 | 239 | 1652 | 2479 | 87 | 5526 | 64.2 | 28 | 1160 |
| HyperTensor | code | 128 | 17 | 128 | 90.5 | 116 | 244 | 1658 | 2457 | 84 | 5526 | 59 | 30 | 1156 |
| HyperTensor | code | 128 | 18 | 128 | 87.3 | 112.1 | 255 | 1721 | 2524 | 0 | 5526 | 57.8 | 32 | 1160 |
| HyperTensor | code | 128 | 19 | 128 | 87.4 | 112.1 | 257 | 1721 | 2528 | 0 | 5526 | 61.1 | 24 | 1156 |
| HyperTensor | code | 128 | 20 | 128 | 90.9 | 116 | 239 | 1647 | 2462 | 88 | 5526 | 56.7 | 34 | 1160 |
| HyperTensor | code | 128 | 21 | 128 | 87.6 | 112 | 250 | 1711 | 2543 | 86 | 5526 | 61.7 | 27 | 1211 |
| HyperTensor | code | 128 | 22 | 128 | 87.5 | 111.7 | 252 | 1715 | 2536 | 0 | 5526 | 58.8 | 23 | 1162 |
| HyperTensor | code | 128 | 23 | 128 | 87.3 | 112.3 | 257 | 1723 | 2544 | 0 | 5526 | 61.5 | 29 | 1166 |
| HyperTensor | code | 128 | 24 | 128 | 89.4 | 115.5 | 250 | 1681 | 2519 | 0 | 5526 | 61.2 | 34 | 1152 |
| HyperTensor | code | 128 | 25 | 128 | 90.6 | 115.9 | 242 | 1655 | 2490 | 0 | 5526 | 57.9 | 24 | 1160 |
| HyperTensor | code | 128 | 26 | 128 | 90 | 115.6 | 242 | 1664 | 2475 | 1 | 5526 | 67.2 | 27 | 1160 |
| HyperTensor | code | 128 | 27 | 128 | 90.3 | 115.8 | 241 | 1658 | 2482 | 89 | 5526 | 57.9 | 34 | 1219 |
| HyperTensor | code | 128 | 28 | 128 | 90.5 | 115.9 | 241 | 1655 | 2459 | 10 | 5526 | 62.3 | 29 | 1223 |
| HyperTensor | code | 128 | 29 | 128 | 87.4 | 112 | 254 | 1718 | 2515 | 1 | 5526 | 60.3 | 37 | 1219 |
| HyperTensor | code | 128 | 30 | 128 | 87.2 | 111.9 | 254 | 1722 | 2548 | 87 | 5526 | 58.2 | 30 | 1160 |
| HyperTensor | code | 512 | 1 | 512 | 95.7 | 101.6 | 242 | 5592 | 6403 | 74.2 | 7500 | 62.2 | 16.8 | 2625 |
| HyperTensor | code | 512 | 2 | 512 | 94.5 | 100.8 | 262 | 5681 | 6528 | 74.5 | 7500 | 66.9 | 20.8 | 2625 |
| HyperTensor | code | 512 | 3 | 512 | 94.7 | 100.6 | 251 | 5656 | 6484 | 79.5 | 7500 | 67.8 | 19 | 2625 |
| HyperTensor | code | 512 | 4 | 512 | 92.7 | 98.9 | 268 | 5789 | 6714 | 74 | 7500 | 60.7 | 23.5 | 2625 |
| HyperTensor | code | 512 | 5 | 512 | 92 | 97.5 | 241 | 5805 | 6882 | 76 | 7500 | 59.3 | 24.5 | 2625 |
| HyperTensor | code | 512 | 6 | 512 | 91 | 96.8 | 254 | 5878 | 6890 | 75.2 | 7500 | 58.7 | 29 | 2625 |
| HyperTensor | code | 512 | 7 | 512 | 87.9 | 93.4 | 269 | 6094 | 7007 | 95.5 | 7500 | 64.2 | 36.2 | 2625 |
| HyperTensor | code | 512 | 8 | 512 | 90 | 95.9 | 268 | 5956 | 6836 | 75.2 | 7500 | 59.2 | 32.2 | 2625 |
| HyperTensor | code | 512 | 9 | 512 | 91.6 | 97.2 | 253 | 5844 | 6740 | 84.5 | 7500 | 65.1 | 29 | 2625 |
| HyperTensor | code | 512 | 10 | 512 | 92.9 | 98.5 | 248 | 5759 | 6699 | 86.8 | 7500 | 67.1 | 26.8 | 2625 |
| HyperTensor | code | 512 | 11 | 512 | 90.9 | 96.9 | 273 | 5904 | 6777 | 96 | 7500 | 65.6 | 29 | 2625 |
| HyperTensor | code | 512 | 12 | 512 | 90.4 | 95.8 | 251 | 5915 | 6755 | 74.5 | 7500 | 61.5 | 29 | 2625 |
| HyperTensor | code | 512 | 13 | 512 | 87 | 92.4 | 275 | 6162 | 7033 | 75 | 7500 | 59.9 | 33.8 | 2625 |
| HyperTensor | code | 512 | 14 | 512 | 87.8 | 93.2 | 271 | 6102 | 6968 | 68.8 | 7500 | 68.2 | 39 | 2625 |
| HyperTensor | code | 512 | 15 | 512 | 88.9 | 94.3 | 261 | 6018 | 6943 | 73.5 | 7503 | 61.3 | 47.2 | 2625 |
| HyperTensor | code | 512 | 16 | 512 | 93.2 | 98.9 | 248 | 5741 | 6618 | 92.2 | 7503 | 72.7 | 24.5 | 2625 |
| HyperTensor | code | 512 | 17 | 512 | 88 | 93.4 | 268 | 6089 | 6967 | 74.5 | 7503 | 62.4 | 30.8 | 2625 |
| HyperTensor | code | 512 | 18 | 512 | 87.6 | 93.2 | 267 | 6109 | 7079 | 75 | 7503 | 67.6 | 32.5 | 2625 |
| HyperTensor | code | 512 | 19 | 512 | 89.9 | 95.9 | 282 | 5977 | 6862 | 75.8 | 7503 | 60.6 | 38 | 2625 |
| HyperTensor | code | 512 | 20 | 512 | 88.7 | 94.2 | 266 | 6040 | 6938 | 96 | 7517 | 67.9 | 30.5 | 2625 |
| HyperTensor | code | 512 | 21 | 512 | 87.4 | 93.3 | 289 | 6147 | 7117 | 74 | 7517 | 51.3 | 33.8 | 2625 |
| HyperTensor | code | 512 | 22 | 512 | 87.8 | 93.4 | 274 | 6108 | 6993 | 87.8 | 7517 | 64.1 | 31 | 2625 |
| HyperTensor | code | 512 | 23 | 512 | 90.5 | 96.3 | 274 | 5932 | 6856 | 95.5 | 7517 | 67 | 27.8 | 2625 |
| HyperTensor | code | 512 | 24 | 512 | 91.8 | 97.8 | 267 | 5846 | 6698 | 96.8 | 7517 | 62.3 | 23.8 | 2625 |
| HyperTensor | code | 512 | 25 | 512 | 93.4 | 99.1 | 242 | 5725 | 6600 | 93 | 7517 | 64 | 21 | 2625 |
| HyperTensor | code | 512 | 26 | 512 | 91.4 | 96.8 | 251 | 5852 | 6695 | 95.2 | 7517 | 67.6 | 21.8 | 2625 |
| HyperTensor | code | 512 | 27 | 512 | 90.5 | 96.7 | 282 | 5940 | 6854 | 75.2 | 7517 | 66.1 | 30.2 | 2625 |
| HyperTensor | code | 512 | 28 | 512 | 87.9 | 93.4 | 267 | 6093 | 6995 | 75.8 | 7517 | 55.7 | 30.8 | 2625 |
| HyperTensor | code | 512 | 29 | 512 | 87.3 | 93 | 285 | 6153 | 7019 | 75.2 | 7517 | 59.3 | 26 | 2625 |
| HyperTensor | code | 512 | 30 | 512 | 93.2 | 99.2 | 266 | 5762 | 6661 | 75.8 | 7517 | 62 | 31.2 | 2625 |
| HyperTensor | long | 32 | 1 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 2 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 3 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 4 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 5 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 6 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 7 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 8 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 9 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 10 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 11 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 12 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 13 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 14 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 15 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 16 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 17 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 18 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 19 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 20 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 21 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 22 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 23 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 24 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 25 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 26 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 27 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 28 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 29 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 32 | 30 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 1 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 2 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 3 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 4 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 5 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 6 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 7 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 8 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 9 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 10 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 11 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 12 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 13 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 14 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 15 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 16 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 17 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 18 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 19 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 20 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 21 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 22 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 23 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 24 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 25 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 26 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 27 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 28 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 29 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 128 | 30 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 1 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 2 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 3 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 4 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 5 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 6 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 7 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 8 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 9 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 10 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 11 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 12 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 13 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 14 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 15 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 16 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 17 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 18 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 19 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 20 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 21 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 22 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 23 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 24 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 25 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 26 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 27 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 28 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 29 | ERR | no HT output | | | | | | | | | |
| HyperTensor | long | 512 | 30 | ERR | no HT output | | | | | | | | | |
| HyperTensor | medium | 32 | 1 | 1 | 2.9 | 128.6 | 265 | 610 | 1448 | 53 | 4070 | 33.9 | 35 | 1232 |
| HyperTensor | medium | 32 | 2 | 1 | 3 | 136.3 | 256 | 593 | 1427 | 2 | 4070 | 25.1 | 35 | 1250 |
| HyperTensor | medium | 32 | 3 | 1 | 3 | 130.4 | 249 | 583 | 1429 | 65 | 4070 | 34.2 | 28 | 1160 |
| HyperTensor | medium | 32 | 4 | 1 | 3.1 | 127.5 | 245 | 571 | 1437 | 0 | 4070 | 34.8 | 26 | 1211 |
| HyperTensor | medium | 32 | 5 | 1 | 3 | 135.3 | 243 | 572 | 1419 | 0 | 4070 | 36.6 | 35 | 1160 |
| HyperTensor | medium | 32 | 6 | 1 | 3 | 140.3 | 246 | 574 | 1426 | 22 | 4070 | 27.9 | 33 | 1160 |
| HyperTensor | medium | 32 | 7 | 1 | 3 | 145.1 | 240 | 568 | 1420 | 0 | 4070 | 36.6 | 32 | 1211 |
| HyperTensor | medium | 32 | 8 | 1 | 3.1 | 145.4 | 237 | 559 | 1417 | 87 | 4070 | 37.4 | 25 | 1211 |
| HyperTensor | medium | 32 | 9 | 1 | 3.1 | 151.7 | 235 | 559 | 1451 | 35 | 4070 | 28.6 | 29 | 1159 |
| HyperTensor | medium | 32 | 10 | 1 | 3.1 | 144.3 | 231 | 549 | 1388 | 1 | 4070 | 34.7 | 43 | 1160 |
| HyperTensor | medium | 32 | 11 | 1 | 3.1 | 146.4 | 238 | 557 | 1420 | 88 | 4070 | 34.8 | 33 | 1215 |
| HyperTensor | medium | 32 | 12 | 1 | 3.1 | 150.4 | 243 | 567 | 1428 | 1 | 4070 | 36.3 | 33 | 1196 |
| HyperTensor | medium | 32 | 13 | 1 | 3.1 | 154.3 | 232 | 553 | 1385 | 16 | 4070 | 30.2 | 26 | 1221 |
| HyperTensor | medium | 32 | 14 | 1 | 3.1 | 145.3 | 238 | 561 | 1404 | 0 | 4070 | 36.6 | 28 | 1160 |
| HyperTensor | medium | 32 | 15 | 1 | 3.1 | 142.4 | 240 | 562 | 1386 | 1 | 4070 | 34.5 | 27 | 1217 |
| HyperTensor | medium | 32 | 16 | 1 | 3.2 | 154.9 | 237 | 553 | 1415 | 1 | 4070 | 36.1 | 24 | 1219 |
| HyperTensor | medium | 32 | 17 | 1 | 3 | 142.9 | 245 | 575 | 1410 | 0 | 4070 | 30.2 | 26 | 1162 |
| HyperTensor | medium | 32 | 18 | 1 | 3 | 145.1 | 247 | 579 | 1407 | 88 | 4070 | 32.7 | 31 | 1219 |
| HyperTensor | medium | 32 | 19 | 1 | 3.1 | 149.7 | 236 | 556 | 1430 | 15 | 4070 | 34.8 | 32 | 1072 |
| HyperTensor | medium | 32 | 20 | 1 | 3 | 130.5 | 252 | 587 | 1449 | 0 | 4070 | 34.4 | 27 | 1160 |
| HyperTensor | medium | 32 | 21 | 1 | 2.5 | 123.5 | 289 | 694 | 1914 | 1 | 4070 | 16.7 | 56 | 1160 |
| HyperTensor | medium | 32 | 22 | 1 | 2.8 | 129.1 | 260 | 616 | 1592 | 6 | 4070 | 16.3 | 43 | 1217 |
| HyperTensor | medium | 32 | 23 | 1 | 2.7 | 106.1 | 280 | 656 | 1710 | 1 | 4071 | 16.2 | 65 | 1165 |
| HyperTensor | medium | 32 | 24 | 1 | 2.9 | 144.7 | 255 | 602 | 1545 | 7 | 4071 | 16.9 | 39 | 1215 |
| HyperTensor | medium | 32 | 25 | 1 | 3 | 132.7 | 251 | 585 | 1492 | 0 | 4071 | 16.2 | 43 | 1211 |
| HyperTensor | medium | 32 | 26 | 1 | 2.8 | 128.3 | 268 | 628 | 1740 | 7 | 4071 | 16.5 | 59 | 1254 |
| HyperTensor | medium | 32 | 27 | 1 | 2.7 | 124.8 | 275 | 642 | 1626 | 5 | 4071 | 16.3 | 43 | 1252 |
| HyperTensor | medium | 32 | 28 | 1 | 2.7 | 138.3 | 276 | 644 | 1505 | 7 | 4071 | 16.6 | 37 | 1211 |
| HyperTensor | medium | 32 | 29 | 1 | 2.9 | 143.4 | 251 | 595 | 1565 | 1 | 4079 | 16.5 | 26 | 1223 |
| HyperTensor | medium | 32 | 30 | 1 | 3.1 | 148.5 | 240 | 563 | 1522 | 3 | 4080 | 16.8 | 48 | 1215 |
| HyperTensor | medium | 128 | 1 | 1 | 2.9 | 136.7 | 254 | 596 | 1542 | 88 | 4080 | 64.3 | 47 | 1230 |
| HyperTensor | medium | 128 | 2 | 1 | 2.9 | 111.1 | 256 | 605 | 1633 | 5 | 4080 | 16.4 | 58 | 1217 |
| HyperTensor | medium | 128 | 3 | 1 | 3 | 129.8 | 246 | 579 | 1514 | 5 | 4080 | 37.2 | 35 | 1149 |
| HyperTensor | medium | 128 | 4 | 1 | 3.1 | 139.3 | 241 | 564 | 1489 | 1 | 4080 | 14.5 | 43 | 1149 |
| HyperTensor | medium | 128 | 5 | 1 | 2.5 | 109.8 | 293 | 697 | 2112 | 5 | 4081 | 19.6 | 60 | 1216 |
| HyperTensor | medium | 128 | 6 | 1 | 2.5 | 119 | 254 | 652 | 1855 | 87 | 4081 | 61.5 | 65 | 1217 |
| HyperTensor | medium | 128 | 7 | 1 | 2.6 | 114.2 | 280 | 662 | 1747 | 89 | 4081 | 62.1 | 49 | 1160 |
| HyperTensor | medium | 128 | 8 | 1 | 2.6 | 128.4 | 279 | 670 | 1729 | 7 | 4081 | 19.4 | 37 | 1160 |
| HyperTensor | medium | 128 | 9 | 1 | 2.7 | 136 | 269 | 636 | 1553 | 6 | 4081 | 16.9 | 38 | 1222 |
| HyperTensor | medium | 128 | 10 | 1 | 2.8 | 126.3 | 264 | 625 | 1610 | 0 | 4082 | 16.4 | 46 | 1211 |
| HyperTensor | medium | 128 | 11 | 1 | 2.7 | 140.1 | 272 | 640 | 1608 | 3 | 4082 | 32.5 | 45 | 1215 |
| HyperTensor | medium | 128 | 12 | 1 | 2.7 | 123.3 | 263 | 631 | 1646 | 22 | 4090 | 44.6 | 38 | 1290 |
| HyperTensor | medium | 128 | 13 | 1 | 2.7 | 127.1 | 267 | 637 | 1689 | 14 | 4082 | 40.6 | 51 | 1211 |
| HyperTensor | medium | 128 | 14 | 1 | 2.8 | 108.9 | 258 | 619 | 1652 | 8 | 4082 | 17 | 38 | 1166 |
| HyperTensor | medium | 128 | 15 | 1 | 3 | 125.1 | 249 | 584 | 1517 | 7 | 4116 | 18 | 37 | 1211 |
| HyperTensor | medium | 128 | 16 | 1 | 2.8 | 130.9 | 269 | 631 | 1655 | 17 | 4116 | 17.9 | 47 | 1239 |
| HyperTensor | medium | 128 | 17 | 1 | 2.9 | 141.1 | 261 | 606 | 1468 | 39 | 4116 | 42.4 | 39 | 1232 |
| HyperTensor | medium | 128 | 18 | 1 | 3.1 | 149.2 | 236 | 555 | 1472 | 87 | 4116 | 57.8 | 34 | 1160 |
| HyperTensor | medium | 128 | 19 | 1 | 3.2 | 150.4 | 234 | 550 | 1414 | 0 | 0 | 0 | 0 | 0 |
| HyperTensor | medium | 128 | 20 | 1 | 2.3 | 105.2 | 341 | 777 | 1655 | 27 | 5880 | 27.6 | 69 | 1250 |
| HyperTensor | medium | 128 | 21 | 1 | 2.2 | 96.4 | 362 | 823 | 1674 | 46 | 5880 | 53.5 | 65 | 1227 |
| HyperTensor | medium | 128 | 22 | 1 | 1.1 | 51.8 | 582 | 1472 | 8212 | 42.3 | 7916 | 29.2 | 91.3 | 2271 |
| HyperTensor | medium | 128 | 23 | 1 | 2 | 100.4 | 390 | 898 | 3452 | 24 | 6662 | 12.8 | 30 | 2252 |
| HyperTensor | medium | 128 | 24 | 1 | 2.1 | 90.1 | 356 | 832 | 2464 | 25 | 4959 | 25 | 48 | 1225 |
| HyperTensor | medium | 128 | 25 | 1 | 2.7 | 89.2 | 271 | 643 | 2167 | 20 | 5599 | 24.3 | 45 | 1201 |
| HyperTensor | medium | 128 | 26 | 1 | 3 | 119.9 | 242 | 573 | 2074 | 68 | 5663 | 47 | 57 | 1160 |
| HyperTensor | medium | 128 | 27 | 1 | 1.8 | 79.2 | 392 | 935 | 2255 | 23 | 5415 | 24.8 | 74 | 1227 |
| HyperTensor | medium | 128 | 28 | 1 | 2.7 | 77 | 270 | 644 | 2357 | 31 | 5663 | 28.1 | 38 | 1087 |
| HyperTensor | medium | 128 | 29 | 1 | 2.2 | 99.7 | 368 | 830 | 2228 | 25 | 5471 | 27.4 | 68 | 1280 |
| HyperTensor | medium | 128 | 30 | 1 | 2.2 | 96.6 | 345 | 808 | 2070 | 17 | 5362 | 33.2 | 70 | 1227 |
| HyperTensor | medium | 512 | 1 | 1 | 1 | 81.3 | 866 | 1823 | 2587 | 55 | 5699 | 36.1 | 76 | 1239 |
| HyperTensor | medium | 512 | 2 | 1 | 1.9 | 103.4 | 377 | 897 | 2142 | 26 | 5256 | 25 | 44 | 1232 |
| HyperTensor | medium | 512 | 3 | 1 | 1.6 | 6.5 | 243 | 865 | 2380 | 26 | 5704 | 28 | 40 | 1146 |
| HyperTensor | medium | 512 | 4 | 1 | 2.6 | 82.3 | 276 | 655 | 2249 | 25 | 5719 | 24.2 | 47 | 1225 |
| HyperTensor | medium | 512 | 5 | 1 | 1.9 | 92.4 | 387 | 910 | 2251 | 23 | 5399 | 28 | 76 | 1211 |
| HyperTensor | medium | 512 | 6 | 1 | 2.1 | 100.7 | 372 | 857 | 2451 | 24 | 5527 | 25.1 | 68 | 1248 |
| HyperTensor | medium | 512 | 7 | 1 | 0.2 | 121.5 | 4790 | 9685 | 6699 | 77.7 | 7496 | 33.2 | 50.7 | 2683 |
| HyperTensor | medium | 512 | 8 | 1 | 2.1 | 99.3 | 353 | 829 | 2302 | 19 | 5392 | 28 | 74 | 1162 |
| HyperTensor | medium | 512 | 9 | 1 | 2.5 | 63.6 | 267 | 660 | 2089 | 24 | 5730 | 47.7 | 70 | 1230 |
| HyperTensor | medium | 512 | 10 | 1 | 2.3 | 103.6 | 333 | 776 | 2380 | 17 | 5554 | 24.3 | 51 | 1262 |
| HyperTensor | medium | 512 | 11 | 1 | 2 | 84.9 | 355 | 852 | 2238 | 16 | 5554 | 21.9 | 70 | 1160 |
| HyperTensor | medium | 512 | 12 | 1 | 3 | 106.5 | 247 | 584 | 2000 | 29 | 5746 | 39.2 | 69 | 1160 |
| HyperTensor | medium | 512 | 13 | 1 | 1.9 | 81.6 | 374 | 902 | 2522 | 20 | 5426 | 22.9 | 60 | 1323 |
| HyperTensor | medium | 512 | 14 | 1 | 2.1 | 92.9 | 354 | 823 | 2336 | 17 | 5554 | 23.7 | 52 | 1215 |
| HyperTensor | medium | 512 | 15 | 1 | 2.7 | 92.2 | 271 | 641 | 2273 | 18 | 5746 | 31.8 | 37 | 1213 |
| HyperTensor | medium | 512 | 16 | 1 | 2 | 76.5 | 377 | 886 | 2286 | 25 | 5562 | 28.3 | 44 | 1307 |
| HyperTensor | medium | 512 | 17 | 1 | 2 | 84.5 | 377 | 883 | 2586 | 25 | 5554 | 23.5 | 57 | 1253 |
| HyperTensor | medium | 512 | 18 | 1 | 1.9 | 84.9 | 383 | 898 | 2448 | 23 | 5562 | 25.7 | 54 | 1371 |
| HyperTensor | medium | 512 | 19 | 1 | 1.9 | 97.9 | 389 | 923 | 2577 | 17 | 5298 | 21.9 | 63 | 1232 |
| HyperTensor | medium | 512 | 20 | 1 | 2.4 | 87 | 312 | 737 | 2323 | 26 | 5746 | 23.9 | 57 | 1253 |
| HyperTensor | medium | 512 | 21 | 1 | 2.6 | 71.2 | 278 | 669 | 2209 | 21 | 5746 | 25.2 | 39 | 1284 |
| HyperTensor | medium | 512 | 22 | 1 | 2.1 | 90.3 | 373 | 854 | 2426 | 24 | 5562 | 24.6 | 32 | 1308 |
| HyperTensor | medium | 512 | 23 | 1 | 2 | 91.7 | 383 | 877 | 2335 | 24 | 5554 | 26 | 42 | 1219 |
| HyperTensor | medium | 512 | 24 | 1 | 2.4 | 82.8 | 295 | 705 | 2294 | 22 | 5746 | 26.1 | 51 | 1215 |
| HyperTensor | medium | 512 | 25 | 1 | 1.9 | 96.3 | 382 | 898 | 2272 | 30 | 5426 | 26.3 | 40 | 1232 |
| HyperTensor | medium | 512 | 26 | 1 | 2.4 | 95.3 | 316 | 739 | 2161 | 15 | 5746 | 26.2 | 52 | 1160 |
| HyperTensor | medium | 512 | 27 | 1 | 2.5 | 74.9 | 290 | 685 | 2040 | 36 | 5764 | 33.8 | 63 | 1237 |
| HyperTensor | medium | 512 | 28 | 1 | 2.1 | 102.1 | 363 | 840 | 2397 | 27 | 5444 | 26.7 | 34 | 1160 |
| HyperTensor | medium | 512 | 29 | 1 | 2.4 | 50.5 | 293 | 709 | 2235 | 20 | 5772 | 17.6 | 25 | 1352 |
| HyperTensor | medium | 512 | 30 | 1 | 2.5 | 77.9 | 292 | 694 | 2325 | 17 | 5764 | 21.5 | 46 | 1211 |
| HyperTensor | short | 32 | 1 | 14 | 36.2 | 111.6 | 170 | 557 | 1710 | 7 | 4063 | 16.4 | 34 | 1149 |
| HyperTensor | short | 32 | 2 | 14 | 37.1 | 112.4 | 167 | 544 | 1516 | 3 | 4063 | 26.4 | 36 | 1221 |
| HyperTensor | short | 32 | 3 | 14 | 37.8 | 113.8 | 167 | 537 | 1470 | 3 | 4063 | 26.1 | 33 | 1217 |
| HyperTensor | short | 32 | 4 | 14 | 35.3 | 102.5 | 176 | 573 | 1544 | 3 | 4063 | 40.6 | 30 | 1160 |
| HyperTensor | short | 32 | 5 | 14 | 38.8 | 116.6 | 163 | 524 | 1537 | 91 | 4063 | 33.3 | 27 | 1160 |
| HyperTensor | short | 32 | 6 | 14 | 37.9 | 116.9 | 160 | 529 | 1496 | 51 | 4063 | 26.5 | 34 | 1211 |
| HyperTensor | short | 32 | 7 | 14 | 34.7 | 102.5 | 181 | 585 | 1565 | 7 | 4063 | 25.7 | 37 | 1157 |
| HyperTensor | short | 32 | 8 | 14 | 37.4 | 112.8 | 167 | 541 | 1499 | 69 | 4063 | 33.9 | 29 | 1165 |
| HyperTensor | short | 32 | 9 | 14 | 40.7 | 123.4 | 155 | 499 | 1434 | 76 | 4063 | 29.1 | 21 | 1166 |
| HyperTensor | short | 32 | 10 | 14 | 40.9 | 122.9 | 156 | 498 | 1405 | 9 | 4063 | 26.2 | 23 | 1217 |
| HyperTensor | short | 32 | 11 | 14 | 37.6 | 112.1 | 165 | 537 | 1447 | 85 | 4063 | 27.4 | 34 | 1160 |
| HyperTensor | short | 32 | 12 | 14 | 38.3 | 110 | 160 | 526 | 1427 | 1 | 4063 | 26.2 | 29 | 1217 |
| HyperTensor | short | 32 | 13 | 14 | 40.9 | 123.4 | 154 | 496 | 1440 | 88 | 4063 | 36.9 | 27 | 1186 |
| HyperTensor | short | 32 | 14 | 14 | 40.1 | 122.7 | 152 | 501 | 1429 | 0 | 4063 | 25.9 | 21 | 1179 |
| HyperTensor | short | 32 | 15 | 14 | 38.4 | 110.4 | 158 | 523 | 1511 | 0 | 4063 | 25.3 | 26 | 1176 |
| HyperTensor | short | 32 | 16 | 14 | 41.2 | 123.6 | 152 | 492 | 1486 | 0 | 4063 | 24.8 | 16 | 1199 |
| HyperTensor | short | 32 | 17 | 14 | 39.3 | 123.4 | 150 | 506 | 1480 | 0 | 4063 | 30.2 | 25 | 1160 |
| HyperTensor | short | 32 | 18 | 14 | 37.4 | 110.8 | 165 | 539 | 1469 | 14 | 4063 | 26.6 | 28 | 1211 |
| HyperTensor | short | 32 | 19 | 14 | 41.3 | 121.1 | 151 | 490 | 1415 | 70 | 4063 | 30.2 | 20 | 1160 |
| HyperTensor | short | 32 | 20 | 14 | 40.7 | 122.8 | 155 | 499 | 1414 | 95 | 4063 | 23.9 | 29 | 1211 |
| HyperTensor | short | 32 | 21 | 14 | 41.1 | 123.4 | 152 | 493 | 1424 | 0 | 4063 | 25 | 32 | 1160 |
| HyperTensor | short | 32 | 22 | 14 | 38.7 | 112.3 | 159 | 521 | 1459 | 88 | 4063 | 39.8 | 26 | 1215 |
| HyperTensor | short | 32 | 23 | 14 | 40.8 | 120.6 | 154 | 497 | 1444 | 87 | 4063 | 33.1 | 35 | 1227 |
| HyperTensor | short | 32 | 24 | 14 | 38.3 | 111.7 | 159 | 525 | 1451 | 1 | 4063 | 26.3 | 27 | 1160 |
| HyperTensor | short | 32 | 25 | 14 | 38.4 | 110.6 | 161 | 526 | 1429 | 88 | 4063 | 36.1 | 26 | 1176 |
| HyperTensor | short | 32 | 26 | 14 | 41.1 | 123.6 | 150 | 491 | 1427 | 1 | 4063 | 25.4 | 27 | 1160 |
| HyperTensor | short | 32 | 27 | 14 | 40.8 | 121.6 | 151 | 494 | 1433 | 1 | 4063 | 25.5 | 20 | 1201 |
| HyperTensor | short | 32 | 28 | 14 | 40.6 | 121.9 | 156 | 501 | 1417 | 1 | 4063 | 26.5 | 28 | 1211 |
| HyperTensor | short | 32 | 29 | 14 | 40.8 | 123.4 | 153 | 496 | 1423 | 0 | 4063 | 38.5 | 25 | 1211 |
| HyperTensor | short | 32 | 30 | 14 | 40.8 | 121.5 | 156 | 499 | 1435 | 76 | 4063 | 26.7 | 17 | 1160 |
| HyperTensor | short | 128 | 1 | 14 | 41.1 | 123.3 | 151 | 492 | 1411 | 88 | 4063 | 54.1 | 16 | 1166 |
| HyperTensor | short | 128 | 2 | 14 | 40.8 | 123.3 | 149 | 492 | 1443 | 78 | 4063 | 26 | 26 | 1202 |
| HyperTensor | short | 128 | 3 | 14 | 38.5 | 110.6 | 163 | 527 | 1474 | 65 | 4063 | 26.8 | 26 | 1211 |
| HyperTensor | short | 128 | 4 | 14 | 39.3 | 112.4 | 156 | 512 | 1483 | 57 | 4063 | 26.6 | 18 | 1160 |
| HyperTensor | short | 128 | 5 | 14 | 40.9 | 123.1 | 150 | 492 | 1465 | 89 | 4063 | 53.6 | 25 | 1160 |
| HyperTensor | short | 128 | 6 | 14 | 40.3 | 121.1 | 151 | 498 | 1418 | 88 | 4063 | 54.4 | 19 | 1175 |
| HyperTensor | short | 128 | 7 | 14 | 40.5 | 123.2 | 154 | 500 | 1426 | 65 | 4063 | 26.8 | 18 | 1170 |
| HyperTensor | short | 128 | 8 | 14 | 40.1 | 122.1 | 157 | 506 | 1429 | 0 | 4063 | 26.6 | 16 | 1211 |
| HyperTensor | short | 128 | 9 | 14 | 38.8 | 109.7 | 159 | 520 | 1511 | 88 | 4063 | 63.6 | 29 | 1160 |
| HyperTensor | short | 128 | 10 | 14 | 40.6 | 119.9 | 156 | 501 | 1432 | 1 | 4063 | 26.7 | 29 | 1211 |
| HyperTensor | short | 128 | 11 | 14 | 40.6 | 117.4 | 151 | 496 | 1410 | 1 | 4063 | 30.4 | 28 | 1225 |
| HyperTensor | short | 128 | 12 | 14 | 40.3 | 121.8 | 152 | 499 | 1421 | 1 | 4063 | 26.5 | 23 | 1184 |
| HyperTensor | short | 128 | 13 | 14 | 38.6 | 111.2 | 162 | 525 | 1441 | 86 | 4063 | 22.6 | 30 | 1211 |
| HyperTensor | short | 128 | 14 | 14 | 41.3 | 122 | 149 | 488 | 1428 | 1 | 4063 | 57 | 32 | 1249 |
| HyperTensor | short | 128 | 15 | 14 | 40.3 | 122.6 | 154 | 501 | 1618 | 86 | 4063 | 54.9 | 38 | 1073 |
| HyperTensor | short | 128 | 16 | 14 | 40.1 | 119.8 | 153 | 502 | 1493 | 1 | 4063 | 30.4 | 35 | 1149 |
| HyperTensor | short | 128 | 17 | 14 | 39 | 111.1 | 158 | 517 | 1429 | 87 | 4063 | 63.9 | 22 | 1211 |
| HyperTensor | short | 128 | 18 | 14 | 39.1 | 110.8 | 154 | 512 | 1435 | 2 | 4063 | 59.8 | 21 | 1243 |
| HyperTensor | short | 128 | 19 | 14 | 38.4 | 112.4 | 154 | 519 | 1459 | 0 | 4063 | 26.4 | 21 | 1221 |
| HyperTensor | short | 128 | 20 | 14 | 39.9 | 123.6 | 154 | 505 | 1416 | 44 | 4063 | 26.8 | 26 | 1160 |
| HyperTensor | short | 128 | 21 | 14 | 41.7 | 121.4 | 148 | 484 | 1449 | 87 | 4063 | 62.2 | 20 | 1181 |
| HyperTensor | short | 128 | 22 | 14 | 40 | 120.7 | 150 | 500 | 1442 | 87 | 4063 | 60.9 | 24 | 1160 |
| HyperTensor | short | 128 | 23 | 14 | 41.2 | 123.3 | 151 | 491 | 1414 | 1 | 4063 | 27.7 | 21 | 1221 |
| HyperTensor | short | 128 | 24 | 14 | 38.9 | 112.2 | 160 | 520 | 1442 | 55 | 4063 | 62.6 | 28 | 1278 |
| HyperTensor | short | 128 | 25 | 14 | 40.7 | 123.4 | 155 | 499 | 1402 | 87 | 4063 | 23.2 | 29 | 1160 |
| HyperTensor | short | 128 | 26 | 14 | 40.6 | 123.4 | 153 | 498 | 1441 | 64 | 4063 | 26.2 | 24 | 1160 |
| HyperTensor | short | 128 | 27 | 14 | 40.5 | 123.7 | 155 | 501 | 1440 | 89 | 4063 | 55.3 | 26 | 1230 |
| HyperTensor | short | 128 | 28 | 14 | 39.8 | 118.5 | 155 | 507 | 1476 | 89 | 4063 | 61 | 34 | 1160 |
| HyperTensor | short | 128 | 29 | 14 | 40.8 | 120.5 | 151 | 494 | 1451 | 0 | 4063 | 26.7 | 17 | 1215 |
| HyperTensor | short | 128 | 30 | 14 | 37.8 | 111 | 164 | 534 | 1472 | 0 | 4063 | 25.9 | 26 | 1160 |
| HyperTensor | short | 512 | 1 | 14 | 40.7 | 119.4 | 151 | 495 | 1414 | 0 | 4063 | 27.7 | 23 | 1167 |
| HyperTensor | short | 512 | 2 | 14 | 39 | 111.8 | 158 | 517 | 1442 | 1 | 4063 | 25.8 | 24 | 1238 |
| HyperTensor | short | 512 | 3 | 14 | 38.9 | 111 | 156 | 516 | 1428 | 2 | 4063 | 30.3 | 21 | 1166 |
| HyperTensor | short | 512 | 4 | 14 | 39.9 | 121.9 | 155 | 506 | 1476 | 99 | 4063 | 25 | 26 | 1182 |
| HyperTensor | short | 512 | 5 | 14 | 40.8 | 120 | 150 | 493 | 1415 | 0 | 4063 | 27 | 24 | 1160 |
| HyperTensor | short | 512 | 6 | 14 | 37 | 111.3 | 165 | 543 | 1466 | 87 | 4063 | 64.3 | 23 | 1223 |
| HyperTensor | short | 512 | 7 | 14 | 38.6 | 99.9 | 147 | 510 | 1504 | 87 | 4074 | 65.4 | 24 | 1160 |
| HyperTensor | short | 512 | 8 | 14 | 40.6 | 122.4 | 155 | 500 | 763846 | 1 | 4074 | 26.5 | 31 | 1160 |
| HyperTensor | short | 512 | 9 | 14 | 37.3 | 105.9 | 165 | 540 | 2757 | 7 | 4063 | 59.3 | 18 | 694 |
| HyperTensor | short | 512 | 10 | 14 | 37.7 | 120.9 | 167 | 538 | 1819 | 86 | 4070 | 63.4 | 40 | 938 |
| HyperTensor | short | 512 | 11 | 14 | 40.6 | 122.2 | 149 | 494 | 1581 | 1 | 4070 | 54.1 | 32 | 1036 |
| HyperTensor | short | 512 | 12 | 14 | 39.5 | 122.4 | 154 | 508 | 1468 | 89 | 4070 | 63.8 | 19 | 1160 |
| HyperTensor | short | 512 | 13 | 14 | 38.1 | 110.9 | 162 | 529 | 1461 | 31 | 4070 | 30.7 | 33 | 1197 |
| HyperTensor | short | 512 | 14 | 14 | 33.2 | 101.8 | 179 | 601 | 1682 | 84 | 4070 | 63 | 45 | 1217 |
| HyperTensor | short | 512 | 15 | 14 | 40 | 120.6 | 152 | 502 | 1449 | 86 | 4070 | 63.4 | 27 | 1218 |
| HyperTensor | short | 512 | 16 | 14 | 39.8 | 120.3 | 154 | 506 | 1506 | 1 | 4070 | 25.9 | 29 | 1219 |
| HyperTensor | short | 512 | 17 | 14 | 40.6 | 120.5 | 156 | 501 | 1470 | 1 | 4070 | 26.9 | 27 | 1160 |
| HyperTensor | short | 512 | 18 | 14 | 41.1 | 123.2 | 154 | 495 | 1463 | 86 | 4070 | 63.7 | 34 | 1211 |
| HyperTensor | short | 512 | 19 | 14 | 38.1 | 111.8 | 166 | 533 | 1447 | 0 | 4070 | 26.9 | 35 | 1160 |
| HyperTensor | short | 512 | 20 | 14 | 39.8 | 119.9 | 151 | 503 | 1623 | 88 | 4070 | 63.8 | 31 | 1160 |
| HyperTensor | short | 512 | 21 | 14 | 40.9 | 121.5 | 152 | 494 | 1469 | 0 | 4070 | 59.1 | 37 | 1223 |
| HyperTensor | short | 512 | 22 | 14 | 40.3 | 121.8 | 158 | 505 | 1482 | 0 | 4070 | 30.6 | 26 | 1160 |
| HyperTensor | short | 512 | 23 | 14 | 39.9 | 123.2 | 152 | 503 | 1458 | 25 | 4070 | 27.1 | 25 | 1155 |
| HyperTensor | short | 512 | 24 | 14 | 39.5 | 119.5 | 158 | 512 | 1460 | 88 | 4070 | 64.4 | 31 | 1213 |
| HyperTensor | short | 512 | 25 | 14 | 40.2 | 118.7 | 154 | 502 | 1480 | 0 | 4070 | 27.1 | 31 | 1174 |
| HyperTensor | short | 512 | 26 | 14 | 38.1 | 110.1 | 159 | 526 | 1512 | 2 | 4070 | 59.3 | 26 | 1160 |
| HyperTensor | short | 512 | 27 | 14 | 38.6 | 115.8 | 163 | 526 | 1633 | 87 | 4070 | 63.8 | 23 | 1194 |
| HyperTensor | short | 512 | 28 | 14 | 41.3 | 121.3 | 151 | 490 | 1512 | 1 | 4070 | 14.2 | 24 | 1149 |
| HyperTensor | short | 512 | 29 | 14 | 39.7 | 119.7 | 159 | 512 | 1437 | 1 | 4070 | 59.4 | 29 | 1250 |
| HyperTensor | short | 512 | 30 | 14 | 38.8 | 111 | 160 | 521 | 1441 | 69 | 4070 | 63.7 | 21 | 1248 |
| Ollama | code | 32 | 1 | 32 | 99.8 | 759.4 | 36.9 | 358 | 688 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 2 | 32 | 109.7 | 2403 | 11.7 | 303 | 649 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 3 | 32 | 110.7 | 2508.2 | 11.2 | 300 | 615 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 4 | 32 | 103.8 | 2773 | 10.1 | 319 | 673 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 5 | 32 | 121.4 | 2896.4 | 9.7 | 273 | 558 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 6 | 32 | 119.7 | 2953.5 | 9.5 | 277 | 622 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 7 | 32 | 121.1 | 2759.4 | 10.1 | 274 | 585 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 8 | 32 | 120.5 | 2923.2 | 9.6 | 275 | 580 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 9 | 32 | 118.5 | 2864 | 9.8 | 280 | 591 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 10 | 32 | 117.9 | 2940.1 | 9.5 | 281 | 595 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 11 | 32 | 115.7 | 2972.7 | 9.4 | 286 | 599 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 12 | 32 | 119.4 | 2868.6 | 9.8 | 278 | 630 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 13 | 32 | 117.9 | 3011.3 | 9.3 | 281 | 659 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 14 | 32 | 110.5 | 2754.8 | 10.2 | 300 | 630 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 15 | 32 | 108.3 | 3055.6 | 9.2 | 305 | 620 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 16 | 32 | 117.1 | 3075.8 | 9.1 | 282 | 547 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 17 | 32 | 119.7 | 2961.1 | 9.5 | 277 | 535 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 18 | 32 | 121.4 | 2913.5 | 9.6 | 273 | 519 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 19 | 32 | 121.7 | 2909.8 | 9.6 | 272 | 554 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 20 | 32 | 121.6 | 2940.2 | 9.5 | 273 | 530 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 21 | 32 | 124.8 | 3055.6 | 9.2 | 266 | 519 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 22 | 32 | 117.7 | 3079.4 | 9.1 | 281 | 517 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 23 | 32 | 117.4 | 3144.6 | 8.9 | 282 | 521 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 24 | 32 | 118.7 | 3029.1 | 9.2 | 279 | 537 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 25 | 32 | 118.2 | 3061.4 | 9.1 | 280 | 522 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 26 | 32 | 120.1 | 3104.6 | 9 | 275 | 528 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 27 | 32 | 116.5 | 2971.8 | 9.4 | 284 | 517 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 28 | 32 | 117.5 | 2966.2 | 9.4 | 282 | 526 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 29 | 32 | 115.2 | 2688.7 | 10.4 | 288 | 532 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 32 | 30 | 32 | 120.4 | 3206.2 | 8.7 | 275 | 524 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 128 | 1 | 128 | 113.8 | 3145.7 | 8.9 | 1133 | 1415 | 0 | 5517 | 61.3 | 29 | 241 |
| Ollama | code | 128 | 2 | 128 | 117.7 | 2910.7 | 9.6 | 1097 | 1376 | 41 | 5517 | 59.6 | 29 | 242 |
| Ollama | code | 128 | 3 | 128 | 117.4 | 2776.6 | 10.1 | 1101 | 1380 | 0 | 5517 | 57 | 24 | 239 |
| Ollama | code | 128 | 4 | 128 | 113.2 | 2965.1 | 9.4 | 1140 | 1422 | 1 | 5517 | 68.5 | 33 | 242 |
| Ollama | code | 128 | 5 | 128 | 115 | 2952.7 | 9.5 | 1122 | 1408 | 66 | 5517 | 71.7 | 29 | 239 |
| Ollama | code | 128 | 6 | 128 | 109.6 | 2633.5 | 10.6 | 1178 | 1481 | 67 | 5517 | 65 | 32 | 242 |
| Ollama | code | 128 | 7 | 128 | 112.9 | 2775.7 | 10.1 | 1144 | 1552 | 1 | 5518 | 16.8 | 52 | 230 |
| Ollama | code | 128 | 8 | 128 | 112.6 | 2843.7 | 9.8 | 1147 | 1567 | 0 | 5526 | 17 | 45 | 230 |
| Ollama | code | 128 | 9 | 128 | 102.2 | 2798.1 | 10 | 1263 | 1748 | 0 | 0 | 0 | 0 | 0 |
| Ollama | code | 128 | 10 | 128 | 102.7 | 1088.5 | 25.7 | 1272 | 1621 | 0 | 5526 | 10 | 22 | 230 |
| Ollama | code | 128 | 11 | 128 | 113.8 | 2920.4 | 9.6 | 1134 | 1465 | 91 | 5526 | 57.8 | 40 | 230 |
| Ollama | code | 128 | 12 | 128 | 117.4 | 3134.5 | 8.9 | 1099 | 1406 | 62 | 5526 | 58.6 | 27 | 233 |
| Ollama | code | 128 | 13 | 128 | 117.4 | 2937.7 | 9.5 | 1100 | 1402 | 88 | 5526 | 57.6 | 23 | 233 |
| Ollama | code | 128 | 14 | 128 | 111.1 | 2972.3 | 9.4 | 1161 | 1444 | 1 | 5526 | 58.5 | 22 | 231 |
| Ollama | code | 128 | 15 | 128 | 117.5 | 2954.5 | 9.5 | 1099 | 1405 | 98 | 5526 | 77.5 | 16 | 231 |
| Ollama | code | 128 | 16 | 128 | 120 | 2904.4 | 9.6 | 1076 | 1460 | 1 | 5526 | 57.6 | 13 | 231 |
| Ollama | code | 128 | 17 | 128 | 116.8 | 2973.3 | 9.4 | 1106 | 1407 | 99 | 5526 | 78.2 | 36 | 231 |
| Ollama | code | 128 | 18 | 128 | 117.2 | 1505.6 | 18.6 | 1111 | 1453 | 88 | 5526 | 54.1 | 36 | 231 |
| Ollama | code | 128 | 19 | 128 | 121 | 3054.8 | 9.2 | 1067 | 1421 | 99 | 5526 | 74.3 | 28 | 183 |
| Ollama | code | 128 | 20 | 128 | 120.2 | 3128.9 | 8.9 | 1074 | 1401 | 0 | 5526 | 54.8 | 22 | 183 |
| Ollama | code | 128 | 21 | 128 | 119 | 2980.7 | 9.4 | 1085 | 1384 | 1 | 5526 | 54 | 13 | 183 |
| Ollama | code | 128 | 22 | 128 | 110.9 | 3078.2 | 9.1 | 1163 | 1474 | 0 | 5526 | 54.4 | 26 | 183 |
| Ollama | code | 128 | 23 | 128 | 116.9 | 2956.4 | 9.5 | 1105 | 1399 | 1 | 5526 | 65.8 | 28 | 183 |
| Ollama | code | 128 | 24 | 128 | 117 | 2955.2 | 9.5 | 1103 | 1387 | 1 | 5526 | 61.2 | 23 | 183 |
| Ollama | code | 128 | 25 | 128 | 117.7 | 3064.6 | 9.1 | 1096 | 1405 | 0 | 5526 | 55 | 24 | 183 |
| Ollama | code | 128 | 26 | 128 | 118.2 | 2977.5 | 9.4 | 1093 | 1419 | 0 | 5526 | 69.9 | 27 | 183 |
| Ollama | code | 128 | 27 | 128 | 118.1 | 3081 | 9.1 | 1093 | 1392 | 46 | 5526 | 61.4 | 20 | 183 |
| Ollama | code | 128 | 28 | 128 | 119.9 | 3012 | 9.3 | 1077 | 1394 | 0 | 5526 | 54.3 | 27 | 183 |
| Ollama | code | 128 | 29 | 128 | 119.5 | 2933.8 | 9.5 | 1081 | 1403 | 1 | 5526 | 63.6 | 23 | 183 |
| Ollama | code | 128 | 30 | 128 | 113.6 | 2954.1 | 9.5 | 1137 | 1463 | 0 | 5526 | 69.1 | 15 | 183 |
| Ollama | code | 512 | 1 | 512 | 117.5 | 2947.4 | 9.5 | 4367 | 4862 | 58.3 | 5526 | 63.2 | 21.3 | 185 |
| Ollama | code | 512 | 2 | 512 | 115.3 | 2196.4 | 12.7 | 4453 | 4843 | 85.3 | 5526 | 66.1 | 24.7 | 185 |
| Ollama | code | 512 | 3 | 512 | 116.3 | 2219.8 | 12.6 | 4414 | 4855 | 90.3 | 5526 | 67.3 | 23.7 | 185 |
| Ollama | code | 512 | 4 | 512 | 114.6 | 2113 | 13.3 | 4481 | 4848 | 90.7 | 5526 | 63.9 | 24 | 183 |
| Ollama | code | 512 | 5 | 512 | 114.3 | 2007.6 | 13.9 | 4492 | 4967 | 86.7 | 5526 | 65.2 | 36 | 183 |
| Ollama | code | 512 | 6 | 512 | 111.5 | 2079.4 | 13.5 | 4605 | 5059 | 61.7 | 5526 | 63.1 | 40.7 | 187 |
| Ollama | code | 512 | 7 | 512 | 114.9 | 1939 | 14.4 | 4470 | 4934 | 59.3 | 5526 | 64.3 | 29.3 | 187 |
| Ollama | code | 512 | 8 | 512 | 110.6 | 2029.2 | 13.8 | 4642 | 5058 | 91.7 | 5526 | 66.2 | 29.7 | 187 |
| Ollama | code | 512 | 9 | 512 | 110.9 | 2217.3 | 12.6 | 4631 | 5075 | 59 | 5526 | 63 | 25.7 | 187 |
| Ollama | code | 512 | 10 | 512 | 111.2 | 2008.4 | 13.9 | 4617 | 5092 | 87 | 5526 | 64.5 | 36 | 187 |
| Ollama | code | 512 | 11 | 512 | 110.5 | 2201 | 12.7 | 4647 | 5108 | 59.7 | 5526 | 61.5 | 36.3 | 187 |
| Ollama | code | 512 | 12 | 512 | 108 | 2040.2 | 13.7 | 4754 | 5236 | 63.3 | 5526 | 61.3 | 42.3 | 186 |
| Ollama | code | 512 | 13 | 512 | 115.2 | 2048.4 | 13.7 | 4459 | 4912 | 58 | 5526 | 64.7 | 27.7 | 186 |
| Ollama | code | 512 | 14 | 512 | 107 | 2060.9 | 13.6 | 4799 | 5279 | 61.3 | 5526 | 60.8 | 41.3 | 184 |
| Ollama | code | 512 | 15 | 512 | 110.9 | 1876.3 | 14.9 | 4631 | 5321 | 69 | 5526 | 56.4 | 36.3 | 184 |
| Ollama | code | 512 | 16 | 512 | 112.6 | 1919.2 | 14.6 | 4563 | 5040 | 61 | 5529 | 49 | 31.7 | 184 |
| Ollama | code | 512 | 17 | 512 | 107.9 | 1949.9 | 14.4 | 4758 | 5227 | 60.7 | 5529 | 61.9 | 37.7 | 184 |
| Ollama | code | 512 | 18 | 512 | 107.7 | 1922.8 | 14.6 | 4768 | 5248 | 61.7 | 5529 | 61.8 | 34.7 | 182 |
| Ollama | code | 512 | 19 | 512 | 107.7 | 1906 | 14.7 | 4767 | 5228 | 83.3 | 5529 | 64.3 | 37 | 182 |
| Ollama | code | 512 | 20 | 512 | 107.6 | 2131.3 | 13.1 | 4772 | 5264 | 89.7 | 5529 | 63.8 | 36 | 182 |
| Ollama | code | 512 | 21 | 512 | 107.5 | 2033.1 | 13.8 | 4776 | 5293 | 62 | 5543 | 49.5 | 45 | 182 |
| Ollama | code | 512 | 22 | 512 | 109.8 | 1881.3 | 14.9 | 4678 | 5176 | 62 | 5543 | 58.4 | 32.3 | 182 |
| Ollama | code | 512 | 23 | 512 | 111.7 | 1984.1 | 14.1 | 4597 | 5142 | 58 | 5543 | 62.1 | 30.7 | 182 |
| Ollama | code | 512 | 24 | 512 | 114.2 | 2002.7 | 14 | 4497 | 4984 | 58.3 | 5543 | 62.1 | 25 | 181 |
| Ollama | code | 512 | 25 | 512 | 112.9 | 1604.1 | 17.5 | 4554 | 5065 | 58 | 5543 | 59.5 | 28.3 | 181 |
| Ollama | code | 512 | 26 | 512 | 110.4 | 1764.2 | 15.9 | 4654 | 5169 | 59 | 5543 | 61.9 | 25.3 | 181 |
| Ollama | code | 512 | 27 | 512 | 107 | 1904.4 | 14.7 | 4798 | 5334 | 62 | 5543 | 59.6 | 25 | 181 |
| Ollama | code | 512 | 28 | 512 | 108.2 | 2041.1 | 13.7 | 4748 | 5259 | 69.3 | 5543 | 63 | 35.7 | 182 |
| Ollama | code | 512 | 29 | 512 | 110.4 | 1784.4 | 15.7 | 4654 | 5196 | 58.3 | 5543 | 60.8 | 32.3 | 187 |
| Ollama | code | 512 | 30 | 512 | 109.9 | 2020.1 | 13.9 | 4671 | 5214 | 57 | 5543 | 62.9 | 33.7 | 181 |
| Ollama | long | 32 | 1 | 32 | 70.4 | 251.3 | 214.9 | 669 | 1024 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 2 | 32 | 71.3 | 471.8 | 114.5 | 563 | 944 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 3 | 32 | 76.2 | 596.5 | 90.5 | 511 | 910 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 4 | 32 | 72.6 | 870.8 | 62 | 503 | 943 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 5 | 32 | 73 | 382.7 | 141.1 | 580 | 1022 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 6 | 32 | 72 | 4025.3 | 13.4 | 458 | 860 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 7 | 32 | 71.4 | 526.3 | 102.6 | 551 | 955 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 8 | 32 | 70.8 | 963.7 | 56 | 508 | 917 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 9 | 32 | 75.3 | 807.1 | 66.9 | 492 | 820 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 10 | 32 | 78.4 | 825.4 | 65.4 | 473 | 878 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 11 | 32 | 71.8 | 820.8 | 65.8 | 511 | 817 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 12 | 32 | 76.4 | 645.7 | 83.6 | 503 | 962 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 13 | 32 | 70.5 | 532.4 | 101.4 | 556 | 953 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 14 | 32 | 71 | 629.4 | 85.8 | 536 | 944 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 15 | 32 | 71.9 | 569.1 | 94.9 | 540 | 943 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 16 | 32 | 70.4 | 466 | 115.9 | 571 | 957 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 17 | 32 | 69.9 | 1040.5 | 51.9 | 510 | 910 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 18 | 32 | 76.4 | 443.6 | 121.7 | 541 | 912 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 19 | 32 | 71 | 602.2 | 89.7 | 540 | 912 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 20 | 32 | 71.5 | 802 | 67.3 | 515 | 917 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 21 | 32 | 71.6 | 874.4 | 61.8 | 509 | 863 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 22 | 32 | 76.4 | 457.2 | 118.1 | 537 | 914 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 23 | 32 | 70.9 | 770.5 | 70.1 | 522 | 922 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 24 | 32 | 67.1 | 432.8 | 124.8 | 602 | 1003 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 25 | 32 | 72 | 3327.4 | 16.2 | 460 | 997 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 26 | 32 | 69.9 | 730.8 | 73.9 | 532 | 946 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 27 | 32 | 70.1 | 559 | 96.6 | 553 | 949 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 28 | 32 | 77.3 | 441.8 | 122.2 | 536 | 926 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 29 | 32 | 75.9 | 425.4 | 126.9 | 548 | 925 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 32 | 30 | 32 | 78.6 | 611 | 88.4 | 496 | 840 | 0 | 0 | 0 | 0 | 0 |
| Ollama | long | 128 | 1 | 128 | 76.6 | 5652.9 | 9.6 | 1681 | 2079 | 92 | 5770 | 56.7 | 33 | 818 |
| Ollama | long | 128 | 2 | 128 | 115.7 | 6061.7 | 8.9 | 1115 | 1466 | 10 | 5438 | 19.9 | 16 | 797 |
| Ollama | long | 128 | 3 | 128 | 111.9 | 5739 | 9.4 | 1153 | 1576 | 1 | 5480 | 17 | 26 | 780 |
| Ollama | long | 128 | 4 | 128 | 109 | 5651.6 | 9.6 | 1184 | 1554 | 7 | 5480 | 17.2 | 37 | 768 |
| Ollama | long | 128 | 5 | 128 | 109.8 | 5614.8 | 9.6 | 1175 | 1550 | 5 | 5480 | 16.9 | 35 | 757 |
| Ollama | long | 128 | 6 | 128 | 103.9 | 4919.2 | 11 | 1243 | 1611 | 6 | 5480 | 16.1 | 34 | 747 |
| Ollama | long | 128 | 7 | 128 | 111 | 5677.2 | 9.5 | 1163 | 1530 | 7 | 5480 | 17 | 30 | 732 |
| Ollama | long | 128 | 8 | 128 | 116.8 | 5627.1 | 9.6 | 1105 | 1455 | 0 | 5480 | 16.5 | 36 | 706 |
| Ollama | long | 128 | 9 | 128 | 110.5 | 5964.5 | 9.1 | 1167 | 1532 | 0 | 5480 | 37.8 | 31 | 700 |
| Ollama | long | 128 | 10 | 128 | 109.7 | 4897.6 | 11 | 1178 | 1543 | 7 | 5480 | 15 | 24 | 691 |
| Ollama | long | 128 | 11 | 128 | 103.6 | 5305.8 | 10.2 | 1246 | 1617 | 6 | 5480 | 24.1 | 22 | 673 |
| Ollama | long | 128 | 12 | 128 | 102.8 | 5963.5 | 9.1 | 1254 | 1605 | 7 | 5480 | 18.1 | 35 | 665 |
| Ollama | long | 128 | 13 | 128 | 107.4 | 5984 | 9 | 1200 | 1572 | 6 | 5480 | 16.8 | 41 | 657 |
| Ollama | long | 128 | 14 | 128 | 118.7 | 4707.4 | 11.5 | 1090 | 1438 | 0 | 5480 | 16.9 | 25 | 643 |
| Ollama | long | 128 | 15 | 128 | 116.9 | 5612.9 | 9.6 | 1104 | 1462 | 8 | 5480 | 18 | 43 | 633 |
| Ollama | long | 128 | 16 | 128 | 109.3 | 5689.7 | 9.5 | 1181 | 1538 | 7 | 5480 | 26.9 | 26 | 618 |
| Ollama | long | 128 | 17 | 128 | 109.4 | 5598.6 | 9.6 | 1180 | 1521 | 58 | 5480 | 55.7 | 26 | 617 |
| Ollama | long | 128 | 18 | 128 | 110.9 | 5693.4 | 9.5 | 1164 | 1560 | 7 | 5480 | 17.3 | 21 | 617 |
| Ollama | long | 128 | 19 | 128 | 108 | 5857.3 | 9.2 | 1194 | 1591 | 7 | 5480 | 22.7 | 31 | 617 |
| Ollama | long | 128 | 20 | 128 | 111.6 | 5555.3 | 9.7 | 1157 | 1552 | 5 | 5480 | 15.6 | 34 | 617 |
| Ollama | long | 128 | 21 | 128 | 109.6 | 5096.6 | 10.6 | 1178 | 1486 | 8 | 5480 | 20.1 | 39 | 619 |
| Ollama | long | 128 | 22 | 128 | 114.7 | 5938.7 | 9.1 | 1125 | 1480 | 8 | 5480 | 26.1 | 55 | 617 |
| Ollama | long | 128 | 23 | 128 | 114.2 | 5253 | 10.3 | 1132 | 1528 | 83 | 5481 | 62.2 | 45 | 617 |
| Ollama | long | 128 | 24 | 128 | 106.5 | 5207.4 | 10.4 | 1212 | 1612 | 0 | 5486 | 17.6 | 51 | 617 |
| Ollama | long | 128 | 25 | 128 | 108.7 | 5818.3 | 9.3 | 1187 | 1565 | 6 | 5466 | 16.5 | 40 | 243 |
| Ollama | long | 128 | 26 | 128 | 109.3 | 5526.4 | 9.8 | 1181 | 1538 | 6 | 5466 | 32.9 | 37 | 243 |
| Ollama | long | 128 | 27 | 128 | 108.8 | 5821.2 | 9.3 | 1186 | 1559 | 6 | 5466 | 38.1 | 26 | 243 |
| Ollama | long | 128 | 28 | 128 | 108.8 | 5261.1 | 10.3 | 1187 | 1541 | 88 | 5466 | 62.7 | 32 | 243 |
| Ollama | long | 128 | 29 | 128 | 117.8 | 6073.7 | 8.9 | 1095 | 1445 | 0 | 5466 | 20.7 | 21 | 243 |
| Ollama | long | 128 | 30 | 128 | 110.6 | 5646.4 | 9.6 | 1167 | 1517 | 1 | 5466 | 26 | 42 | 243 |
| Ollama | long | 512 | 1 | 512 | 107.1 | 6030.8 | 9 | 4790 | 5304 | 60 | 5474 | 47.7 | 31.7 | 247 |
| Ollama | long | 512 | 2 | 512 | 113.6 | 2906.4 | 18.6 | 4528 | 5051 | 73.3 | 5474 | 63.3 | 29.3 | 250 |
| Ollama | long | 512 | 3 | 512 | 109.7 | 3597.5 | 15 | 4682 | 5210 | 59.3 | 5474 | 48.4 | 39.7 | 251 |
| Ollama | long | 512 | 4 | 512 | 114.1 | 3897.9 | 13.9 | 4501 | 5058 | 58.3 | 5474 | 60.2 | 24 | 252 |
| Ollama | long | 512 | 5 | 512 | 114.7 | 4058.9 | 13.3 | 4476 | 4915 | 58 | 5474 | 56.9 | 26.7 | 252 |
| Ollama | long | 512 | 6 | 512 | 110.6 | 4239.2 | 12.7 | 4640 | 5103 | 59 | 5474 | 51.1 | 27 | 253 |
| Ollama | long | 512 | 7 | 512 | 115.9 | 4055.5 | 13.3 | 4429 | 4881 | 73.7 | 5474 | 57.1 | 30.3 | 253 |
| Ollama | long | 512 | 8 | 512 | 116.7 | 4124.4 | 13.1 | 4402 | 4856 | 78.3 | 5474 | 53 | 24.3 | 252 |
| Ollama | long | 512 | 9 | 512 | 114.8 | 4213.4 | 12.8 | 4473 | 4930 | 58.7 | 5474 | 58.3 | 31.3 | 253 |
| Ollama | long | 512 | 10 | 512 | 108.8 | 3072.4 | 17.6 | 4722 | 5195 | 59 | 5474 | 48.3 | 35.7 | 253 |
| Ollama | long | 512 | 11 | 512 | 111.6 | 3876.6 | 13.9 | 4600 | 5078 | 61 | 5474 | 56.7 | 27.7 | 253 |
| Ollama | long | 512 | 12 | 512 | 110.3 | 4071.8 | 13.3 | 4653 | 5131 | 67.7 | 5474 | 51.6 | 27.7 | 254 |
| Ollama | long | 512 | 13 | 512 | 111.8 | 3640.6 | 14.8 | 4596 | 5082 | 57 | 5474 | 56.5 | 35.3 | 254 |
| Ollama | long | 512 | 14 | 512 | 113.2 | 4108.8 | 13.1 | 4535 | 4965 | 70 | 5474 | 52.6 | 27.3 | 254 |
| Ollama | long | 512 | 15 | 512 | 112.8 | 4140.5 | 13 | 4554 | 5013 | 60.3 | 5474 | 56.3 | 29 | 254 |
| Ollama | long | 512 | 16 | 512 | 107 | 4069.1 | 13.3 | 4800 | 5245 | 62 | 5474 | 49.6 | 29.7 | 254 |
| Ollama | long | 512 | 17 | 512 | 111.2 | 3774.8 | 14.3 | 4620 | 5173 | 60 | 5522 | 48.9 | 33.3 | 253 |
| Ollama | long | 512 | 18 | 512 | 110 | 3460.5 | 15.6 | 4669 | 5261 | 59 | 5518 | 47.6 | 41.3 | 253 |
| Ollama | long | 512 | 19 | 512 | 108.1 | 3597.6 | 15 | 4750 | 5209 | 61.7 | 5500 | 51.1 | 30.7 | 253 |
| Ollama | long | 512 | 20 | 512 | 108.9 | 4015.5 | 13.4 | 4714 | 5186 | 89 | 5500 | 63.3 | 30.7 | 253 |
| Ollama | long | 512 | 21 | 512 | 111.9 | 4038.4 | 13.4 | 4588 | 5062 | 60 | 5500 | 48.2 | 27.3 | 254 |
| Ollama | long | 512 | 22 | 512 | 106.9 | 4074.2 | 13.3 | 4801 | 5279 | 59.3 | 5500 | 61.9 | 46.3 | 254 |
| Ollama | long | 512 | 23 | 512 | 107.4 | 4286.5 | 12.6 | 4779 | 5240 | 55.3 | 5500 | 46.9 | 39.7 | 254 |
| Ollama | long | 512 | 24 | 512 | 108.5 | 3766.1 | 14.3 | 4731 | 5190 | 77.7 | 5500 | 58.8 | 32 | 254 |
| Ollama | long | 512 | 25 | 512 | 107 | 3797.8 | 14.2 | 4801 | 5250 | 61.7 | 5500 | 50.9 | 29 | 254 |
| Ollama | long | 512 | 26 | 512 | 109.1 | 3969.4 | 13.6 | 4706 | 5299 | 55 | 5500 | 46.1 | 36.3 | 254 |
| Ollama | long | 512 | 27 | 512 | 110.9 | 3340.8 | 16.2 | 4631 | 5203 | 61.3 | 5500 | 49.8 | 36.7 | 240 |
| Ollama | long | 512 | 28 | 512 | 113.9 | 3772.2 | 14.3 | 4508 | 5036 | 58.7 | 5500 | 49.6 | 29.7 | 240 |
| Ollama | long | 512 | 29 | 512 | 115.7 | 3612.2 | 14.9 | 4440 | 4977 | 58.3 | 5500 | 49.1 | 21.3 | 240 |
| Ollama | long | 512 | 30 | 512 | 107.4 | 2926.8 | 18.5 | 4785 | 5348 | 62.7 | 5500 | 48.2 | 36.7 | 240 |
| Ollama | medium | 32 | 1 | 32 | 116.2 | 1325.1 | 20.4 | 296 | 556 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 2 | 2 | 190.9 | 2937 | 9.2 | 20 | 267 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 3 | 32 | 114.8 | 2587 | 10.4 | 289 | 538 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 4 | 2 | 236 | 2995.8 | 9 | 17 | 253 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 5 | 32 | 122.5 | 2859 | 9.4 | 271 | 523 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 6 | 32 | 121.2 | 2817.2 | 9.6 | 274 | 531 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 7 | 32 | 122.6 | 3019.5 | 8.9 | 270 | 530 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 8 | 32 | 119.2 | 2879.4 | 9.4 | 278 | 530 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 9 | 32 | 118.8 | 2839 | 9.5 | 279 | 533 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 10 | 2 | 238.1 | 2858 | 9.4 | 18 | 251 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 11 | 32 | 117.6 | 2884.4 | 9.4 | 282 | 520 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 12 | 2 | 236.7 | 3024.3 | 8.9 | 17 | 247 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 13 | 32 | 118.4 | 2893.3 | 9.3 | 280 | 527 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 14 | 2 | 233.8 | 2865.4 | 9.4 | 18 | 264 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 15 | 32 | 120.5 | 2717.6 | 9.9 | 275 | 2553 | 1 | 4070 | 27.8 | 2 | 1040 |
| Ollama | medium | 32 | 16 | 32 | 120.8 | 2571.3 | 10.5 | 276 | 521 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 17 | 2 | 238.4 | 2891.4 | 9.3 | 18 | 252 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 18 | 32 | 128.4 | 3092.7 | 8.7 | 258 | 522 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 19 | 2 | 220.4 | 2815.7 | 9.6 | 19 | 253 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 20 | 32 | 116.8 | 3010.5 | 9 | 283 | 530 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 21 | 2 | 227.9 | 2620.7 | 10.3 | 19 | 313 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 22 | 32 | 119 | 2681.5 | 10.1 | 279 | 658 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 23 | 2 | 231.6 | 2905.3 | 9.3 | 18 | 362 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 24 | 2 | 190.3 | 2334.9 | 11.6 | 22 | 323 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 25 | 32 | 117 | 2944.5 | 9.2 | 283 | 566 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 26 | 32 | 110.4 | 2433.8 | 11.1 | 301 | 617 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 27 | 2 | 223.2 | 2391.9 | 11.3 | 20 | 296 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 28 | 2 | 209.8 | 2893.2 | 9.3 | 19 | 321 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 29 | 32 | 96.5 | 2773.4 | 9.7 | 341 | 674 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 32 | 30 | 32 | 118.8 | 2861.3 | 9.4 | 279 | 584 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 1 | 119 | 108.1 | 2971.6 | 9.1 | 1109 | 1448 | 3 | 4080 | 53.3 | 45 | 959 |
| Ollama | medium | 128 | 2 | 72 | 116.1 | 2795.3 | 9.7 | 630 | 1002 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 3 | 2 | 224 | 2835.5 | 9.5 | 18 | 372 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 4 | 2 | 230.9 | 2859.3 | 9.4 | 18 | 352 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 5 | 99 | 105.5 | 2746.8 | 9.8 | 948 | 1338 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 6 | 118 | 115.2 | 2829.1 | 9.5 | 1034 | 3403 | 5.5 | 4081 | 15.7 | 23.5 | 389 |
| Ollama | medium | 128 | 7 | 128 | 111.6 | 2752.8 | 9.8 | 1157 | 1482 | 6 | 4081 | 17.2 | 37 | 304 |
| Ollama | medium | 128 | 8 | 128 | 111 | 2418.2 | 11.2 | 1164 | 1566 | 8 | 4081 | 14.8 | 40 | 259 |
| Ollama | medium | 128 | 9 | 128 | 110.5 | 2478.9 | 10.9 | 1169 | 1490 | 6 | 4082 | 43.1 | 38 | 260 |
| Ollama | medium | 128 | 10 | 114 | 110.2 | 2606.4 | 10.4 | 1045 | 1400 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 11 | 89 | 109.2 | 2449.4 | 11 | 826 | 1217 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 12 | 2 | 191.7 | 2677.6 | 10.1 | 21 | 366 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 13 | 90 | 107.1 | 2809.5 | 9.6 | 850 | 1219 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 14 | 2 | 237 | 2962 | 9.1 | 18 | 325 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 15 | 93 | 109.8 | 2455.4 | 11 | 858 | 1210 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 16 | 111 | 111.2 | 2759.2 | 9.8 | 1008 | 1372 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 17 | 94 | 113 | 2929.8 | 9.2 | 841 | 1134 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 18 | 128 | 113.3 | 2959.1 | 9.1 | 1139 | 1435 | 34 | 4116 | 57.6 | 21 | 232 |
| Ollama | medium | 128 | 19 | 109 | 115.3 | 3021.5 | 8.9 | 954 | 1236 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 20 | 128 | 74.7 | 1798.4 | 15 | 1728 | 2102 | 65 | 5880 | 40.3 | 43 | 217 |
| Ollama | medium | 128 | 21 | 105 | 59.6 | 1596.7 | 16.9 | 1780 | 4171 | 55.5 | 7848 | 39.2 | 51 | 216 |
| Ollama | medium | 128 | 22 | 94 | 61.2 | 2164.1 | 12.5 | 1549 | 1901 | 98 | 7848 | 50.7 | 56 | 218 |
| Ollama | medium | 128 | 23 | 2 | 121.6 | 5.8 | 4676 | 4692 | 7382 | 24.5 | 2093 | 13.2 | 64.5 | 156 |
| Ollama | medium | 128 | 24 | 111 | 52.5 | 91.8 | 294.1 | 2407 | 2854 | 25 | 4703 | 27.6 | 43 | 3717 |
| Ollama | medium | 128 | 25 | 2 | 188 | 95.4 | 283.1 | 294 | 708 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 128 | 26 | 128 | 76.3 | 236.9 | 114 | 1791 | 2256 | 24 | 5151 | 26 | 36 | 3596 |
| Ollama | medium | 128 | 27 | 125 | 75.3 | 960.3 | 28.1 | 1688 | 2106 | 17 | 5471 | 26.2 | 51 | 3604 |
| Ollama | medium | 128 | 28 | 115 | 69.9 | 329 | 82.1 | 1727 | 2192 | 28 | 5288 | 25.4 | 57 | 3530 |
| Ollama | medium | 128 | 29 | 128 | 75.5 | 348.5 | 77.5 | 1773 | 2179 | 19 | 5343 | 25.9 | 50 | 3539 |
| Ollama | medium | 128 | 30 | 2 | 140.1 | 269.2 | 100.3 | 115 | 452 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 1 | 2 | 194.3 | 511.5 | 52.8 | 63 | 429 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 2 | 156 | 75.7 | 194.7 | 138.7 | 2201 | 2716 | 17 | 5128 | 25.1 | 53 | 3285 |
| Ollama | medium | 512 | 3 | 99 | 69.7 | 260.7 | 103.6 | 1524 | 2089 | 17 | 5384 | 22.6 | 57 | 3113 |
| Ollama | medium | 512 | 4 | 2 | 140.1 | 450.3 | 60 | 74 | 442 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 5 | 102 | 69.3 | 228.3 | 118.3 | 1589 | 2063 | 21 | 5271 | 28.5 | 47 | 2689 |
| Ollama | medium | 512 | 6 | 133 | 69 | 265.9 | 101.5 | 2030 | 2509 | 27 | 5399 | 25.3 | 52 | 1900 |
| Ollama | medium | 512 | 7 | 165 | 23 | 7.1 | 3793.8 | 10952 | 11617 | 84.7 | 5724 | 28.6 | 46.2 | 876 |
| Ollama | medium | 512 | 8 | 105 | 74.1 | 72.7 | 371.6 | 1789 | 2233 | 26 | 5264 | 27.9 | 51 | 509 |
| Ollama | medium | 512 | 9 | 103 | 74.3 | 431.6 | 62.6 | 1448 | 1863 | 17 | 5538 | 23.8 | 46 | 629 |
| Ollama | medium | 512 | 10 | 174 | 74.3 | 387.4 | 69.7 | 2413 | 2939 | 19 | 5426 | 22.9 | 69 | 656 |
| Ollama | medium | 512 | 11 | 2 | 202.4 | 331.5 | 81.5 | 91 | 478 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 12 | 123 | 70.3 | 478.6 | 56.4 | 1805 | 2255 | 28 | 5554 | 28.1 | 51 | 632 |
| Ollama | medium | 512 | 13 | 113 | 74.7 | 73.5 | 367.3 | 1880 | 2370 | 17 | 5170 | 22.8 | 41 | 581 |
| Ollama | medium | 512 | 14 | 136 | 75.7 | 231.1 | 116.8 | 1912 | 2400 | 22 | 5426 | 29.7 | 33 | 560 |
| Ollama | medium | 512 | 15 | 2 | 138.2 | 369.4 | 73.1 | 88 | 457 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 16 | 2 | 152.9 | 87.2 | 309.7 | 323 | 778 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 17 | 2 | 159.4 | 2088.8 | 12.9 | 25 | 407 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 18 | 130 | 70.1 | 210.1 | 128.5 | 1982 | 2563 | 18 | 5426 | 23.9 | 51 | 488 |
| Ollama | medium | 512 | 19 | 102 | 69.9 | 162.3 | 166.4 | 1627 | 2089 | 25 | 5170 | 24.6 | 44 | 775 |
| Ollama | medium | 512 | 20 | 105 | 70.6 | 370.5 | 72.9 | 1560 | 1979 | 15 | 5426 | 24.7 | 27 | 766 |
| Ollama | medium | 512 | 21 | 94 | 68.9 | 407.6 | 66.2 | 1430 | 1865 | 23 | 5554 | 25.4 | 39 | 734 |
| Ollama | medium | 512 | 22 | 2 | 136.2 | 293.1 | 92.1 | 107 | 473 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 23 | 75 | 68.6 | 409.8 | 65.9 | 1159 | 1590 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 24 | 2 | 207.1 | 256.2 | 105.4 | 115 | 563 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 25 | 2 | 184.6 | 268.7 | 100.5 | 111 | 520 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 26 | 2 | 124.4 | 1620 | 16.7 | 33 | 413 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 27 | 2 | 128.4 | 271.1 | 99.6 | 115 | 473 | 0 | 0 | 0 | 0 | 0 |
| Ollama | medium | 512 | 28 | 115 | 68.9 | 191.8 | 140.8 | 1810 | 2261 | 20 | 5316 | 24.7 | 34 | 761 |
| Ollama | medium | 512 | 29 | 93 | 68.6 | 450.7 | 59.9 | 1416 | 1881 | 28 | 5572 | 28 | 40 | 782 |
| Ollama | medium | 512 | 30 | 108 | 69.7 | 543.6 | 49.7 | 1599 | 2011 | 15 | 5572 | 22.7 | 26 | 749 |
| Ollama | short | 32 | 1 | 32 | 112.3 | 2055.4 | 9.7 | 295 | 555 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 2 | 32 | 103.7 | 1927.4 | 10.4 | 319 | 582 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 3 | 32 | 114.3 | 2134 | 9.4 | 289 | 539 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 4 | 32 | 107.8 | 1846.8 | 10.8 | 308 | 552 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 5 | 32 | 113.3 | 1859.2 | 10.8 | 293 | 547 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 6 | 32 | 113.3 | 2228.1 | 9 | 291 | 540 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 7 | 32 | 115.5 | 2166.9 | 9.2 | 286 | 539 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 8 | 32 | 115 | 1995.2 | 10 | 288 | 552 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 9 | 32 | 109.3 | 2241.4 | 8.9 | 302 | 548 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 10 | 32 | 122.5 | 2254.7 | 8.9 | 270 | 524 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 11 | 32 | 117.2 | 2224.7 | 9 | 282 | 526 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 12 | 32 | 123.4 | 2248.5 | 8.9 | 268 | 521 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 13 | 32 | 122.9 | 2220.3 | 9 | 269 | 524 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 14 | 32 | 118.6 | 2172.8 | 9.2 | 279 | 523 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 15 | 32 | 119.3 | 2219.6 | 9 | 277 | 575 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 16 | 32 | 122.7 | 2248.5 | 8.9 | 270 | 525 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 17 | 32 | 116.5 | 1925.5 | 10.4 | 285 | 525 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 18 | 32 | 118.1 | 2136.2 | 9.4 | 280 | 520 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 19 | 32 | 120.8 | 2121.2 | 9.4 | 274 | 517 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 20 | 32 | 126.1 | 2143.2 | 9.3 | 263 | 527 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 21 | 32 | 116.3 | 2226 | 9 | 284 | 533 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 22 | 32 | 117.1 | 2124.3 | 9.4 | 283 | 520 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 23 | 32 | 115.5 | 2171.6 | 9.2 | 286 | 524 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 24 | 32 | 118.6 | 2122.9 | 9.4 | 279 | 527 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 25 | 32 | 119.7 | 2141.4 | 9.3 | 277 | 525 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 26 | 32 | 117.8 | 2251 | 8.9 | 281 | 521 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 27 | 32 | 121.1 | 2084.2 | 9.6 | 274 | 520 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 28 | 32 | 116.6 | 2087.7 | 9.6 | 284 | 519 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 29 | 32 | 122.2 | 1021.5 | 19.6 | 282 | 525 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 32 | 30 | 32 | 119.1 | 2121 | 9.4 | 278 | 518 | 0 | 0 | 0 | 0 | 0 |
| Ollama | short | 128 | 1 | 128 | 115.7 | 1913.3 | 10.5 | 1117 | 1393 | 94 | 4063 | 24.7 | 25 | 1035 |
| Ollama | short | 128 | 2 | 128 | 115.9 | 2113 | 9.5 | 1114 | 1406 | 29 | 4063 | 26.5 | 28 | 1035 |
| Ollama | short | 128 | 3 | 128 | 116 | 2145.3 | 9.3 | 1113 | 1389 | 89 | 4063 | 29.4 | 31 | 1043 |
| Ollama | short | 128 | 4 | 128 | 115.8 | 2077.2 | 9.6 | 1115 | 1394 | 99 | 4063 | 30.2 | 28 | 1037 |
| Ollama | short | 128 | 5 | 128 | 117.3 | 2114.6 | 9.5 | 1100 | 1389 | 99 | 4063 | 27.5 | 31 | 1040 |
| Ollama | short | 128 | 6 | 128 | 117.4 | 1991.4 | 10 | 1100 | 1396 | 1 | 4063 | 30 | 25 | 1037 |
| Ollama | short | 128 | 7 | 128 | 117.5 | 2128.5 | 9.4 | 1098 | 1376 | 69 | 4063 | 30.5 | 20 | 1040 |
| Ollama | short | 128 | 8 | 128 | 116 | 2245.1 | 8.9 | 1112 | 1398 | 1 | 4063 | 25.6 | 21 | 1037 |
| Ollama | short | 128 | 9 | 128 | 115.7 | 2247.2 | 8.9 | 1115 | 1381 | 0 | 4063 | 30.3 | 29 | 1037 |
| Ollama | short | 128 | 10 | 128 | 117.4 | 2110.7 | 9.5 | 1099 | 1381 | 0 | 4063 | 27.5 | 23 | 1040 |
| Ollama | short | 128 | 11 | 128 | 115.6 | 2114 | 9.5 | 1117 | 1381 | 1 | 4063 | 30.5 | 25 | 1040 |
| Ollama | short | 128 | 12 | 128 | 113 | 2234.9 | 8.9 | 1142 | 1425 | 0 | 4063 | 30.1 | 25 | 1037 |
| Ollama | short | 128 | 13 | 128 | 116.1 | 2215.3 | 9 | 1112 | 1387 | 0 | 4063 | 28.7 | 30 | 1038 |
| Ollama | short | 128 | 14 | 128 | 115 | 2140.1 | 9.3 | 1122 | 1390 | 0 | 4063 | 31.4 | 27 | 1040 |
| Ollama | short | 128 | 15 | 128 | 117.5 | 2215.4 | 9 | 1098 | 1406 | 75 | 4063 | 26.7 | 24 | 1038 |
| Ollama | short | 128 | 16 | 128 | 118.4 | 2128.7 | 9.4 | 1091 | 1384 | 0 | 4063 | 31.6 | 29 | 1040 |
| Ollama | short | 128 | 17 | 128 | 122.4 | 2098.3 | 9.5 | 1056 | 1386 | 1 | 4063 | 26.8 | 29 | 1038 |
| Ollama | short | 128 | 18 | 128 | 116.2 | 2113.5 | 9.5 | 1111 | 1383 | 1 | 4063 | 26.6 | 31 | 1038 |
| Ollama | short | 128 | 19 | 128 | 120.6 | 2155.7 | 9.3 | 1071 | 1383 | 92 | 4063 | 24.3 | 21 | 1038 |
| Ollama | short | 128 | 20 | 128 | 116.9 | 1141 | 17.5 | 1113 | 1386 | 1 | 4063 | 54.7 | 25 | 1039 |
| Ollama | short | 128 | 21 | 128 | 113.5 | 2246.1 | 8.9 | 1136 | 1391 | 0 | 4063 | 26.7 | 21 | 1040 |
| Ollama | short | 128 | 22 | 128 | 115.5 | 1649.1 | 12.1 | 1120 | 1390 | 1 | 4063 | 30.8 | 17 | 1040 |
| Ollama | short | 128 | 23 | 128 | 116.3 | 2245.7 | 8.9 | 1109 | 1386 | 1 | 4063 | 30.4 | 27 | 1038 |
| Ollama | short | 128 | 24 | 128 | 117 | 2246 | 8.9 | 1103 | 1385 | 77 | 4063 | 63.5 | 24 | 1040 |
| Ollama | short | 128 | 25 | 128 | 118.8 | 2127.5 | 9.4 | 1087 | 1386 | 92 | 4063 | 27 | 22 | 1038 |
| Ollama | short | 128 | 26 | 128 | 116.7 | 2116.2 | 9.5 | 1106 | 1389 | 87 | 4063 | 52.6 | 18 | 1038 |
| Ollama | short | 128 | 27 | 128 | 117 | 2089.7 | 9.6 | 1103 | 1380 | 78 | 4063 | 30.6 | 27 | 1038 |
| Ollama | short | 128 | 28 | 128 | 119.8 | 2092.2 | 9.6 | 1078 | 1377 | 6 | 4063 | 25.8 | 29 | 1039 |
| Ollama | short | 128 | 29 | 128 | 117.5 | 2237.6 | 8.9 | 1098 | 1400 | 6 | 4063 | 31.9 | 32 | 1038 |
| Ollama | short | 128 | 30 | 128 | 120.4 | 2223.7 | 9 | 1072 | 1389 | 0 | 4063 | 29.3 | 23 | 1039 |
| Ollama | short | 512 | 1 | 353 | 115.6 | 2107.7 | 9.5 | 3062 | 3429 | 44.5 | 4063 | 47.4 | 29 | 1040 |
| Ollama | short | 512 | 2 | 343 | 119.3 | 1593.4 | 12.6 | 2888 | 3333 | 89 | 4063 | 64.5 | 24.5 | 1041 |
| Ollama | short | 512 | 3 | 333 | 114.9 | 1599.9 | 12.5 | 2911 | 3275 | 44 | 4063 | 46.8 | 27.5 | 1042 |
| Ollama | short | 512 | 4 | 276 | 116.1 | 1574.4 | 12.7 | 2391 | 2777 | 91 | 4063 | 29.3 | 28 | 1039 |
| Ollama | short | 512 | 5 | 246 | 115.7 | 1240.1 | 16.1 | 2143 | 2462 | 0 | 4063 | 30.7 | 18 | 1042 |
| Ollama | short | 512 | 6 | 387 | 117 | 1513.8 | 13.2 | 3321 | 3765 | 44.5 | 4063 | 45.4 | 24.5 | 1042 |
| Ollama | short | 512 | 7 | 315 | 117.4 | 1256.6 | 15.9 | 2699 | 3171 | 0 | 4074 | 53.1 | 46 | 1040 |
| Ollama | short | 512 | 8 | 290 | 115.1 | 712.4 | 28.1 | 2547 | 10050 | 22.2 | 4063 | 16 | 37.4 | 1033 |
| Ollama | short | 512 | 9 | 322 | 109.8 | 707.1 | 28.3 | 2961 | 5414 | 30 | 4063 | 41.3 | 16 | 1041 |
| Ollama | short | 512 | 10 | 292 | 112 | 1330.6 | 15 | 2623 | 3004 | 1 | 4063 | 25.1 | 43 | 1037 |
| Ollama | short | 512 | 11 | 312 | 109.2 | 1311.2 | 15.3 | 2873 | 3262 | 46.5 | 4070 | 37.8 | 37.5 | 1040 |
| Ollama | short | 512 | 12 | 369 | 116.7 | 1263.2 | 15.8 | 3178 | 3578 | 45 | 4070 | 45 | 23.5 | 1040 |
| Ollama | short | 512 | 13 | 284 | 114.2 | 1604.9 | 12.5 | 2500 | 2888 | 27 | 4070 | 26.7 | 49 | 1040 |
| Ollama | short | 512 | 14 | 385 | 115.9 | 1483 | 13.5 | 3335 | 3734 | 45.5 | 4070 | 39.1 | 28.5 | 1041 |
| Ollama | short | 512 | 15 | 290 | 111.5 | 1302.8 | 15.4 | 2617 | 2965 | 4 | 4070 | 25.1 | 45 | 1041 |
| Ollama | short | 512 | 16 | 292 | 117 | 1525.6 | 13.1 | 2508 | 2942 | 1 | 4070 | 30.6 | 28 | 1041 |
| Ollama | short | 512 | 17 | 265 | 115.4 | 1511.4 | 13.2 | 2309 | 2670 | 0 | 4070 | 30.7 | 27 | 1041 |
| Ollama | short | 512 | 18 | 294 | 116.1 | 1651.2 | 12.1 | 2544 | 2904 | 0 | 4070 | 30.6 | 17 | 1040 |
| Ollama | short | 512 | 19 | 384 | 113.5 | 1534.7 | 13 | 3397 | 3783 | 43.5 | 4070 | 45 | 34.5 | 1041 |
| Ollama | short | 512 | 20 | 298 | 113.8 | 1585.4 | 12.6 | 2631 | 2951 | 2 | 4070 | 25.9 | 26 | 1039 |
| Ollama | short | 512 | 21 | 299 | 116.2 | 1579.8 | 12.7 | 2586 | 2956 | 0 | 4070 | 26.5 | 23 | 1041 |
| Ollama | short | 512 | 22 | 310 | 116.1 | 1575.6 | 12.7 | 2683 | 3064 | 1 | 4070 | 30.8 | 24 | 1041 |
| Ollama | short | 512 | 23 | 338 | 115.6 | 1307.5 | 15.3 | 2940 | 3310 | 74.5 | 4070 | 45.1 | 21.5 | 1040 |
| Ollama | short | 512 | 24 | 340 | 115 | 1586.6 | 12.6 | 2968 | 5369 | 41.7 | 4070 | 31.7 | 15 | 1039 |
| Ollama | short | 512 | 25 | 305 | 115.9 | 1520.2 | 13.2 | 2644 | 3061 | 88 | 4070 | 30.8 | 32 | 1038 |
| Ollama | short | 512 | 26 | 341 | 116.7 | 1538.8 | 13 | 2934 | 3362 | 55.5 | 4070 | 62.7 | 20 | 1041 |
| Ollama | short | 512 | 27 | 305 | 114.7 | 1521.8 | 13.1 | 2672 | 3015 | 2 | 4070 | 26.9 | 26 | 1042 |
| Ollama | short | 512 | 28 | 281 | 115.6 | 1377.6 | 14.5 | 2446 | 2826 | 0 | 4070 | 26.1 | 22 | 1042 |
| Ollama | short | 512 | 29 | 285 | 115.6 | 1526 | 13.1 | 2478 | 2823 | 0 | 4070 | 26.9 | 41 | 1042 |
| Ollama | short | 512 | 30 | 301 | 116.6 | 1470 | 13.6 | 2595 | 2977 | 0 | 4070 | 25.9 | 26 | 1041 |

