# TensorOS Inference Benchmark Report

**Date:** 2026-04-17 10:58

**CPU:** AMD Ryzen 9 7940HS w/ Radeon 780M Graphics     

**GPU:** NVIDIA GeForce RTX 4070 Laptop GPU

**Models:** smollm2-135m-instruct Q8_0 (138 MB) + Gemma-4-E2B Q4_0 (3.2 GB)

**Trials:** 2 per condition | **Token counts:** 40 / 128 / 512

**Column definitions:**

- **Decode t/s** = generation tokens per second (decode phase only)
- **Prefill t/s** = prompt tokens processed per second
- **TTFT ms** = Time To First Token (prefill latency)
- **Gen ms** = decode wall-time (excludes prefill)
- **Total ms** = gen ms + TTFT ms

---

## Summary -- Average Across All Conditions

| Runtime | Backend | Model | Decode t/s | Prefill t/s | TTFT ms | Gen ms |
|---------|---------|-------|:----------:|:-----------:|:-------:|:------:|
| Geodessical | CPU | gemma4-2b | 7 | 12 | 1107.9 | 20449.2 |
| Geodessical | CPU | smollm2-135m | 34.3 | 43.2 | 473.2 | 3724.1 |
| Geodessical | GPU | gemma4-2b | 33.8 | 101.2 | 165.7 | 804.5 |
| Geodessical | GPU | smollm2-135m | 243 | 315.2 | 74.2 | 539.3 |
| Ollama | CPU | gemma4-2b | 26.5 | 719.4 | 83.2 | 7184.7 |
| Ollama | CPU | smollm2-135m | 148.5 | 4758.5 | 12.2 | 870.8 |
| Ollama | GPU | gemma4-2b | 103.6 | 2719.1 | 13.2 | 2236.1 |
| Ollama | GPU | smollm2-135m | 726.1 | 17122.7 | 5.3 | 160.6 |

---

## Model: gemma4-2b

### Prompt: long

| Runtime | Backend | N | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |
|---------|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|
| Geodessical | GPU | 40 | 7 | 9.8 | 84.7 | 245.5 | 718.5 | 964 |
| Geodessical | GPU | 128 | 7 | 4 | 85.9 | 234 | 1724.5 | 1958.5 |
| Geodessical | GPU | 512 | 7 | 4 | 85.4 | 241 | 1751 | 1992 |
| Geodessical | CPU | 40 | 40 | 8.3 | 12.4 | 1588.5 | 4801.5 | 6390 |
| Geodessical | CPU | 128 | 128 | 10.7 | 12.3 | 1595 | 11979 | 13574 |
| Geodessical | CPU | 512 | 512 | 11 | 11.4 | 1563.5 | 46618 | 48181.5 |
| Ollama | GPU | 40 | 40 | 120.8 | 3305.6 | 15.6 | 331.5 | 347 |
| Ollama | GPU | 128 | 128 | 78 | 4204.4 | 10.3 | 1649 | 1660 |
| Ollama | GPU | 512 | 512 | 72.6 | 3391.6 | 14 | 7048 | 7062 |
| Ollama | CPU | 40 | 40 | 25.3 | 578.1 | 225.2 | 1586 | 1811 |
| Ollama | CPU | 128 | 128 | 29.2 | 1190.8 | 36.1 | 4372.5 | 4409 |
| Ollama | CPU | 512 | 512 | 28.6 | 1179.6 | 36.5 | 17894 | 17930.5 |

### Prompt: short

| Runtime | Backend | N | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |
|---------|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|
| Geodessical | GPU | 40 | 13 | 61.4 | 116.8 | 92 | 212 | 304 |
| Geodessical | GPU | 128 | 13 | 62.2 | 118 | 90 | 209 | 299 |
| Geodessical | GPU | 512 | 13 | 61.4 | 116.6 | 91.5 | 212 | 303.5 |
| Geodessical | CPU | 40 | 33 | 8.6 | 12.4 | 632.5 | 3854.5 | 4487 |
| Geodessical | CPU | 128 | 33 | 2.9 | 11.8 | 635 | 11438.5 | 12073.5 |
| Geodessical | CPU | 512 | 33 | 0.8 | 11.8 | 633 | 44004 | 44637 |
| Ollama | GPU | 40 | 40 | 115.6 | 1480.6 | 19 | 346 | 365 |
| Ollama | GPU | 128 | 128 | 117.8 | 2072.4 | 9.6 | 1087 | 1096.5 |
| Ollama | GPU | 512 | 345 | 116.8 | 1860 | 11 | 2955 | 2966.5 |
| Ollama | CPU | 40 | 40 | 28.8 | 313.4 | 124.4 | 1391.5 | 1515.5 |
| Ollama | CPU | 128 | 128 | 23 | 554.8 | 36.1 | 5573.5 | 5609 |
| Ollama | CPU | 512 | 295 | 24.2 | 499.4 | 40.8 | 12290.5 | 12331 |

### GPU vs CPU Speedup -- gemma4-2b

| Runtime | N | GPU Decode t/s | CPU Decode t/s | Speedup |
|---------|--:|:--------------:|:--------------:|:-------:|
| Geodessical | 40 | 35.6 | 8.4 | 4.2x |
| Geodessical | 128 | 33.1 | 6.8 | 4.9x |
| Geodessical | 512 | 32.7 | 5.9 | 5.5x |
| Ollama | 40 | 118.2 | 27 | 4.4x |
| Ollama | 128 | 97.9 | 26.2 | 3.7x |
| Ollama | 512 | 94.7 | 26.4 | 3.6x |

---

## Model: smollm2-135m

### Prompt: long

| Runtime | Backend | N | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |
|---------|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|
| Geodessical | GPU | 40 | 40 | 174.3 | 322.6 | 105 | 229.5 | 334.5 |
| Geodessical | GPU | 128 | 128 | 252.8 | 322.4 | 109 | 506.5 | 615.5 |
| Geodessical | GPU | 512 | 512 | 271.4 | 287.3 | 104.5 | 1887 | 1991.5 |
| Geodessical | CPU | 40 | 40 | 25 | 42.8 | 671 | 1605.5 | 2276.5 |
| Geodessical | CPU | 128 | 128 | 35 | 43 | 681 | 3654 | 4335 |
| Geodessical | CPU | 512 | 512 | 39.8 | 42 | 679.5 | 12850 | 13529.5 |
| Ollama | GPU | 40 | 40 | 608.8 | 14086.4 | 6.2 | 67.5 | 74 |
| Ollama | GPU | 128 | 128 | 732 | 27908.8 | 2.1 | 175 | 177 |
| Ollama | GPU | 512 | 307 | 759.8 | 22950.9 | 2.6 | 415.5 | 418 |
| Ollama | CPU | 40 | 40 | 154.1 | 4690.5 | 15 | 260 | 275 |
| Ollama | CPU | 128 | 128 | 149 | 7275.6 | 8 | 859.5 | 867.5 |
| Ollama | CPU | 512 | 512 | 174.9 | 6930.1 | 8.4 | 2938.5 | 2947 |

### Prompt: short

| Runtime | Backend | N | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |
|---------|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|
| Geodessical | GPU | 40 | 40 | 228.5 | 301 | 42 | 176 | 218 |
| Geodessical | GPU | 128 | 58 | 266 | 329.2 | 42 | 218 | 260 |
| Geodessical | GPU | 512 | 58 | 264.8 | 328.6 | 42.5 | 219 | 261.5 |
| Geodessical | CPU | 40 | 40 | 33.7 | 43.7 | 273.5 | 1189 | 1462.5 |
| Geodessical | CPU | 128 | 55 | 36 | 43.6 | 265.5 | 1526.5 | 1792 |
| Geodessical | CPU | 512 | 55 | 36.2 | 44 | 269 | 1519.5 | 1788.5 |
| Ollama | GPU | 40 | 38 | 808.2 | 7453 | 14.4 | 47.5 | 62 |
| Ollama | GPU | 128 | 78 | 718.1 | 13554.3 | 2.6 | 108.5 | 110 |
| Ollama | GPU | 512 | 109 | 729.4 | 14998.6 | 2.4 | 149.5 | 152 |
| Ollama | CPU | 40 | 40 | 144.7 | 2803.7 | 17.5 | 277 | 294.5 |
| Ollama | CPU | 128 | 75 | 137.4 | 3854.6 | 9.1 | 549 | 558 |
| Ollama | CPU | 512 | 46 | 130.6 | 2996.4 | 15.4 | 341 | 356.5 |

### GPU vs CPU Speedup -- smollm2-135m

| Runtime | N | GPU Decode t/s | CPU Decode t/s | Speedup |
|---------|--:|:--------------:|:--------------:|:-------:|
| Geodessical | 40 | 201.4 | 29.3 | 6.9x |
| Geodessical | 128 | 259.4 | 35.6 | 7.3x |
| Geodessical | 512 | 268.1 | 38 | 7.1x |
| Ollama | 40 | 708.5 | 149.4 | 4.7x |
| Ollama | 128 | 725.1 | 143.2 | 5.1x |
| Ollama | 512 | 744.7 | 152.7 | 4.9x |

---

## Raw Results (all individual trials)

| Runtime | Backend | Model | Prompt | N | Trial | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |
|---------|---------|-------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|
| Geodessical | CPU | gemma4-2b | long | 40 | 1 | 40 | 8.2 | 12.3 | 1609 | 4851 | 6460 |
| Geodessical | CPU | gemma4-2b | long | 40 | 2 | 40 | 8.4 | 12.6 | 1568 | 4752 | 6320 |
| Geodessical | CPU | gemma4-2b | long | 128 | 1 | 128 | 10.8 | 12.4 | 1573 | 11893 | 13466 |
| Geodessical | CPU | gemma4-2b | long | 128 | 2 | 128 | 10.6 | 12.2 | 1617 | 12065 | 13682 |
| Geodessical | CPU | gemma4-2b | long | 512 | 1 | 512 | 11.4 | 11.9 | 1560 | 44771 | 46331 |
| Geodessical | CPU | gemma4-2b | long | 512 | 2 | 512 | 10.6 | 10.9 | 1567 | 48465 | 50032 |
| Geodessical | CPU | gemma4-2b | short | 40 | 1 | 33 | 8.5 | 12.4 | 642 | 3880 | 4522 |
| Geodessical | CPU | gemma4-2b | short | 40 | 2 | 33 | 8.6 | 12.5 | 623 | 3829 | 4452 |
| Geodessical | CPU | gemma4-2b | short | 128 | 1 | 33 | 3 | 12.1 | 622 | 11183 | 11805 |
| Geodessical | CPU | gemma4-2b | short | 128 | 2 | 33 | 2.8 | 11.6 | 648 | 11694 | 12342 |
| Geodessical | CPU | gemma4-2b | short | 512 | 1 | 33 | 0.7 | 11.8 | 629 | 44127 | 44756 |
| Geodessical | CPU | gemma4-2b | short | 512 | 2 | 33 | 0.8 | 11.8 | 637 | 43881 | 44518 |
| Geodessical | CPU | smollm2-135m | long | 40 | 1 | 40 | 24.6 | 41.9 | 673 | 1628 | 2301 |
| Geodessical | CPU | smollm2-135m | long | 40 | 2 | 40 | 25.3 | 43.8 | 669 | 1583 | 2252 |
| Geodessical | CPU | smollm2-135m | long | 128 | 1 | 128 | 35.6 | 43.5 | 657 | 3599 | 4256 |
| Geodessical | CPU | smollm2-135m | long | 128 | 2 | 128 | 34.5 | 42.6 | 705 | 3709 | 4414 |
| Geodessical | CPU | smollm2-135m | long | 512 | 1 | 512 | 39.9 | 42.1 | 687 | 12837 | 13524 |
| Geodessical | CPU | smollm2-135m | long | 512 | 2 | 512 | 39.8 | 42 | 672 | 12863 | 13535 |
| Geodessical | CPU | smollm2-135m | short | 40 | 1 | 40 | 33.7 | 43.7 | 271 | 1186 | 1457 |
| Geodessical | CPU | smollm2-135m | short | 40 | 2 | 40 | 33.6 | 43.7 | 276 | 1192 | 1468 |
| Geodessical | CPU | smollm2-135m | short | 128 | 1 | 55 | 36 | 43.4 | 264 | 1529 | 1793 |
| Geodessical | CPU | smollm2-135m | short | 128 | 2 | 55 | 36.1 | 43.8 | 267 | 1524 | 1791 |
| Geodessical | CPU | smollm2-135m | short | 512 | 1 | 55 | 36.5 | 44.4 | 270 | 1508 | 1778 |
| Geodessical | CPU | smollm2-135m | short | 512 | 2 | 55 | 35.9 | 43.6 | 268 | 1531 | 1799 |
| Geodessical | GPU | gemma4-2b | long | 40 | 1 | 7 | 9.7 | 85 | 251 | 723 | 974 |
| Geodessical | GPU | gemma4-2b | long | 40 | 2 | 7 | 9.8 | 84.4 | 240 | 714 | 954 |
| Geodessical | GPU | gemma4-2b | long | 128 | 1 | 7 | 4.1 | 86.7 | 235 | 1712 | 1947 |
| Geodessical | GPU | gemma4-2b | long | 128 | 2 | 7 | 4 | 85.1 | 233 | 1737 | 1970 |
| Geodessical | GPU | gemma4-2b | long | 512 | 1 | 7 | 4 | 85.7 | 240 | 1745 | 1985 |
| Geodessical | GPU | gemma4-2b | long | 512 | 2 | 7 | 4 | 85.2 | 242 | 1757 | 1999 |
| Geodessical | GPU | gemma4-2b | short | 40 | 1 | 13 | 62.5 | 120.7 | 92 | 208 | 300 |
| Geodessical | GPU | gemma4-2b | short | 40 | 2 | 13 | 60.2 | 112.8 | 92 | 216 | 308 |
| Geodessical | GPU | gemma4-2b | short | 128 | 1 | 13 | 63.7 | 123.6 | 91 | 204 | 295 |
| Geodessical | GPU | gemma4-2b | short | 128 | 2 | 13 | 60.7 | 112.3 | 89 | 214 | 303 |
| Geodessical | GPU | gemma4-2b | short | 512 | 1 | 13 | 63.1 | 121.6 | 91 | 206 | 297 |
| Geodessical | GPU | gemma4-2b | short | 512 | 2 | 13 | 59.6 | 111.6 | 92 | 218 | 310 |
| Geodessical | GPU | smollm2-135m | long | 40 | 1 | 40 | 176.2 | 323.5 | 103 | 227 | 330 |
| Geodessical | GPU | smollm2-135m | long | 40 | 2 | 40 | 172.4 | 321.8 | 107 | 232 | 339 |
| Geodessical | GPU | smollm2-135m | long | 128 | 1 | 128 | 253.5 | 322.2 | 107 | 505 | 612 |
| Geodessical | GPU | smollm2-135m | long | 128 | 2 | 128 | 252 | 322.5 | 111 | 508 | 619 |
| Geodessical | GPU | smollm2-135m | long | 512 | 1 | 512 | 271.5 | 287.3 | 104 | 1886 | 1990 |
| Geodessical | GPU | smollm2-135m | long | 512 | 2 | 512 | 271.2 | 287.3 | 105 | 1888 | 1993 |
| Geodessical | GPU | smollm2-135m | short | 40 | 1 | 40 | 211.6 | 274.1 | 43 | 189 | 232 |
| Geodessical | GPU | smollm2-135m | short | 40 | 2 | 40 | 245.4 | 327.8 | 41 | 163 | 204 |
| Geodessical | GPU | smollm2-135m | short | 128 | 1 | 58 | 267.3 | 329.7 | 41 | 217 | 258 |
| Geodessical | GPU | smollm2-135m | short | 128 | 2 | 58 | 264.8 | 328.8 | 43 | 219 | 262 |
| Geodessical | GPU | smollm2-135m | short | 512 | 1 | 58 | 264.8 | 329.5 | 43 | 219 | 262 |
| Geodessical | GPU | smollm2-135m | short | 512 | 2 | 58 | 264.8 | 327.7 | 42 | 219 | 261 |
| Ollama | CPU | gemma4-2b | long | 40 | 1 | 40 | 24 | 105 | 409.4 | 1669 | 2078 |
| Ollama | CPU | gemma4-2b | long | 40 | 2 | 40 | 26.6 | 1051.2 | 40.9 | 1503 | 1544 |
| Ollama | CPU | gemma4-2b | long | 128 | 1 | 128 | 29.4 | 1168.4 | 36.8 | 4349 | 4386 |
| Ollama | CPU | gemma4-2b | long | 128 | 2 | 128 | 29.1 | 1213.2 | 35.4 | 4396 | 4432 |
| Ollama | CPU | gemma4-2b | long | 512 | 1 | 512 | 28.6 | 1154 | 37.3 | 17877 | 17914 |
| Ollama | CPU | gemma4-2b | long | 512 | 2 | 512 | 28.6 | 1205.3 | 35.7 | 17911 | 17947 |
| Ollama | CPU | gemma4-2b | short | 40 | 1 | 40 | 28.1 | 94.7 | 211.2 | 1421 | 1632 |
| Ollama | CPU | gemma4-2b | short | 40 | 2 | 40 | 29.4 | 532.1 | 37.6 | 1362 | 1399 |
| Ollama | CPU | gemma4-2b | short | 128 | 1 | 128 | 24.7 | 547 | 36.6 | 5173 | 5209 |
| Ollama | CPU | gemma4-2b | short | 128 | 2 | 128 | 21.4 | 562.5 | 35.6 | 5974 | 6009 |
| Ollama | CPU | gemma4-2b | short | 512 | 1 | 267 | 25.5 | 430.6 | 46.4 | 10465 | 10511 |
| Ollama | CPU | gemma4-2b | short | 512 | 2 | 323 | 22.9 | 568.3 | 35.2 | 14116 | 14151 |
| Ollama | CPU | smollm2-135m | long | 40 | 1 | 40 | 147.1 | 2723.7 | 21.3 | 272 | 293 |
| Ollama | CPU | smollm2-135m | long | 40 | 2 | 40 | 161.2 | 6657.4 | 8.7 | 248 | 257 |
| Ollama | CPU | smollm2-135m | long | 128 | 1 | 128 | 145.3 | 6674.9 | 8.7 | 881 | 890 |
| Ollama | CPU | smollm2-135m | long | 128 | 2 | 128 | 152.8 | 7876.2 | 7.4 | 838 | 845 |
| Ollama | CPU | smollm2-135m | long | 512 | 1 | 512 | 164.4 | 7049.8 | 8.2 | 3115 | 3123 |
| Ollama | CPU | smollm2-135m | long | 512 | 2 | 512 | 185.4 | 6810.4 | 8.5 | 2762 | 2771 |
| Ollama | CPU | smollm2-135m | short | 40 | 1 | 40 | 136.3 | 1303.2 | 26.9 | 293 | 320 |
| Ollama | CPU | smollm2-135m | short | 40 | 2 | 40 | 153.1 | 4304.2 | 8.1 | 261 | 269 |
| Ollama | CPU | smollm2-135m | short | 128 | 1 | 65 | 142.3 | 3734.6 | 9.4 | 457 | 466 |
| Ollama | CPU | smollm2-135m | short | 128 | 2 | 85 | 132.6 | 3974.5 | 8.8 | 641 | 650 |
| Ollama | CPU | smollm2-135m | short | 512 | 1 | 13 | 122 | 1519.5 | 23 | 107 | 130 |
| Ollama | CPU | smollm2-135m | short | 512 | 2 | 80 | 139.1 | 4473.4 | 7.8 | 575 | 583 |
| Ollama | GPU | gemma4-2b | long | 40 | 1 | 40 | 120.2 | 1965.5 | 21.9 | 333 | 355 |
| Ollama | GPU | gemma4-2b | long | 40 | 2 | 40 | 121.3 | 4645.7 | 9.3 | 330 | 339 |
| Ollama | GPU | gemma4-2b | long | 128 | 1 | 128 | 83.8 | 4585.4 | 9.4 | 1527 | 1537 |
| Ollama | GPU | gemma4-2b | long | 128 | 2 | 128 | 72.3 | 3823.4 | 11.2 | 1771 | 1783 |
| Ollama | GPU | gemma4-2b | long | 512 | 1 | 512 | 72.6 | 4461.7 | 9.6 | 7056 | 7066 |
| Ollama | GPU | gemma4-2b | long | 512 | 2 | 512 | 72.7 | 2321.5 | 18.5 | 7040 | 7058 |
| Ollama | GPU | gemma4-2b | short | 40 | 1 | 40 | 114.6 | 688.5 | 29.1 | 349 | 378 |
| Ollama | GPU | gemma4-2b | short | 40 | 2 | 40 | 116.6 | 2272.8 | 8.8 | 343 | 352 |
| Ollama | GPU | gemma4-2b | short | 128 | 1 | 128 | 117.7 | 2146.1 | 9.3 | 1088 | 1097 |
| Ollama | GPU | gemma4-2b | short | 128 | 2 | 128 | 117.8 | 1998.8 | 10 | 1086 | 1096 |
| Ollama | GPU | gemma4-2b | short | 512 | 1 | 361 | 116.2 | 2118.5 | 9.4 | 3106 | 3116 |
| Ollama | GPU | gemma4-2b | short | 512 | 2 | 329 | 117.3 | 1601.6 | 12.5 | 2804 | 2817 |
| Ollama | GPU | smollm2-135m | long | 40 | 1 | 40 | 507.6 | 5832.6 | 9.9 | 79 | 89 |
| Ollama | GPU | smollm2-135m | long | 40 | 2 | 40 | 710 | 22340.3 | 2.6 | 56 | 59 |
| Ollama | GPU | smollm2-135m | long | 128 | 1 | 128 | 703.9 | 27943.7 | 2.1 | 182 | 184 |
| Ollama | GPU | smollm2-135m | long | 128 | 2 | 128 | 760.2 | 27873.9 | 2.1 | 168 | 170 |
| Ollama | GPU | smollm2-135m | long | 512 | 1 | 102 | 790.4 | 24649.4 | 2.4 | 129 | 131 |
| Ollama | GPU | smollm2-135m | long | 512 | 2 | 512 | 729.3 | 21252.4 | 2.7 | 702 | 705 |
| Ollama | GPU | smollm2-135m | short | 40 | 1 | 40 | 851.9 | 1330.2 | 26.3 | 47 | 73 |
| Ollama | GPU | smollm2-135m | short | 40 | 2 | 37 | 764.4 | 13575.9 | 2.6 | 48 | 51 |
| Ollama | GPU | smollm2-135m | short | 128 | 1 | 87 | 836.5 | 13554.3 | 2.6 | 104 | 107 |
| Ollama | GPU | smollm2-135m | short | 128 | 2 | 68 | 599.7 | -- | -- | 113 | 113 |
| Ollama | GPU | smollm2-135m | short | 512 | 1 | 109 | 747.6 | 15841.4 | 2.2 | 146 | 148 |
| Ollama | GPU | smollm2-135m | short | 512 | 2 | 109 | 711.3 | 14155.7 | 2.5 | 153 | 156 |

---

## End-to-End Latency

**E2E = TTFT + Gen ms** (wall time from prompt submission to last token).
**ms/tok** = E2E / tokens_generated (user-perceived cost per output token).

### gemma4-2b -- E2E Total ms

| Prompt | N | Geo-CPU ms | Geo-GPU ms | Oll-CPU ms | Oll-GPU ms | Fastest |
|--------|--:|-----------:|-----------:|-----------:|-----------:|---------|
| short | 40 | 4487 | 304 | 1515.5 | 365 | Geo-GPU |
| short | 128 | 12073.5 | 299 | 5609 | 1096.5 | Geo-GPU |
| short | 512 | 44637 | 303.5 | 12331 | 2966.5 | Geo-GPU |
| long | 40 | 6390 | 964 | 1811 | 347 | Oll-GPU |
| long | 128 | 13574 | 1958.5 | 4409 | 1660 | Oll-GPU |
| long | 512 | 48181.5 | 1992 | 17930.5 | 7062 | Geo-GPU |

### gemma4-2b -- ms per Output Token (lower = better)

| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |
|--------|--:|--------:|--------:|--------:|--------:|
| short | 40 | 136 | 23.4 | 37.9 | 9.1 |
| short | 128 | 365.8 | 23 | 43.8 | 8.6 |
| short | 512 | 1352.6 | 23.3 | 41.6 | 8.6 |
| long | 40 | 159.8 | 137.7 | 45.3 | 8.7 |
| long | 128 | 106 | 279.8 | 34.4 | 13 |
| long | 512 | 94.1 | 284.6 | 35 | 13.8 |

### gemma4-2b -- TTFT ms (Time to First Token)

| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |
|--------|--:|--------:|--------:|--------:|--------:|
| short | 40 | 632.5 | 92 | 124.4 | 19 |
| short | 128 | 635 | 90 | 36.1 | 9.6 |
| short | 512 | 633 | 91.5 | 40.8 | 11 |
| long | 40 | 1588.5 | 245.5 | 225.2 | 15.6 |
| long | 128 | 1595 | 234 | 36.1 | 10.3 |
| long | 512 | 1563.5 | 241 | 36.5 | 14 |

### smollm2-135m -- E2E Total ms

| Prompt | N | Geo-CPU ms | Geo-GPU ms | Oll-CPU ms | Oll-GPU ms | Fastest |
|--------|--:|-----------:|-----------:|-----------:|-----------:|---------|
| short | 40 | 1462.5 | 218 | 294.5 | 62 | Oll-GPU |
| short | 128 | 1792 | 260 | 558 | 110 | Oll-GPU |
| short | 512 | 1788.5 | 261.5 | 356.5 | 152 | Oll-GPU |
| long | 40 | 2276.5 | 334.5 | 275 | 74 | Oll-GPU |
| long | 128 | 4335 | 615.5 | 867.5 | 177 | Oll-GPU |
| long | 512 | 13529.5 | 1991.5 | 2947 | 418 | Oll-GPU |

### smollm2-135m -- ms per Output Token (lower = better)

| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |
|--------|--:|--------:|--------:|--------:|--------:|
| short | 40 | 36.6 | 5.4 | 7.4 | 1.6 |
| short | 128 | 32.6 | 4.4 | 7.4 | 1.4 |
| short | 512 | 32.5 | 4.5 | 8.6 | 1.4 |
| long | 40 | 56.9 | 8.4 | 6.8 | 1.8 |
| long | 128 | 33.8 | 4.8 | 6.8 | 1.4 |
| long | 512 | 26.4 | 3.9 | 5.8 | 1.4 |

### smollm2-135m -- TTFT ms (Time to First Token)

| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |
|--------|--:|--------:|--------:|--------:|--------:|
| short | 40 | 273.5 | 42 | 17.5 | 14.4 |
| short | 128 | 265.5 | 42 | 9.1 | 2.6 |
| short | 512 | 269 | 42.5 | 15.4 | 2.4 |
| long | 40 | 671 | 105 | 15 | 6.2 |
| long | 128 | 681 | 109 | 8 | 2.1 |
| long | 512 | 679.5 | 104.5 | 8.4 | 2.6 |

