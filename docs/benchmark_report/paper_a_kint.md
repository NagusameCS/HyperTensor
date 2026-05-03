## Paper A — k_int Generalisation Across Architectures

Summary: across the sampled models, intrinsic rank k_int (95% joint-Gram variance) stays below d and is often in the 0.5–0.7 d range, supporting rank-limited attention compression.

| Model | d | Mean k_int | k_int/d | Min k_int | Max k_int | Layers sampled |
|-------|---|-----------|---------|-----------|-----------|----------------|
| SmolLM2-135M | 576 | 299.4 | 0.5198 | 122 | 378 | 5 |
| Gemma4-2B | 1536 | 947.6 | 0.6169 | 867 | 1121 | 7 |
| Qwen3.5-35B | 2048 | 1385.0 | 0.6763 | 1325 | 1475 | 5 |
| Qwen3.5-MoE-30B-A3.5B | 2048 | 1385.0 | 0.6763 | 1325 | 1475 | 5 |
| Gemma3-4B | 2560 | 1460.5 | 0.5705 | 1261 | 1635 | 6 |
| Gemma3-12B | 3840 | 2499.6 | 0.6509 | 1719 | 2885 | 7 |
| Gemma4-27B | 5376 | 3803.2 | 0.7074 | 3165 | 4342 | 6 |
| Gemma4-31B | 5376 | 3803.2 | 0.7074 | 3165 | 4342 | 6 |

### Per-layer k_int detail

SmolLM2-135M (d=576)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 122 | 0.2118 |
| 5 | 364 | 0.6319 |
| 10 | 378 | 0.6562 |
| 15 | 315 | 0.5469 |
| 20 | 318 | 0.5521 |

Gemma4-2B (d=1536)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1002 | 0.6523 |
| 4 | 1121 | 0.7298 |
| 8 | 918 | 0.5977 |
| 12 | 867 | 0.5645 |
| 16 | 921 | 0.5996 |
| 20 | 917 | 0.5970 |
| 25 | 887 | 0.5775 |

Qwen3.5-35B (d=2048)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1382 | 0.6748 |
| 10 | 1391 | 0.6792 |
| 20 | 1352 | 0.6602 |
| 30 | 1325 | 0.6470 |
| 39 | 1475 | 0.7202 |

Qwen3.5-MoE-30B-A3.5B (d=2048)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1382 | 0.6748 |
| 10 | 1391 | 0.6792 |
| 20 | 1352 | 0.6602 |
| 30 | 1325 | 0.6470 |
| 39 | 1475 | 0.7202 |

Gemma3-4B (d=2560)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1564 | 0.6109 |
| 6 | 1450 | 0.5664 |
| 12 | 1367 | 0.5340 |
| 18 | 1486 | 0.5805 |
| 24 | 1635 | 0.6387 |
| 30 | 1261 | 0.4926 |

Gemma3-12B (d=3840)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 1719 | 0.4477 |
| 8 | 2559 | 0.6664 |
| 16 | 2433 | 0.6336 |
| 24 | 2263 | 0.5893 |
| 32 | 2821 | 0.7346 |
| 40 | 2885 | 0.7513 |
| 45 | 2817 | 0.7336 |

Gemma4-27B (d=5376)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 3416 | 0.6354 |
| 12 | 3165 | 0.5887 |
| 24 | 3896 | 0.7247 |
| 36 | 3998 | 0.7437 |
| 48 | 4342 | 0.8077 |
| 58 | 4002 | 0.7444 |

Gemma4-31B (d=5376)

| Layer | k_int | k_int/d |
|-------|-------|---------|
| 0 | 3416 | 0.6354 |
| 12 | 3165 | 0.5887 |
| 24 | 3896 | 0.7247 |
| 36 | 3998 | 0.7437 |
| 48 | 4342 | 0.8077 |
| 58 | 4002 | 0.7444 |

### Data-Quality Notes

Some model pairs have identical per-layer k_int vectors. This may reflect shared checkpoints, aliasing in model selection, or a pipeline mapping issue and should be verified before publication.

- Identical k_int profile: Gemma4-27B, Gemma4-31B
- Identical k_int profile: Qwen3.5-35B, Qwen3.5-MoE-30B-A3.5B

### LaTeX Generalisation Table (copy-paste)

```latex
\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Model & $d$ & $\bar{k}_\mathrm{int}$ & $\bar{k}_\mathrm{int}/d$ \\
\midrule
  SmolLM2-135M & 576 & 299.4 & 0.5198 \\
  Gemma4-2B & 1536 & 947.6 & 0.6169 \\
  Qwen3.5-35B & 2048 & 1385.0 & 0.6763 \\
  Qwen3.5-MoE-30B-A3.5B & 2048 & 1385.0 & 0.6763 \\
  Gemma3-4B & 2560 & 1460.5 & 0.5705 \\
  Gemma3-12B & 3840 & 2499.6 & 0.6509 \\
  Gemma4-27B & 5376 & 3803.2 & 0.7074 \\
  Gemma4-31B & 5376 & 3803.2 & 0.7074 \\
\bottomrule
\end{tabular}
\caption{Intrinsic rank $k_\mathrm{int}$ at 95\% joint Gram variance}
\end{table}
```
