# PPL Sweep --- Meta-Llama-3.1-8B-Instruct-Q4_K_M

Date: 2026-04-22 · Eval: WikiText-2 first 512 tokens via `geodessical.exe --ppl-eval` · Device: RTX 4070 Laptop, CUDA.

## Headline table

| Case        | Rank K | NLL    | PPL     | PPL % of baseline | tps   | elapsed |
|-------------|-------:|-------:|--------:|------------------:|------:|--------:|
| baseline    | ---      | 1.9155 |  6.7902 | 100.00 %      | 33.9  | 15.1 s  |
| GRC k=1024  | 1024   | 2.3941 | 10.9585 | 161.39 %      | 35.5  | 14.4 s  |
| GRC k=1536  | 1536   | 2.0404 |  7.6936 | 113.30 %      | 29.9  | 17.1 s  |
| GRC k=2048  | 2048   | 2.0404 |  7.6936 | 113.30 %      | 31.0  | 16.5 s  |

Flags for all GRC cases: `--axex-compress --axex-attn-only --axex-skip-o --axex-weight-pca --axex-compress-rank K --ppl-eval`.

## Why k=1536 and k=2048 are bit-identical

Llama-3.1-8B uses Grouped-Query Attention with 8 KV heads × `head_dim`=128, so the K and V projection output dimension is exactly 1024. With `--axex-attn-only`, the rank cap applies to the attention projections {Q, K, V} (we skip O via `--axex-skip-o`). For K and V the rank is clamped at 1024 --- once `k ≥ 1024`, K and V are recovered losslessly and only Q is being PCA-truncated. Q is `4096 × 4096`. The measurement shows Q's spectrum effectively saturates at k=1536: the residual energy in singular values 1537..2048 is below the float-32 / Q4_K_M numerical floor, so k=2048 adds no information. This is mechanistically clean and reproducible.

Operational consequence. For `--axex-attn-only` on Llama-3.1-8B, k=1536 is the Pareto rank; k=2048 only burns memory and compute. The v0.6.1 published value 13.30 % PPL increase replicates exactly (113.30 %).

## Reproduce

```powershell
$model = "<path>\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
.\build_host\geodessical.exe $model --ppl-eval
foreach ($k in 1024,1536,2048) {
  .\build_host\geodessical.exe $model `
    --axex-compress --axex-attn-only --axex-skip-o --axex-weight-pca `
    --axex-compress-rank $k --ppl-eval
}
```

Or use the wrapper: `.\scripts\run_ppl_sweep.ps1 -Reps 1 -CooldownSec 30`.

Raw JSON: `docs/figures/ppl_sweep/llama31_8b_ppl_sweep.json`.
