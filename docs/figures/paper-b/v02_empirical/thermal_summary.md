# Thermal Rank --- Sustained-Decode Empirical Trace

Hardware: RTX 4070 Laptop (8 GB VRAM, ~40 TFLOPS FP16 peak); model: Meta-Llama-3.1-8B-Instruct Q4_K_M (8.31 B params, 4693 MB on-disk).
Telemetry: `nvidia-smi` polled at 1 Hz (N=94 samples, 86 active, 8 idle).

## Active-window summary (util ≥ 20 %)

| metric | min | p50 | mean | p90 | max |
|---|---:|---:|---:|---:|---:|
| GPU temp (°C) | 41 | 64 | 61.2 | 74 | 75 |
| SM clock (MHz) | 315 | 2235 | 1708.8 | 2235 | 2235 |
| MEM clock (MHz) | 405 | 8001 | 5956.5 | 8001 | 8001 |
| Power (W) | 2 | 100 | 66.3 | 109 | 116 |

## Sustained-decode drift (early third vs late third of active window)

| | SM MHz | Power W | GPU °C |
|---|---:|---:|---:|
| early | 2135 | 76.4 | 58.8 |
| late  | 724 | 22.8 | 54.5 |
| Δ     | -1411 (-66.1 %) | --- | -4.2 |

## Decode throughput drift

- First decode: 25.8 tok/s
- Last decode:  28.0 tok/s
- Δ: +8.5 %

## Energy efficiency

- Mean decode rate: 27.5 tok/s
- Mean active power: 66.3 W
- Tokens per joule (TpJ) ≈ 0.415

## Interpretation for paper-B §Thermal Rank

The Thermal Rank module (`runtime/nn/thermal_rank.c`) consumes exactly the telemetry channels measured here (NVML `nvmlDeviceGetTemperature`, `nvmlDeviceGetPowerUsage`). The thresholds in code default to T_low=65 °C -> full rank, T_high=85 °C -> min rank, with a linear interpolation between. The empirically observed active-window distribution (above) shows whether the workload reaches the actuation band. If the p90 active temp sits below T_low, the rank is held at full and Thermal Rank reduces to a no-op for this workload (which is the correct behaviour --- no throttling pressure means no rank reduction is needed); the feature only differentiates from fixed-rank operation under thermal load.

_Generated from `thermal_sustained_8b.csv` (94 samples, 86 active)._