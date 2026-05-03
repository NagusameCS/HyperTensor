# Decode Substitution Summary (v0.4-pre)

Date: 2026-04-27

## Three-bucket metric

Per decode step, classify by nearest cached anchor distance `d`:

- Correctable: `d <= rho` (Jacobi correction trusted)
- Hit-only: `rho < d <= eps_star` (cache hit but use fallback solve)
- Miss: `d > eps_star` (full forward)

Projection model:

`t_step_avg = f_corr * (t_lookup + t_jacobi) + f_hit_only * (t_lookup + 0.5*t_full) + f_miss * (t_lookup + t_full)`

with `eps_star=3.0`, `rho=0.4`, `k=32/64`.

## Runtime-cloud results (real exported clouds)

| Model | Cache hit (<=eps*) | Correctable (<=rho) | Hit-only | Miss | Projected speedup | Baseline TPS -> GTC TPS |
|---|---:|---:|---:|---:|---:|---:|
| smollm2-135m | 100.0% | 0.0% | 100.0% | 0.0% | 1.972x | 200.0 -> 394.4 |
| phi-3.5-mini | 100.0% | 0.0% | 100.0% | 0.0% | 1.969x | 200.0 -> 393.8 |
| gemma-4-e2b | 100.0% | 0.0% | 100.0% | 0.0% | 1.988x | 107.5 -> 213.8 |

Sources:
- smollm2-135m_decode_substitution.json
- phi-3.5-mini_decode_substitution.json
- gemma-4-e2b_decode_substitution.json

## Dense local validation (contract test)

The runtime clouds are sparse. To test the actual correction regime, we sampled
queries inside rho around anchors and compared Jacobi prediction to fresh
geodesic truth.

| rho | mean rel err | p95 rel err | Jacobi us | Geodesic us | speedup vs geodesic |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 1.65e-8 | 9.61e-8 | 2.215 | 355.04 | 160.3x |
| 0.10 | 3.38e-8 | 2.79e-7 | 1.849 | 339.89 | 183.9x |
| 0.20 | 8.08e-8 | 5.35e-7 | 2.192 | 350.56 | 159.9x |
| 0.40 | 1.43e-7 | 1.35e-6 | 2.274 | 360.89 | 158.7x |

Best measured Jacobi cost: 1.849 us.
Relative to 5.0 ms baseline decode step, that is 2704.6x per correctable step.

Source:
- smollm2-135m_decode_substitution_dense.json

## Conclusion

GTC correction quality is excellent in-regime (`d <= rho`).
The practical blocker is cloud density: 64-point phase exports are too coarse,
so `correctable_rate_within_rho` is currently 0% despite 100% cache-hit rate.
This cleanly separates a geometry-valid mechanism from a telemetry-density
constraint.
