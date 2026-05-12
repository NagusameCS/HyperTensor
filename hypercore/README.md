# hypercore — HyperTensor Geometric Core

Riemannian geometry primitives for transformer analysis, compression,
hallucination detection, and geodesic trajectory computation.

From the HyperTensor project (Papers I–XVIII).

## Install

```bash
pip install hypercore
```

## Modules

| Module | Description |
|--------|-------------|
| `GeodesicMetric` | Riemannian metric tensor, Christoffel symbols, geodesic integration |
| `HallucinationGuard` | Four-condition hallucination boundary detection |
| `GenerationMetrics` | Token-collapse, geodesic half-life, topological compression |

## Quick Start

```python
from hypercore import GeodesicMetric, HallucinationGuard

# Build a Riemannian metric from hidden states
metric = GeodesicMetric(k_manifold=32)
metric.fit(hidden_states)

# Compute geodesic distance between two points
d = metric.geodesic_distance(h_a, h_b)

# Check if a generation is likely hallucinated
guard = HallucinationGuard(coverage_radius=0.15)
is_hallucination, reason = guard.check(
    query_projection=query_k,
    nearest_trajectories=trajectories,
    jury_confidence=jury_J,
)
```

## Advanced

For the full HyperTensor stack including AxiomGauge, ThermalRank, OnlineOja, TreeDrafter, and safety red-team modules, install the main package:

```bash
pip install hypertensor
```

## License

MIT — see [LICENSE](https://github.com/NagusameCS/HyperTensor/blob/main/LICENSE)
