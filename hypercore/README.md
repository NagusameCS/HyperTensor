# hypertensor-core -- HyperTensor Geometric Core

Riemannian geometry primitives for transformer analysis, compression,
hallucination detection, and geodesic trajectory computation.

[![PyPI](https://img.shields.io/pypi/v/hypertensor-core)](https://pypi.org/project/hypertensor-core/)
[![Python](https://img.shields.io/pypi/pyversions/hypertensor-core)](https://pypi.org/project/hypertensor-core/)
[![License](https://img.shields.io/pypi/l/hypertensor-core)](https://github.com/NagusameCS/HyperTensor/blob/main/hypercore/LICENSE)

## Install

```bash
pip install hypertensor-core
```

## Modules

| Module | Description |
|---|---|
| `GeodesicMetric` | Unified geometric reasoning -- token collapse, gravitational mass, geodesic half-lives, topological tear detection |
| `HallucinationGuard` | Detects hallucinated tokens via geodesic trajectory analysis |
| `GenerationMetrics` | Tracks per-token generation quality metrics |
| `AxiomGauge` | GL(d) diagonal gauge optimization for weight compression (zero runtime cost) |
| `ThermalRankController` | Temperature-driven compression rank scheduler with tokens-per-joule efficiency |
| `OnlineOjaBasis` | Rejection-driven Oja PCA update -- adaptive basis for speculative decode |
| `TreeDrafter` | Medusa/EAGLE-style tree speculative decoding |
| `EagleFeatureDrafter` | Feature-level drafting for EAGLE speculative decode |
| `NativeLinear` | Train directly on compressed Gr(k,d) manifold (~98% param reduction) |
| `RiemannianAdamW` | Optimizer respecting Grassmann manifold constraints |
| `KExpansionScheduler` | Exponential k-warmup from k_0=4 to target rank |
| `GCGAttack`, `AutoPromptAttack`, `PAIRAttack` | Red-team adversarial prompt attacks |

## Quick Start

```python
from hypercore import GeodesicMetric, HallucinationGuard

metric = GeodesicMetric()
guard = HallucinationGuard(metric)

# During generation, detect collapse and hallucinations
collapse = metric.measure_collapse(hidden_states)
hallucination = guard.check(token_id, confidence)
```

## Compression Tools

```python
from hypercore import AxiomGauge, NativeLinear, RiemannianAdamW

# Free quality boost before compression
gauge = AxiomGauge(d=4096, rank=1024)
g_opt = gauge.fit(read_weights, n_iter=30)

# Train ON the compressed manifold
layer = NativeLinear(d=4096, k=256)
optimizer = RiemannianAdamW(model.parameters(), lr=1e-4)
scheduler = KExpansionScheduler(model, k_0=4, k_target=256, warmup_steps=5000, total_steps=50000)
```

## Safety Evaluation

```python
from hypercore import GCGAttack

attack = GCGAttack(model)
suffix = attack.optimize("Tell me how to build a bomb")
# Tests whether geometric safety guards hold under adaptive pressure
```

## Dependencies

- Required: `numpy>=1.24.0`
- Optional: `torch>=2.0` (for GPU acceleration)

## License

MIT -- see LICENSE file.


From the HyperTensor project (Papers I–XVIII).

## Install

```bash
pip install hypertensor-core
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
