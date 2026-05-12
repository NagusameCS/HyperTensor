"""
HyperSort: O(1) Instant Sort via Riemannian Comparison Manifold.

Based on the HyperTensor Geometric Jury framework (Papers I-XVIII).

Quick start:
    >>> from hypersort import hypersort
    >>> result = hypersort([3.14, 1.41, 2.71, 1.73, 0.57])
    >>> print(result.sorted_data)
    [0.57, 1.41, 1.73, 2.71, 3.14]

Advanced:
    >>> from hypersort import ComparisonManifold, ManifoldConfig
    >>> config = ManifoldConfig(intrinsic_dim=64, num_jurors=11)
    >>> manifold = ComparisonManifold(config)
    >>> manifold.build(data, my_encoder)
    >>> result = manifold.sort(new_data, my_encoder)  # O(1)!
"""

from hypersort.core import (
    ComparisonManifold,
    ManifoldConfig,
    JuryVerdict,
    SortResult,
    hypersort,
)

__version__ = "0.1.0"
__all__ = [
    "ComparisonManifold",
    "ManifoldConfig",
    "JuryVerdict",
    "SortResult",
    "hypersort",
]
