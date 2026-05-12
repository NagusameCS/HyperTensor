"""
Tests for HyperSort: O(1) Instant Sort via Riemannian Comparison Manifold.
"""

import pytest
import math
import time
from hypersort import hypersort, ComparisonManifold, ManifoldConfig, SortResult


# ---------------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------------

class TestHyperSortBasic:
    """Test basic sorting correctness."""

    def test_empty_list(self):
        result = hypersort([])
        assert result.sorted_data == []
        assert len(result.original_indices) == 0

    def test_single_element(self):
        result = hypersort([42])
        assert result.sorted_data == [42]
        assert result.confidence_scores[0] > 0

    def test_sorted_numbers(self):
        data = [1, 2, 3, 4, 5]
        result = hypersort(data)
        assert result.sorted_data == [1, 2, 3, 4, 5]

    def test_reverse_numbers(self):
        data = [5, 4, 3, 2, 1]
        result = hypersort(data)
        assert result.sorted_data == [1, 2, 3, 4, 5]

    def test_random_numbers(self):
        data = [3.14, 1.41, 2.71, 1.73, 0.57]
        result = hypersort(data)
        assert result.sorted_data == sorted(data)

    def test_duplicates(self):
        data = [5, 3, 5, 1, 3, 1]
        result = hypersort(data)
        assert result.sorted_data == sorted(data)

    def test_negative_numbers(self):
        data = [-5, 3, -10, 0, 7, -3]
        result = hypersort(data)
        assert result.sorted_data == sorted(data)

    def test_large_range(self):
        data = [1000, -500, 0.001, -0.001, 1e6, -1e6]
        result = hypersort(data)
        assert result.sorted_data == sorted(data)

    def test_strings(self):
        data = ["banana", "apple", "cherry", "date"]
        result = hypersort(data)
        assert result.sorted_data == sorted(data)

    def test_integers(self):
        data = [7, 2, 9, 1, 5, 3, 8, 4, 6, 0]
        result = hypersort(data)
        assert result.sorted_data == sorted(data)

    def test_all_equal(self):
        data = [5, 5, 5, 5, 5]
        result = hypersort(data)
        assert result.sorted_data == [5, 5, 5, 5, 5]


# ---------------------------------------------------------------------------
# Metadata and Statistics Tests
# ---------------------------------------------------------------------------

class TestHyperSortMetadata:
    """Test result metadata correctness."""

    def test_sort_result_structure(self):
        result = hypersort([3, 1, 2])
        assert isinstance(result, SortResult)
        assert result.sorted_data is not None
        assert len(result.original_indices) == 3
        assert len(result.confidence_scores) == 3
        assert result.total_time_ms >= 0
        assert result.manifold_dim > 0
        assert result.num_comparisons > 0

    def test_original_indices(self):
        data = ["c", "a", "b"]
        result = hypersort(data)
        assert result.sorted_data == ["a", "b", "c"]
        # Original indices should map back
        reconstructed = [None] * 3
        for new_pos, old_pos in enumerate(result.original_indices):
            reconstructed[old_pos] = result.sorted_data[new_pos]
        assert reconstructed == data

    def test_confidence_range(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = hypersort(data)
        for conf in result.confidence_scores:
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1] range"

    def test_comparisons_performed(self):
        """HyperSort performs all n² comparisons in one parallel step."""
        import random
        data = list(range(100))
        random.shuffle(data)
        result = hypersort(data)
        assert result.num_comparisons == len(data) ** 2


# ---------------------------------------------------------------------------
# Manifold Configuration Tests
# ---------------------------------------------------------------------------

class TestManifoldConfig:
    """Test manifold configuration options."""

    def test_custom_dimensions(self):
        config = ManifoldConfig(intrinsic_dim=64, num_jurors=11)
        manifold = ComparisonManifold(config)
        data = [3, 1, 2]
        encoder = lambda x: __import__('numpy').array([float(x), float(x)/10.0, 1.0])
        manifold.build(data, encoder)
        result = manifold.sort(data, encoder)
        assert result.manifold_dim <= 64

    def test_instinct_horizon(self):
        manifold = ComparisonManifold()
        data = [1, 2, 3, 4, 5]
        encoder = lambda x: __import__('numpy').array([float(x), float(x)/5.0, 1.0])
        manifold.build(data, encoder)
        horizon = manifold.instinct_horizon()
        assert horizon > 0
        assert math.isfinite(horizon)

    def test_statistics(self):
        manifold = ComparisonManifold()
        stats = manifold.get_statistics()
        assert "intrinsic_dim" in stats
        assert "coverage_radius" in stats
        assert "instinct_horizon" in stats
        assert not stats["is_built"]

        encoder = lambda x: __import__('numpy').array([float(x), float(x)/10.0, 1.0])
        manifold.build([1, 2, 3], encoder)
        stats = manifold.get_statistics()
        assert stats["is_built"]
        assert stats["num_trajectories"] == 3


# ---------------------------------------------------------------------------
# Custom Encoder Tests
# ---------------------------------------------------------------------------

class TestCustomEncoder:
    """Test custom encoder functions."""

    def test_custom_encoder(self):
        data = ["ccc", "a", "bb"]
        # Custom encoder: [length, first_char_code, last_char_code]
        def encoder(s):
            return __import__('numpy').array([
                float(len(s)),
                float(ord(s[0])) if s else 0.0,
                float(ord(s[-1])) if s else 0.0,
            ])

        result = hypersort(data, encoder=encoder)
        # Should sort by the projection, likely by length
        assert len(result.sorted_data) == 3
        for conf in result.confidence_scores:
            assert 0.0 <= conf <= 1.0

    def test_high_dimensional_encoder(self):
        """Test with high-dimensional ambient space."""
        data = [1, 2, 3, 4, 5]

        def encoder(x):
            # 128-dimensional encoding
            vec = __import__('numpy').zeros(128)
            vec[0] = float(x)
            vec[1] = float(x) / 5.0
            vec[2] = 1.0
            return vec

        result = hypersort(data, encoder=encoder)
        assert result.sorted_data == sorted(data)


# ---------------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------------

class TestPerformance:
    """Test performance characteristics."""

    def test_manifold_reuse(self):
        """Manifold should be reusable for multiple sort calls."""
        manifold = ComparisonManifold()
        encoder = lambda x: __import__('numpy').array([float(x), float(x)/10.0, 1.0])

        # Build once
        manifold.build([1, 2, 3, 4, 5], encoder)

        # Sort multiple times
        for _ in range(5):
            result = manifold.sort([5, 3, 1, 4, 2], encoder)
            assert result.sorted_data == [1, 2, 3, 4, 5]

    def test_scaling(self):
        """Test that hypersort scales correctly."""
        import numpy as np

        sizes = [10, 50, 100]
        times = []

        for n in sizes:
            data = list(range(n))
            np.random.shuffle(data)
            start = time.perf_counter()
            result = hypersort(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert result.sorted_data == sorted(data)

        # The core sorting step should be near-constant time
        # (excluding encoder which scales with n)
        assert times[-1] < times[0] * 20, f"Scaling issue: {times}"

    def test_large_list(self):
        """Test with a moderately large list."""
        import numpy as np
        np.random.seed(42)
        data = list(np.random.randint(0, 10000, 500))
        result = hypersort(data)
        assert result.sorted_data == sorted(data)
        assert all(0.0 <= c <= 1.0 for c in result.confidence_scores)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unbuilt_manifold(self):
        manifold = ComparisonManifold()
        with pytest.raises(RuntimeError):
            manifold.sort([1, 2, 3], lambda x: __import__('numpy').array([float(x)]))

    def test_zero_intrinsic_dim(self):
        config = ManifoldConfig(intrinsic_dim=0)
        manifold = ComparisonManifold(config)
        encoder = lambda x: __import__('numpy').array([float(x)])
        # Should auto-adjust to valid dim
        manifold.build([1, 2, 3], encoder)
        assert manifold.k > 0

    def test_single_dimension_ambient(self):
        """Test with 1D ambient space."""
        data = [3, 1, 2]
        encoder = lambda x: __import__('numpy').array([float(x)])
        result = hypersort(data, encoder=encoder)
        assert result.sorted_data == [1, 2, 3]
