"""Tests for pc_rai.features.roughness module."""

import numpy as np
import pytest


def test_uniform_slope_zero_roughness_knn():
    """Points with identical slopes should have ~0 roughness (k-NN)."""
    from pc_rai.features.roughness import calculate_roughness_knn
    from pc_rai.utils.spatial import SpatialIndex

    # Create grid of points
    points = np.random.uniform(0, 1, (100, 3))
    # All have same slope
    slopes = np.full(100, 45.0)

    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_knn(slopes, index, k=10)

    assert np.allclose(roughness, 0, atol=0.01)


def test_variable_slope_nonzero_roughness_knn():
    """Points with varying slopes should have positive roughness (k-NN)."""
    from pc_rai.features.roughness import calculate_roughness_knn
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 1, (100, 3))
    # Varying slopes
    slopes = np.random.uniform(30, 60, 100)

    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_knn(slopes, index, k=10)

    assert roughness.mean() > 0


def test_uniform_slope_zero_roughness_radius():
    """Points with identical slopes should have ~0 roughness (radius)."""
    from pc_rai.features.roughness import calculate_roughness_radius
    from pc_rai.utils.spatial import SpatialIndex

    # Create dense grid of points
    points = np.random.uniform(0, 1, (500, 3))
    # All have same slope
    slopes = np.full(500, 45.0)

    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_radius(slopes, index, radius=0.3, min_neighbors=5)

    # Valid points should have ~0 roughness
    valid = ~np.isnan(roughness)
    assert np.allclose(roughness[valid], 0, atol=0.01)


def test_variable_slope_nonzero_roughness_radius():
    """Points with varying slopes should have positive roughness (radius)."""
    from pc_rai.features.roughness import calculate_roughness_radius
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 1, (500, 3))
    # Varying slopes
    slopes = np.random.uniform(30, 60, 500)

    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_radius(slopes, index, radius=0.3, min_neighbors=5)

    valid = ~np.isnan(roughness)
    assert roughness[valid].mean() > 0


def test_insufficient_neighbors():
    """Sparse points should return NaN for radius method."""
    from pc_rai.features.roughness import calculate_roughness_radius
    from pc_rai.utils.spatial import SpatialIndex

    # Very spread out points
    points = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], dtype=np.float64)
    slopes = np.array([45, 45, 45])

    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_radius(slopes, index, radius=0.5, min_neighbors=2)

    # Should be NaN due to insufficient neighbors
    assert np.all(np.isnan(roughness))


def test_knn_always_returns_values():
    """K-NN method should always return valid values (not NaN)."""
    from pc_rai.features.roughness import calculate_roughness_knn
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 10, (50, 3))
    slopes = np.random.uniform(30, 60, 50)

    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_knn(slopes, index, k=10)

    # Should have no NaN values
    assert not np.any(np.isnan(roughness))
    assert np.all(counts == 10)


def test_calculate_all_roughness_both_methods():
    """Test calculating all roughness metrics with both methods."""
    from pc_rai.features.roughness import calculate_all_roughness
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 1, (200, 3))
    slopes = np.random.uniform(30, 60, 200)

    index = SpatialIndex(points)
    results = calculate_all_roughness(
        slopes, index,
        radius_small=0.2, radius_large=0.4,
        k_small=10, k_large=30,
        methods=["radius", "knn"]
    )

    assert "roughness_small_radius" in results
    assert "roughness_large_radius" in results
    assert "roughness_small_knn" in results
    assert "roughness_large_knn" in results
    assert "neighbor_count_small" in results
    assert "neighbor_count_large" in results


def test_calculate_all_roughness_radius_only():
    """Test calculating roughness with radius method only."""
    from pc_rai.features.roughness import calculate_all_roughness
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 1, (200, 3))
    slopes = np.random.uniform(30, 60, 200)

    index = SpatialIndex(points)
    results = calculate_all_roughness(
        slopes, index,
        methods=["radius"]
    )

    assert "roughness_small_radius" in results
    assert "roughness_large_radius" in results
    assert "roughness_small_knn" not in results
    assert "roughness_large_knn" not in results


def test_calculate_all_roughness_knn_only():
    """Test calculating roughness with k-NN method only."""
    from pc_rai.features.roughness import calculate_all_roughness
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 1, (200, 3))
    slopes = np.random.uniform(30, 60, 200)

    index = SpatialIndex(points)
    results = calculate_all_roughness(
        slopes, index,
        methods=["knn"]
    )

    assert "roughness_small_knn" in results
    assert "roughness_large_knn" in results
    assert "roughness_small_radius" not in results
    assert "roughness_large_radius" not in results


def test_larger_radius_more_neighbors():
    """Larger radius should generally find more neighbors."""
    from pc_rai.features.roughness import calculate_roughness_radius
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 1, (500, 3))
    slopes = np.random.uniform(30, 60, 500)

    index = SpatialIndex(points)

    _, counts_small = calculate_roughness_radius(slopes, index, radius=0.1, min_neighbors=1)
    _, counts_large = calculate_roughness_radius(slopes, index, radius=0.3, min_neighbors=1)

    assert counts_large.mean() > counts_small.mean()


def test_roughness_statistics():
    """Test roughness statistics calculation."""
    from pc_rai.features.roughness import roughness_statistics

    roughness = np.array([5.0, 10.0, 15.0, 20.0, np.nan])
    stats = roughness_statistics(roughness, name="test")

    assert stats["test_mean"] == 12.5
    assert stats["test_min"] == 5.0
    assert stats["test_max"] == 20.0
    assert stats["test_n_valid"] == 4
    assert stats["test_pct_valid"] == 80.0


def test_roughness_statistics_all_nan():
    """Test roughness statistics with all NaN values."""
    from pc_rai.features.roughness import roughness_statistics

    roughness = np.array([np.nan, np.nan, np.nan])
    stats = roughness_statistics(roughness, name="test")

    assert np.isnan(stats["test_mean"])
    assert stats["test_n_valid"] == 0
    assert stats["test_pct_valid"] == 0.0


def test_default_parameters():
    """Test roughness calculation with default Markus et al. 2023 parameters."""
    from pc_rai.features.roughness import calculate_all_roughness
    from pc_rai.utils.spatial import SpatialIndex

    # Create a reasonably dense point cloud (1 point per ~0.01 m^3)
    n_points = 1000
    points = np.random.uniform(0, 2, (n_points, 3))
    slopes = np.random.uniform(30, 60, n_points)

    index = SpatialIndex(points)

    # Use default parameters
    results = calculate_all_roughness(slopes, index)

    # Check all expected outputs exist
    assert "roughness_small_radius" in results
    assert "roughness_large_radius" in results
    assert "roughness_small_knn" in results
    assert "roughness_large_knn" in results
