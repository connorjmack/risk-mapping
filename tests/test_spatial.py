"""Tests for pc_rai.utils.spatial module."""

import numpy as np
import pytest


def test_spatial_index_creation():
    """Test SpatialIndex creation."""
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 10, (1000, 3))
    index = SpatialIndex(points)

    assert index.n_points == 1000
    assert index.tree is not None


def test_spatial_index_invalid_shape():
    """Test SpatialIndex rejects invalid point shapes."""
    from pc_rai.utils.spatial import SpatialIndex

    # Wrong number of columns
    with pytest.raises(ValueError, match="must have shape"):
        SpatialIndex(np.array([[0, 0], [1, 1]]))

    # 1D array
    with pytest.raises(ValueError, match="must have shape"):
        SpatialIndex(np.array([0, 0, 0]))


def test_radius_query():
    """Test radius neighbor query."""
    from pc_rai.utils.spatial import SpatialIndex

    # Create grid of points with known spacing
    x = np.linspace(0, 1, 10)
    points = np.array([[i, 0, 0] for i in x])
    index = SpatialIndex(points)

    neighbors, counts = index.query_radius(0.15, return_counts=True, show_progress=False)

    # Each point should have ~1-2 neighbors within 0.15 (spacing is ~0.111)
    assert all(c >= 1 for c in counts)
    assert len(neighbors) == 10


def test_radius_query_self_included():
    """Test that radius query includes self as neighbor."""
    from pc_rai.utils.spatial import SpatialIndex

    points = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float64)
    index = SpatialIndex(points)

    neighbors, counts = index.query_radius(0.1, return_counts=True, show_progress=False)

    # Each point should have at least itself as neighbor
    assert all(c >= 1 for c in counts)


def test_knn_query():
    """Test k-NN neighbor query."""
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 10, (100, 3))
    index = SpatialIndex(points)

    distances, indices = index.query_knn(k=5, show_progress=False)

    assert distances.shape == (100, 5)
    assert indices.shape == (100, 5)
    # First neighbor should be self (distance 0)
    assert np.allclose(distances[:, 0], 0)


def test_knn_query_k_larger_than_n():
    """Test k-NN when k > n_points."""
    from pc_rai.utils.spatial import SpatialIndex

    points = np.random.uniform(0, 10, (10, 3))
    index = SpatialIndex(points)

    # Request more neighbors than points
    distances, indices = index.query_knn(k=20, show_progress=False)

    # Should cap at n_points
    assert distances.shape == (10, 10)
    assert indices.shape == (10, 10)


def test_query_radius_single():
    """Test single-point radius query."""
    from pc_rai.utils.spatial import SpatialIndex

    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    index = SpatialIndex(points)

    query_point = np.array([0.5, 0, 0])
    neighbors = index.query_radius_single(query_point, 0.6)

    # Should find points at 0 and 1
    assert len(neighbors) == 2


def test_query_knn_single():
    """Test single-point k-NN query."""
    from pc_rai.utils.spatial import SpatialIndex

    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    index = SpatialIndex(points)

    query_point = np.array([0.5, 0, 0])
    distances, indices = index.query_knn_single(query_point, k=2)

    assert len(distances) == 2
    assert len(indices) == 2
    # Closest should be distance 0.5
    assert np.isclose(distances[0], 0.5)


def test_compute_neighbor_stats():
    """Test neighbor statistics computation."""
    from pc_rai.utils.spatial import SpatialIndex, compute_neighbor_stats

    # Create clustered points
    points = np.random.uniform(0, 1, (100, 3))
    index = SpatialIndex(points)

    # All same value -> zero std
    values = np.full(100, 45.0)
    neighbors, _ = index.query_radius(0.5, show_progress=False)
    std_dev, counts = compute_neighbor_stats(values, neighbors, min_neighbors=3)

    assert np.allclose(std_dev[~np.isnan(std_dev)], 0, atol=0.01)


def test_compute_neighbor_stats_varied():
    """Test neighbor statistics with varied values."""
    from pc_rai.utils.spatial import SpatialIndex, compute_neighbor_stats

    points = np.random.uniform(0, 1, (100, 3))
    index = SpatialIndex(points)

    # Varied values -> positive std
    values = np.random.uniform(30, 60, 100)
    neighbors, _ = index.query_radius(0.5, show_progress=False)
    std_dev, counts = compute_neighbor_stats(values, neighbors, min_neighbors=3)

    valid = ~np.isnan(std_dev)
    assert np.mean(std_dev[valid]) > 0


def test_compute_neighbor_stats_insufficient():
    """Test neighbor statistics with insufficient neighbors."""
    from pc_rai.utils.spatial import compute_neighbor_stats

    values = np.array([45.0, 45.0, 45.0])
    # Each point has only itself as neighbor
    neighbors = [np.array([0]), np.array([1]), np.array([2])]

    std_dev, counts = compute_neighbor_stats(values, neighbors, min_neighbors=3)

    # All should be NaN due to insufficient neighbors
    assert np.all(np.isnan(std_dev))


def test_compute_neighbor_stats_knn():
    """Test k-NN neighbor statistics computation."""
    from pc_rai.utils.spatial import SpatialIndex, compute_neighbor_stats_knn

    points = np.random.uniform(0, 1, (100, 3))
    index = SpatialIndex(points)

    # All same value -> zero std
    values = np.full(100, 45.0)
    _, indices = index.query_knn(k=10, show_progress=False)
    std_dev = compute_neighbor_stats_knn(values, indices)

    assert np.allclose(std_dev, 0, atol=0.01)


def test_compute_neighbor_stats_knn_varied():
    """Test k-NN neighbor statistics with varied values."""
    from pc_rai.utils.spatial import SpatialIndex, compute_neighbor_stats_knn

    points = np.random.uniform(0, 1, (100, 3))
    index = SpatialIndex(points)

    # Varied values -> positive std
    values = np.random.uniform(30, 60, 100)
    _, indices = index.query_knn(k=10, show_progress=False)
    std_dev = compute_neighbor_stats_knn(values, indices)

    assert np.mean(std_dev) > 0
