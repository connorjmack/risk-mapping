"""Tests for ML feature extraction module (pc_rai/ml/feature_extraction.py)."""

import numpy as np
import pytest

from pc_rai.ml.feature_extraction import (
    compute_eigenvalue_features,
    compute_relative_height,
    compute_roughness,
    compute_slope,
    extract_features,
    voxel_subsample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_cloud():
    """Points on a horizontal plane with upward normals."""
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    y = np.random.uniform(0, 10, n)
    z = np.zeros(n)
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    return xyz, normals


@pytest.fixture
def cliff_cloud():
    """Points on a vertical cliff face with horizontal normals."""
    np.random.seed(42)
    n = 200
    x = np.zeros(n)
    y = np.random.uniform(0, 10, n)
    z = np.random.uniform(0, 20, n)
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 0] = -1.0  # Pointing west (toward observer)
    return xyz, normals


@pytest.fixture
def dense_grid():
    """Dense regular grid for subsampling tests."""
    # 10x10x10 grid with 0.1m spacing = 1000 points in 1m³
    coords = np.arange(0, 1.0, 0.1)
    xx, yy, zz = np.meshgrid(coords, coords, coords)
    xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    return xyz


# ---------------------------------------------------------------------------
# voxel_subsample
# ---------------------------------------------------------------------------


class TestVoxelSubsample:
    def test_reduces_point_count(self, dense_grid):
        subsampled = voxel_subsample(dense_grid, voxel_size=0.5)
        assert len(subsampled) < len(dense_grid)

    def test_empty_input(self):
        xyz = np.empty((0, 3), dtype=np.float64)
        result = voxel_subsample(xyz, voxel_size=0.5)
        assert len(result) == 0

    def test_return_indices(self, dense_grid):
        subsampled, indices = voxel_subsample(dense_grid, voxel_size=0.5, return_indices=True)
        assert len(subsampled) == len(indices)
        # Indices should be valid into the original array
        assert indices.max() < len(dense_grid)
        # Reconstructed points should match
        np.testing.assert_array_equal(subsampled, dense_grid[indices])

    def test_large_voxel_gives_fewer_points(self, dense_grid):
        sub_small = voxel_subsample(dense_grid, voxel_size=0.2)
        sub_large = voxel_subsample(dense_grid, voxel_size=0.5)
        assert len(sub_large) <= len(sub_small)

    def test_single_point(self):
        xyz = np.array([[1.0, 2.0, 3.0]])
        result = voxel_subsample(xyz, voxel_size=0.5)
        assert len(result) == 1

    def test_output_shape(self, dense_grid):
        result = voxel_subsample(dense_grid, voxel_size=0.5)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_preserves_bounds(self, dense_grid):
        result = voxel_subsample(dense_grid, voxel_size=0.5)
        # Subsampled points should be within original bounds
        assert result[:, 0].min() >= dense_grid[:, 0].min()
        assert result[:, 0].max() <= dense_grid[:, 0].max()


# ---------------------------------------------------------------------------
# compute_slope
# ---------------------------------------------------------------------------


class TestComputeSlope:
    def test_horizontal_surface(self, flat_cloud):
        _, normals = flat_cloud
        slope = compute_slope(normals)
        np.testing.assert_allclose(slope, 0.0, atol=0.1)

    def test_vertical_surface(self, cliff_cloud):
        _, normals = cliff_cloud
        slope = compute_slope(normals)
        np.testing.assert_allclose(slope, 90.0, atol=0.1)

    def test_output_range(self):
        # Random unit normals
        np.random.seed(42)
        normals = np.random.randn(100, 3).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        slope = compute_slope(normals)
        assert slope.min() >= 0
        assert slope.max() <= 180

    def test_output_dtype(self, flat_cloud):
        _, normals = flat_cloud
        slope = compute_slope(normals)
        assert slope.dtype == np.float32

    def test_overhang(self):
        """Normal pointing downward -> slope > 90°."""
        normals = np.array([[0, 0, -1]], dtype=np.float32)
        slope = compute_slope(normals)
        np.testing.assert_allclose(slope, 180.0, atol=0.1)


# ---------------------------------------------------------------------------
# compute_roughness
# ---------------------------------------------------------------------------


class TestComputeRoughness:
    def test_flat_surface_low_roughness(self, flat_cloud):
        xyz, normals = flat_cloud
        slope = compute_slope(normals)
        roughness = compute_roughness(slope, xyz, radius=3.0, min_neighbors=5)
        # Flat surface: all slopes ~0, so std dev should be ~0
        valid = roughness[~np.isnan(roughness)]
        assert valid.mean() < 1.0  # Very low roughness

    def test_nan_for_insufficient_neighbors(self):
        # Points far apart with small radius
        xyz = np.array([[0, 0, 0], [100, 100, 100]], dtype=np.float64)
        slope = np.array([45.0, 45.0], dtype=np.float32)
        roughness = compute_roughness(slope, xyz, radius=1.0, min_neighbors=5)
        assert np.isnan(roughness).all()

    def test_output_shape(self, flat_cloud):
        xyz, normals = flat_cloud
        slope = compute_slope(normals)
        roughness = compute_roughness(slope, xyz, radius=3.0)
        assert roughness.shape == slope.shape

    def test_roughness_non_negative(self, flat_cloud):
        xyz, normals = flat_cloud
        slope = compute_slope(normals)
        roughness = compute_roughness(slope, xyz, radius=3.0)
        valid = roughness[~np.isnan(roughness)]
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# compute_relative_height
# ---------------------------------------------------------------------------


class TestComputeRelativeHeight:
    def test_flat_surface_zero_height(self, flat_cloud):
        xyz, _ = flat_cloud
        height = compute_relative_height(xyz, horizontal_radius=20.0)
        # All Z=0, so relative height should be 0
        np.testing.assert_allclose(height, 0.0, atol=0.01)

    def test_positive_height(self, cliff_cloud):
        xyz, _ = cliff_cloud
        height = compute_relative_height(xyz, horizontal_radius=20.0)
        assert (height >= 0).all()

    def test_tallest_point_has_max_height(self):
        xyz = np.array([
            [0, 0, 0],
            [0, 0, 5],
            [0, 0, 10],
        ], dtype=np.float64)
        height = compute_relative_height(xyz, horizontal_radius=20.0)
        assert height[2] == pytest.approx(10.0)

    def test_output_dtype(self, flat_cloud):
        xyz, _ = flat_cloud
        height = compute_relative_height(xyz)
        assert height.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_eigenvalue_features
# ---------------------------------------------------------------------------


class TestComputeEigenvalueFeatures:
    def test_returns_four_features(self, flat_cloud):
        xyz, _ = flat_cloud
        result = compute_eigenvalue_features(xyz, radius=3.0, verbose=False)
        assert "planarity" in result
        assert "linearity" in result
        assert "sphericity" in result
        assert "curvature" in result

    def test_features_in_0_1(self, flat_cloud):
        xyz, _ = flat_cloud
        result = compute_eigenvalue_features(xyz, radius=3.0, verbose=False)
        for name, values in result.items():
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                assert valid.min() >= -0.01, f"{name} has values below 0"
                assert valid.max() <= 1.01, f"{name} has values above 1"

    def test_flat_surface_high_planarity(self, flat_cloud):
        xyz, _ = flat_cloud
        result = compute_eigenvalue_features(xyz, radius=3.0, verbose=False)
        valid = result["planarity"][~np.isnan(result["planarity"])]
        # Flat surface should have high planarity
        assert valid.mean() > 0.3

    def test_nan_for_isolated_points(self):
        xyz = np.array([[0, 0, 0], [100, 100, 100]], dtype=np.float64)
        result = compute_eigenvalue_features(xyz, radius=1.0, min_neighbors=5, verbose=False)
        assert np.isnan(result["planarity"]).all()

    def test_output_shapes(self, flat_cloud):
        xyz, _ = flat_cloud
        result = compute_eigenvalue_features(xyz, radius=3.0, verbose=False)
        for name, values in result.items():
            assert values.shape == (len(xyz),), f"{name} shape mismatch"


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_all_features(self, flat_cloud):
        xyz, normals = flat_cloud
        result = extract_features(
            xyz, normals, radius_small=3.0, radius_large=5.0, verbose=False
        )
        expected_keys = {
            "slope", "roughness_small", "roughness_large",
            "roughness_ratio", "height",
            "planarity", "linearity", "sphericity", "curvature",
        }
        assert set(result.keys()) == expected_keys

    def test_all_arrays_correct_length(self, flat_cloud):
        xyz, normals = flat_cloud
        result = extract_features(
            xyz, normals, radius_small=3.0, radius_large=5.0, verbose=False
        )
        for name, values in result.items():
            assert len(values) == len(xyz), f"{name} length mismatch"

    def test_slope_range(self, flat_cloud):
        xyz, normals = flat_cloud
        result = extract_features(
            xyz, normals, radius_small=3.0, radius_large=5.0, verbose=False
        )
        assert result["slope"].min() >= 0
        assert result["slope"].max() <= 180

    def test_height_non_negative(self, flat_cloud):
        xyz, normals = flat_cloud
        result = extract_features(
            xyz, normals, radius_small=3.0, radius_large=5.0, verbose=False
        )
        assert (result["height"] >= 0).all()
