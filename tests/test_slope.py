"""Tests for pc_rai.features.slope module."""

import numpy as np
import pytest


def test_horizontal_surface():
    """Normal pointing up = 0° slope."""
    from pc_rai.features.slope import calculate_slope

    normals = np.array([[0, 0, 1]], dtype=np.float32)  # pointing up
    slope = calculate_slope(normals)

    assert np.isclose(slope[0], 0, atol=0.01)


def test_vertical_surface():
    """Normal pointing sideways = 90° slope."""
    from pc_rai.features.slope import calculate_slope

    normals = np.array([[1, 0, 0]], dtype=np.float32)  # pointing in X
    slope = calculate_slope(normals)

    assert np.isclose(slope[0], 90, atol=0.01)


def test_vertical_surface_y():
    """Normal pointing in Y direction = 90° slope."""
    from pc_rai.features.slope import calculate_slope

    normals = np.array([[0, 1, 0]], dtype=np.float32)  # pointing in Y
    slope = calculate_slope(normals)

    assert np.isclose(slope[0], 90, atol=0.01)


def test_45_degree_slope():
    """Normal at 45° from vertical."""
    from pc_rai.features.slope import calculate_slope

    # Normal at 45° angle
    normals = np.array([[0.707, 0, 0.707]], dtype=np.float32)
    slope = calculate_slope(normals)

    assert np.isclose(slope[0], 45, atol=0.5)


def test_overhang():
    """Normal pointing down-ish = >90° slope."""
    from pc_rai.features.slope import calculate_slope

    # 45° below horizontal (135° from up)
    normals = np.array([[0.707, 0, -0.707]], dtype=np.float32)
    slope = calculate_slope(normals)

    assert slope[0] > 90
    assert np.isclose(slope[0], 135, atol=0.5)


def test_inverted():
    """Normal pointing straight down = 180° slope."""
    from pc_rai.features.slope import calculate_slope

    normals = np.array([[0, 0, -1]], dtype=np.float32)
    slope = calculate_slope(normals)

    assert np.isclose(slope[0], 180, atol=0.01)


def test_vectorized():
    """Test with multiple normals."""
    from pc_rai.features.slope import calculate_slope

    normals = np.array([
        [0, 0, 1],    # 0°
        [1, 0, 0],    # 90°
        [0, 0, -1],   # 180°
    ], dtype=np.float32)

    slope = calculate_slope(normals)

    assert len(slope) == 3
    assert np.isclose(slope[0], 0, atol=0.01)
    assert np.isclose(slope[1], 90, atol=0.01)
    assert np.isclose(slope[2], 180, atol=0.01)


def test_custom_up_vector():
    """Test with custom up vector."""
    from pc_rai.features.slope import calculate_slope

    # Normal pointing in +Y, with +Y as up
    normals = np.array([[0, 1, 0]], dtype=np.float32)
    slope = calculate_slope(normals, up_vector=(0, 1, 0))

    assert np.isclose(slope[0], 0, atol=0.01)


def test_invalid_normals_shape():
    """Test rejection of invalid normals shape."""
    from pc_rai.features.slope import calculate_slope

    with pytest.raises(ValueError, match="must have shape"):
        calculate_slope(np.array([[0, 0], [1, 1]]))


def test_identify_overhangs():
    """Test overhang identification."""
    from pc_rai.features.slope import identify_overhangs

    slopes = np.array([45, 89, 91, 120, 180])
    mask = identify_overhangs(slopes)

    assert list(mask) == [False, False, True, True, True]


def test_identify_overhangs_custom_threshold():
    """Test overhang identification with custom threshold."""
    from pc_rai.features.slope import identify_overhangs

    slopes = np.array([45, 89, 91, 120, 180])
    mask = identify_overhangs(slopes, threshold=100)

    assert list(mask) == [False, False, False, True, True]


def test_identify_steep_slopes():
    """Test steep slope identification."""
    from pc_rai.features.slope import identify_steep_slopes

    slopes = np.array([30, 45, 60, 89, 91, 120])
    mask = identify_steep_slopes(slopes, min_slope=45, max_slope=90)

    assert list(mask) == [False, True, True, True, False, False]


def test_slope_statistics():
    """Test slope statistics calculation."""
    from pc_rai.features.slope import slope_statistics

    slopes = np.array([0, 30, 60, 90, 120])
    stats = slope_statistics(slopes)

    assert stats["mean"] == 60.0
    assert stats["min"] == 0.0
    assert stats["max"] == 120.0
    assert stats["n_overhang"] == 1
    assert stats["pct_overhang"] == 20.0


def test_slope_statistics_with_nan():
    """Test slope statistics with NaN values."""
    from pc_rai.features.slope import slope_statistics

    slopes = np.array([0, 30, np.nan, 90, 120])
    stats = slope_statistics(slopes)

    # Should ignore NaN values
    assert stats["mean"] == 60.0
    assert stats["n_overhang"] == 1


def test_slope_statistics_all_nan():
    """Test slope statistics with all NaN values."""
    from pc_rai.features.slope import slope_statistics

    slopes = np.array([np.nan, np.nan, np.nan])
    stats = slope_statistics(slopes)

    assert np.isnan(stats["mean"])
    assert stats["n_overhang"] == 0


def test_large_array():
    """Test slope calculation on large array."""
    from pc_rai.features.slope import calculate_slope

    n = 100000
    normals = np.random.randn(n, 3).astype(np.float32)
    # Normalize
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    slope = calculate_slope(normals)

    assert len(slope) == n
    assert slope.dtype == np.float32
    assert np.all(slope >= 0)
    assert np.all(slope <= 180)
