"""Tests for polygon geometry module (pc_rai/ml/polygons.py)."""

import numpy as np
import pytest

from pc_rai.ml.polygons import Polygon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_square():
    """A 1x1 polygon at the origin."""
    vertices = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0],
    ], dtype=np.float64)
    return Polygon(
        polygon_id=0,
        vertices=vertices,
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
    )


@pytest.fixture
def offset_square():
    """A 1x1 polygon at (10, 10)."""
    vertices = np.array([
        [10, 10], [11, 10], [11, 11], [10, 11], [10, 10],
    ], dtype=np.float64)
    return Polygon(
        polygon_id=100,
        vertices=vertices,
        x_min=10.0, x_max=11.0,
        y_min=10.0, y_max=11.0,
    )


# ---------------------------------------------------------------------------
# Polygon.contains_point
# ---------------------------------------------------------------------------


class TestContainsPoint:
    def test_interior_point(self, unit_square):
        assert unit_square.contains_point(0.5, 0.5)

    def test_exterior_point(self, unit_square):
        assert not unit_square.contains_point(2.0, 2.0)

    def test_far_exterior(self, unit_square):
        assert not unit_square.contains_point(-10.0, -10.0)


# ---------------------------------------------------------------------------
# Polygon.points_inside
# ---------------------------------------------------------------------------


class TestPointsInside:
    def test_all_inside(self, unit_square):
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.2, 0.5, 0.8])
        mask = unit_square.points_inside(x, y)
        assert mask.all()

    def test_all_outside(self, unit_square):
        x = np.array([2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0])
        mask = unit_square.points_inside(x, y)
        assert not mask.any()

    def test_mixed(self, unit_square):
        x = np.array([0.5, 2.0, 0.5])
        y = np.array([0.5, 2.0, 0.5])
        mask = unit_square.points_inside(x, y)
        assert mask[0] and not mask[1] and mask[2]

    def test_empty_input(self, unit_square):
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        mask = unit_square.points_inside(x, y)
        assert len(mask) == 0

    def test_many_points(self, unit_square):
        """Vectorized operation handles many points."""
        np.random.seed(42)
        x = np.random.uniform(-1, 2, 1000)
        y = np.random.uniform(-1, 2, 1000)
        mask = unit_square.points_inside(x, y)
        # ~1/9 of the [-1,2]x[-1,2] area is inside the unit square
        assert 50 < mask.sum() < 500

    def test_bbox_optimization(self, offset_square):
        """Points outside bbox are quickly rejected."""
        x = np.array([0.5, 10.5])
        y = np.array([0.5, 10.5])
        mask = offset_square.points_inside(x, y)
        assert not mask[0]  # Outside bbox entirely
        assert mask[1]      # Inside polygon

    def test_output_shape(self, unit_square):
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        mask = unit_square.points_inside(x, y)
        assert mask.shape == (3,)
        assert mask.dtype == bool
