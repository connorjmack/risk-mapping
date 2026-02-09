"""Tests for polygon aggregation module (pc_rai/ml/polygon_aggregation.py).

Supplements existing test_polygon_indexing.py (load_polygons) and
test_polygon_features_output.py (CSV validation) with unit tests for
the aggregation logic itself.
"""

import numpy as np
import pytest

from pc_rai.ml.polygon_aggregation import (
    AGG_STATS,
    FEATURE_COLUMNS,
    aggregate_features,
    extract_all_locations,
    extract_location,
    extract_mop_range,
    get_elevation_zone,
    get_overlapping_locations,
    points_in_polygon_vectorized,
)


# ---------------------------------------------------------------------------
# aggregate_features
# ---------------------------------------------------------------------------


class TestAggregateFeatures:
    def test_basic_stats(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = aggregate_features(values)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["p50"] == pytest.approx(3.0)

    def test_all_stats_present(self):
        values = np.array([1.0, 2.0, 3.0])
        stats = aggregate_features(values)
        for stat_name in AGG_STATS:
            assert stat_name in stats

    def test_nan_values_excluded(self):
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        stats = aggregate_features(values)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)

    def test_all_nan_returns_nan(self):
        values = np.array([np.nan, np.nan, np.nan])
        stats = aggregate_features(values)
        for stat_name in AGG_STATS:
            assert np.isnan(stats[stat_name])

    def test_single_value(self):
        values = np.array([42.0])
        stats = aggregate_features(values)
        assert stats["mean"] == pytest.approx(42.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_empty_array(self):
        values = np.array([])
        stats = aggregate_features(values)
        for stat_name in AGG_STATS:
            assert np.isnan(stats[stat_name])


# ---------------------------------------------------------------------------
# get_elevation_zone
# ---------------------------------------------------------------------------


class TestGetElevationZone:
    def test_three_zones(self):
        z = np.array([0.0, 3.0, 6.0, 9.0])
        zones = get_elevation_zone(z, z_min=0.0, z_max=9.0)
        assert zones[0] == 0  # Lower
        assert zones[1] == 0  # Lower (exactly at 1/3 boundary)
        assert zones[2] == 1  # Middle (exactly at 2/3 boundary)
        assert zones[3] == 2  # Upper

    def test_flat_surface_all_middle(self):
        z = np.array([5.0, 5.01, 5.02])
        zones = get_elevation_zone(z, z_min=5.0, z_max=5.05)
        assert (zones == 1).all()  # All middle for flat

    def test_output_dtype(self):
        z = np.array([0.0, 5.0, 10.0])
        zones = get_elevation_zone(z, 0.0, 10.0)
        assert zones.dtype == np.int32

    def test_output_values(self):
        z = np.linspace(0, 10, 100)
        zones = get_elevation_zone(z, 0.0, 10.0)
        assert set(np.unique(zones)).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# points_in_polygon_vectorized
# ---------------------------------------------------------------------------


class TestPointsInPolygonVectorized:
    def test_unit_square(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)
        points = np.array([[0.5, 0.5], [2.0, 2.0], [0.1, 0.9]])
        result = points_in_polygon_vectorized(points, poly)
        assert result[0]      # Inside
        assert not result[1]  # Outside
        assert result[2]      # Inside

    def test_empty_points(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.empty((0, 2))
        result = points_in_polygon_vectorized(points, poly)
        assert len(result) == 0

    def test_triangle(self):
        poly = np.array([[0, 0], [2, 0], [1, 2], [0, 0]], dtype=np.float64)
        points = np.array([[1.0, 0.5], [1.0, 3.0]])
        result = points_in_polygon_vectorized(points, poly)
        assert result[0]      # Inside triangle
        assert not result[1]  # Above triangle


# ---------------------------------------------------------------------------
# extract_location / extract_all_locations
# ---------------------------------------------------------------------------


class TestExtractLocation:
    def test_single_location(self):
        assert extract_location("20230101_DelMar_stuff.laz") == "DelMar"

    def test_torrey(self):
        assert extract_location("20230101_Torrey_scan.laz") == "Torrey"

    def test_unknown(self):
        assert extract_location("20230101_random.laz") == "Unknown"

    def test_case_insensitive(self):
        assert extract_location("20230101_DELMAR_scan.laz") == "DelMar"


class TestExtractAllLocations:
    def test_multiple_locations(self):
        result = extract_all_locations("20181121_00567_00622_NoWaves_Torrey_DelMar_scan")
        assert "Torrey" in result
        assert "DelMar" in result

    def test_cardiff_maps_to_solana(self):
        result = extract_all_locations("20200101_Cardiff_scan")
        assert "Solana" in result

    def test_no_duplicates(self):
        result = extract_all_locations("20200101_Solana_Cardiff_scan")
        assert result.count("Solana") == 1


# ---------------------------------------------------------------------------
# extract_mop_range
# ---------------------------------------------------------------------------


class TestExtractMopRange:
    def test_standard_filename(self):
        result = extract_mop_range("20181121_00567_00622_NoWaves")
        assert result == (567, 622)

    def test_no_mop_range(self):
        result = extract_mop_range("delmar_scan.laz")
        assert result is None

    def test_too_few_parts(self):
        result = extract_mop_range("scan")
        assert result is None

    def test_non_numeric_parts(self):
        result = extract_mop_range("20230101_abc_def_scan")
        assert result is None


# ---------------------------------------------------------------------------
# get_overlapping_locations
# ---------------------------------------------------------------------------


class TestGetOverlappingLocations:
    def test_single_location(self):
        result = get_overlapping_locations(600, 615)
        assert result == ["DelMar"]

    def test_multiple_locations(self):
        result = get_overlapping_locations(567, 622)
        assert "Torrey" in result
        assert "DelMar" in result

    def test_no_overlap(self):
        result = get_overlapping_locations(0, 100)
        assert result == []

    def test_exact_boundary_no_overlap(self):
        # Boundaries are exclusive, so touching exactly shouldn't match
        result = get_overlapping_locations(620, 637)
        assert result == []

    def test_all_locations(self):
        result = get_overlapping_locations(0, 10000)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_feature_columns_defined(self):
        assert len(FEATURE_COLUMNS) > 0
        assert "slope" in FEATURE_COLUMNS
        assert "height" in FEATURE_COLUMNS

    def test_agg_stats_defined(self):
        assert "mean" in AGG_STATS
        assert "std" in AGG_STATS
        assert "p50" in AGG_STATS
