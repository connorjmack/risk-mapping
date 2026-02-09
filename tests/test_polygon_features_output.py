"""Tests to verify polygon_features.csv output after Step 3 re-run.

Validates that the polygon indexing fix produces correct local
alongshore_m coordinates (0-based per beach, not MOP coordinates).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# Minimum unique polygons for a location to count as having real coverage
# (excludes boundary-overlap slivers like a single polygon from an adjacent survey)
MIN_POLYGONS_FOR_COVERAGE = 20

# Expected polygon counts per location (from shapefiles)
EXPECTED_POLYGON_COUNTS = {
    "DelMar": 2285,
    "Blacks": 4705,
    "Torrey": 1399,
    "Encinitas": 5607,
    "SanElijo": 2501,
    "Solana": 2898,
}

# MOP ranges — alongshore_m should NOT be in these ranges
MOP_RANGES = {
    "Blacks": (520, 567),
    "Torrey": (567, 581),
    "DelMar": (595, 620),
    "Solana": (637, 666),
    "SanElijo": (683, 708),
    "Encinitas": (708, 764),
}


@pytest.fixture
def features_df(request):
    """Load polygon features CSV, skip if not available."""
    features_path = Path(request.config.getoption("--features-csv"))
    if not features_path.exists():
        pytest.skip(f"Features CSV not found at {features_path}")
    return pd.read_csv(features_path)


class TestAlongshoreCoordinates:
    """Verify alongshore_m uses local 0-based meters, not MOP."""

    def test_alongshore_starts_near_zero(self, features_df):
        """Each location's min alongshore_m should be near 0, not MOP values.

        Threshold is 50m to allow for survey data not covering the
        southernmost polygons of a shapefile. Locations with trivial
        coverage (boundary slivers) are skipped.
        """
        for location, group in features_df.groupby("location"):
            if group["alongshore_m"].nunique() < MIN_POLYGONS_FOR_COVERAGE:
                continue
            min_along = group["alongshore_m"].min()
            assert min_along < 50.0, (
                f"{location}: min alongshore_m = {min_along:.1f}, "
                f"expected near 0 (got MOP-scale value?)"
            )

    def test_alongshore_not_in_mop_range(self, features_df):
        """alongshore_m should NOT look like global MOP coordinates."""
        for location, group in features_df.groupby("location"):
            if location not in MOP_RANGES:
                continue
            mop_min, mop_max = MOP_RANGES[location]
            min_along = group["alongshore_m"].min()
            max_along = group["alongshore_m"].max()

            # If min is near MOP range, the fix didn't work
            in_mop_range = mop_min - 5 <= min_along <= mop_max + 5
            assert not in_mop_range, (
                f"{location}: alongshore_m range [{min_along:.1f}, {max_along:.1f}] "
                f"looks like MOP coordinates ({mop_min}-{mop_max}). Fix not applied?"
            )

    def test_delmar_alongshore_integer_spaced(self, features_df):
        """DelMar alongshore_m should be integer-valued (Id - min_Id)."""
        dm = features_df[features_df["location"] == "DelMar"]
        if len(dm) == 0:
            pytest.skip("No DelMar data in features")

        unique_along = dm["alongshore_m"].unique()
        # Check that values are integers (or very close)
        remainders = unique_along % 1.0
        n_integer = np.sum((remainders < 0.01) | (remainders > 0.99))
        pct_integer = n_integer / len(unique_along)
        assert pct_integer > 0.95, (
            f"DelMar: only {pct_integer:.0%} of alongshore_m values are integers. "
            f"Expected integer spacing from Id field."
        )

    def test_delmar_max_alongshore(self, features_df):
        """DelMar max alongshore_m should be ~2284 (Id 2759 - 475)."""
        dm = features_df[features_df["location"] == "DelMar"]
        if len(dm) == 0:
            pytest.skip("No DelMar data in features")

        max_along = dm["alongshore_m"].max()
        assert 2200 <= max_along <= 2300, (
            f"DelMar: max alongshore_m = {max_along:.1f}, expected ~2284"
        )

    def test_other_beaches_continuous_spacing(self, features_df):
        """Non-DelMar beaches should have ~1m spacing (not 100m MOP gaps)."""
        for location in ["Torrey", "Encinitas", "SanElijo", "Solana", "Blacks"]:
            loc_df = features_df[features_df["location"] == location]
            if len(loc_df) == 0:
                continue

            unique_along = np.sort(loc_df["alongshore_m"].unique())
            if len(unique_along) < 2:
                continue

            # Check median spacing between consecutive unique values
            diffs = np.diff(unique_along)
            median_spacing = np.median(diffs)
            assert median_spacing < 3.0, (
                f"{location}: median polygon spacing = {median_spacing:.1f}m, "
                f"expected ~1m (got MOP-level ~100m spacing?)"
            )


class TestPolygonResolution:
    """Verify sub-transect resolution is preserved (not collapsed to integer MOP)."""

    def test_unique_polygon_count_per_survey(self, features_df):
        """Each survey-location with real coverage should have many unique polygons."""
        for (loc, survey), group in features_df.groupby(["location", "survey_file"]):
            n_unique = group["alongshore_m"].nunique()

            # Skip survey-location pairs with trivial coverage
            # (e.g., a single boundary polygon from an adjacent survey)
            if n_unique < MIN_POLYGONS_FOR_COVERAGE:
                continue

            # With 1m resolution, a real survey should have >50 unique polygons
            # Integer MOP would give only ~25-60 unique values
            if loc in EXPECTED_POLYGON_COUNTS:
                expected_max = EXPECTED_POLYGON_COUNTS[loc]
                assert n_unique > 50, (
                    f"{loc}/{survey}: only {n_unique} unique polygons. "
                    f"Expected many more at 1m resolution (max possible: {expected_max})"
                )

    def test_non_delmar_not_integer_mop(self, features_df):
        """Non-DelMar locations should NOT have all-integer alongshore values
        (that would indicate MOP rounding)."""
        for location in ["Torrey", "Encinitas", "SanElijo", "Solana", "Blacks"]:
            loc_df = features_df[features_df["location"] == location]
            unique_along = loc_df["alongshore_m"].unique()

            # Need sufficient data for this heuristic to be meaningful
            if len(unique_along) < MIN_POLYGONS_FOR_COVERAGE:
                continue

            remainders = unique_along % 1.0
            n_integer = np.sum((remainders < 0.01) | (remainders > 0.99))
            pct_integer = n_integer / len(unique_along)

            # centroid_y - min_centroid_y produces non-integer values
            assert pct_integer < 0.5, (
                f"{location}: {pct_integer:.0%} of alongshore_m values are integers. "
                f"Expected non-integer values from centroid Y distances."
            )


class TestDataQuality:
    """Basic quality checks on the output CSV."""

    def test_required_columns_exist(self, features_df):
        """Check all expected columns are present."""
        required = [
            "survey_date", "survey_file", "location",
            "polygon_id", "alongshore_m", "zone", "zone_idx",
            "n_points", "z_min", "z_max", "z_mean", "z_range",
        ]
        for col in required:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_zones_are_valid(self, features_df):
        """Zone should be lower/middle/upper with idx 0/1/2."""
        assert set(features_df["zone"].unique()) <= {"lower", "middle", "upper"}
        assert set(features_df["zone_idx"].unique()) <= {0, 1, 2}

    def test_feature_columns_exist(self, features_df):
        """Should have aggregated feature columns (slope_mean, etc.)."""
        feature_prefixes = ["slope", "roughness_small", "roughness_large"]
        stat_suffixes = ["mean", "std", "min", "max", "p10", "p50", "p90"]

        for prefix in feature_prefixes:
            for suffix in stat_suffixes:
                col = f"{prefix}_{suffix}"
                assert col in features_df.columns, (
                    f"Missing feature column: {col}"
                )

    def test_no_negative_point_counts(self, features_df):
        """n_points should be positive for all rows."""
        assert (features_df["n_points"] > 0).all()

    def test_elevation_consistency(self, features_df):
        """z_min <= z_mean <= z_max for all rows."""
        assert (features_df["z_min"] <= features_df["z_mean"] + 0.01).all()
        assert (features_df["z_mean"] <= features_df["z_max"] + 0.01).all()

    def test_multiple_locations(self, features_df):
        """Output should contain multiple locations."""
        n_locations = features_df["location"].nunique()
        assert n_locations >= 2, (
            f"Only {n_locations} location(s) in output. "
            f"Expected multiple beaches."
        )

    def test_row_count_reasonable(self, features_df):
        """Should have substantial data (at least 1000 polygon-zones)."""
        assert len(features_df) > 1000, (
            f"Only {len(features_df)} rows. Expected many more "
            f"with 1m resolution × 3 zones × multiple surveys."
        )

    def test_slope_values_reasonable(self, features_df):
        """Mean slope should be in 0-180 degree range."""
        assert (features_df["slope_mean"] >= 0).all()
        assert (features_df["slope_mean"] <= 180).all()

    def test_polygon_id_preserved(self, features_df):
        """polygon_id should still contain MOP-like values for reference."""
        for location in ["Torrey", "Encinitas", "SanElijo"]:
            loc_df = features_df[features_df["location"] == location]
            if len(loc_df) == 0:
                continue

            mop_min, mop_max = MOP_RANGES[location]
            pid_min = loc_df["polygon_id"].min()
            pid_max = loc_df["polygon_id"].max()

            # polygon_id should still be in MOP range
            assert pid_min >= mop_min - 1, (
                f"{location}: polygon_id min={pid_min}, "
                f"expected >= {mop_min} (MOP range)"
            )
            assert pid_max <= mop_max + 1, (
                f"{location}: polygon_id max={pid_max}, "
                f"expected <= {mop_max} (MOP range)"
            )


class TestEventMatchability:
    """Verify the output would work with create_labels() for event matching."""

    def test_alongshore_range_covers_events(self, features_df):
        """Each location's alongshore range should be wide enough
        to overlap with event coordinates."""
        # Events use 0-based local meters, so polygon alongshore
        # should span a similar range
        for location, group in features_df.groupby("location"):
            if group["alongshore_m"].nunique() < MIN_POLYGONS_FOR_COVERAGE:
                continue
            max_along = group["alongshore_m"].max()
            # Each beach should span at least several hundred meters
            assert max_along > 100, (
                f"{location}: max alongshore_m = {max_along:.1f}m, "
                f"too small to match events"
            )

    def test_alongshore_and_polygon_id_differ(self, features_df):
        """alongshore_m and polygon_id should be different columns
        (local meters vs MOP coordinates)."""
        for location in ["Torrey", "Encinitas", "SanElijo"]:
            loc_df = features_df[features_df["location"] == location]
            if len(loc_df) == 0:
                continue

            # They should not be identical
            same = (loc_df["alongshore_m"] == loc_df["polygon_id"]).all()
            assert not same, (
                f"{location}: alongshore_m == polygon_id for all rows. "
                f"The indexing fix should make these different."
            )
