"""Tests for polygon indexing fix — local alongshore_m coordinates."""

import pytest
from pathlib import Path

try:
    from pc_rai.ml.polygon_aggregation import load_polygons
    HAS_POLYGON_MODULE = True
except ImportError:
    HAS_POLYGON_MODULE = False


SHAPEFILE_DIR = Path("utiliies/polygons_1m")


@pytest.fixture
def shapefile_dir():
    """Provide shapefile directory, skip if not available."""
    if not SHAPEFILE_DIR.exists():
        pytest.skip("Shapefile directory not available")
    return SHAPEFILE_DIR


@pytest.mark.skipif(not HAS_POLYGON_MODULE, reason="polygon_aggregation not importable")
class TestLoadPolygonsAlongshore:
    """Test that load_polygons returns correct local alongshore_m values."""

    def test_delmar_alongshore_zero_based(self, shapefile_dir):
        """DelMar alongshore_m should be 0-based integers from Id field."""
        polygons, _ = load_polygons(shapefile_dir, "DelMar", verbose=False)
        along_values = [p["alongshore_m"] for p in polygons]

        assert min(along_values) == 0
        # Id range is 475-2759, so max = 2759 - 475 = 2284
        assert max(along_values) == 2284

    def test_other_beach_alongshore_zero_based(self, shapefile_dir):
        """Non-DelMar beaches should have alongshore_m starting at 0."""
        for location in ["Torrey", "Blacks", "Encinitas", "SanElijo", "Solana"]:
            subdir = {
                "Torrey": "TorreyPolygons567to581at1m",
                "Blacks": "BlacksPolygons520to567at1m",
                "Encinitas": "EncinitasPolygons708to764at1m",
                "SanElijo": "SanElijoPolygons683to708at1m",
                "Solana": "SolanaPolygons637to666at1m",
            }[location]
            if not (shapefile_dir / subdir).exists():
                continue

            polygons, _ = load_polygons(shapefile_dir, location, verbose=False)
            along_values = [p["alongshore_m"] for p in polygons]

            assert min(along_values) == 0.0, (
                f"{location}: min alongshore_m should be 0, got {min(along_values)}"
            )
            # Extent should be > 1000m for all beaches
            assert max(along_values) > 1000, (
                f"{location}: max alongshore_m only {max(along_values):.1f}, "
                f"expected > 1000m"
            )

    def test_polygon_has_both_fields(self, shapefile_dir):
        """Each polygon dict should have both mop_id and alongshore_m."""
        polygons, _ = load_polygons(shapefile_dir, "DelMar", verbose=False)
        for p in polygons[:5]:
            assert "mop_id" in p, "polygon missing 'mop_id' key"
            assert "alongshore_m" in p, "polygon missing 'alongshore_m' key"

    def test_delmar_mop_differs_from_alongshore(self, shapefile_dir):
        """For DelMar, mop_id and alongshore_m should differ."""
        polygons, _ = load_polygons(shapefile_dir, "DelMar", verbose=False)
        # mop_id is in 595-620 range, alongshore_m is 0-2284
        p = polygons[0]
        assert p["mop_id"] != p["alongshore_m"]

    def test_alongshore_monotonic(self, shapefile_dir):
        """Alongshore values should increase south-to-north (mostly)."""
        polygons, _ = load_polygons(shapefile_dir, "Torrey", verbose=False)

        # Sort by centroid Y to check ordering
        import numpy as np

        ys = [p["points"][:, 1].mean() for p in polygons]
        along = [p["alongshore_m"] for p in polygons]

        # Correlation between Y and alongshore_m should be very high
        corr = np.corrcoef(ys, along)[0, 1]
        assert corr > 0.99, f"Expected high Y-alongshore correlation, got {corr:.4f}"

    def test_torrey_expected_extent(self, shapefile_dir):
        """Torrey should have ~1399 polygons spanning ~1400m."""
        subdir = shapefile_dir / "TorreyPolygons567to581at1m"
        if not subdir.exists():
            pytest.skip("Torrey shapefiles not available")

        polygons, _ = load_polygons(shapefile_dir, "Torrey", verbose=False)
        along_values = [p["alongshore_m"] for p in polygons]

        assert len(polygons) == 1399
        assert max(along_values) > 1300
        assert max(along_values) < 1500

    def test_encinitas_expected_extent(self, shapefile_dir):
        """Encinitas should have ~5607 polygons."""
        subdir = shapefile_dir / "EncinitasPolygons708to764at1m"
        if not subdir.exists():
            pytest.skip("Encinitas shapefiles not available")

        polygons, _ = load_polygons(shapefile_dir, "Encinitas", verbose=False)
        along_values = [p["alongshore_m"] for p in polygons]

        assert len(polygons) == 5607
        assert max(along_values) > 5000
