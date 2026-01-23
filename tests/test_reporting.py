"""Tests for reporting module."""

import json
import numpy as np
import pytest
from pathlib import Path


class TestClassificationStats:
    """Tests for classification statistics calculation."""

    def test_classification_stats_basic(self):
        """Test basic classification statistics."""
        from pc_rai.reporting.statistics import calculate_classification_stats

        # Create classes with known distribution
        classes = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.uint8)
        stats = calculate_classification_stats(classes)

        assert stats["total"] == 10
        assert stats["by_class"][0]["count"] == 2
        assert stats["by_class"][1]["count"] == 3
        assert stats["by_class"][2]["count"] == 5
        assert stats["by_class"][0]["percent"] == 20.0
        assert stats["by_class"][1]["percent"] == 30.0
        assert stats["by_class"][2]["percent"] == 50.0

    def test_classification_stats_all_classes(self):
        """Test statistics for all 8 classes."""
        from pc_rai.reporting.statistics import calculate_classification_stats

        classes = np.arange(8).astype(np.uint8)
        stats = calculate_classification_stats(classes)

        assert len(stats["by_class"]) == 8
        for code in range(8):
            assert stats["by_class"][code]["count"] == 1
            assert stats["by_class"][code]["percent"] == 12.5

    def test_classification_stats_empty(self):
        """Test statistics with empty array."""
        from pc_rai.reporting.statistics import calculate_classification_stats

        classes = np.array([], dtype=np.uint8)
        stats = calculate_classification_stats(classes)

        assert stats["total"] == 0
        assert all(stats["by_class"][c]["count"] == 0 for c in range(8))

    def test_classification_stats_has_names(self):
        """Test statistics include class names."""
        from pc_rai.reporting.statistics import calculate_classification_stats

        classes = np.array([1], dtype=np.uint8)
        stats = calculate_classification_stats(classes)

        assert stats["by_class"][1]["name"] == "Talus"
        assert stats["by_class"][1]["abbrev"] == "T"


class TestFeatureStats:
    """Tests for feature statistics calculation."""

    def test_feature_stats_basic(self):
        """Test basic feature statistics."""
        from pc_rai.reporting.statistics import calculate_feature_stats

        values = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        stats = calculate_feature_stats(values, "test_feature")

        assert stats["name"] == "test_feature"
        assert stats["count"] == 5
        assert stats["nan_count"] == 0
        assert stats["mean"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0

    def test_feature_stats_with_nan(self):
        """Test feature statistics with NaN values."""
        from pc_rai.reporting.statistics import calculate_feature_stats

        values = np.array([10, 20, np.nan, 40, np.nan], dtype=np.float32)
        stats = calculate_feature_stats(values, "test_feature")

        assert stats["count"] == 3
        assert stats["nan_count"] == 2
        # Mean of 10, 20, 40 = 23.333...
        assert abs(stats["mean"] - 23.3333) < 0.01

    def test_feature_stats_all_nan(self):
        """Test feature statistics with all NaN values."""
        from pc_rai.reporting.statistics import calculate_feature_stats

        values = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        stats = calculate_feature_stats(values, "test_feature")

        assert stats["count"] == 0
        assert stats["nan_count"] == 3
        assert stats["mean"] is None
        assert stats["std"] is None

    def test_feature_stats_percentiles(self):
        """Test feature statistics include percentiles."""
        from pc_rai.reporting.statistics import calculate_feature_stats

        values = np.arange(100, dtype=np.float32)
        stats = calculate_feature_stats(values, "test_feature")

        assert "percentiles" in stats
        assert 50 in stats["percentiles"]
        # 50th percentile of 0-99 is approximately 49.5
        assert abs(stats["percentiles"][50] - 49.5) < 1


class TestMethodAgreement:
    """Tests for method agreement calculation."""

    def test_agreement_perfect_match(self):
        """Test agreement with identical classifications."""
        from pc_rai.reporting.statistics import calculate_method_agreement

        classes = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        agreement = calculate_method_agreement(classes, classes)

        assert agreement["agreement_count"] == 8
        assert agreement["agreement_pct"] == 100.0
        assert agreement["cohens_kappa"] == 1.0

    def test_agreement_no_match(self):
        """Test agreement with no matching classifications."""
        from pc_rai.reporting.statistics import calculate_method_agreement

        classes_radius = np.zeros(100, dtype=np.uint8)
        classes_knn = np.ones(100, dtype=np.uint8)
        agreement = calculate_method_agreement(classes_radius, classes_knn)

        assert agreement["agreement_count"] == 0
        assert agreement["agreement_pct"] == 0.0

    def test_agreement_partial(self):
        """Test agreement with partial matching."""
        from pc_rai.reporting.statistics import calculate_method_agreement

        classes_radius = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8)
        classes_knn = np.array([0, 1, 1, 2, 2, 3], dtype=np.uint8)
        agreement = calculate_method_agreement(classes_radius, classes_knn)

        # 3 matches: indices 0, 2, 4
        assert agreement["agreement_count"] == 3
        assert agreement["agreement_pct"] == 50.0

    def test_agreement_confusion_matrix(self):
        """Test confusion matrix is correct shape."""
        from pc_rai.reporting.statistics import calculate_method_agreement

        classes_radius = np.random.randint(0, 8, 100).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, 100).astype(np.uint8)
        agreement = calculate_method_agreement(classes_radius, classes_knn)

        confusion = np.array(agreement["confusion_matrix"])
        assert confusion.shape == (8, 8)
        assert confusion.sum() == 100

    def test_agreement_kappa_calculation(self):
        """Test Cohen's kappa is in valid range."""
        from pc_rai.reporting.statistics import calculate_method_agreement

        classes_radius = np.random.randint(0, 8, 1000).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, 1000).astype(np.uint8)
        agreement = calculate_method_agreement(classes_radius, classes_knn)

        # Kappa should be between -1 and 1
        assert -1 <= agreement["cohens_kappa"] <= 1


class TestCalculateAllStatistics:
    """Tests for combined statistics calculation."""

    def test_calculate_all_basic(self):
        """Test calculate_all_statistics with basic data."""
        from pc_rai.reporting.statistics import calculate_all_statistics

        n = 100
        slope = np.random.uniform(0, 90, n).astype(np.float32)
        r_small = np.random.uniform(0, 20, n).astype(np.float32)
        r_large = np.random.uniform(0, 25, n).astype(np.float32)
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        stats = calculate_all_statistics(
            slope_deg=slope,
            roughness_small_radius=r_small,
            roughness_large_radius=r_large,
            classes_radius=classes,
        )

        assert stats["n_points"] == n
        assert "slope_deg" in stats["features"]
        assert "roughness_small_radius" in stats["features"]
        assert stats["classification_radius"] is not None

    def test_calculate_all_with_both_methods(self):
        """Test calculate_all_statistics with both methods."""
        from pc_rai.reporting.statistics import calculate_all_statistics

        n = 100
        slope = np.random.uniform(0, 90, n).astype(np.float32)
        classes_radius = np.random.randint(0, 8, n).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, n).astype(np.uint8)

        stats = calculate_all_statistics(
            slope_deg=slope,
            classes_radius=classes_radius,
            classes_knn=classes_knn,
        )

        assert stats["classification_radius"] is not None
        assert stats["classification_knn"] is not None
        assert stats["method_agreement"] is not None


class TestMarkdownReport:
    """Tests for Markdown report generation."""

    def test_write_markdown_report(self, tmp_path):
        """Test Markdown report is written correctly."""
        from pc_rai.reporting.statistics import calculate_all_statistics
        from pc_rai.reporting.report_writer import write_markdown_report

        # Create test data
        n = 100
        slope = np.random.uniform(0, 90, n).astype(np.float32)
        classes = np.random.randint(0, 8, n).astype(np.uint8)
        stats = calculate_all_statistics(slope_deg=slope, classes_radius=classes)

        output_path = tmp_path / "report.md"
        config_summary = {"methods": ["radius"], "radius_small": "0.175m"}

        write_markdown_report(stats, output_path, "test.las", config_summary)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# RAI Classification Report" in content
        assert "test.las" in content
        assert "100" in content

    def test_write_markdown_with_extent(self, tmp_path):
        """Test Markdown report includes extent information."""
        from pc_rai.reporting.statistics import calculate_all_statistics
        from pc_rai.reporting.report_writer import write_markdown_report

        slope = np.random.uniform(0, 90, 100).astype(np.float32)
        stats = calculate_all_statistics(slope_deg=slope)

        output_path = tmp_path / "report.md"
        extent = {"x": (0, 100), "y": (0, 50), "z": (0, 25)}

        write_markdown_report(
            stats, output_path, "test.las", {}, extent=extent
        )

        content = output_path.read_text()
        assert "Extent X" in content
        assert "100.00" in content

    def test_write_markdown_with_timing(self, tmp_path):
        """Test Markdown report includes timing information."""
        from pc_rai.reporting.statistics import calculate_all_statistics
        from pc_rai.reporting.report_writer import write_markdown_report

        slope = np.random.uniform(0, 90, 100).astype(np.float32)
        stats = calculate_all_statistics(slope_deg=slope)

        output_path = tmp_path / "report.md"
        timing = {"normals": 10.5, "roughness": 25.3, "total": 35.8}

        write_markdown_report(
            stats, output_path, "test.las", {}, timing=timing
        )

        content = output_path.read_text()
        assert "Processing Time" in content
        assert "10.50s" in content


class TestJsonReport:
    """Tests for JSON report generation."""

    def test_write_json_report(self, tmp_path):
        """Test JSON report is written correctly."""
        from pc_rai.reporting.statistics import calculate_all_statistics
        from pc_rai.reporting.report_writer import write_json_report

        n = 100
        slope = np.random.uniform(0, 90, n).astype(np.float32)
        classes = np.random.randint(0, 8, n).astype(np.uint8)
        stats = calculate_all_statistics(slope_deg=slope, classes_radius=classes)

        output_path = tmp_path / "report.json"
        config_summary = {"methods": ["radius"]}

        write_json_report(stats, output_path, "test.las", config_summary)

        assert output_path.exists()

        # Verify valid JSON
        with open(output_path) as f:
            data = json.load(f)

        assert data["input"]["file"] == "test.las"
        assert data["input"]["n_points"] == 100
        assert "features" in data["statistics"]

    def test_json_report_serialization(self, tmp_path):
        """Test JSON report handles numpy types correctly."""
        from pc_rai.reporting.statistics import calculate_all_statistics
        from pc_rai.reporting.report_writer import write_json_report

        # Use numpy types that need serialization
        slope = np.array([1, 2, 3], dtype=np.float32)
        classes = np.array([0, 1, 2], dtype=np.uint8)
        stats = calculate_all_statistics(slope_deg=slope, classes_radius=classes)

        output_path = tmp_path / "report.json"
        write_json_report(stats, output_path, "test.las", {})

        # Should not raise
        with open(output_path) as f:
            json.load(f)


class TestConfigSummary:
    """Tests for config summary generation."""

    def test_generate_config_summary(self):
        """Test config summary generation."""
        from pc_rai.config import RAIConfig
        from pc_rai.reporting.report_writer import generate_config_summary

        config = RAIConfig(
            radius_small=0.175,
            radius_large=0.425,
            k_small=30,
            k_large=100,
        )

        summary = generate_config_summary(config)

        assert "radius_small" in summary
        assert "0.175m" in summary["radius_small"]
        assert summary["k_small"] == 30
