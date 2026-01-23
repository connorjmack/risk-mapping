"""Tests for multi-panel figure generation."""

import numpy as np
import pytest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


class TestComparisonFigure:
    """Tests for comparison figure generation."""

    def test_comparison_creates_figure(self):
        """Test comparison figure returns a figure."""
        from pc_rai.visualization.figures import create_comparison_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes_radius = np.random.randint(0, 8, n).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, n).astype(np.uint8)

        fig = create_comparison_figure(
            xyz, classes_radius, classes_knn, max_points=100
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_saves_file(self, tmp_path):
        """Test comparison figure saves to file."""
        from pc_rai.visualization.figures import create_comparison_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes_radius = np.random.randint(0, 8, n).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, n).astype(np.uint8)

        output_path = tmp_path / "comparison.png"
        fig = create_comparison_figure(
            xyz,
            classes_radius,
            classes_knn,
            output_path=str(output_path),
            max_points=100,
            dpi=72,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_comparison_has_two_subplots(self):
        """Test comparison figure has two 3D subplots."""
        from pc_rai.visualization.figures import create_comparison_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes_radius = np.random.randint(0, 8, n).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, n).astype(np.uint8)

        fig = create_comparison_figure(
            xyz, classes_radius, classes_knn, max_points=100
        )

        # Should have at least 2 axes (the 3D subplots)
        axes_3d = [ax for ax in fig.axes if hasattr(ax, "get_zlim")]
        assert len(axes_3d) == 2
        plt.close(fig)


class TestSummaryFigure:
    """Tests for summary figure generation."""

    def test_summary_creates_figure(self):
        """Test summary figure returns a figure."""
        from pc_rai.visualization.figures import create_summary_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        slope = np.random.uniform(0, 90, n)
        r_small = np.random.uniform(0, 20, n)
        r_large = np.random.uniform(0, 25, n)
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        fig = create_summary_figure(
            xyz, slope, r_small, r_large, classes, max_points=100
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_summary_saves_file(self, tmp_path):
        """Test summary figure saves to file."""
        from pc_rai.visualization.figures import create_summary_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        slope = np.random.uniform(0, 90, n)
        r_small = np.random.uniform(0, 20, n)
        r_large = np.random.uniform(0, 25, n)
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        output_path = tmp_path / "summary.png"
        fig = create_summary_figure(
            xyz,
            slope,
            r_small,
            r_large,
            classes,
            output_path=str(output_path),
            max_points=100,
            dpi=72,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_summary_has_four_subplots(self):
        """Test summary figure has four 3D subplots."""
        from pc_rai.visualization.figures import create_summary_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        slope = np.random.uniform(0, 90, n)
        r_small = np.random.uniform(0, 20, n)
        r_large = np.random.uniform(0, 25, n)
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        fig = create_summary_figure(
            xyz, slope, r_small, r_large, classes, max_points=100
        )

        # Should have 4 3D axes
        axes_3d = [ax for ax in fig.axes if hasattr(ax, "get_zlim")]
        assert len(axes_3d) == 4
        plt.close(fig)

    def test_summary_handles_nan(self):
        """Test summary figure handles NaN values in roughness."""
        from pc_rai.visualization.figures import create_summary_figure

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        slope = np.random.uniform(0, 90, n)
        r_small = np.random.uniform(0, 20, n)
        r_large = np.random.uniform(0, 25, n)
        # Add NaN values
        r_small[::10] = np.nan
        r_large[::10] = np.nan
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        fig = create_summary_figure(
            xyz, slope, r_small, r_large, classes, max_points=100
        )
        plt.close(fig)


class TestHistogramFigure:
    """Tests for histogram figure generation."""

    def test_histogram_creates_figure(self):
        """Test histogram figure returns a figure."""
        from pc_rai.visualization.figures import create_histogram_figure

        classes = np.random.randint(0, 8, 1000).astype(np.uint8)

        fig = create_histogram_figure(classes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_histogram_saves_file(self, tmp_path):
        """Test histogram figure saves to file."""
        from pc_rai.visualization.figures import create_histogram_figure

        classes = np.random.randint(0, 8, 1000).astype(np.uint8)

        output_path = tmp_path / "histogram.png"
        fig = create_histogram_figure(classes, output_path=str(output_path), dpi=72)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_histogram_all_classes(self):
        """Test histogram includes all 8 classes."""
        from pc_rai.visualization.figures import create_histogram_figure

        # Create classes with all 8 values
        classes = np.arange(8).astype(np.uint8)

        fig = create_histogram_figure(classes)
        ax = fig.axes[0]

        # Should have 8 bars
        bars = ax.patches
        assert len(bars) == 8
        plt.close(fig)

    def test_histogram_single_class(self):
        """Test histogram works with single class."""
        from pc_rai.visualization.figures import create_histogram_figure

        classes = np.zeros(100, dtype=np.uint8)  # All unclassified

        fig = create_histogram_figure(classes)
        plt.close(fig)

    def test_histogram_with_percentages(self):
        """Test histogram shows percentage labels."""
        from pc_rai.visualization.figures import create_histogram_figure

        classes = np.random.randint(0, 8, 1000).astype(np.uint8)

        fig = create_histogram_figure(classes, show_percentages=True)
        ax = fig.axes[0]

        # Should have text annotations
        texts = [t for t in ax.texts if "%" in t.get_text()]
        assert len(texts) > 0
        plt.close(fig)

    def test_histogram_without_percentages(self):
        """Test histogram without percentage labels."""
        from pc_rai.visualization.figures import create_histogram_figure

        classes = np.random.randint(0, 8, 1000).astype(np.uint8)

        fig = create_histogram_figure(classes, show_percentages=False)
        ax = fig.axes[0]

        # Should have no percentage annotations (only total count)
        texts = [t for t in ax.texts if "%" in t.get_text()]
        assert len(texts) == 0
        plt.close(fig)


class TestMethodAgreementFigure:
    """Tests for method agreement figure generation."""

    def test_agreement_creates_figure(self):
        """Test agreement figure returns a figure."""
        from pc_rai.visualization.figures import create_method_agreement_figure

        classes_radius = np.random.randint(0, 8, 1000).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, 1000).astype(np.uint8)

        fig = create_method_agreement_figure(classes_radius, classes_knn)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_agreement_saves_file(self, tmp_path):
        """Test agreement figure saves to file."""
        from pc_rai.visualization.figures import create_method_agreement_figure

        classes_radius = np.random.randint(0, 8, 1000).astype(np.uint8)
        classes_knn = np.random.randint(0, 8, 1000).astype(np.uint8)

        output_path = tmp_path / "agreement.png"
        fig = create_method_agreement_figure(
            classes_radius, classes_knn, output_path=str(output_path), dpi=72
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_agreement_perfect_match(self):
        """Test agreement figure with identical classifications."""
        from pc_rai.visualization.figures import create_method_agreement_figure

        classes = np.random.randint(0, 8, 1000).astype(np.uint8)

        fig = create_method_agreement_figure(classes, classes)
        # Title should show 100% agreement
        assert "100.0%" in fig.axes[0].get_title()
        plt.close(fig)

    def test_agreement_no_match(self):
        """Test agreement figure with no matching classifications."""
        from pc_rai.visualization.figures import create_method_agreement_figure

        # Radius is 0, k-NN is 1 for all points
        classes_radius = np.zeros(1000, dtype=np.uint8)
        classes_knn = np.ones(1000, dtype=np.uint8)

        fig = create_method_agreement_figure(classes_radius, classes_knn)
        # Title should show 0% agreement
        assert "0.0%" in fig.axes[0].get_title()
        plt.close(fig)
