"""Tests for visualization module."""

import numpy as np
import pytest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


class TestRAIColormap:
    """Tests for RAI colormap creation."""

    def test_create_colormap(self):
        """Test colormap has correct number of colors."""
        from pc_rai.visualization.render_3d import create_rai_colormap

        cmap = create_rai_colormap()
        assert cmap.N == 8

    def test_colormap_name(self):
        """Test colormap has correct name."""
        from pc_rai.visualization.render_3d import create_rai_colormap

        cmap = create_rai_colormap()
        assert cmap.name == "rai_classes"


class TestHexToRGB:
    """Tests for hex color conversion."""

    def test_hex_to_rgb_basic(self):
        """Test basic hex color conversion."""
        from pc_rai.visualization.render_3d import hex_to_rgb

        # White
        assert hex_to_rgb("#FFFFFF") == (1.0, 1.0, 1.0)
        # Black
        assert hex_to_rgb("#000000") == (0.0, 0.0, 0.0)
        # Red
        assert hex_to_rgb("#FF0000") == (1.0, 0.0, 0.0)

    def test_hex_to_rgb_without_hash(self):
        """Test conversion without leading hash."""
        from pc_rai.visualization.render_3d import hex_to_rgb

        assert hex_to_rgb("FFFFFF") == (1.0, 1.0, 1.0)


class TestGetClassColors:
    """Tests for class-to-color mapping."""

    def test_class_colors_shape(self):
        """Test output shape is correct."""
        from pc_rai.visualization.render_3d import get_class_colors

        classes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        colors = get_class_colors(classes)
        assert colors.shape == (8, 3)

    def test_class_colors_range(self):
        """Test colors are in valid range."""
        from pc_rai.visualization.render_3d import get_class_colors

        classes = np.random.randint(0, 8, 100).astype(np.uint8)
        colors = get_class_colors(classes)
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)

    def test_class_colors_consistency(self):
        """Test same class gets same color."""
        from pc_rai.visualization.render_3d import get_class_colors

        classes = np.array([1, 1, 1, 2, 2])
        colors = get_class_colors(classes)
        # All class 1 points should have same color
        assert np.allclose(colors[0], colors[1])
        assert np.allclose(colors[0], colors[2])
        # Class 2 should be different
        assert not np.allclose(colors[0], colors[3])


class TestGetViewParams:
    """Tests for view parameter retrieval."""

    def test_front_view(self):
        """Test front view parameters."""
        from pc_rai.visualization.render_3d import get_view_params

        bounds = {"x": (0, 10), "y": (0, 10), "z": (0, 5)}
        params = get_view_params("front", bounds)
        assert "elev" in params
        assert "azim" in params

    def test_oblique_view(self):
        """Test oblique view parameters."""
        from pc_rai.visualization.render_3d import get_view_params

        bounds = {"x": (0, 10), "y": (0, 10), "z": (0, 5)}
        params = get_view_params("oblique", bounds)
        assert params["elev"] > 0  # Oblique has elevated view

    def test_invalid_view(self):
        """Test error on invalid view name."""
        from pc_rai.visualization.render_3d import get_view_params

        bounds = {"x": (0, 10), "y": (0, 10), "z": (0, 5)}
        with pytest.raises(ValueError, match="Unknown view"):
            get_view_params("invalid_view", bounds)


class TestRenderClassification:
    """Tests for classification rendering."""

    def test_render_creates_figure(self):
        """Test render returns a figure."""
        from pc_rai.visualization.render_3d import render_classification

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        fig = render_classification(xyz, classes, max_points=100)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_saves_file(self, tmp_path):
        """Test render saves to file."""
        from pc_rai.visualization.render_3d import render_classification

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        output_path = tmp_path / "test_classification.png"
        fig = render_classification(
            xyz, classes, output_path=str(output_path), max_points=100, dpi=72
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_render_all_views(self, tmp_path):
        """Test all view angles work."""
        from pc_rai.visualization.render_3d import render_classification

        n = 50
        xyz = np.random.uniform(0, 10, (n, 3))
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        for view in ["front", "oblique", "top", "side"]:
            fig = render_classification(xyz, classes, view=view, max_points=50)
            plt.close(fig)

    def test_render_with_legend(self):
        """Test legend is added when requested."""
        from pc_rai.visualization.render_3d import render_classification

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        fig = render_classification(xyz, classes, show_legend=True, max_points=100)
        ax = fig.axes[0]
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_render_without_legend(self):
        """Test legend is not added when disabled."""
        from pc_rai.visualization.render_3d import render_classification

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        classes = np.random.randint(0, 8, n).astype(np.uint8)

        fig = render_classification(xyz, classes, show_legend=False, max_points=100)
        ax = fig.axes[0]
        assert ax.get_legend() is None
        plt.close(fig)


class TestRenderContinuous:
    """Tests for continuous value rendering."""

    def test_render_continuous_creates_figure(self):
        """Test continuous render returns a figure."""
        from pc_rai.visualization.render_3d import render_continuous

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        values = np.random.uniform(0, 90, n)

        fig = render_continuous(xyz, values, max_points=100)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_continuous_saves_file(self, tmp_path):
        """Test continuous render saves to file."""
        from pc_rai.visualization.render_3d import render_continuous

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        values = np.random.uniform(0, 90, n)

        output_path = tmp_path / "test_continuous.png"
        fig = render_continuous(
            xyz, values, output_path=str(output_path), max_points=100, dpi=72
        )

        assert output_path.exists()
        plt.close(fig)

    def test_render_with_nan_values(self):
        """Test rendering handles NaN values."""
        from pc_rai.visualization.render_3d import render_continuous

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        values = np.random.uniform(0, 90, n)
        values[::10] = np.nan  # Add some NaN values

        fig = render_continuous(xyz, values, max_points=100)
        plt.close(fig)

    def test_custom_colormap(self):
        """Test custom colormap is applied."""
        from pc_rai.visualization.render_3d import render_continuous

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        values = np.random.uniform(0, 90, n)

        fig = render_continuous(xyz, values, cmap="plasma", max_points=100)
        plt.close(fig)

    def test_custom_vmin_vmax(self):
        """Test custom value range is applied."""
        from pc_rai.visualization.render_3d import render_continuous

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        values = np.random.uniform(20, 70, n)

        fig = render_continuous(xyz, values, vmin=0, vmax=90, max_points=100)
        plt.close(fig)


class TestRenderSlope:
    """Tests for slope rendering convenience function."""

    def test_render_slope(self, tmp_path):
        """Test slope rendering."""
        from pc_rai.visualization.render_3d import render_slope

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        slope = np.random.uniform(0, 180, n)

        output_path = tmp_path / "test_slope.png"
        fig = render_slope(xyz, slope, output_path=str(output_path), max_points=100, dpi=72)

        assert output_path.exists()
        plt.close(fig)


class TestRenderRoughness:
    """Tests for roughness rendering convenience function."""

    def test_render_roughness(self, tmp_path):
        """Test roughness rendering."""
        from pc_rai.visualization.render_3d import render_roughness

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        roughness = np.random.uniform(0, 30, n)

        output_path = tmp_path / "test_roughness.png"
        fig = render_roughness(
            xyz, roughness, output_path=str(output_path), max_points=100, dpi=72
        )

        assert output_path.exists()
        plt.close(fig)

    def test_render_roughness_with_nan(self):
        """Test roughness rendering with NaN values."""
        from pc_rai.visualization.render_3d import render_roughness

        n = 100
        xyz = np.random.uniform(0, 10, (n, 3))
        roughness = np.random.uniform(0, 30, n)
        roughness[::5] = np.nan

        fig = render_roughness(xyz, roughness, max_points=100)
        plt.close(fig)


class TestSubsampling:
    """Tests for point subsampling."""

    def test_subsampling_reduces_points(self):
        """Test subsampling limits point count."""
        from pc_rai.visualization.render_3d import _subsample_points

        n = 10000
        xyz = np.random.uniform(0, 10, (n, 3))
        colors = np.random.uniform(0, 1, (n, 3))

        xyz_sub, colors_sub = _subsample_points(xyz, colors, max_points=1000)

        assert len(xyz_sub) == 1000
        assert len(colors_sub) == 1000

    def test_subsampling_preserves_small_clouds(self):
        """Test subsampling doesn't reduce small clouds."""
        from pc_rai.visualization.render_3d import _subsample_points

        n = 500
        xyz = np.random.uniform(0, 10, (n, 3))
        colors = np.random.uniform(0, 1, (n, 3))

        xyz_sub, colors_sub = _subsample_points(xyz, colors, max_points=1000)

        assert len(xyz_sub) == n
        assert np.array_equal(xyz_sub, xyz)

    def test_subsampling_reproducible(self):
        """Test subsampling is reproducible with same seed."""
        from pc_rai.visualization.render_3d import _subsample_points

        n = 10000
        xyz = np.random.uniform(0, 10, (n, 3))
        colors = np.random.uniform(0, 1, (n, 3))

        xyz_1, _ = _subsample_points(xyz, colors, max_points=1000, seed=42)
        xyz_2, _ = _subsample_points(xyz, colors, max_points=1000, seed=42)

        assert np.array_equal(xyz_1, xyz_2)
