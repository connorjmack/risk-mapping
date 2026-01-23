"""End-to-end integration tests for PC-RAI."""

import numpy as np
import pytest
from pathlib import Path
import laspy


@pytest.fixture
def synthetic_cliff_las(tmp_path):
    """Create a synthetic cliff point cloud with normals.

    Creates a cliff-like structure with:
    - Bottom: talus (low slope, smooth)
    - Middle: vertical cliff face
    - Top: flat area with some roughness
    """
    n_per_zone = 500

    # Zone 1: Talus at bottom (gentle slope ~25 degrees)
    x1 = np.random.uniform(0, 10, n_per_zone)
    y1 = np.random.uniform(0, 2, n_per_zone)
    z1 = y1 * 0.5 + np.random.normal(0, 0.02, n_per_zone)
    # Normals pointing mostly up with slight tilt
    normals1 = np.zeros((n_per_zone, 3), dtype=np.float32)
    normals1[:, 1] = -0.4  # Tilted toward viewer
    normals1[:, 2] = 0.9
    normals1 /= np.linalg.norm(normals1, axis=1, keepdims=True)

    # Zone 2: Cliff face (vertical ~85-90 degrees)
    x2 = np.random.uniform(0, 10, n_per_zone)
    y2 = np.full(n_per_zone, 2.0) + np.random.normal(0, 0.05, n_per_zone)
    z2 = np.random.uniform(1, 8, n_per_zone)
    # Normals pointing outward (horizontal)
    normals2 = np.zeros((n_per_zone, 3), dtype=np.float32)
    normals2[:, 1] = -1.0  # Pointing toward viewer
    normals2 += np.random.normal(0, 0.05, (n_per_zone, 3)).astype(np.float32)
    normals2 /= np.linalg.norm(normals2, axis=1, keepdims=True)

    # Zone 3: Top plateau (horizontal)
    x3 = np.random.uniform(0, 10, n_per_zone)
    y3 = np.random.uniform(2.5, 5, n_per_zone)
    z3 = 8.0 + np.random.normal(0, 0.1, n_per_zone)
    # Normals pointing up
    normals3 = np.zeros((n_per_zone, 3), dtype=np.float32)
    normals3[:, 2] = 1.0
    normals3 += np.random.normal(0, 0.03, (n_per_zone, 3)).astype(np.float32)
    normals3 /= np.linalg.norm(normals3, axis=1, keepdims=True)

    # Zone 4: Overhang section (slope > 90 degrees)
    n_overhang = 200
    x4 = np.random.uniform(3, 7, n_overhang)
    y4 = np.full(n_overhang, 1.8) + np.random.normal(0, 0.02, n_overhang)
    z4 = np.random.uniform(5, 7, n_overhang)
    # Normals pointing down and outward (overhang)
    normals4 = np.zeros((n_overhang, 3), dtype=np.float32)
    normals4[:, 1] = -0.7
    normals4[:, 2] = -0.7  # Pointing downward
    normals4 /= np.linalg.norm(normals4, axis=1, keepdims=True)

    # Combine all zones
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    z = np.concatenate([z1, z2, z3, z4])
    normals = np.vstack([normals1, normals2, normals3, normals4])

    n_total = len(x)

    # Create LAS file
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = x.astype(np.float64)
    las.y = y.astype(np.float64)
    las.z = z.astype(np.float64)

    # Add normals as extra dimensions
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
    las["NormalX"] = normals[:, 0]
    las["NormalY"] = normals[:, 1]
    las["NormalZ"] = normals[:, 2]

    filepath = tmp_path / "synthetic_cliff.las"
    las.write(filepath)
    return filepath


@pytest.fixture
def simple_las_with_normals(tmp_path):
    """Create a simple horizontal surface for quick tests."""
    n = 200
    x = np.repeat(np.linspace(0, 5, 20), 10).astype(np.float64)
    y = np.tile(np.linspace(0, 5, 10), 20).astype(np.float64)
    z = np.zeros(n, dtype=np.float64) + np.random.normal(0, 0.02, n)

    # Normals pointing up
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0

    las = laspy.create(point_format=0, file_version="1.4")
    las.x = x
    las.y = y
    las.z = z
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
    las["NormalX"] = normals[:, 0]
    las["NormalY"] = normals[:, 1]
    las["NormalZ"] = normals[:, 2]

    filepath = tmp_path / "simple_surface.las"
    las.write(filepath)
    return filepath


class TestFullPipeline:
    """Integration tests for the full processing pipeline."""

    def test_full_pipeline_synthetic_cliff(self, synthetic_cliff_las, tmp_path):
        """Test complete pipeline on synthetic cliff data."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        output_dir = tmp_path / "output"

        config = RAIConfig(
            methods=["radius", "knn"],
            compress_output=False,
            visualization_dpi=72,
            visualization_views=["front"],
        )

        classifier = RAIClassifier(config)
        result = classifier.process_file(
            synthetic_cliff_las,
            output_dir,
            compute_normals=False,  # Use pre-computed normals
            generate_visualizations=True,
            generate_report=True,
        )

        # Check outputs exist
        assert (output_dir / "synthetic_cliff_rai.las").exists()
        assert (output_dir / "synthetic_cliff_classification_radius_front.png").exists()
        assert (output_dir / "synthetic_cliff_report.md").exists()
        assert (output_dir / "synthetic_cliff_report.json").exists()

        # Check result validity
        assert result.n_points == 1700  # 500*3 + 200
        assert not np.all(np.isnan(result.slope_deg))

        # Check we got multiple classes (should have talus, cliff, overhang)
        unique_classes_radius = np.unique(result.rai_class_radius)
        unique_classes_knn = np.unique(result.rai_class_knn)
        assert len(unique_classes_radius) >= 2, f"Only got classes: {unique_classes_radius}"
        assert len(unique_classes_knn) >= 2, f"Only got classes: {unique_classes_knn}"

        # Check statistics are populated
        assert "classification_radius" in result.statistics
        assert "classification_knn" in result.statistics
        assert "method_agreement" in result.statistics

    def test_pipeline_radius_only(self, simple_las_with_normals, tmp_path):
        """Test pipeline with radius method only."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        output_dir = tmp_path / "output"

        config = RAIConfig(
            methods=["radius"],
            compress_output=False,
        )

        classifier = RAIClassifier(config)
        result = classifier.process_file(
            simple_las_with_normals,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=False,
        )

        assert result.rai_class_radius is not None
        assert result.rai_class_knn is None
        assert result.roughness_small_radius is not None
        assert result.roughness_small_knn is None

    def test_pipeline_knn_only(self, simple_las_with_normals, tmp_path):
        """Test pipeline with k-NN method only."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        output_dir = tmp_path / "output"

        config = RAIConfig(
            methods=["knn"],
            compress_output=False,
        )

        classifier = RAIClassifier(config)
        result = classifier.process_file(
            simple_las_with_normals,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=False,
        )

        assert result.rai_class_radius is None
        assert result.rai_class_knn is not None
        assert result.roughness_small_radius is None
        assert result.roughness_small_knn is not None

    def test_output_las_has_all_attributes(self, simple_las_with_normals, tmp_path):
        """Test that output LAS file contains all expected attributes."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        output_dir = tmp_path / "output"

        config = RAIConfig(
            methods=["radius", "knn"],
            compress_output=False,
        )

        classifier = RAIClassifier(config)
        classifier.process_file(
            simple_las_with_normals,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=False,
        )

        # Load output and check attributes
        output_las = laspy.read(output_dir / "simple_surface_rai.las")
        dim_names = [dim.name for dim in output_las.point_format.extra_dimensions]

        expected_dims = [
            "slope_deg",
            "roughness_small_radius",
            "roughness_large_radius",
            "roughness_small_knn",
            "roughness_large_knn",
            "rai_class_radius",
            "rai_class_knn",
        ]

        for dim in expected_dims:
            assert dim in dim_names, f"Missing dimension: {dim}"


class TestCLIIntegration:
    """Integration tests for CLI end-to-end."""

    def test_cli_process_basic(self, synthetic_cliff_las, tmp_path):
        """Test CLI process command."""
        from pc_rai.cli import main

        output_dir = tmp_path / "cli_output"

        result = main([
            "process",
            str(synthetic_cliff_las),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "both",
            "--no-visualize",
            "--no-report",
        ])

        assert result == 0
        assert output_dir.exists()

        # Check output file exists (either .las or .laz)
        output_files = list(output_dir.glob("*_rai.las")) + list(output_dir.glob("*_rai.laz"))
        assert len(output_files) == 1

    def test_cli_process_with_all_outputs(self, synthetic_cliff_las, tmp_path):
        """Test CLI generates all outputs when enabled."""
        from pc_rai.cli import main

        output_dir = tmp_path / "cli_output"

        result = main([
            "process",
            str(synthetic_cliff_las),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "radius",
        ])

        assert result == 0

        # Check for visualizations
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) > 0, "No PNG files generated"

        # Check for reports
        assert (output_dir / "synthetic_cliff_report.md").exists()
        assert (output_dir / "synthetic_cliff_report.json").exists()

    def test_cli_verbose_output(self, simple_las_with_normals, tmp_path, capsys):
        """Test CLI verbose mode provides output."""
        from pc_rai.cli import main

        output_dir = tmp_path / "cli_output"

        result = main([
            "process",
            str(simple_las_with_normals),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "radius",
            "--no-visualize",
            "--no-report",
            "-v",
        ])

        assert result == 0

        captured = capsys.readouterr()
        # Verbose mode should print summary info
        assert "Completed" in captured.out

    def test_cli_visualize_command(self, synthetic_cliff_las, tmp_path):
        """Test CLI visualize command on processed file."""
        from pc_rai.cli import main

        # First process the file
        process_dir = tmp_path / "processed"
        result = main([
            "process",
            str(synthetic_cliff_las),
            "-o", str(process_dir),
            "--skip-normals",
            "--methods", "radius",
            "--no-visualize",
            "--no-report",
        ])
        assert result == 0

        # Find the processed file
        processed_files = list(process_dir.glob("*_rai.las")) + list(process_dir.glob("*_rai.laz"))
        assert len(processed_files) == 1
        processed_file = processed_files[0]

        # Now run visualize command
        viz_dir = tmp_path / "visualizations"
        result = main([
            "visualize",
            str(processed_file),
            "-o", str(viz_dir),
            "--views", "front", "oblique",
            "--dpi", "72",
        ])

        assert result == 0
        assert viz_dir.exists()

        # Should have generated visualizations
        png_files = list(viz_dir.glob("*.png"))
        assert len(png_files) > 0


class TestBatchProcessing:
    """Integration tests for batch processing."""

    def test_batch_process_multiple_files(self, tmp_path):
        """Test batch processing of multiple files."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        # Create multiple input files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(3):
            n = 100
            x = np.random.uniform(0, 5, n).astype(np.float64)
            y = np.random.uniform(0, 5, n).astype(np.float64)
            z = np.random.uniform(0, 1, n).astype(np.float64)

            normals = np.zeros((n, 3), dtype=np.float32)
            normals[:, 2] = 1.0

            las = laspy.create(point_format=0, file_version="1.4")
            las.x = x
            las.y = y
            las.z = z
            las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
            las["NormalX"] = normals[:, 0]
            las["NormalY"] = normals[:, 1]
            las["NormalZ"] = normals[:, 2]
            las.write(input_dir / f"file_{i}.las")

        # Process batch
        output_dir = tmp_path / "output"
        config = RAIConfig(methods=["radius"], compress_output=False)
        classifier = RAIClassifier(config)

        input_files = list(input_dir.glob("*.las"))
        results = classifier.process_batch(
            input_files,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=False,
        )

        assert len(results) == 3
        assert all(r.n_points == 100 for r in results)

        # Check output files
        output_files = list(output_dir.glob("*_rai.las"))
        assert len(output_files) == 3


class TestReportContent:
    """Tests for report content validity."""

    def test_json_report_structure(self, simple_las_with_normals, tmp_path):
        """Test JSON report has expected structure."""
        import json
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        output_dir = tmp_path / "output"

        config = RAIConfig(methods=["radius", "knn"], compress_output=False)
        classifier = RAIClassifier(config)
        classifier.process_file(
            simple_las_with_normals,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=True,
        )

        with open(output_dir / "simple_surface_report.json") as f:
            report = json.load(f)

        # Check required sections
        assert "input" in report
        assert "config" in report
        assert "statistics" in report
        assert "timing" in report

        # Check input section
        assert report["input"]["n_points"] == 200

        # Check statistics section
        assert "features" in report["statistics"]
        assert "classification_radius" in report["statistics"]

    def test_markdown_report_content(self, simple_las_with_normals, tmp_path):
        """Test Markdown report has expected content."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        output_dir = tmp_path / "output"

        config = RAIConfig(methods=["radius"], compress_output=False)
        classifier = RAIClassifier(config)
        classifier.process_file(
            simple_las_with_normals,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=True,
        )

        report_path = output_dir / "simple_surface_report.md"
        content = report_path.read_text()

        # Check key sections exist
        assert "# RAI Classification Report" in content
        assert "## Input" in content
        assert "## Configuration" in content
        assert "## Classification Results" in content or "## Feature Statistics" in content
        assert "200" in content  # Point count
