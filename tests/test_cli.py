"""Tests for pc_rai.cli module."""

import pytest
from pathlib import Path


def test_parser_process():
    """Test parsing process command arguments."""
    from pc_rai.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["process", "input.las", "-o", "output/"])

    assert args.command == "process"
    assert args.input == Path("input.las")
    assert args.output == Path("output/")
    assert args.methods == "knn"  # default (knn only for performance)
    assert not args.batch
    assert not args.skip_normals


def test_parser_process_all_options():
    """Test parsing process command with all options."""
    from pc_rai.cli import create_parser

    parser = create_parser()
    args = parser.parse_args([
        "process",
        "input.las",
        "-o", "output/",
        "-c", "config.yaml",
        "--batch",
        "--skip-normals",
        "--methods", "knn",
        "--no-visualize",
        "--no-report",
        "-v",
    ])

    assert args.command == "process"
    assert args.config == Path("config.yaml")
    assert args.batch
    assert args.skip_normals
    assert args.methods == "knn"
    assert args.no_visualize
    assert args.no_report
    assert args.verbose


def test_parser_visualize():
    """Test parsing visualize command arguments."""
    from pc_rai.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["visualize", "processed.las", "-o", "viz_output/"])

    assert args.command == "visualize"
    assert args.input == Path("processed.las")
    assert args.output == Path("viz_output/")


def test_parser_help():
    """Test that --help exits cleanly."""
    from pc_rai.cli import create_parser

    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_main_no_command(capsys):
    """Test main with no command shows help and returns 1."""
    from pc_rai.cli import main

    result = main([])
    assert result == 1

    # Verify help was printed
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "Commands" in captured.out


def test_main_version(capsys):
    """Test --version flag."""
    from pc_rai.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_run_process_missing_input(capsys):
    """Test process command fails with missing input file."""
    from pc_rai.cli import main

    result = main(["process", "nonexistent.las", "-o", "output/"])
    assert result == 1

    captured = capsys.readouterr()
    assert "not found" in captured.err.lower() or "error" in captured.err.lower()


def test_run_process_missing_config(tmp_path, capsys):
    """Test process command fails with missing config file."""
    from pc_rai.cli import main

    # Create a dummy input file
    input_file = tmp_path / "test.las"
    input_file.touch()

    result = main([
        "process",
        str(input_file),
        "-o", str(tmp_path / "output"),
        "-c", "nonexistent_config.yaml"
    ])
    assert result == 1

    captured = capsys.readouterr()
    assert "config" in captured.err.lower() or "not found" in captured.err.lower()


def test_run_visualize_missing_input(capsys):
    """Test visualize command fails with missing input file."""
    from pc_rai.cli import main

    result = main(["visualize", "nonexistent.las", "-o", "output/"])
    assert result == 1

    captured = capsys.readouterr()
    assert "not found" in captured.err.lower() or "error" in captured.err.lower()


def test_run_process_batch_requires_directory(tmp_path, capsys):
    """Test --batch flag requires a directory input."""
    from pc_rai.cli import main

    # Create a file instead of directory
    input_file = tmp_path / "test.las"
    input_file.touch()

    result = main([
        "process",
        str(input_file),
        "-o", str(tmp_path / "output"),
        "--batch"
    ])
    assert result == 1

    captured = capsys.readouterr()
    assert "directory" in captured.err.lower()


def test_run_process_batch_empty_directory(tmp_path, capsys):
    """Test batch processing fails with empty directory."""
    from pc_rai.cli import main

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    result = main([
        "process",
        str(input_dir),
        "-o", str(tmp_path / "output"),
        "--batch"
    ])
    assert result == 1

    captured = capsys.readouterr()
    assert "no las" in captured.err.lower() or "not found" in captured.err.lower()


class TestCLIIntegration:
    """Integration tests for CLI with real files."""

    @pytest.fixture
    def synthetic_las_file(self, tmp_path):
        """Create a synthetic LAS file with normals for testing."""
        import laspy
        import numpy as np

        n = 100
        # Create a simple grid
        x = np.repeat(np.linspace(0, 5, 10), 10).astype(np.float64)
        y = np.tile(np.linspace(0, 5, 10), 10).astype(np.float64)
        z = np.zeros(n, dtype=np.float64) + np.random.normal(0, 0.05, n)

        # Create LAS file
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = x
        las.y = y
        las.z = z

        # Add normals
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))

        # Normals pointing up
        las["NormalX"] = np.zeros(n, dtype=np.float32)
        las["NormalY"] = np.zeros(n, dtype=np.float32)
        las["NormalZ"] = np.ones(n, dtype=np.float32)

        input_path = tmp_path / "test_input.las"
        las.write(input_path)

        return input_path

    def test_process_single_file(self, tmp_path, synthetic_las_file):
        """Test processing a single file through CLI."""
        from pc_rai.cli import main

        output_dir = tmp_path / "output"

        result = main([
            "process",
            str(synthetic_las_file),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "radius",
            "--no-visualize",
            "--no-report",
        ])

        assert result == 0
        assert output_dir.exists()
        # Check for either .las or .laz output (now in rai subdirectory)
        rai_dir = output_dir / "rai"
        output_files = list(rai_dir.glob("*.las")) + list(rai_dir.glob("*.laz"))
        assert len(output_files) == 1

    def test_process_with_visualizations(self, tmp_path, synthetic_las_file):
        """Test processing generates visualizations."""
        from pc_rai.cli import main

        output_dir = tmp_path / "output"

        result = main([
            "process",
            str(synthetic_las_file),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "radius",
            "--no-report",
        ])

        assert result == 0
        # Check for visualization files (now in figures/<date> subdirectory)
        from datetime import date
        figures_dir = output_dir / "figures" / date.today().isoformat()
        png_files = list(figures_dir.glob("*.png"))
        assert len(png_files) > 0

    def test_process_with_report(self, tmp_path, synthetic_las_file):
        """Test processing generates reports."""
        from pc_rai.cli import main

        output_dir = tmp_path / "output"

        result = main([
            "process",
            str(synthetic_las_file),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "radius",
            "--no-visualize",
        ])

        assert result == 0
        # Check for report files (now in reports/<date> subdirectory)
        from datetime import date
        reports_dir = output_dir / "reports" / date.today().isoformat()
        assert (reports_dir / "test_input_report.md").exists()
        assert (reports_dir / "test_input_report.json").exists()

    def test_visualize_processed_file(self, tmp_path, synthetic_las_file):
        """Test visualize command on processed file."""
        from pc_rai.cli import main

        # First, process the file
        output_dir = tmp_path / "processed"
        result = main([
            "process",
            str(synthetic_las_file),
            "-o", str(output_dir),
            "--skip-normals",
            "--methods", "radius",
            "--no-visualize",
            "--no-report",
        ])
        assert result == 0

        # Find processed file (may be .las or .laz, now in rai subdirectory)
        rai_dir = output_dir / "rai"
        processed_files = list(rai_dir.glob("*_rai.las")) + list(rai_dir.glob("*_rai.laz"))
        assert len(processed_files) == 1
        processed_file = processed_files[0]

        # Now visualize it
        viz_dir = tmp_path / "visualizations"
        result = main([
            "visualize",
            str(processed_file),
            "-o", str(viz_dir),
            "--views", "front",
            "--dpi", "72",
        ])

        assert result == 0
        assert viz_dir.exists()
        # Visualizations go into figures/<date> subdirectory
        from datetime import date
        figures_subdir = viz_dir / "figures" / date.today().isoformat()
        png_files = list(figures_subdir.glob("*.png"))
        assert len(png_files) > 0
