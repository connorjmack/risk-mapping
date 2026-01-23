"""Tests for RAI Classifier module."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def synthetic_cloud_with_normals():
    """Create synthetic point cloud with normals for testing."""
    from pc_rai.io.las_reader import PointCloud

    n = 500
    # Create a simple grid
    x = np.repeat(np.linspace(0, 10, 50), 10)
    y = np.tile(np.linspace(0, 10, 10), 50)
    z = np.zeros(n) + np.random.normal(0, 0.1, n)

    xyz = np.column_stack([x, y, z]).astype(np.float64)

    # Mostly upward-pointing normals (horizontal surface)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    # Add some noise
    normals += np.random.normal(0, 0.05, (n, 3)).astype(np.float32)
    # Renormalize
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return PointCloud(xyz=xyz, normals=normals, source_file=Path("synthetic.las"))


@pytest.fixture
def varied_cloud_with_normals():
    """Create point cloud with varied slopes for classification testing."""
    from pc_rai.io.las_reader import PointCloud

    # Create dense grid for each zone (need sufficient neighbors)
    n_per_zone = 200

    # Zone 1: Horizontal (talus-like) - low slope, smooth
    x1 = np.repeat(np.linspace(0, 2, 20), 10)
    y1 = np.tile(np.linspace(0, 2, 10), 20)
    z1 = np.zeros(n_per_zone) + np.random.normal(0, 0.01, n_per_zone)
    normals1 = np.zeros((n_per_zone, 3), dtype=np.float32)
    normals1[:, 2] = 1.0  # Pointing up

    # Zone 2: Vertical (cliff-like) - 90 degree slope
    x2 = np.repeat(np.linspace(3, 5, 20), 10)
    y2 = np.tile(np.linspace(0, 2, 10), 20)
    z2 = np.linspace(0, 5, n_per_zone)
    normals2 = np.zeros((n_per_zone, 3), dtype=np.float32)
    normals2[:, 0] = 1.0  # Pointing sideways

    # Zone 3: Overhang - >90 degree slope
    x3 = np.repeat(np.linspace(6, 8, 20), 10)
    y3 = np.tile(np.linspace(0, 2, 10), 20)
    z3 = np.linspace(0, 5, n_per_zone)
    normals3 = np.zeros((n_per_zone, 3), dtype=np.float32)
    normals3[:, 2] = -0.5
    normals3[:, 0] = 0.866  # ~120 degree slope

    # Combine
    xyz = np.vstack([
        np.column_stack([x1, y1, z1]),
        np.column_stack([x2, y2, z2]),
        np.column_stack([x3, y3, z3]),
    ]).astype(np.float64)

    normals = np.vstack([normals1, normals2, normals3])
    # Normalize
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return PointCloud(xyz=xyz, normals=normals, source_file=Path("varied.las"))


class TestRAIResult:
    """Tests for RAIResult dataclass."""

    def test_result_creation(self):
        """Test RAIResult can be created."""
        from pc_rai.classifier import RAIResult

        slope = np.array([45.0, 60.0, 90.0])
        result = RAIResult(
            source_file="test.las",
            n_points=3,
            slope_deg=slope,
        )

        assert result.n_points == 3
        assert result.source_file == "test.las"
        assert len(result.slope_deg) == 3

    def test_result_optional_fields(self):
        """Test optional fields default to None."""
        from pc_rai.classifier import RAIResult

        result = RAIResult(
            source_file="test.las",
            n_points=10,
            slope_deg=np.zeros(10),
        )

        assert result.rai_class_radius is None
        assert result.rai_class_knn is None
        assert result.roughness_small_radius is None


class TestRAIClassifier:
    """Tests for RAIClassifier class."""

    def test_classifier_init_default(self):
        """Test classifier initializes with default config."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        classifier = RAIClassifier()
        assert isinstance(classifier.config, RAIConfig)

    def test_classifier_init_custom_config(self):
        """Test classifier initializes with custom config."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        config = RAIConfig(radius_small=0.2, k_small=50)
        classifier = RAIClassifier(config)

        assert classifier.config.radius_small == 0.2
        assert classifier.config.k_small == 50

    def test_process_synthetic(self, synthetic_cloud_with_normals):
        """Test processing synthetic point cloud."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        config = RAIConfig(methods=["radius", "knn"])
        classifier = RAIClassifier(config)

        result = classifier.process(synthetic_cloud_with_normals, compute_normals=False)

        assert result.n_points == 500
        assert len(result.slope_deg) == 500
        assert result.rai_class_radius is not None
        assert result.rai_class_knn is not None
        assert "slope" in result.timing
        assert "total" in result.timing

    def test_process_radius_only(self, synthetic_cloud_with_normals):
        """Test processing with radius method only."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        config = RAIConfig(methods=["radius"])
        classifier = RAIClassifier(config)

        result = classifier.process(synthetic_cloud_with_normals, compute_normals=False)

        assert result.rai_class_radius is not None
        assert result.rai_class_knn is None
        assert result.roughness_small_radius is not None
        assert result.roughness_small_knn is None

    def test_process_knn_only(self, synthetic_cloud_with_normals):
        """Test processing with k-NN method only."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        config = RAIConfig(methods=["knn"])
        classifier = RAIClassifier(config)

        result = classifier.process(synthetic_cloud_with_normals, compute_normals=False)

        assert result.rai_class_radius is None
        assert result.rai_class_knn is not None
        assert result.roughness_small_radius is None
        assert result.roughness_small_knn is not None

    def test_process_varied_cloud(self, varied_cloud_with_normals):
        """Test processing cloud with varied slopes produces multiple classes."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        # Use k-NN method which always has sufficient neighbors
        config = RAIConfig(methods=["knn"], k_small=20, k_large=50)
        classifier = RAIClassifier(config)

        result = classifier.process(varied_cloud_with_normals, compute_normals=False)

        # Should have multiple classes (including overhangs)
        unique_classes = np.unique(result.rai_class_knn)
        # We expect at least: Talus/Intact (horizontal), some cliff class, Overhang
        assert len(unique_classes) >= 2

    def test_process_statistics(self, synthetic_cloud_with_normals):
        """Test statistics are computed."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        config = RAIConfig(methods=["radius", "knn"])
        classifier = RAIClassifier(config)

        result = classifier.process(synthetic_cloud_with_normals, compute_normals=False)

        assert "n_points" in result.statistics
        assert "features" in result.statistics
        assert "classification_radius" in result.statistics
        assert "classification_knn" in result.statistics
        assert "method_agreement" in result.statistics

    def test_process_no_normals_error(self):
        """Test error when cloud has no normals."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.io.las_reader import PointCloud

        xyz = np.random.uniform(0, 10, (100, 3))
        cloud = PointCloud(xyz=xyz)  # No normals

        classifier = RAIClassifier()

        with pytest.raises(ValueError, match="no normals"):
            classifier.process(cloud, compute_normals=False)


class TestRAIClassifierFile:
    """Tests for file-based processing."""

    def test_process_file_synthetic(self, tmp_path, synthetic_cloud_with_normals):
        """Test processing file end-to-end."""
        import laspy
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        # Create input LAS file
        input_path = tmp_path / "input.las"
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = synthetic_cloud_with_normals.xyz[:, 0]
        las.y = synthetic_cloud_with_normals.xyz[:, 1]
        las.z = synthetic_cloud_with_normals.xyz[:, 2]

        # Add normals as extra dimensions
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
        las["NormalX"] = synthetic_cloud_with_normals.normals[:, 0]
        las["NormalY"] = synthetic_cloud_with_normals.normals[:, 1]
        las["NormalZ"] = synthetic_cloud_with_normals.normals[:, 2]
        las.write(input_path)

        # Process
        output_dir = tmp_path / "output"
        config = RAIConfig(methods=["radius"], compress_output=False)
        classifier = RAIClassifier(config)

        result = classifier.process_file(
            input_path,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=False,
        )

        # Check result
        assert result.n_points == 500
        assert result.rai_class_radius is not None

        # Check output file
        assert (output_dir / "input_rai.las").exists()

    def test_process_file_with_visualizations(self, tmp_path, synthetic_cloud_with_normals):
        """Test file processing generates visualizations."""
        import laspy
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        # Create input LAS file
        input_path = tmp_path / "input.las"
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = synthetic_cloud_with_normals.xyz[:, 0]
        las.y = synthetic_cloud_with_normals.xyz[:, 1]
        las.z = synthetic_cloud_with_normals.xyz[:, 2]
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
        las["NormalX"] = synthetic_cloud_with_normals.normals[:, 0]
        las["NormalY"] = synthetic_cloud_with_normals.normals[:, 1]
        las["NormalZ"] = synthetic_cloud_with_normals.normals[:, 2]
        las.write(input_path)

        # Process with visualizations
        output_dir = tmp_path / "output"
        config = RAIConfig(
            methods=["radius"],
            compress_output=False,
            visualization_dpi=72,  # Low res for fast tests
            visualization_views=["front"],
        )
        classifier = RAIClassifier(config)

        result = classifier.process_file(
            input_path,
            output_dir,
            compute_normals=False,
            generate_visualizations=True,
            generate_report=False,
        )

        # Check visualization files
        assert (output_dir / "input_classification_radius_front.png").exists()
        assert (output_dir / "input_slope.png").exists()
        assert (output_dir / "input_histogram.png").exists()

    def test_process_file_with_reports(self, tmp_path, synthetic_cloud_with_normals):
        """Test file processing generates reports."""
        import laspy
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        # Create input LAS file
        input_path = tmp_path / "input.las"
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = synthetic_cloud_with_normals.xyz[:, 0]
        las.y = synthetic_cloud_with_normals.xyz[:, 1]
        las.z = synthetic_cloud_with_normals.xyz[:, 2]
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
        las["NormalX"] = synthetic_cloud_with_normals.normals[:, 0]
        las["NormalY"] = synthetic_cloud_with_normals.normals[:, 1]
        las["NormalZ"] = synthetic_cloud_with_normals.normals[:, 2]
        las.write(input_path)

        # Process with reports
        output_dir = tmp_path / "output"
        config = RAIConfig(methods=["radius"], compress_output=False)
        classifier = RAIClassifier(config)

        result = classifier.process_file(
            input_path,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=True,
        )

        # Check report files
        assert (output_dir / "input_report.md").exists()
        assert (output_dir / "input_report.json").exists()

        # Verify report content
        import json
        with open(output_dir / "input_report.json") as f:
            report = json.load(f)
        assert report["input"]["n_points"] == 500


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_process_batch(self, tmp_path, synthetic_cloud_with_normals):
        """Test batch processing multiple files."""
        import laspy
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        # Create multiple input files
        input_files = []
        for i in range(3):
            input_path = tmp_path / f"input_{i}.las"
            las = laspy.create(point_format=0, file_version="1.4")
            las.x = synthetic_cloud_with_normals.xyz[:100, 0]  # Smaller for speed
            las.y = synthetic_cloud_with_normals.xyz[:100, 1]
            las.z = synthetic_cloud_with_normals.xyz[:100, 2]
            las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
            las["NormalX"] = synthetic_cloud_with_normals.normals[:100, 0]
            las["NormalY"] = synthetic_cloud_with_normals.normals[:100, 1]
            las["NormalZ"] = synthetic_cloud_with_normals.normals[:100, 2]
            las.write(input_path)
            input_files.append(input_path)

        # Process batch
        output_dir = tmp_path / "output"
        config = RAIConfig(methods=["radius"], compress_output=False)
        classifier = RAIClassifier(config)

        results = classifier.process_batch(
            input_files,
            output_dir,
            compute_normals=False,
            generate_visualizations=False,
            generate_report=False,
        )

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert all(r.n_points == 100 for r in results)
