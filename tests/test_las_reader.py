"""Tests for pc_rai.io.las_reader module."""

import numpy as np
import pytest
from pathlib import Path


def test_point_cloud_creation():
    """Test PointCloud dataclass creation."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    cloud = PointCloud(xyz=xyz)

    assert cloud.n_points == 3
    assert not cloud.has_normals
    assert cloud.source_file is None


def test_point_cloud_with_normals():
    """Test PointCloud with normal vectors."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

    cloud = PointCloud(xyz=xyz, normals=normals)

    assert cloud.n_points == 2
    assert cloud.has_normals
    assert cloud.normals.shape == (2, 3)


def test_point_cloud_bounds():
    """Test PointCloud.bounds property."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float64)
    cloud = PointCloud(xyz=xyz)

    bounds = cloud.bounds
    assert bounds["x"] == (0, 10)
    assert bounds["y"] == (0, 20)
    assert bounds["z"] == (0, 30)


def test_point_cloud_centroid():
    """Test PointCloud.centroid property."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64)
    cloud = PointCloud(xyz=xyz)

    centroid = cloud.centroid
    assert np.allclose(centroid, [5, 5, 5])


def test_point_cloud_extent():
    """Test PointCloud.extent property."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float64)
    cloud = PointCloud(xyz=xyz)

    extent = cloud.extent
    assert extent["x"] == 10
    assert extent["y"] == 20
    assert extent["z"] == 30


def test_point_cloud_invalid_xyz_shape():
    """Test PointCloud rejects invalid xyz shape."""
    from pc_rai.io.las_reader import PointCloud

    # Wrong number of columns
    with pytest.raises(ValueError, match="xyz must have shape"):
        PointCloud(xyz=np.array([[0, 0], [1, 1]]))

    # 1D array
    with pytest.raises(ValueError, match="xyz must have shape"):
        PointCloud(xyz=np.array([0, 0, 0]))


def test_point_cloud_invalid_normals_shape():
    """Test PointCloud rejects invalid normals shape."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)

    # Wrong number of columns
    with pytest.raises(ValueError, match="normals must have shape"):
        PointCloud(xyz=xyz, normals=np.array([[0, 0], [0, 0]]))


def test_point_cloud_mismatched_lengths():
    """Test PointCloud rejects mismatched xyz and normals lengths."""
    from pc_rai.io.las_reader import PointCloud

    xyz = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    normals = np.array([[0, 0, 1]], dtype=np.float32)  # Only 1 normal

    with pytest.raises(ValueError, match="normals length"):
        PointCloud(xyz=xyz, normals=normals)


def test_load_synthetic_las(tmp_path):
    """Create a minimal LAS file and load it."""
    import laspy
    from pc_rai.io.las_reader import load_point_cloud

    # Create synthetic data
    n_points = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    # Save
    filepath = tmp_path / "test.las"
    las.write(filepath)

    # Load and verify
    cloud = load_point_cloud(filepath)

    assert cloud.n_points == n_points
    assert cloud.xyz.shape == (n_points, 3)
    assert not cloud.has_normals
    assert cloud.source_file == filepath


def test_load_las_with_normals(tmp_path):
    """Test loading LAS file with normal vectors."""
    import laspy
    from pc_rai.io.las_reader import load_point_cloud

    n_points = 50
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    # Add normals as extra dimensions (CloudCompare format)
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))

    las["NormalX"] = np.zeros(n_points, dtype=np.float32)
    las["NormalY"] = np.zeros(n_points, dtype=np.float32)
    las["NormalZ"] = np.ones(n_points, dtype=np.float32)

    filepath = tmp_path / "test_normals.las"
    las.write(filepath)

    # Load and verify
    cloud = load_point_cloud(filepath)

    assert cloud.n_points == n_points
    assert cloud.has_normals
    assert cloud.normals.shape == (n_points, 3)
    assert np.allclose(cloud.normals[:, 2], 1.0)


def test_load_nonexistent_file():
    """Test loading non-existent file raises FileNotFoundError."""
    from pc_rai.io.las_reader import load_point_cloud

    with pytest.raises(FileNotFoundError):
        load_point_cloud(Path("/nonexistent/file.las"))


def test_has_valid_normals_true(tmp_path):
    """Test has_valid_normals returns True when normals present."""
    import laspy
    from pc_rai.io.las_reader import has_valid_normals

    las = laspy.create(point_format=0, file_version="1.4")
    las.x = [0, 1]
    las.y = [0, 1]
    las.z = [0, 1]

    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))

    assert has_valid_normals(las)


def test_has_valid_normals_false(tmp_path):
    """Test has_valid_normals returns False when normals absent."""
    import laspy
    from pc_rai.io.las_reader import has_valid_normals

    las = laspy.create(point_format=0, file_version="1.4")
    las.x = [0, 1]
    las.y = [0, 1]
    las.z = [0, 1]

    assert not has_valid_normals(las)


def test_get_las_info(tmp_path):
    """Test get_las_info returns correct metadata."""
    import laspy
    from pc_rai.io.las_reader import get_las_info

    n_points = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    filepath = tmp_path / "test_info.las"
    las.write(filepath)

    info = get_las_info(filepath)

    assert info["point_count"] == n_points
    assert info["version"] == "1.4"
    assert "bounds" in info
    assert "x" in info["bounds"]


def test_load_preserves_las_data(tmp_path):
    """Test that loading preserves original LAS data for round-trip."""
    import laspy
    from pc_rai.io.las_reader import load_point_cloud

    n_points = 50
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    filepath = tmp_path / "test_preserve.las"
    las.write(filepath)

    cloud = load_point_cloud(filepath)

    # Should have preserved LAS data
    assert cloud._las_data is not None
    assert len(cloud._las_data.x) == n_points
