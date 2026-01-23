"""Tests for pc_rai.normals.cloudcompare module."""

import numpy as np
import pytest
from pathlib import Path


def test_find_cloudcompare():
    """Test CloudCompare detection."""
    from pc_rai.normals.cloudcompare import find_cloudcompare

    cc_path = find_cloudcompare()
    # May or may not be found depending on system
    if cc_path is not None:
        assert isinstance(cc_path, Path)


def test_is_cloudcompare_available():
    """Test CloudCompare availability check."""
    from pc_rai.normals.cloudcompare import is_cloudcompare_available

    result = is_cloudcompare_available()
    assert isinstance(result, bool)


def test_cloudcompare_not_found_error(tmp_path):
    """Test CloudCompareNotFoundError is raised correctly."""
    import laspy
    from pc_rai.normals.cloudcompare import (
        CloudCompareNotFoundError,
        compute_normals_cloudcompare,
    )

    # Create a valid input file
    n_points = 10
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    input_path = tmp_path / "input.las"
    las.write(input_path)

    with pytest.raises(CloudCompareNotFoundError):
        compute_normals_cloudcompare(
            input_path,
            tmp_path / "output.las",
            cloudcompare_path="/nonexistent/path/CloudCompare",
        )


def test_input_file_not_found():
    """Test FileNotFoundError for missing input."""
    from pc_rai.normals.cloudcompare import (
        find_cloudcompare,
        compute_normals_cloudcompare,
    )

    cc_path = find_cloudcompare()
    if cc_path is None:
        pytest.skip("CloudCompare not found on system")

    with pytest.raises(FileNotFoundError):
        compute_normals_cloudcompare(
            Path("/nonexistent/input.las"),
            Path("/tmp/output.las"),
        )


def test_extract_normals_from_las(tmp_path):
    """Test extracting normals from LAS file."""
    import laspy
    from pc_rai.normals.cloudcompare import extract_normals_from_las

    # Create LAS file with normals
    n_points = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    # Add normals (CloudCompare format)
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))

    las["NormalX"] = np.zeros(n_points, dtype=np.float32)
    las["NormalY"] = np.zeros(n_points, dtype=np.float32)
    las["NormalZ"] = np.ones(n_points, dtype=np.float32)

    filepath = tmp_path / "test_normals.las"
    las.write(filepath)

    # Extract normals
    normals = extract_normals_from_las(filepath)

    assert normals is not None
    assert normals.shape == (n_points, 3)
    assert np.allclose(normals[:, 2], 1.0)


def test_extract_normals_no_normals(tmp_path):
    """Test extracting normals from file without normals returns None."""
    import laspy
    from pc_rai.normals.cloudcompare import extract_normals_from_las

    # Create LAS file without normals
    n_points = 50
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    filepath = tmp_path / "test_no_normals.las"
    las.write(filepath)

    normals = extract_normals_from_las(filepath)
    assert normals is None


def test_extract_normals_nonexistent_file():
    """Test extracting normals from nonexistent file returns None."""
    from pc_rai.normals.cloudcompare import extract_normals_from_las

    normals = extract_normals_from_las(Path("/nonexistent/file.las"))
    assert normals is None


def test_extract_normals_lowercase_names(tmp_path):
    """Test extracting normals with lowercase naming convention."""
    import laspy
    from pc_rai.normals.cloudcompare import extract_normals_from_las

    n_points = 50
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)

    # Lowercase naming
    las.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32))

    las["nx"] = np.ones(n_points, dtype=np.float32)
    las["ny"] = np.zeros(n_points, dtype=np.float32)
    las["nz"] = np.zeros(n_points, dtype=np.float32)

    filepath = tmp_path / "test_lowercase.las"
    las.write(filepath)

    normals = extract_normals_from_las(filepath)

    assert normals is not None
    assert np.allclose(normals[:, 0], 1.0)


# Integration tests that require CloudCompare
@pytest.mark.requires_cloudcompare
def test_compute_normals_integration(tmp_path):
    """Full integration test with CloudCompare."""
    import laspy
    from pc_rai.normals.cloudcompare import (
        find_cloudcompare,
        compute_normals_cloudcompare,
        extract_normals_from_las,
    )

    if find_cloudcompare() is None:
        pytest.skip("CloudCompare not installed")

    # Create simple test file (flat surface)
    n_points = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 0.1, n_points)  # Mostly flat

    input_path = tmp_path / "input.las"
    output_path = tmp_path / "output.las"
    las.write(input_path)

    # Compute normals
    success = compute_normals_cloudcompare(input_path, output_path)
    assert success
    assert output_path.exists()

    # Verify normals were computed
    normals = extract_normals_from_las(output_path)
    assert normals is not None
    assert normals.shape == (n_points, 3)

    # For a flat surface, normals should point mostly up (Z component ~ 1)
    assert np.mean(np.abs(normals[:, 2])) > 0.9


@pytest.mark.requires_cloudcompare
def test_compute_normals_for_cloud(tmp_path):
    """Test convenience function for computing normals."""
    import laspy
    from pc_rai.normals.cloudcompare import (
        find_cloudcompare,
        compute_normals_for_cloud,
    )

    if find_cloudcompare() is None:
        pytest.skip("CloudCompare not installed")

    # Create test file
    n_points = 50
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 5, n_points)
    las.y = np.random.uniform(0, 5, n_points)
    las.z = np.random.uniform(0, 0.1, n_points)

    input_path = tmp_path / "test_cloud.las"
    las.write(input_path)

    # Compute normals
    normals, output_path = compute_normals_for_cloud(
        input_path,
        radius=0.5,
        mst_neighbors=6,
        cleanup=False,
    )

    assert normals is not None
    assert normals.shape == (n_points, 3)
    assert output_path.exists()


@pytest.mark.requires_cloudcompare
def test_compute_normals_vertical_surface(tmp_path):
    """Test normal computation on vertical surface."""
    import laspy
    from pc_rai.normals.cloudcompare import (
        find_cloudcompare,
        compute_normals_cloudcompare,
        extract_normals_from_las,
    )

    if find_cloudcompare() is None:
        pytest.skip("CloudCompare not installed")

    # Create vertical surface (points in YZ plane)
    n_points = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 0.1, n_points)  # Thin in X
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 10, n_points)

    input_path = tmp_path / "vertical.las"
    output_path = tmp_path / "vertical_normals.las"
    las.write(input_path)

    success = compute_normals_cloudcompare(input_path, output_path)
    assert success

    normals = extract_normals_from_las(output_path)
    assert normals is not None

    # For vertical surface, normals should point in X direction
    assert np.mean(np.abs(normals[:, 0])) > 0.9


def test_cloudcompare_error_types():
    """Test that error classes are properly defined."""
    from pc_rai.normals.cloudcompare import (
        CloudCompareError,
        CloudCompareNotFoundError,
    )

    # CloudCompareNotFoundError should be a subclass of CloudCompareError
    assert issubclass(CloudCompareNotFoundError, CloudCompareError)

    # Test error instantiation
    err1 = CloudCompareError("test error")
    assert str(err1) == "test error"

    err2 = CloudCompareNotFoundError("not found")
    assert str(err2) == "not found"
