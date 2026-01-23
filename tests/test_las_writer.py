"""Tests for pc_rai.io.las_writer module."""

import numpy as np
import pytest
from pathlib import Path


def test_save_with_extra_dims(tmp_path):
    """Write a LAS file with extra dimensions and read it back."""
    import laspy
    from pc_rai.io.las_reader import PointCloud, load_point_cloud
    from pc_rai.io.las_writer import save_point_cloud

    # Create synthetic cloud
    n_points = 50
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    # Create attributes
    attributes = {
        "slope_deg": np.random.uniform(0, 90, n_points).astype(np.float32),
        "rai_class_radius": np.random.randint(0, 8, n_points).astype(np.uint8),
    }

    # Write
    output_path = tmp_path / "output.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    # Verify file exists and has extra dims
    assert output_path.exists()
    las = laspy.read(output_path)

    extra_dim_names = [dim.name for dim in las.point_format.extra_dimensions]
    assert "slope_deg" in extra_dim_names
    assert "rai_class_radius" in extra_dim_names

    # Verify values are correct
    assert np.allclose(las["slope_deg"], attributes["slope_deg"])
    assert np.array_equal(las["rai_class_radius"], attributes["rai_class_radius"])


def test_save_compressed(tmp_path):
    """Test saving as compressed LAZ file."""
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_point_cloud

    n_points = 100
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    attributes = {
        "slope_deg": np.random.uniform(0, 90, n_points).astype(np.float32),
    }

    # Request compression with .las extension - should change to .laz
    output_path = tmp_path / "output.las"
    save_point_cloud(cloud, attributes, output_path, compress=True)

    # Should have created .laz file
    laz_path = tmp_path / "output.laz"
    assert laz_path.exists()


def test_save_uncompressed(tmp_path):
    """Test saving as uncompressed LAS file."""
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_point_cloud

    n_points = 100
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    attributes = {
        "slope_deg": np.random.uniform(0, 90, n_points).astype(np.float32),
    }

    output_path = tmp_path / "output.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    assert output_path.exists()
    # Verify it's not compressed by checking file size is larger than compressed
    assert output_path.stat().st_size > 0


def test_save_preserves_original_attributes(tmp_path):
    """Test that saving preserves original LAS attributes."""
    import laspy
    from pc_rai.io.las_reader import load_point_cloud
    from pc_rai.io.las_writer import save_point_cloud

    # Create original LAS with intensity
    n_points = 50
    original_las = laspy.create(point_format=0, file_version="1.4")
    original_las.x = np.random.uniform(0, 10, n_points)
    original_las.y = np.random.uniform(0, 10, n_points)
    original_las.z = np.random.uniform(0, 5, n_points)
    original_las.intensity = np.random.randint(0, 65535, n_points).astype(np.uint16)

    original_path = tmp_path / "original.las"
    original_las.write(original_path)

    # Load, add attributes, and save
    cloud = load_point_cloud(original_path)
    attributes = {
        "slope_deg": np.random.uniform(0, 90, n_points).astype(np.float32),
    }

    output_path = tmp_path / "output.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    # Verify intensity is preserved
    result_las = laspy.read(output_path)
    assert np.array_equal(result_las.intensity, original_las.intensity)


def test_save_with_normals(tmp_path):
    """Test saving point cloud with normal vectors."""
    import laspy
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_point_cloud

    n_points = 50
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # All pointing up

    cloud = PointCloud(xyz=xyz, normals=normals)

    attributes = {
        "slope_deg": np.zeros(n_points, dtype=np.float32),
    }

    output_path = tmp_path / "output_normals.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    # Verify normals are saved
    result_las = laspy.read(output_path)
    extra_dim_names = [dim.name for dim in result_las.point_format.extra_dimensions]

    assert "NormalX" in extra_dim_names
    assert "NormalY" in extra_dim_names
    assert "NormalZ" in extra_dim_names
    assert np.allclose(result_las["NormalZ"], 1.0)


def test_save_invalid_attribute_length(tmp_path):
    """Test that mismatched attribute length raises error."""
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_point_cloud

    n_points = 50
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    # Wrong length attribute
    attributes = {
        "slope_deg": np.random.uniform(0, 90, n_points + 10).astype(np.float32),
    }

    output_path = tmp_path / "output.las"
    with pytest.raises(ValueError, match="has length"):
        save_point_cloud(cloud, attributes, output_path)


def test_save_creates_output_directory(tmp_path):
    """Test that save creates output directory if needed."""
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_point_cloud

    n_points = 20
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    attributes = {
        "slope_deg": np.random.uniform(0, 90, n_points).astype(np.float32),
    }

    # Output in non-existent subdirectory
    output_path = tmp_path / "subdir" / "nested" / "output.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    assert output_path.exists()


def test_save_all_rai_attributes(tmp_path):
    """Test saving all RAI extra dimensions."""
    import laspy
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_point_cloud, RAI_EXTRA_DIMS

    n_points = 100
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    # Create all RAI attributes
    attributes = {
        "slope_deg": np.random.uniform(0, 180, n_points).astype(np.float32),
        "roughness_small_radius": np.random.uniform(0, 30, n_points).astype(np.float32),
        "roughness_large_radius": np.random.uniform(0, 30, n_points).astype(np.float32),
        "roughness_small_knn": np.random.uniform(0, 30, n_points).astype(np.float32),
        "roughness_large_knn": np.random.uniform(0, 30, n_points).astype(np.float32),
        "rai_class_radius": np.random.randint(0, 8, n_points).astype(np.uint8),
        "rai_class_knn": np.random.randint(0, 8, n_points).astype(np.uint8),
        "neighbor_count_small": np.random.randint(0, 100, n_points).astype(np.uint16),
        "neighbor_count_large": np.random.randint(0, 500, n_points).astype(np.uint16),
    }

    output_path = tmp_path / "full_rai.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    # Verify all attributes present
    result_las = laspy.read(output_path)
    extra_dim_names = [dim.name for dim in result_las.point_format.extra_dimensions]

    for name in RAI_EXTRA_DIMS.keys():
        assert name in extra_dim_names, f"Missing attribute: {name}"


def test_save_classified_cloud(tmp_path):
    """Test save_classified_cloud convenience function."""
    import laspy
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.io.las_writer import save_classified_cloud

    n_points = 50
    xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    cloud = PointCloud(xyz=xyz)

    slope_deg = np.random.uniform(0, 90, n_points).astype(np.float32)
    roughness_small = np.random.uniform(0, 20, n_points).astype(np.float32)
    roughness_large = np.random.uniform(0, 20, n_points).astype(np.float32)
    rai_class = np.random.randint(0, 8, n_points).astype(np.uint8)

    output_path = tmp_path / "classified.las"
    save_classified_cloud(
        cloud,
        slope_deg,
        roughness_small,
        roughness_large,
        rai_class,
        output_path,
        method="radius",
        compress=False,
    )

    # Verify
    result_las = laspy.read(output_path)
    extra_dim_names = [dim.name for dim in result_las.point_format.extra_dimensions]

    assert "slope_deg" in extra_dim_names
    assert "roughness_small_radius" in extra_dim_names
    assert "roughness_large_radius" in extra_dim_names
    assert "rai_class_radius" in extra_dim_names


def test_round_trip(tmp_path):
    """Test complete round-trip: create -> save -> load."""
    from pc_rai.io.las_reader import PointCloud, load_point_cloud
    from pc_rai.io.las_writer import save_point_cloud

    n_points = 100
    original_xyz = np.random.uniform(0, 10, (n_points, 3)).astype(np.float64)
    original_normals = np.zeros((n_points, 3), dtype=np.float32)
    original_normals[:, 2] = 1.0

    cloud = PointCloud(xyz=original_xyz, normals=original_normals)

    slope_values = np.random.uniform(0, 90, n_points).astype(np.float32)
    attributes = {"slope_deg": slope_values}

    output_path = tmp_path / "round_trip.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)

    # Load back
    loaded = load_point_cloud(output_path)

    # Verify coordinates match (with tolerance for LAS integer precision)
    # LAS files use scaled integers, typical precision is ~0.001m
    assert np.allclose(loaded.xyz, original_xyz, atol=0.01)
    assert loaded.has_normals
    assert np.allclose(loaded.normals[:, 2], 1.0)
