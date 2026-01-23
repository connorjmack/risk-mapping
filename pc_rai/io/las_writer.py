"""
LAS/LAZ file writer for PC-RAI.

Provides save_point_cloud function for writing point clouds with
RAI classification attributes as extra dimensions.
"""

from pathlib import Path
from typing import Dict, Tuple

import laspy
import numpy as np

from pc_rai.io.las_reader import PointCloud


# Extra dimensions to add for RAI processing results
# Format: (name, (dtype_str, description))
# Note: LAS description field is limited to 32 characters
RAI_EXTRA_DIMS: Dict[str, Tuple[str, str]] = {
    "slope_deg": ("f4", "Slope angle (degrees)"),
    "roughness_small_radius": ("f4", "Small roughness (radius)"),
    "roughness_large_radius": ("f4", "Large roughness (radius)"),
    "roughness_small_knn": ("f4", "Small roughness (kNN)"),
    "roughness_large_knn": ("f4", "Large roughness (kNN)"),
    "rai_class_radius": ("u1", "RAI class (radius)"),
    "rai_class_knn": ("u1", "RAI class (kNN)"),
    "neighbor_count_small": ("u2", "Neighbor count (small)"),
    "neighbor_count_large": ("u2", "Neighbor count (large)"),
}

# Mapping from numpy dtype strings to laspy types
DTYPE_MAP = {
    "f4": np.float32,
    "f8": np.float64,
    "u1": np.uint8,
    "u2": np.uint16,
    "u4": np.uint32,
    "i1": np.int8,
    "i2": np.int16,
    "i4": np.int32,
}


def save_point_cloud(
    cloud: PointCloud,
    attributes: Dict[str, np.ndarray],
    output_path: Path,
    compress: bool = True,
) -> None:
    """
    Save point cloud with RAI attributes as extra dimensions.

    Parameters
    ----------
    cloud : PointCloud
        Original point cloud (preserves all original attributes if available).
    attributes : dict
        Dictionary mapping attribute names to numpy arrays.
        Keys should match RAI_EXTRA_DIMS or be custom names.
    output_path : Path
        Output file path (.las or .laz).
    compress : bool
        If True, save as LAZ (compressed). Default True.

    Raises
    ------
    ValueError
        If attribute arrays have wrong length or unknown dtype.
    """
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate attributes
    for name, arr in attributes.items():
        if len(arr) != cloud.n_points:
            raise ValueError(
                f"Attribute '{name}' has length {len(arr)}, "
                f"expected {cloud.n_points}"
            )

    # Create output LAS based on original or new
    if cloud._las_data is not None:
        # Copy from original to preserve all attributes
        las = _copy_las_data(cloud._las_data)
    else:
        # Create new LAS file
        las = _create_new_las(cloud)

    # Add extra dimensions for RAI attributes
    for name, arr in attributes.items():
        _add_extra_dimension(las, name, arr)

    # Handle compression based on file extension and compress flag
    if compress and not output_path.suffix.lower() == ".laz":
        output_path = output_path.with_suffix(".laz")
    elif not compress and output_path.suffix.lower() == ".laz":
        output_path = output_path.with_suffix(".las")

    # Write file
    las.write(output_path)


def _copy_las_data(original: laspy.LasData) -> laspy.LasData:
    """Create a copy of LAS data for modification."""
    # Create new LAS with same point format
    las = laspy.create(
        point_format=original.point_format,
        file_version=original.header.version,
    )

    # Copy header information
    las.header.offsets = original.header.offsets
    las.header.scales = original.header.scales

    # Copy all point records
    las.points = original.points.copy()

    return las


def _create_new_las(cloud: PointCloud) -> laspy.LasData:
    """Create a new LAS file from PointCloud data."""
    # Use point format 0 (basic XYZ + intensity)
    las = laspy.create(point_format=0, file_version="1.4")

    # Set coordinates
    las.x = cloud.xyz[:, 0]
    las.y = cloud.xyz[:, 1]
    las.z = cloud.xyz[:, 2]

    # Add normals if present
    if cloud.has_normals:
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))

        las["NormalX"] = cloud.normals[:, 0]
        las["NormalY"] = cloud.normals[:, 1]
        las["NormalZ"] = cloud.normals[:, 2]

    return las


def _add_extra_dimension(
    las: laspy.LasData,
    name: str,
    values: np.ndarray,
) -> None:
    """Add an extra dimension to LAS file.

    Parameters
    ----------
    las : laspy.LasData
        LAS data object to modify.
    name : str
        Name of the extra dimension.
    values : np.ndarray
        Values to store.
    """
    # Check if dimension already exists
    existing_dims = [dim.name for dim in las.point_format.extra_dimensions]

    if name in existing_dims:
        # Update existing dimension
        las[name] = values
        return

    # Determine dtype
    if name in RAI_EXTRA_DIMS:
        dtype_str, description = RAI_EXTRA_DIMS[name]
        dtype = DTYPE_MAP[dtype_str]
    else:
        # Infer from array
        dtype = values.dtype
        description = ""

    # Add new extra dimension
    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name=name,
            type=dtype,
            description=description,
        )
    )

    # Set values (convert to correct dtype if needed)
    las[name] = values.astype(dtype)


def save_classified_cloud(
    cloud: PointCloud,
    slope_deg: np.ndarray,
    roughness_small: np.ndarray,
    roughness_large: np.ndarray,
    rai_class: np.ndarray,
    output_path: Path,
    method: str = "radius",
    neighbor_counts: Dict[str, np.ndarray] = None,
    compress: bool = True,
) -> None:
    """
    Convenience function to save a fully classified point cloud.

    Parameters
    ----------
    cloud : PointCloud
        Original point cloud.
    slope_deg : np.ndarray
        Slope angles in degrees.
    roughness_small : np.ndarray
        Small-scale roughness values.
    roughness_large : np.ndarray
        Large-scale roughness values.
    rai_class : np.ndarray
        RAI classification codes (0-7).
    output_path : Path
        Output file path.
    method : str
        Method used: "radius" or "knn".
    neighbor_counts : dict, optional
        Dictionary with 'small' and 'large' neighbor count arrays.
    compress : bool
        If True, save as LAZ.
    """
    # Build attributes dict
    attributes = {
        "slope_deg": slope_deg.astype(np.float32),
    }

    # Add roughness and class with method suffix
    if method == "radius":
        attributes["roughness_small_radius"] = roughness_small.astype(np.float32)
        attributes["roughness_large_radius"] = roughness_large.astype(np.float32)
        attributes["rai_class_radius"] = rai_class.astype(np.uint8)
    else:
        attributes["roughness_small_knn"] = roughness_small.astype(np.float32)
        attributes["roughness_large_knn"] = roughness_large.astype(np.float32)
        attributes["rai_class_knn"] = rai_class.astype(np.uint8)

    # Add neighbor counts if provided
    if neighbor_counts is not None:
        if "small" in neighbor_counts:
            attributes["neighbor_count_small"] = neighbor_counts["small"].astype(
                np.uint16
            )
        if "large" in neighbor_counts:
            attributes["neighbor_count_large"] = neighbor_counts["large"].astype(
                np.uint16
            )

    save_point_cloud(cloud, attributes, output_path, compress=compress)
