"""
LAS/LAZ file reader for PC-RAI.

Provides the PointCloud dataclass and load_point_cloud function for
reading LiDAR point cloud files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import laspy
import numpy as np


@dataclass
class PointCloud:
    """Container for point cloud data.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) array of XYZ coordinates in float64.
    normals : np.ndarray, optional
        (N, 3) array of unit normal vectors in float32.
    source_file : Path, optional
        Path to the source LAS/LAZ file.

    Attributes
    ----------
    _las_data : laspy.LasData, optional
        Original LAS data for preserving attributes during output.
    """

    xyz: np.ndarray
    normals: Optional[np.ndarray] = None
    source_file: Optional[Path] = None

    # Store original LAS data for preservation
    _las_data: Optional[laspy.LasData] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate array shapes and types."""
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {self.xyz.shape}")

        if self.normals is not None:
            if self.normals.ndim != 2 or self.normals.shape[1] != 3:
                raise ValueError(
                    f"normals must have shape (N, 3), got {self.normals.shape}"
                )
            if len(self.normals) != len(self.xyz):
                raise ValueError(
                    f"normals length ({len(self.normals)}) must match "
                    f"xyz length ({len(self.xyz)})"
                )

    @property
    def n_points(self) -> int:
        """Return number of points in the cloud."""
        return len(self.xyz)

    @property
    def has_normals(self) -> bool:
        """Return True if normals are present."""
        return self.normals is not None

    @property
    def bounds(self) -> Dict[str, tuple]:
        """Return min/max for each dimension.

        Returns
        -------
        dict
            Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples.
        """
        return {
            "x": (float(self.xyz[:, 0].min()), float(self.xyz[:, 0].max())),
            "y": (float(self.xyz[:, 1].min()), float(self.xyz[:, 1].max())),
            "z": (float(self.xyz[:, 2].min()), float(self.xyz[:, 2].max())),
        }

    @property
    def centroid(self) -> np.ndarray:
        """Return centroid of the point cloud."""
        return self.xyz.mean(axis=0)

    @property
    def extent(self) -> Dict[str, float]:
        """Return extent (range) for each dimension."""
        bounds = self.bounds
        return {
            "x": bounds["x"][1] - bounds["x"][0],
            "y": bounds["y"][1] - bounds["y"][0],
            "z": bounds["z"][1] - bounds["z"][0],
        }


def load_point_cloud(filepath: Path) -> PointCloud:
    """
    Load a LAS/LAZ file into a PointCloud object.

    Extracts XYZ coordinates and normal vectors if present.
    Preserves original LAS data for later output.

    Parameters
    ----------
    filepath : Path
        Path to LAS or LAZ file.

    Returns
    -------
    PointCloud
        Point cloud object with XYZ and optional normals.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be read as a valid LAS/LAZ file.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        las = laspy.read(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read LAS file: {filepath}. Error: {e}") from e

    # Extract XYZ coordinates
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    # Extract normals if present
    normals = None
    if has_valid_normals(las):
        normals = extract_normals(las)

    return PointCloud(
        xyz=xyz,
        normals=normals,
        source_file=filepath,
        _las_data=las,
    )


def has_valid_normals(las: laspy.LasData) -> bool:
    """
    Check if LAS file has valid normal vectors.

    CloudCompare typically saves normals as NormalX, NormalY, NormalZ
    extra dimensions. Some software uses Nx, Ny, Nz or nx, ny, nz.

    Parameters
    ----------
    las : laspy.LasData
        Loaded LAS file data.

    Returns
    -------
    bool
        True if valid normals are present.
    """
    # Get list of extra dimension names
    extra_dim_names = [dim.name for dim in las.point_format.extra_dimensions]

    # Check for common normal naming conventions
    normal_name_sets = [
        ("NormalX", "NormalY", "NormalZ"),
        ("normalx", "normaly", "normalz"),
        ("Nx", "Ny", "Nz"),
        ("nx", "ny", "nz"),
    ]

    for nx_name, ny_name, nz_name in normal_name_sets:
        if all(name in extra_dim_names for name in (nx_name, ny_name, nz_name)):
            return True

    return False


def extract_normals(las: laspy.LasData) -> np.ndarray:
    """
    Extract normal vectors from LAS file.

    Parameters
    ----------
    las : laspy.LasData
        Loaded LAS file data with normal dimensions.

    Returns
    -------
    np.ndarray
        (N, 3) array of normal vectors as float32.

    Raises
    ------
    ValueError
        If normals cannot be found in the file.
    """
    extra_dim_names = [dim.name for dim in las.point_format.extra_dimensions]

    # Try each naming convention
    normal_name_sets = [
        ("NormalX", "NormalY", "NormalZ"),
        ("normalx", "normaly", "normalz"),
        ("Nx", "Ny", "Nz"),
        ("nx", "ny", "nz"),
    ]

    for nx_name, ny_name, nz_name in normal_name_sets:
        if all(name in extra_dim_names for name in (nx_name, ny_name, nz_name)):
            nx = np.array(las[nx_name], dtype=np.float32)
            ny = np.array(las[ny_name], dtype=np.float32)
            nz = np.array(las[nz_name], dtype=np.float32)
            return np.column_stack([nx, ny, nz])

    raise ValueError("Could not find normal vectors in LAS file")


def get_las_info(filepath: Path) -> Dict[str, Any]:
    """
    Get summary information about a LAS file without fully loading it.

    Parameters
    ----------
    filepath : Path
        Path to LAS or LAZ file.

    Returns
    -------
    dict
        Dictionary containing file information.
    """
    filepath = Path(filepath)

    with laspy.open(filepath) as f:
        header = f.header
        info = {
            "filepath": str(filepath),
            "point_count": header.point_count,
            "point_format": header.point_format.id,
            "version": f"{header.version.major}.{header.version.minor}",
            "bounds": {
                "x": (header.x_min, header.x_max),
                "y": (header.y_min, header.y_max),
                "z": (header.z_min, header.z_max),
            },
            "scale": (header.x_scale, header.y_scale, header.z_scale),
            "offset": (header.x_offset, header.y_offset, header.z_offset),
            "extra_dimensions": [
                dim.name for dim in header.point_format.extra_dimensions
            ],
        }

    return info
