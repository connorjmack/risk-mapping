"""
Spatial indexing utilities for PC-RAI.

Provides the SpatialIndex class wrapping scipy's cKDTree for efficient
neighbor queries used in roughness calculation.
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


class SpatialIndex:
    """Wrapper around scipy cKDTree for neighbor queries.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of XYZ coordinates.

    Attributes
    ----------
    points : np.ndarray
        Original point coordinates.
    tree : cKDTree
        Spatial index structure.
    n_points : int
        Number of points in the index.
    """

    def __init__(self, points: np.ndarray):
        """
        Build KD-tree from points.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of XYZ coordinates.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.tree = cKDTree(self.points)
        self.n_points = len(points)

    def query_radius(
        self,
        radius: float,
        return_counts: bool = False,
        show_progress: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Query all neighbors within radius for each point.

        Parameters
        ----------
        radius : float
            Search radius in same units as points.
        return_counts : bool
            If True, also return neighbor counts.
        show_progress : bool
            If True, display progress bar.

        Returns
        -------
        neighbors : list of np.ndarray
            List of neighbor indices for each point (includes self).
        counts : np.ndarray
            Number of neighbors per point (only if return_counts=True).
        """
        # Query all points at once - more efficient than looping
        neighbors = self.tree.query_ball_tree(self.tree, radius)

        # Convert to list of arrays
        neighbors = [np.array(n, dtype=np.int64) for n in neighbors]

        if return_counts:
            counts = np.array([len(n) for n in neighbors], dtype=np.int32)
            return neighbors, counts

        return neighbors, np.array([len(n) for n in neighbors], dtype=np.int32)

    def query_knn(
        self,
        k: int,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k nearest neighbors for each point.

        Parameters
        ----------
        k : int
            Number of neighbors (including self).
        show_progress : bool
            If True, display progress bar.

        Returns
        -------
        distances : np.ndarray
            (N, k) distances to neighbors.
        indices : np.ndarray
            (N, k) indices of neighbors.
        """
        if k > self.n_points:
            k = self.n_points

        # Query all points at once
        distances, indices = self.tree.query(self.points, k=k)

        return distances.astype(np.float64), indices.astype(np.int64)

    def query_radius_single(self, point: np.ndarray, radius: float) -> np.ndarray:
        """
        Query neighbors within radius for a single point.

        Parameters
        ----------
        point : np.ndarray
            (3,) query point coordinates.
        radius : float
            Search radius.

        Returns
        -------
        indices : np.ndarray
            Indices of neighbors within radius.
        """
        indices = self.tree.query_ball_point(point, radius)
        return np.array(indices, dtype=np.int64)

    def query_knn_single(
        self, point: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k nearest neighbors for a single point.

        Parameters
        ----------
        point : np.ndarray
            (3,) query point coordinates.
        k : int
            Number of neighbors.

        Returns
        -------
        distances : np.ndarray
            (k,) distances to neighbors.
        indices : np.ndarray
            (k,) indices of neighbors.
        """
        distances, indices = self.tree.query(point, k=k)
        return np.atleast_1d(distances), np.atleast_1d(indices)


def compute_neighbor_stats(
    values: np.ndarray,
    neighbors: List[np.ndarray],
    min_neighbors: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute standard deviation of values for each point's neighborhood.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of values (e.g., slope angles).
    neighbors : list of np.ndarray
        List of neighbor indices for each point.
    min_neighbors : int
        Minimum neighbors for valid calculation.

    Returns
    -------
    std_dev : np.ndarray
        (N,) standard deviation values (NaN where insufficient neighbors).
    counts : np.ndarray
        (N,) neighbor counts.
    """
    n_points = len(values)
    std_dev = np.full(n_points, np.nan, dtype=np.float32)
    counts = np.zeros(n_points, dtype=np.int32)

    for i, neighbor_idx in enumerate(neighbors):
        counts[i] = len(neighbor_idx)
        if counts[i] >= min_neighbors:
            neighbor_values = values[neighbor_idx]
            std_dev[i] = np.std(neighbor_values)

    return std_dev, counts


def compute_neighbor_stats_knn(
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Compute standard deviation of values for k-NN neighborhoods.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of values (e.g., slope angles).
    indices : np.ndarray
        (N, k) neighbor indices from k-NN query.

    Returns
    -------
    std_dev : np.ndarray
        (N,) standard deviation values.
    """
    # Gather neighbor values: (N, k)
    neighbor_values = values[indices]
    # Compute std along neighbor axis
    std_dev = np.std(neighbor_values, axis=1).astype(np.float32)
    return std_dev


def smooth_values_radius(
    values: np.ndarray,
    spatial_index: "SpatialIndex",
    radius: float,
) -> np.ndarray:
    """
    Smooth values by averaging within a radius neighborhood.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of values to smooth (e.g., slope angles).
    spatial_index : SpatialIndex
        Pre-built spatial index for the points.
    radius : float
        Smoothing radius in same units as points.

    Returns
    -------
    smoothed : np.ndarray
        (N,) smoothed values (mean of neighbors within radius).
    """
    neighbors, _ = spatial_index.query_radius(radius, return_counts=True)

    smoothed = np.zeros_like(values, dtype=np.float32)
    for i, neighbor_idx in enumerate(neighbors):
        if len(neighbor_idx) > 0:
            smoothed[i] = np.mean(values[neighbor_idx])
        else:
            smoothed[i] = values[i]  # Keep original if no neighbors

    return smoothed


def voxel_subsample(
    xyz: np.ndarray,
    voxel_size: float,
    normals: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample point cloud using voxel grid.

    For each voxel, keeps the point closest to the voxel center.
    This avoids systematic bias from scan-line ordering that can
    create visible striping artifacts.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    voxel_size : float
        Voxel grid spacing in same units as xyz.
    normals : np.ndarray, optional
        (N, 3) normal vectors. If provided, returns subsampled normals.

    Returns
    -------
    xyz_sub : np.ndarray
        (M, 3) subsampled point coordinates.
    normals_sub : np.ndarray or None
        (M, 3) subsampled normals if input normals provided, else None.
    indices : np.ndarray
        (M,) indices of kept points in original array.
    """
    # Compute voxel indices for each point
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    # Compute voxel centers
    voxel_centers = (voxel_indices + 0.5) * voxel_size

    # Distance from each point to its voxel center
    dist_to_center = np.linalg.norm(xyz - voxel_centers, axis=1)

    # Create unique voxel keys using a large prime multiplier approach
    keys = (
        voxel_indices[:, 0] * 73856093 +
        voxel_indices[:, 1] * 19349663 +
        voxel_indices[:, 2] * 83492791
    )

    # For each unique voxel, find the point closest to center
    # Sort by key first, then by distance within each key
    sort_order = np.lexsort((dist_to_center, keys))
    sorted_keys = keys[sort_order]

    # Find first occurrence of each unique key (which has smallest distance due to lexsort)
    unique_mask = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
    unique_indices = sort_order[unique_mask]

    # Sort indices to maintain spatial coherence in output
    unique_indices = np.sort(unique_indices)

    xyz_sub = xyz[unique_indices]
    normals_sub = normals[unique_indices] if normals is not None else None

    return xyz_sub, normals_sub, unique_indices
