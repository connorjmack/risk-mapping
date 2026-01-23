"""
Roughness calculation for PC-RAI.

Calculates multi-scale surface roughness as the standard deviation of
slope angles within local neighborhoods.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pc_rai.utils.spatial import SpatialIndex, compute_neighbor_stats, compute_neighbor_stats_knn


def calculate_roughness_radius(
    slope_deg: np.ndarray,
    spatial_index: SpatialIndex,
    radius: float,
    min_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate roughness as std dev of slope within radius.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    spatial_index : SpatialIndex
        Pre-built spatial index.
    radius : float
        Search radius in same units as points.
    min_neighbors : int
        Minimum neighbors for valid calculation.

    Returns
    -------
    roughness : np.ndarray
        (N,) roughness values in degrees (NaN where insufficient neighbors).
    neighbor_counts : np.ndarray
        (N,) number of neighbors found.
    """
    # Query neighbors within radius
    neighbors, counts = spatial_index.query_radius(radius, return_counts=True)

    # Compute standard deviation of slope for each neighborhood
    roughness, _ = compute_neighbor_stats(slope_deg, neighbors, min_neighbors)

    return roughness, counts


def calculate_roughness_knn(
    slope_deg: np.ndarray,
    spatial_index: SpatialIndex,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate roughness as std dev of slope for k nearest neighbors.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    spatial_index : SpatialIndex
        Pre-built spatial index.
    k : int
        Number of neighbors.

    Returns
    -------
    roughness : np.ndarray
        (N,) roughness values in degrees.
    neighbor_counts : np.ndarray
        (N,) all values equal to k (or n_points if k > n_points).
    """
    # Query k nearest neighbors
    _, indices = spatial_index.query_knn(k)

    # Compute standard deviation of slope for each neighborhood
    roughness = compute_neighbor_stats_knn(slope_deg, indices)

    # Neighbor count is always k (or capped at n_points)
    actual_k = indices.shape[1]
    neighbor_counts = np.full(len(slope_deg), actual_k, dtype=np.int32)

    return roughness, neighbor_counts


def calculate_all_roughness(
    slope_deg: np.ndarray,
    spatial_index: SpatialIndex,
    radius_small: float = 0.175,
    radius_large: float = 0.425,
    k_small: int = 30,
    k_large: int = 100,
    min_neighbors: int = 5,
    methods: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate all roughness metrics.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    spatial_index : SpatialIndex
        Pre-built spatial index.
    radius_small : float
        Small-scale radius (default 0.175m from Markus et al. 2023).
    radius_large : float
        Large-scale radius (default 0.425m from Markus et al. 2023).
    k_small : int
        Small-scale k for k-NN method.
    k_large : int
        Large-scale k for k-NN method.
    min_neighbors : int
        Minimum neighbors for valid radius calculation.
    methods : list, optional
        Methods to compute: ["radius"], ["knn"], or ["radius", "knn"].
        Default is both.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - roughness_small_radius (if "radius" in methods)
        - roughness_large_radius (if "radius" in methods)
        - roughness_small_knn (if "knn" in methods)
        - roughness_large_knn (if "knn" in methods)
        - neighbor_count_small
        - neighbor_count_large
    """
    if methods is None:
        methods = ["radius", "knn"]

    results = {}

    # Initialize neighbor counts (will be overwritten by actual calculations)
    n_points = len(slope_deg)
    results["neighbor_count_small"] = np.zeros(n_points, dtype=np.int32)
    results["neighbor_count_large"] = np.zeros(n_points, dtype=np.int32)

    if "radius" in methods:
        # Small-scale radius roughness
        r_small, counts_small = calculate_roughness_radius(
            slope_deg, spatial_index, radius_small, min_neighbors
        )
        results["roughness_small_radius"] = r_small
        results["neighbor_count_small"] = counts_small

        # Large-scale radius roughness
        r_large, counts_large = calculate_roughness_radius(
            slope_deg, spatial_index, radius_large, min_neighbors
        )
        results["roughness_large_radius"] = r_large
        results["neighbor_count_large"] = counts_large

    if "knn" in methods:
        # Small-scale k-NN roughness
        r_small_knn, counts_small_knn = calculate_roughness_knn(
            slope_deg, spatial_index, k_small
        )
        results["roughness_small_knn"] = r_small_knn

        # Large-scale k-NN roughness
        r_large_knn, counts_large_knn = calculate_roughness_knn(
            slope_deg, spatial_index, k_large
        )
        results["roughness_large_knn"] = r_large_knn

        # Use k-NN counts if radius not computed
        if "radius" not in methods:
            results["neighbor_count_small"] = counts_small_knn
            results["neighbor_count_large"] = counts_large_knn

    return results


def roughness_statistics(roughness: np.ndarray, name: str = "roughness") -> dict:
    """
    Calculate summary statistics for roughness values.

    Parameters
    ----------
    roughness : np.ndarray
        (N,) roughness values in degrees.
    name : str
        Name for the roughness metric.

    Returns
    -------
    stats : dict
        Dictionary with statistics.
    """
    valid = ~np.isnan(roughness)
    n_valid = np.sum(valid)

    if n_valid == 0:
        return {
            f"{name}_mean": np.nan,
            f"{name}_std": np.nan,
            f"{name}_min": np.nan,
            f"{name}_max": np.nan,
            f"{name}_median": np.nan,
            f"{name}_n_valid": 0,
            f"{name}_pct_valid": 0.0,
        }

    valid_values = roughness[valid]

    return {
        f"{name}_mean": float(np.mean(valid_values)),
        f"{name}_std": float(np.std(valid_values)),
        f"{name}_min": float(np.min(valid_values)),
        f"{name}_max": float(np.max(valid_values)),
        f"{name}_median": float(np.median(valid_values)),
        f"{name}_n_valid": int(n_valid),
        f"{name}_pct_valid": 100.0 * n_valid / len(roughness),
    }
