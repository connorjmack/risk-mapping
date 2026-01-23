"""
Slope calculation for PC-RAI.

Calculates slope angle from surface normal vectors relative to an up vector.
"""

from typing import Tuple

import numpy as np


def calculate_slope(
    normals: np.ndarray,
    up_vector: Tuple[float, float, float] = (0, 0, 1),
) -> np.ndarray:
    """
    Calculate slope angle from normal vectors.

    Slope is the angle between the surface normal and the up vector.
    - 0° = horizontal surface (normal points up)
    - 90° = vertical surface
    - >90° = overhanging surface
    - 180° = inverted surface (normal points down)

    Parameters
    ----------
    normals : np.ndarray
        (N, 3) array of unit normal vectors.
    up_vector : tuple
        Direction considered "up" (default: +Z).

    Returns
    -------
    slope_deg : np.ndarray
        (N,) array of slope angles in degrees.
    """
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must have shape (N, 3), got {normals.shape}")

    # Normalize up vector
    up = np.array(up_vector, dtype=np.float64)
    up = up / np.linalg.norm(up)

    # Compute dot product with up vector
    # For unit normals: dot = cos(angle)
    dot_product = np.dot(normals, up)

    # Clamp to [-1, 1] to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians, then convert to degrees
    slope_rad = np.arccos(dot_product)
    slope_deg = np.degrees(slope_rad)

    return slope_deg.astype(np.float32)


def identify_overhangs(
    slope_deg: np.ndarray,
    threshold: float = 90.0,
) -> np.ndarray:
    """
    Return boolean mask of overhanging points.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    threshold : float
        Slope angle threshold for overhang (default 90°).

    Returns
    -------
    mask : np.ndarray
        (N,) boolean array, True for overhanging points.
    """
    return slope_deg > threshold


def identify_steep_slopes(
    slope_deg: np.ndarray,
    min_slope: float = 45.0,
    max_slope: float = 90.0,
) -> np.ndarray:
    """
    Return boolean mask of steep (but not overhanging) slopes.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    min_slope : float
        Minimum slope angle.
    max_slope : float
        Maximum slope angle.

    Returns
    -------
    mask : np.ndarray
        (N,) boolean array.
    """
    return (slope_deg >= min_slope) & (slope_deg <= max_slope)


def slope_statistics(slope_deg: np.ndarray) -> dict:
    """
    Calculate summary statistics for slope angles.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.

    Returns
    -------
    stats : dict
        Dictionary with mean, std, min, max, and percentiles.
    """
    valid = ~np.isnan(slope_deg)
    if not np.any(valid):
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "n_overhang": 0,
            "pct_overhang": 0.0,
        }

    valid_slopes = slope_deg[valid]
    n_overhang = int(np.sum(valid_slopes > 90))

    return {
        "mean": float(np.mean(valid_slopes)),
        "std": float(np.std(valid_slopes)),
        "min": float(np.min(valid_slopes)),
        "max": float(np.max(valid_slopes)),
        "median": float(np.median(valid_slopes)),
        "p25": float(np.percentile(valid_slopes, 25)),
        "p75": float(np.percentile(valid_slopes, 75)),
        "n_overhang": n_overhang,
        "pct_overhang": 100.0 * n_overhang / len(valid_slopes),
    }
