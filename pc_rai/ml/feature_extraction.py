"""
Feature extraction for ML training pipeline.

Subsamples point clouds and computes per-point features (slope, roughness, height)
for use in polygon-level aggregation and model training.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import laspy
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def voxel_subsample(
    xyz: np.ndarray,
    voxel_size: float = 0.5,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Subsample point cloud using voxel grid.

    Divides space into cubic voxels and keeps one point per occupied voxel
    (the point closest to the voxel center).

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) array of XYZ coordinates.
    voxel_size : float
        Side length of voxel cubes in same units as coordinates.
    return_indices : bool
        If True, also return indices of selected points.

    Returns
    -------
    subsampled : np.ndarray
        (M, 3) array of subsampled coordinates where M <= N.
    indices : np.ndarray, optional
        (M,) indices into original array (if return_indices=True).
    """
    if xyz.shape[0] == 0:
        if return_indices:
            return xyz, np.array([], dtype=np.int64)
        return xyz

    # Compute voxel indices for each point
    min_coords = xyz.min(axis=0)
    voxel_indices = ((xyz - min_coords) / voxel_size).astype(np.int64)

    # Create unique voxel keys
    # Use a large multiplier to create unique keys from 3D indices
    max_idx = voxel_indices.max(axis=0) + 1
    keys = (
        voxel_indices[:, 0] * (max_idx[1] * max_idx[2])
        + voxel_indices[:, 1] * max_idx[2]
        + voxel_indices[:, 2]
    )

    # Find unique voxels and get one point per voxel
    # Use lexsort to group by voxel, then take first point in each group
    unique_keys, unique_indices = np.unique(keys, return_index=True)

    # Get subsampled points
    selected_indices = unique_indices
    subsampled = xyz[selected_indices]

    if return_indices:
        return subsampled, selected_indices
    return subsampled


def estimate_normals_pca(
    xyz: np.ndarray,
    k_neighbors: int = 30,
    orient_toward: str = "west",
    verbose: bool = True,
) -> np.ndarray:
    """Estimate normals using PCA on local neighborhoods.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    k_neighbors : int
        Number of neighbors for PCA.
    orient_toward : str
        Direction to orient normals: "west" (-X), "up" (+Z), or "origin".
    verbose : bool
        Print progress.

    Returns
    -------
    normals : np.ndarray
        (N, 3) unit normal vectors.
    """
    n_points = len(xyz)
    normals = np.zeros((n_points, 3), dtype=np.float32)

    if verbose:
        logger.info(f"  Estimating normals using PCA (k={k_neighbors})...")

    # Build KD-tree
    tree = cKDTree(xyz)

    # Query k nearest neighbors for each point
    _, indices = tree.query(xyz, k=k_neighbors)

    # Compute normal for each point using PCA
    for i in range(n_points):
        neighbors = xyz[indices[i]]
        # Center the neighborhood
        centered = neighbors - neighbors.mean(axis=0)
        # Compute covariance matrix
        cov = np.dot(centered.T, centered) / len(neighbors)
        # Eigendecomposition (smallest eigenvector is normal)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is eigenvector with smallest eigenvalue
        normals[i] = eigenvectors[:, 0]

        # Progress every 100k points
        if verbose and (i + 1) % 100000 == 0:
            logger.info(f"    Processed {i+1:,}/{n_points:,} points...")

    # Orient normals consistently
    if orient_toward == "west":
        # Cliff faces typically face west (-X direction)
        # Flip normals that point east (+X)
        mask = normals[:, 0] > 0
        normals[mask] *= -1
    elif orient_toward == "up":
        # Flip normals that point down (-Z)
        mask = normals[:, 2] < 0
        normals[mask] *= -1
    elif orient_toward == "origin":
        # Point toward centroid
        centroid = xyz.mean(axis=0)
        to_centroid = centroid - xyz
        dot = np.sum(normals * to_centroid, axis=1)
        mask = dot < 0
        normals[mask] *= -1

    return normals


def compute_slope(normals: np.ndarray) -> np.ndarray:
    """Compute slope angle from normal vectors.

    Parameters
    ----------
    normals : np.ndarray
        (N, 3) array of unit normal vectors.

    Returns
    -------
    slope_deg : np.ndarray
        (N,) slope angles in degrees (0=horizontal, 90=vertical, >90=overhang).
    """
    # Slope is angle from vertical (Z-up)
    # For unit normals: slope = arccos(Nz)
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    slope_rad = np.arccos(nz)
    return np.degrees(slope_rad).astype(np.float32)


def compute_roughness(
    slope_deg: np.ndarray,
    xyz: np.ndarray,
    radius: float,
    min_neighbors: int = 5,
) -> np.ndarray:
    """Compute roughness as std dev of slope within radius.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    xyz : np.ndarray
        (N, 3) point coordinates for neighbor queries.
    radius : float
        Search radius.
    min_neighbors : int
        Minimum neighbors for valid result.

    Returns
    -------
    roughness : np.ndarray
        (N,) roughness values in degrees (NaN where insufficient neighbors).
    """
    tree = cKDTree(xyz)
    roughness = np.full(len(xyz), np.nan, dtype=np.float32)

    # Query all neighbors within radius
    neighbors_list = tree.query_ball_tree(tree, radius)

    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) >= min_neighbors:
            roughness[i] = np.std(slope_deg[neighbors])

    return roughness


def compute_relative_height(
    xyz: np.ndarray,
    horizontal_radius: float = 5.0,
) -> np.ndarray:
    """Compute height relative to local minimum within horizontal radius.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    horizontal_radius : float
        Radius for finding local minimum (horizontal distance only).

    Returns
    -------
    height : np.ndarray
        (N,) relative height values (Z - local Z_min).
    """
    # Build 2D tree for horizontal neighbor queries
    xy = xyz[:, :2]
    tree = cKDTree(xy)

    height = np.zeros(len(xyz), dtype=np.float32)

    # Query neighbors within horizontal radius
    neighbors_list = tree.query_ball_tree(tree, horizontal_radius)

    for i, neighbors in enumerate(neighbors_list):
        local_z_min = xyz[neighbors, 2].min()
        height[i] = xyz[i, 2] - local_z_min

    return height


def extract_features(
    xyz: np.ndarray,
    normals: np.ndarray,
    radius_small: float = 0.5,
    radius_large: float = 2.0,
    height_radius: float = 5.0,
    min_neighbors: int = 5,
    verbose: bool = True,
) -> dict:
    """Extract all features at each point.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    normals : np.ndarray
        (N, 3) unit normal vectors.
    radius_small : float
        Radius for small-scale roughness.
    radius_large : float
        Radius for large-scale roughness.
    height_radius : float
        Horizontal radius for relative height calculation.
    min_neighbors : int
        Minimum neighbors for roughness calculation.
    verbose : bool
        Print progress.

    Returns
    -------
    features : dict
        Dictionary with arrays: slope, roughness_small, roughness_large,
        roughness_ratio, height.
    """
    n_points = len(xyz)

    if verbose:
        logger.info(f"Extracting features for {n_points:,} points...")

    # 1. Slope from normals
    if verbose:
        logger.info("  Computing slope...")
    slope = compute_slope(normals)

    # 2. Small-scale roughness
    if verbose:
        logger.info(f"  Computing roughness (r={radius_small}m)...")
    roughness_small = compute_roughness(slope, xyz, radius_small, min_neighbors)

    # 3. Large-scale roughness
    if verbose:
        logger.info(f"  Computing roughness (r={radius_large}m)...")
    roughness_large = compute_roughness(slope, xyz, radius_large, min_neighbors)

    # 4. Roughness ratio (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        roughness_ratio = roughness_small / roughness_large
        roughness_ratio = np.where(
            np.isfinite(roughness_ratio), roughness_ratio, np.nan
        ).astype(np.float32)

    # 5. Relative height
    if verbose:
        logger.info(f"  Computing relative height (r={height_radius}m)...")
    height = compute_relative_height(xyz, height_radius)

    if verbose:
        valid_small = np.sum(~np.isnan(roughness_small))
        valid_large = np.sum(~np.isnan(roughness_large))
        logger.info(f"  Valid roughness: small={valid_small:,}, large={valid_large:,}")

    return {
        "slope": slope,
        "roughness_small": roughness_small,
        "roughness_large": roughness_large,
        "roughness_ratio": roughness_ratio,
        "height": height,
    }


def subsample_survey(
    las_path: Union[str, Path],
    output_dir: Union[str, Path],
    voxel_size: float = 0.5,
    verbose: bool = True,
) -> Optional[Path]:
    """Subsample a survey without computing features.

    Use this to prepare point clouds for external normal computation
    (e.g., CloudComPy) before running full feature extraction.

    Parameters
    ----------
    las_path : str or Path
        Path to input LAS/LAZ file.
    output_dir : str or Path
        Output directory for subsampled file.
    voxel_size : float
        Voxel size for subsampling (default: 0.5m).
    verbose : bool
        Print progress.

    Returns
    -------
    output_path : Path or None
        Path to output file, or None if processing failed.
    """
    las_path = Path(las_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSubsampling: {las_path.name}")

    # Load LAS file
    try:
        las = laspy.read(las_path)
    except Exception as e:
        logger.error(f"Failed to read {las_path}: {e}")
        return None

    # Get coordinates
    xyz = np.column_stack([las.x, las.y, las.z])
    n_original = len(xyz)

    if verbose:
        print(f"  Original points: {n_original:,}")

    # Subsample
    if verbose:
        print(f"  Subsampling with voxel size {voxel_size}m...")

    xyz_sub, indices = voxel_subsample(xyz, voxel_size, return_indices=True)
    n_subsampled = len(xyz_sub)

    if verbose:
        reduction = 100 * (1 - n_subsampled / n_original)
        print(f"  Subsampled points: {n_subsampled:,} ({reduction:.1f}% reduction)")

    # Create output LAS - preserve intensity if available
    output_las = laspy.create(point_format=0, file_version="1.4")
    output_las.x = xyz_sub[:, 0]
    output_las.y = xyz_sub[:, 1]
    output_las.z = xyz_sub[:, 2]

    # Copy intensity if present
    if hasattr(las, 'intensity'):
        output_las.intensity = las.intensity[indices]

    # Generate output filename
    stem = las_path.stem
    if "_noveg" in stem:
        stem = stem.replace("_noveg", "_subsampled")
    else:
        stem = stem + "_subsampled"

    output_path = output_dir / f"{stem}.laz"

    # Write
    output_las.write(output_path)

    if verbose:
        print(f"  Saved: {output_path.name}")

    return output_path


def process_survey(
    las_path: Union[str, Path],
    output_dir: Union[str, Path],
    voxel_size: float = 0.5,
    radius_small: float = 0.5,
    radius_large: float = 2.0,
    height_radius: float = 5.0,
    min_neighbors: int = 5,
    normal_k: int = 30,
    compute_normals_if_missing: bool = True,
    subsample_only: bool = False,
    verbose: bool = True,
) -> Optional[Path]:
    """Process a single survey: subsample and extract features.

    Parameters
    ----------
    las_path : str or Path
        Path to input LAS/LAZ file (must have normals).
    output_dir : str or Path
        Output directory for subsampled file with features.
    voxel_size : float
        Voxel size for subsampling (default: 0.5m).
    radius_small : float
        Radius for small-scale roughness.
    radius_large : float
        Radius for large-scale roughness.
    height_radius : float
        Horizontal radius for relative height.
    min_neighbors : int
        Minimum neighbors for roughness.
    verbose : bool
        Print progress.

    Returns
    -------
    output_path : Path or None
        Path to output file, or None if processing failed.
    """
    las_path = Path(las_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nProcessing: {las_path.name}")

    # Load LAS file
    try:
        las = laspy.read(las_path)
    except Exception as e:
        logger.error(f"Failed to read {las_path}: {e}")
        return None

    # Get coordinates
    xyz = np.column_stack([las.x, las.y, las.z])
    n_original = len(xyz)

    if verbose:
        print(f"  Original points: {n_original:,}")

    # If subsample_only, just subsample and save (no normals/features)
    if subsample_only:
        return subsample_survey(las_path, output_dir, voxel_size, verbose=False)

    # Check for normals
    normal_names = ["NormalX", "NormalY", "NormalZ"]
    has_normals = all(name in las.point_format.dimension_names for name in normal_names)

    if not has_normals:
        # Try lowercase
        normal_names_lower = ["normalx", "normaly", "normalz"]
        has_normals = all(
            name in las.point_format.dimension_names for name in normal_names_lower
        )
        if has_normals:
            normal_names = normal_names_lower

    if not has_normals:
        if compute_normals_if_missing:
            if verbose:
                print(f"  No normals found - will compute using PCA after subsampling...")

            # Subsample first (computing normals on full cloud is too slow)
            if verbose:
                print(f"  Subsampling with voxel size {voxel_size}m...")
            xyz_sub, indices = voxel_subsample(xyz, voxel_size, return_indices=True)
            n_subsampled = len(xyz_sub)
            if verbose:
                reduction = 100 * (1 - n_subsampled / n_original)
                print(f"  Subsampled points: {n_subsampled:,} ({reduction:.1f}% reduction)")

            # Compute normals on subsampled cloud
            normals_sub = estimate_normals_pca(
                xyz_sub, k_neighbors=normal_k, orient_toward="west", verbose=verbose
            )
        else:
            logger.error(f"No normals found in {las_path.name}")
            print(f"  ERROR: No normals found. Run with --compute-normals or use CloudComPy.")
            return None
    else:
        # Get normals from file
        normals = np.column_stack([
            las[normal_names[0]],
            las[normal_names[1]],
            las[normal_names[2]],
        ]).astype(np.float32)

        # Normalize normals (some may not be unit length)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms

        # Subsample
        if verbose:
            print(f"  Subsampling with voxel size {voxel_size}m...")

        xyz_sub, indices = voxel_subsample(xyz, voxel_size, return_indices=True)
        normals_sub = normals[indices]
        n_subsampled = len(xyz_sub)

        if verbose:
            reduction = 100 * (1 - n_subsampled / n_original)
            print(f"  Subsampled points: {n_subsampled:,} ({reduction:.1f}% reduction)")

    # Extract features
    features = extract_features(
        xyz_sub,
        normals_sub,
        radius_small=radius_small,
        radius_large=radius_large,
        height_radius=height_radius,
        min_neighbors=min_neighbors,
        verbose=verbose,
    )

    # Create output LAS
    output_las = laspy.create(point_format=0, file_version="1.4")

    # Set coordinates
    output_las.x = xyz_sub[:, 0]
    output_las.y = xyz_sub[:, 1]
    output_las.z = xyz_sub[:, 2]

    # Add normals as extra dimensions
    for name, data in [
        ("NormalX", normals_sub[:, 0]),
        ("NormalY", normals_sub[:, 1]),
        ("NormalZ", normals_sub[:, 2]),
    ]:
        output_las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
        output_las[name] = data.astype(np.float32)

    # Add feature dimensions
    for name, data in features.items():
        output_las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
        output_las[name] = data.astype(np.float32)

    # Generate output filename
    stem = las_path.stem
    if "_noveg" in stem:
        stem = stem.replace("_noveg", "_subsampled_features")
    else:
        stem = stem + "_subsampled_features"

    output_path = output_dir / f"{stem}.laz"

    # Write
    output_las.write(output_path)

    if verbose:
        print(f"  Saved: {output_path.name}")

    return output_path


def process_survey_list(
    survey_list_path: Union[str, Path],
    output_dir: Union[str, Path],
    voxel_size: float = 0.5,
    radius_small: float = 0.5,
    radius_large: float = 2.0,
    skip_existing: bool = True,
    subsample_only: bool = False,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Process multiple surveys from a list file.

    Parameters
    ----------
    survey_list_path : str or Path
        Path to text file with one survey path per line.
    output_dir : str or Path
        Output directory for subsampled files.
    voxel_size : float
        Voxel size for subsampling.
    radius_small : float
        Radius for small-scale roughness.
    radius_large : float
        Radius for large-scale roughness.
    skip_existing : bool
        Skip surveys that already have output files.
    verbose : bool
        Print progress.

    Returns
    -------
    Tuple[int, int]
        (n_processed, n_failed) counts.
    """
    survey_list_path = Path(survey_list_path)
    output_dir = Path(output_dir)

    # Read survey paths
    with open(survey_list_path) as f:
        survey_paths = [line.strip() for line in f if line.strip()]

    if verbose:
        print(f"Found {len(survey_paths)} surveys to process")

    n_processed = 0
    n_failed = 0
    n_skipped = 0

    for i, survey_path in enumerate(survey_paths):
        survey_path = Path(survey_path)

        # Check if output exists
        if skip_existing:
            stem = survey_path.stem
            if subsample_only:
                suffix = "_subsampled"
            else:
                suffix = "_subsampled_features"
            if "_noveg" in stem:
                stem = stem.replace("_noveg", suffix)
            else:
                stem = stem + suffix
            output_path = output_dir / f"{stem}.laz"
            if output_path.exists():
                if verbose:
                    print(f"[{i+1}/{len(survey_paths)}] Skipping (exists): {survey_path.name}")
                n_skipped += 1
                continue

        if verbose:
            print(f"\n[{i+1}/{len(survey_paths)}] Processing: {survey_path.name}")

        result = process_survey(
            survey_path,
            output_dir,
            voxel_size=voxel_size,
            radius_small=radius_small,
            radius_large=radius_large,
            subsample_only=subsample_only,
            verbose=verbose,
        )

        if result is not None:
            n_processed += 1
        else:
            n_failed += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing complete:")
        print(f"  Processed: {n_processed}")
        print(f"  Skipped: {n_skipped}")
        print(f"  Failed: {n_failed}")

    return n_processed, n_failed
