#!/usr/bin/env python3
"""
Compute per-point features on subsampled LAZ files (in-place).

Reads each LAZ file in a directory (must already have normals), computes
slope, roughness, relative height, and eigenvalue features, then overwrites
the file with the new dimensions added.

Features computed (6 base features after ablation-informed selection):
    - slope:           angle from vertical (degrees, from normals)
    - roughness_small: std dev of slope at 1.0m radius
    - roughness_large: std dev of slope at 2.5m radius
    - height:          Z relative to local minimum (5.0m horizontal radius)
    - linearity:       (λ1 - λ2) / λ1 — edges/joints (biggest cross-site gain)
    - curvature:       λ3 / (λ1 + λ2 + λ3) — surface variation

Dropped from original 9 (ablation study showed negligible/negative contribution):
    - roughness_ratio: derived from small/large, RF learns implicitly, slightly hurts
    - sphericity:      mathematically redundant (planarity + linearity + sphericity ≈ 1)
    - planarity:       redundant once linearity included (+0.002 marginal)

Usage:
    # Activate the project venv (needs laspy, numpy, scipy)
    python scripts/compute_features.py data/laz_files/ -v

    # Process mounted network drive
    python scripts/compute_features.py /Volumes/group/.../data/laz_files/ -v
"""

import argparse
import sys
import time
from pathlib import Path

# Direct import to avoid pc_rai.ml.__init__ pulling in shapefile/sklearn
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "feature_extraction",
    Path(__file__).parent.parent / "pc_rai" / "ml" / "feature_extraction.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_slope = _mod.compute_slope
compute_roughness = _mod.compute_roughness
compute_relative_height = _mod.compute_relative_height

import laspy
import numpy as np
from scipy.spatial import cKDTree

# Features that will be written to each LAZ file
FEATURE_NAMES = [
    "slope",
    "roughness_small",
    "roughness_large",
    "height",
    "linearity",
    "curvature",
]


def _get_dims(las_path: Path) -> set:
    """Get dimension names from a LAZ file header."""
    try:
        with laspy.open(las_path) as f:
            return set(f.header.point_format.dimension_names)
    except Exception:
        return set()


def has_features(las_path: Path) -> bool:
    """Check if a LAZ file already has all feature dimensions."""
    dims = _get_dims(las_path)
    return all(name in dims for name in FEATURE_NAMES)


def has_normals(las_path: Path) -> bool:
    """Check if a LAZ file has normal vectors."""
    dims = _get_dims(las_path)
    return (
        all(n in dims for n in ["NormalX", "NormalY", "NormalZ"])
        or all(n in dims for n in ["normalx", "normaly", "normalz"])
    )


def compute_eigenvalue_features_slim(
    xyz: np.ndarray,
    radius: float = 2.5,
    min_neighbors: int = 5,
    verbose: bool = True,
) -> dict:
    """Compute linearity and curvature from local PCA eigenvalues.

    Only computes the two features retained after ablation study
    (linearity and curvature). Skips planarity and sphericity.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    radius : float
        Search radius for local neighborhood.
    min_neighbors : int
        Minimum neighbors for valid computation.
    verbose : bool
        Print progress.

    Returns
    -------
    features : dict
        Dictionary with 'linearity' and 'curvature' arrays.
    """
    n_points = len(xyz)
    linearity = np.full(n_points, np.nan, dtype=np.float32)
    curvature = np.full(n_points, np.nan, dtype=np.float32)

    if verbose:
        print(f"    Computing eigenvalue features (r={radius}m)...")

    tree = cKDTree(xyz)
    neighbors_list = tree.query_ball_tree(tree, radius)

    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) < min_neighbors:
            continue

        pts = xyz[neighbors]
        centered = pts - pts.mean(axis=0)
        cov = np.dot(centered.T, centered) / len(neighbors)
        eigenvalues, _ = np.linalg.eigh(cov)

        # Sort descending: λ1 >= λ2 >= λ3
        eigenvalues = eigenvalues[::-1]
        l1, l2, l3 = eigenvalues

        if l1 < 1e-10:
            continue

        linearity[i] = (l1 - l2) / l1
        sum_eig = l1 + l2 + l3
        curvature[i] = l3 / sum_eig if sum_eig > 1e-10 else np.nan

        if verbose and (i + 1) % 100000 == 0:
            print(f"      {i+1:,}/{n_points:,} points...")

    if verbose:
        valid = int(np.sum(~np.isnan(linearity)))
        print(f"    Valid eigenvalue features: {valid:,}/{n_points:,}")

    return {"linearity": linearity, "curvature": curvature}


def compute_features_for_file(
    las_path: Path,
    radius_small: float = 1.0,
    radius_large: float = 2.5,
    height_radius: float = 5.0,
    min_neighbors: int = 5,
    verbose: bool = True,
) -> bool:
    """Compute features for a single LAZ file and overwrite in-place.

    Parameters
    ----------
    las_path : Path
        Path to LAZ file with normals.
    radius_small : float
        Radius for small-scale roughness.
    radius_large : float
        Radius for large-scale roughness.
    height_radius : float
        Horizontal radius for relative height.
    min_neighbors : int
        Minimum neighbors for roughness/eigenvalue computation.
    verbose : bool
        Print progress.

    Returns
    -------
    bool
        True if successful.
    """
    if verbose:
        print(f"  Loading {las_path.name}...")

    las = laspy.read(las_path)
    xyz = np.column_stack([las.x, las.y, las.z])
    n_points = len(xyz)

    if verbose:
        print(f"    {n_points:,} points")

    # Get normals
    normal_names = ["NormalX", "NormalY", "NormalZ"]
    dims = set(las.point_format.dimension_names)

    if not all(n in dims for n in normal_names):
        # Try lowercase
        normal_names = ["normalx", "normaly", "normalz"]
        if not all(n in dims for n in normal_names):
            print(f"    ERROR: No normals found in {las_path.name}")
            return False

    normals = np.column_stack([
        las[normal_names[0]],
        las[normal_names[1]],
        las[normal_names[2]],
    ]).astype(np.float32)

    # Normalize (some may not be unit length)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # 1. Slope
    if verbose:
        print(f"    Computing slope...")
    slope = compute_slope(normals)

    # 2. Roughness (small + large)
    if verbose:
        print(f"    Computing roughness (r={radius_small}m)...")
    roughness_small = compute_roughness(slope, xyz, radius_small, min_neighbors)

    if verbose:
        print(f"    Computing roughness (r={radius_large}m)...")
    roughness_large = compute_roughness(slope, xyz, radius_large, min_neighbors)

    # 3. Relative height
    if verbose:
        print(f"    Computing relative height (r={height_radius}m)...")
    height = compute_relative_height(xyz, height_radius)

    # 4. Eigenvalue features (linearity + curvature only)
    eigen = compute_eigenvalue_features_slim(
        xyz, radius=radius_large, min_neighbors=min_neighbors, verbose=verbose
    )

    # Build new LAS with features added
    output_las = laspy.create(
        point_format=las.point_format, file_version=las.header.version
    )

    # Copy all existing dimensions
    for dim in las.point_format.dimension_names:
        output_las[dim] = las[dim]

    # Copy header spatial info
    output_las.header.offsets = las.header.offsets
    output_las.header.scales = las.header.scales

    # Add feature dimensions
    features = {
        "slope": slope,
        "roughness_small": roughness_small,
        "roughness_large": roughness_large,
        "height": height,
        "linearity": eigen["linearity"],
        "curvature": eigen["curvature"],
    }

    for name, data in features.items():
        if name not in dims:
            output_las.add_extra_dim(
                laspy.ExtraBytesParams(name=name, type=np.float32)
            )
        output_las[name] = data.astype(np.float32)

    # Write to temp, then replace
    tmp_path = las_path.with_suffix(".tmp.laz")
    output_las.write(tmp_path)
    tmp_path.replace(las_path)

    if verbose:
        print(f"    Saved (in-place): {las_path.name}")

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-point features on subsampled LAZ files (in-place)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing LAZ files with normals",
    )
    parser.add_argument(
        "--radius-small",
        type=float,
        default=1.0,
        help="Radius for small-scale roughness (default: 1.0m)",
    )
    parser.add_argument(
        "--radius-large",
        type=float,
        default=2.5,
        help="Radius for large-scale roughness (default: 2.5m)",
    )
    parser.add_argument(
        "--height-radius",
        type=float,
        default=5.0,
        help="Horizontal radius for relative height (default: 5.0m)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors for roughness/eigenvalue (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    # Find LAZ files
    laz_files = sorted(args.input_dir.glob("*.laz"))
    if not laz_files:
        print(f"Error: No LAZ files found in {args.input_dir}", file=sys.stderr)
        return 1

    # Triage files: already done, ready (has normals), not ready (no normals)
    done = []
    ready = []
    no_normals = []
    for f in laz_files:
        if has_features(f):
            done.append(f)
        elif has_normals(f):
            ready.append(f)
        else:
            no_normals.append(f)

    print(f"Found {len(laz_files)} LAZ files in {args.input_dir}")
    print(f"Features to compute: {', '.join(FEATURE_NAMES)}")
    print(f"  Already have features: {len(done)}")
    print(f"  Ready (have normals):  {len(ready)}")
    print(f"  Waiting for normals:   {len(no_normals)}")
    print("-" * 60)

    if not ready:
        if no_normals:
            print("Nothing ready yet — all remaining files are waiting for normals.")
            print("Re-run after more normals are computed.")
        else:
            print("All files already have features. Nothing to do.")
        return 0

    n_processed = 0
    n_failed = 0
    failed_files = []
    t_start = time.time()

    for i, laz_path in enumerate(ready):
        print(f"\n[{i+1}/{len(ready)}] {laz_path.name}")

        try:
            ok = compute_features_for_file(
                laz_path,
                radius_small=args.radius_small,
                radius_large=args.radius_large,
                height_radius=args.height_radius,
                min_neighbors=args.min_neighbors,
                verbose=args.verbose,
            )
            if ok:
                n_processed += 1
            else:
                n_failed += 1
                failed_files.append((laz_path.name, "returned False"))
        except (OSError, TimeoutError) as e:
            print(f"    I/O ERROR: {e}")
            print(f"    Skipping and continuing...")
            n_failed += 1
            failed_files.append((laz_path.name, str(e)))
            try:
                tmp = laz_path.with_suffix(".tmp.laz")
                if tmp.exists():
                    tmp.unlink()
            except (OSError, TimeoutError):
                pass
        except Exception as e:
            print(f"    ERROR: {e}")
            n_failed += 1
            failed_files.append((laz_path.name, str(e)))
            try:
                tmp = laz_path.with_suffix(".tmp.laz")
                if tmp.exists():
                    tmp.unlink()
            except (OSError, TimeoutError):
                pass

        # Progress every 25 files
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(ready) - i - 1) / rate
            print(f"\n  --- Progress: {i+1}/{len(ready)} "
                  f"({n_processed} ok, {n_failed} failed) "
                  f"~{remaining/60:.0f} min remaining ---")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"  Processed: {n_processed}")
    print(f"  Already had features: {len(done)}")
    print(f"  Waiting for normals: {len(no_normals)}")
    print(f"  Failed: {n_failed}")

    if no_normals:
        print(f"\nRe-run after more normals are computed to process remaining {len(no_normals)} files.")

    if failed_files:
        print(f"\nFailed files:")
        for fname, reason in failed_files:
            print(f"  {fname}: {reason}")

    return 1 if n_processed == 0 and len(to_process) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
