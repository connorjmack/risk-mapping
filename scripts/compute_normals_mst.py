#!/usr/bin/env python3
"""
Compute normals for LAS files using CloudComPy with MST orientation.

This script processes LAS/LAZ files in a directory, computes surface normals,
orients them using Minimum Spanning Tree (MST), and saves the results.

Requirements:
    - CloudComPy311 must be installed and the conda environment activated
    - Run with: . /path/to/CloudComPy311/bin/condaCloud.zsh activate cloud-compy

Usage:
    python compute_normals_mst.py <input_dir> [--output-dir <output_dir>] [--radius <radius>]

Example:
    python compute_normals_mst.py tests/test_data --output-dir output/normals --radius 0.1
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import cloudComPy as cc
except ImportError as e:
    print("Error: CloudComPy not found.", file=sys.stderr)
    print("Make sure you have activated the CloudComPy conda environment:", file=sys.stderr)
    print("  . /path/to/CloudComPy311/bin/condaCloud.zsh activate cloud-compy", file=sys.stderr)
    sys.exit(1)

import numpy as np


def _orient_normals_west(cloud) -> int:
    """
    Flip normals to prefer pointing west (-X direction).

    After MST orientation, some normals may point east (+X) instead of west (-X).
    This function flips those normals to ensure consistent westward orientation,
    which is typical for rock face scans where the scanner is positioned west of the cliff.

    Parameters
    ----------
    cloud : ccPointCloud
        Point cloud with computed normals.

    Returns
    -------
    int
        Number of normals that were flipped.
    """
    n_points = cloud.size()

    # Get normals as numpy array (N, 3)
    normals = cloud.normalsToNpArrayCopy()

    # Find normals pointing east (positive X)
    east_facing = normals[:, 0] > 0
    flipped_count = int(east_facing.sum())

    # Flip those normals (negate all components)
    normals[east_facing] *= -1

    # Set normals back to cloud
    cloud.normalsFromNpArrayCopy(normals)

    print(f"    Flipped {flipped_count:,} of {n_points:,} normals ({100*flipped_count/n_points:.1f}%)")
    return flipped_count


def compute_normals_mst(
    input_path: Path,
    output_path: Path,
    radius: Optional[float] = 1.0,
    mst_neighbors: int = 12,
    prefer_west: bool = True,
) -> bool:
    """
    Compute normals for a point cloud using CloudComPy with MST orientation.

    Parameters
    ----------
    input_path : Path
        Input LAS/LAZ file path.
    output_path : Path
        Output LAS/LAZ file path.
    radius : float, optional
        Local radius for normal estimation. If None, CloudComPy auto-determines.
    mst_neighbors : int
        Number of neighbors for MST orientation (default: 12).
    prefer_west : bool
        If True, flip normals to prefer pointing west (-X direction).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    print(f"Processing: {input_path.name}")

    # Load point cloud
    cloud = cc.loadPointCloud(str(input_path))
    if cloud is None:
        print(f"  Error: Failed to load {input_path}", file=sys.stderr)
        return False

    n_points = cloud.size()
    print(f"  Loaded {n_points:,} points")

    # Check if normals already exist
    if cloud.hasNormals():
        print("  Warning: Cloud already has normals, recomputing...")
        cloud.unallocateNorms()

    # Compute normals
    print("  Computing normals...")
    if radius is not None:
        # Use specified radius
        cc.computeNormals([cloud], model=cc.LOCAL_MODEL_TYPES.LS, defaultRadius=radius)
    else:
        # Auto-determine radius
        cc.computeNormals([cloud])

    if not cloud.hasNormals():
        print(f"  Error: Failed to compute normals for {input_path}", file=sys.stderr)
        return False

    # Orient normals using MST
    print(f"  Orienting normals with MST (neighbors={mst_neighbors})...")
    success = cloud.orientNormalsWithMST(mst_neighbors)
    if not success:
        print(f"  Warning: MST orientation may have failed for {input_path}", file=sys.stderr)

    # Flip normals to prefer west direction (-X)
    if prefer_west:
        print("  Orienting normals toward west (-X)...")
        _orient_normals_west(cloud)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save result
    print(f"  Saving to: {output_path}")
    ret = cc.SavePointCloud(cloud, str(output_path))

    if ret != cc.CC_FILE_ERROR.CC_FERR_NO_ERROR:
        print(f"  Error: Failed to save {output_path} (error code: {ret})", file=sys.stderr)
        return False

    print("  Done!")
    return True


def _has_normals_on_disk(path: Path) -> bool:
    """Check if a LAS/LAZ file already has normals without fully loading it."""
    cloud = cc.loadPointCloud(str(path))
    if cloud is None:
        return False
    return cloud.hasNormals()


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    radius: Optional[float] = None,
    mst_neighbors: int = 12,
    suffix: str = "_normals",
    prefer_west: bool = True,
    in_place: bool = False,
) -> tuple[int, int]:
    """
    Process all LAS/LAZ files in a directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing LAS/LAZ files.
    output_dir : Path, optional
        Output directory. If None, saves alongside input files with suffix.
    radius : float, optional
        Local radius for normal estimation.
    mst_neighbors : int
        Number of neighbors for MST orientation.
    suffix : str
        Suffix to add to output filenames.
    prefer_west : bool
        If True, flip normals to prefer pointing west (-X direction).
    in_place : bool
        If True, overwrite input files with normals added. Skips files
        that already contain normals.

    Returns
    -------
    tuple[int, int]
        (success_count, failure_count)
    """
    # Find all LAS/LAZ files
    las_files = list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz"))
    las_files = sorted(las_files)

    if not las_files:
        print(f"No LAS/LAZ files found in {input_dir}", file=sys.stderr)
        return 0, 0

    print(f"Found {len(las_files)} LAS/LAZ files in {input_dir}")
    if in_place:
        print("Mode: in-place (overwriting input files)")
    print("-" * 60)

    success_count = 0
    failure_count = 0
    skipped_count = 0

    for i, input_path in enumerate(las_files):
        try:
            if in_place:
                # Skip files that already have normals
                try:
                    already_done = _has_normals_on_disk(input_path)
                except (OSError, TimeoutError) as e:
                    print(f"[{i+1}/{len(las_files)}] SKIP (I/O error checking normals): {input_path.name}: {e}")
                    print()
                    failure_count += 1
                    continue

                if already_done:
                    print(f"[{i+1}/{len(las_files)}] Skipping (already has normals): {input_path.name}")
                    print()
                    skipped_count += 1
                    success_count += 1
                    continue

                # Write to temp file, then replace original
                tmp_path = input_path.with_suffix(".tmp.laz")
                print(f"[{i+1}/{len(las_files)}]", end=" ")
                if compute_normals_mst(input_path, tmp_path, radius, mst_neighbors, prefer_west):
                    tmp_path.replace(input_path)
                    success_count += 1
                else:
                    # Clean up temp file on failure
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except (OSError, TimeoutError):
                        print(f"  Warning: could not clean up {tmp_path.name}")
                    failure_count += 1
            else:
                # Original behavior: write to output dir with suffix
                if output_dir is not None:
                    output_path = output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
                else:
                    output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"

                # Skip if output already exists
                if output_path.exists():
                    print(f"[{i+1}/{len(las_files)}] Skipping {input_path.name}: output already exists")
                    print()
                    success_count += 1
                    continue

                print(f"[{i+1}/{len(las_files)}]", end=" ")
                if compute_normals_mst(input_path, output_path, radius, mst_neighbors, prefer_west):
                    success_count += 1
                else:
                    failure_count += 1

        except (OSError, TimeoutError) as e:
            print(f"\n  I/O ERROR on {input_path.name}: {e}")
            print(f"  Skipping file and continuing...")
            failure_count += 1

        print()

    if skipped_count > 0:
        print(f"Skipped {skipped_count} files that already had normals")

    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Compute normals for LAS files using CloudComPy with MST orientation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tests/test_data
  %(prog)s tests/test_data --output-dir output/normals
  %(prog)s tests/test_data --radius 0.1 --mst-neighbors 10

Note: CloudComPy conda environment must be activated before running this script.
        """,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing LAS/LAZ files to process",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input with '_normals' suffix)",
    )
    parser.add_argument(
        "-r", "--radius",
        type=float,
        default=1.0,
        help="Local radius for normal estimation (default: 1.0 meter)",
    )
    parser.add_argument(
        "-n", "--mst-neighbors",
        type=int,
        default=12,
        help="Number of neighbors for MST orientation (default: 12)",
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str,
        default="_normals",
        help="Suffix for output filenames (default: '_normals')",
    )
    parser.add_argument(
        "--no-west",
        action="store_true",
        help="Disable westward (-X) normal orientation (default: normals biased west)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files with normals added (skips files that already have normals)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Process files
    success, failure = process_directory(
        args.input_dir,
        args.output_dir,
        args.radius,
        args.mst_neighbors,
        args.suffix,
        prefer_west=not args.no_west,
        in_place=args.in_place,
    )

    # Summary
    print("=" * 60)
    print(f"Summary: {success} succeeded, {failure} failed out of {success + failure} files")

    sys.exit(0 if failure == 0 else 1)


if __name__ == "__main__":
    main()
