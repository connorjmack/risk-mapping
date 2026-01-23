#!/usr/bin/env python3
"""
Compute normals for LAS files using CloudComPy with MST orientation.

This script processes LAS/LAZ files in a directory, computes surface normals,
orients them using Minimum Spanning Tree (MST), and saves the results.

Requirements:
    - CloudComPy311 must be installed and the conda environment activated
    - Run with: . /path/to/CloudComPy311/bin/condaCloud.zsh activate CloudComPy311

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
    print("  . /path/to/CloudComPy311/bin/condaCloud.zsh activate CloudComPy311", file=sys.stderr)
    sys.exit(1)


def compute_normals_mst(
    input_path: Path,
    output_path: Path,
    radius: Optional[float] = None,
    mst_neighbors: int = 6,
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
        Number of neighbors for MST orientation (default: 6).

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


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    radius: Optional[float] = None,
    mst_neighbors: int = 6,
    suffix: str = "_normals",
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
    print("-" * 60)

    success_count = 0
    failure_count = 0

    for input_path in las_files:
        # Determine output path
        if output_dir is not None:
            output_path = output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
        else:
            output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"

        # Process file
        if compute_normals_mst(input_path, output_path, radius, mst_neighbors):
            success_count += 1
        else:
            failure_count += 1

        print()

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
        default=None,
        help="Local radius for normal estimation (default: auto-determined)",
    )
    parser.add_argument(
        "-n", "--mst-neighbors",
        type=int,
        default=6,
        help="Number of neighbors for MST orientation (default: 6)",
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str,
        default="_normals",
        help="Suffix for output filenames (default: '_normals')",
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
    )

    # Summary
    print("=" * 60)
    print(f"Summary: {success} succeeded, {failure} failed out of {success + failure} files")

    sys.exit(0 if failure == 0 else 1)


if __name__ == "__main__":
    main()
