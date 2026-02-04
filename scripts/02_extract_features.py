#!/usr/bin/env python3
"""
Step 2: Subsample & Extract Point-Level Features

Subsamples point clouds to a voxel grid and computes per-point features
(slope, roughness, height) for subsequent polygon aggregation.

Usage:
    python scripts/02_extract_features.py \
        --survey-list data/test_pre_event_surveys.survey_list.txt \
        --output-dir data/test_subsampled/ \
        --voxel-size 0.5

Or process a single file:
    python scripts/02_extract_features.py \
        --input data/test_data/no_veg/20181121_...noveg.las \
        --output-dir data/test_subsampled/ \
        --voxel-size 0.5

Or process all files in a directory:
    python scripts/02_extract_features.py \
        --input-dir data/test_subsampled_normals/ \
        --output-dir data/test_features/ \
        --voxel-size 0.5

Output:
    LAZ files with extra dimensions:
    - NormalX, NormalY, NormalZ (preserved from input)
    - slope (degrees, 0=horizontal, 90=vertical)
    - roughness_small (std dev of slope at small scale)
    - roughness_large (std dev of slope at large scale)
    - roughness_ratio (small/large)
    - height (Z relative to local minimum)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pc_rai.ml.feature_extraction import (
    process_survey,
    process_survey_list,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Subsample point clouds and extract features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--survey-list",
        type=Path,
        help="Path to text file with survey paths (one per line)",
    )
    input_group.add_argument(
        "--input",
        type=Path,
        help="Path to single LAS/LAZ file to process",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Path to directory containing LAS/LAZ files to process",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for subsampled files",
    )

    # Processing parameters
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
        help="Voxel size for subsampling in meters (default: 0.5)",
    )

    parser.add_argument(
        "--radius-small",
        type=float,
        default=0.5,
        help="Radius for small-scale roughness in meters (default: 0.5)",
    )

    parser.add_argument(
        "--radius-large",
        type=float,
        default=2.0,
        help="Radius for large-scale roughness in meters (default: 2.0)",
    )

    parser.add_argument(
        "--height-radius",
        type=float,
        default=5.0,
        help="Horizontal radius for relative height in meters (default: 5.0)",
    )

    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors for roughness calculation (default: 5)",
    )

    parser.add_argument(
        "--normal-k",
        type=int,
        default=30,
        help="Neighbors for PCA normal estimation (default: 30)",
    )

    parser.add_argument(
        "--no-compute-normals",
        action="store_true",
        help="Don't compute normals if missing (fail instead)",
    )

    parser.add_argument(
        "--subsample-only",
        action="store_true",
        help="Only subsample, don't compute normals or features (for external normal computation)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess files even if output exists",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
        verbose = False
    elif args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )
        verbose = True
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )
        verbose = True

    # Validate inputs
    if args.survey_list and not args.survey_list.exists():
        print(f"Error: Survey list not found: {args.survey_list}", file=sys.stderr)
        return 1

    if args.input and not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.input_dir and not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    try:
        if args.input:
            # Process single file
            result = process_survey(
                args.input,
                args.output_dir,
                voxel_size=args.voxel_size,
                radius_small=args.radius_small,
                radius_large=args.radius_large,
                height_radius=args.height_radius,
                min_neighbors=args.min_neighbors,
                normal_k=args.normal_k,
                compute_normals_if_missing=not args.no_compute_normals,
                subsample_only=args.subsample_only,
                verbose=verbose,
            )

            if result is None:
                print("Error: Processing failed", file=sys.stderr)
                return 1

            if verbose:
                print(f"\nOutput: {result}")

        elif args.input_dir:
            # Process all LAS/LAZ files in directory
            las_files = sorted(
                list(args.input_dir.glob("*.las")) + list(args.input_dir.glob("*.laz"))
            )

            if not las_files:
                print(f"Error: No LAS/LAZ files found in {args.input_dir}", file=sys.stderr)
                return 1

            if verbose:
                print(f"Found {len(las_files)} LAS/LAZ files in {args.input_dir}")

            n_processed = 0
            n_failed = 0

            for i, las_file in enumerate(las_files):
                if verbose:
                    print(f"\n[{i+1}/{len(las_files)}] {las_file.name}")

                result = process_survey(
                    las_file,
                    args.output_dir,
                    voxel_size=args.voxel_size,
                    radius_small=args.radius_small,
                    radius_large=args.radius_large,
                    height_radius=args.height_radius,
                    min_neighbors=args.min_neighbors,
                    normal_k=args.normal_k,
                    compute_normals_if_missing=not args.no_compute_normals,
                    subsample_only=args.subsample_only,
                    verbose=verbose,
                )

                if result is not None:
                    n_processed += 1
                else:
                    n_failed += 1

            if verbose:
                print(f"\n{'='*60}")
                print(f"Processed: {n_processed}, Failed: {n_failed}")

            if n_failed > 0:
                print(f"\nWarning: {n_failed} files failed to process", file=sys.stderr)
                return 1 if n_processed == 0 else 0

        else:
            # Process survey list
            n_processed, n_failed = process_survey_list(
                args.survey_list,
                args.output_dir,
                voxel_size=args.voxel_size,
                radius_small=args.radius_small,
                radius_large=args.radius_large,
                skip_existing=not args.no_skip_existing,
                subsample_only=args.subsample_only,
                verbose=verbose,
            )

            if n_failed > 0:
                print(f"\nWarning: {n_failed} files failed to process", file=sys.stderr)
                return 1 if n_processed == 0 else 0

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
