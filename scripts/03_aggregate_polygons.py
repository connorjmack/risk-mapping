#!/usr/bin/env python3
"""
Step 3: Aggregate Point Features to Polygon-Zone Level

Aggregates point-level features from LAZ files into polygon-zones using
shapefiles for spatial matching. Each polygon is split by relative elevation
zones (lower/middle/upper cliff).

Usage:
    python scripts/03_aggregate_polygons.py \
        --input-dir data/test_features/ \
        --shapefile-dir utiliies/polygons_1m/ \
        --output data/polygon_features.csv

Output:
    CSV with one row per polygon-zone-survey containing:
    - Survey metadata (date, file, location)
    - Polygon metadata (polygon_id, alongshore_m, zone, elevation stats)
    - Aggregated features (mean, std, min, max, p10, p50, p90 for each feature)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pc_rai.ml.polygon_aggregation import aggregate_survey, aggregate_survey_batch, extract_location


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate point features to polygon-zone level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing LAZ files with features",
    )
    input_group.add_argument(
        "--input",
        type=Path,
        help="Single LAZ file to process",
    )

    parser.add_argument(
        "--shapefile-dir",
        type=Path,
        default=Path("utiliies/polygons_1m"),
        help="Directory containing polygon shapefiles (default: utiliies/polygons_1m/)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output CSV path",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum points per polygon-zone (default: 5)",
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
    if args.input_dir and not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    if args.input and not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.shapefile_dir.exists():
        print(f"Error: Shapefile directory not found: {args.shapefile_dir}", file=sys.stderr)
        return 1

    try:
        if args.input:
            # Process single file
            filename = args.input.stem
            location = extract_location(filename)

            if location == "Unknown":
                print(f"Error: Could not determine location from filename: {filename}", file=sys.stderr)
                return 1

            df = aggregate_survey(
                args.input,
                shapefile_dir=args.shapefile_dir,
                location=location,
                min_points_per_zone=args.min_points,
                verbose=verbose,
            )

            if df is None:
                print("Error: Aggregation failed", file=sys.stderr)
                return 1

            # Add survey metadata
            df["survey_date"] = filename[:8]
            df["survey_file"] = args.input.name
            df["location"] = location

            # Save
            args.output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output, index=False)

            if verbose:
                print(f"\nSaved: {args.output}")
                print(f"  Rows: {len(df)}")

        else:
            # Process directory
            df = aggregate_survey_batch(
                args.input_dir,
                args.output,
                shapefile_dir=args.shapefile_dir,
                min_points_per_zone=args.min_points,
                verbose=verbose,
            )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
