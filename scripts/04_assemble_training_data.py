#!/usr/bin/env python3
"""
Step 4: Assemble Training Data (Case-Control)

Joins polygon features with event labels to create a training dataset.

- Cases: Polygon-zones where rockfall events occurred (within 1 year after survey)
- Controls: Polygon-zones with no subsequent events

Usage:
    python scripts/04_assemble_training_data.py \
        --features data/polygon_features.csv \
        --surveys data/test_pre_event_surveys.csv \
        --output data/training_data.csv

Options:
    --min-volume    Minimum event volume to include (default: 5.0 m³)
    --control-ratio Ratio of controls to cases (default: 1.0 = balanced)
    --no-balance    Don't balance controls with cases (use all data)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pc_rai.ml.training_data import (
    assemble_training_data,
    get_feature_columns,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble training data from polygon features and event labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to polygon features CSV",
    )

    parser.add_argument(
        "--surveys",
        type=Path,
        required=True,
        help="Path to pre-event surveys CSV",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for training data CSV",
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=5.0,
        help="Minimum event volume in m³ (default: 5.0)",
    )

    parser.add_argument(
        "--control-ratio",
        type=float,
        default=1.0,
        help="Ratio of controls to cases (default: 1.0 = equal)",
    )

    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance controls with cases (use all data)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
    if not args.features.exists():
        print(f"Error: Features file not found: {args.features}", file=sys.stderr)
        return 1

    if not args.surveys.exists():
        print(f"Error: Surveys file not found: {args.surveys}", file=sys.stderr)
        return 1

    try:
        df = assemble_training_data(
            features_path=args.features,
            surveys_path=args.surveys,
            output_path=args.output,
            min_volume=args.min_volume,
            control_ratio=args.control_ratio,
            balance=not args.no_balance,
            random_state=args.seed,
            verbose=verbose,
        )

        # Print feature summary
        if verbose:
            feature_cols = get_feature_columns(df)
            print(f"\nFeature columns ({len(feature_cols)}):")
            for col in feature_cols[:10]:
                print(f"  - {col}")
            if len(feature_cols) > 10:
                print(f"  ... and {len(feature_cols) - 10} more")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
