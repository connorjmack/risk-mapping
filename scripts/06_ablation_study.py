#!/usr/bin/env python3
"""
Step 6: Cumulative Feature Ablation Study

Trains Random Forest models with progressively more feature groups to show
the marginal contribution of each group to predictive performance.

Outputs:
    - ablation_results.csv: Metrics at each cumulative step
    - ablation_curve.png: Manuscript-ready figure

Usage:
    python scripts/06_ablation_study.py \
        --input data/training_data.csv \
        --output output/ablation/

    # With GroupKFold cross-validation by location
    python scripts/06_ablation_study.py \
        --input data/training_data.csv \
        --output output/ablation/ \
        --group-by location

    # Fewer CV folds for faster iteration
    python scripts/06_ablation_study.py \
        --input data/training_data.csv \
        --output output/ablation/ \
        --cv-splits 3
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from pc_rai.ml.ablation import run_ablation, plot_ablation, DEFAULT_FEATURE_ORDER
from pc_rai.ml.config import MLConfig
from pc_rai.ml.training_data import get_feature_columns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run cumulative feature ablation study for rockfall prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to training data CSV (from Step 4)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for results (CSV + figure)",
    )

    # Model hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest (default: 100)",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features with StandardScaler before training",
    )

    # Cross-validation
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Column for GroupKFold CV (e.g., 'location'). "
             "If not set, uses StratifiedKFold.",
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
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
    if not args.input.exists():
        print(f"Error: Training data not found: {args.input}", file=sys.stderr)
        return 1

    try:
        # Load training data
        df = pd.read_csv(args.input)

        if verbose:
            print(f"Loaded training data: {args.input}")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")

        # Get feature columns
        feature_cols = get_feature_columns(df)
        if len(feature_cols) == 0:
            print("Error: No feature columns found in training data", file=sys.stderr)
            return 1

        if verbose:
            print(f"  Features: {len(feature_cols)}")
            print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

        # Groups for cross-validation
        groups = None
        if args.group_by:
            if args.group_by not in df.columns:
                print(
                    f"Error: Group column '{args.group_by}' not found. "
                    f"Available: {list(df.columns[:10])}...",
                    file=sys.stderr,
                )
                return 1
            groups = df[args.group_by].values
            if verbose:
                unique_groups = np.unique(groups)
                print(f"  Groups ({args.group_by}): {list(unique_groups)}")

        # Configure model
        config = MLConfig(
            n_estimators=args.n_estimators,
            cv_n_splits=args.cv_splits,
        )

        # Run ablation
        results = run_ablation(
            df,
            config=config,
            groups=groups,
            normalize=args.normalize,
            verbose=verbose,
        )

        # Save results
        args.output.mkdir(parents=True, exist_ok=True)

        csv_path = args.output / "ablation_results.csv"
        results.to_dataframe().to_csv(csv_path, index=False)

        fig_path = args.output / "ablation_curve.png"
        plot_ablation(results, fig_path)

        if verbose:
            print(f"\nResults saved:")
            print(f"  CSV: {csv_path}")
            print(f"  Figure: {fig_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
