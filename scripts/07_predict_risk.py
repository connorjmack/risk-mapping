#!/usr/bin/env python3
"""
Step 7: Generate Risk Predictions and Maps

Loads trained Random Forest model and generates risk predictions for polygon-zones.
Visualizes spatial patterns to validate model behavior and identify high-risk areas.

Outputs:
    - risk_predictions.csv: Polygon-level risk scores (0-1 probability)
    - risk_map_<Location>.png: 2D heatmap per beach (alongshore × zone)
    - risk_summary.txt: Statistics per location

Usage:
    # Predict on all surveys
    python scripts/07_predict_risk.py \
        --model models/rf_model.joblib \
        --features data/polygon_features.csv \
        --output output/risk_maps/

    # Use most recent survey per location only
    python scripts/07_predict_risk.py \
        --model models/rf_model.joblib \
        --features data/polygon_features.csv \
        --output output/risk_maps/ \
        --recent

    # Specific location
    python scripts/07_predict_risk.py \
        --model models/rf_model.joblib \
        --features data/polygon_features.csv \
        --output output/risk_maps/ \
        --location DelMar \
        --recent
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from pc_rai.ml.training_data import get_feature_columns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate risk predictions and maps from trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.joblib file)",
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to polygon features CSV (from Step 3)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for predictions and maps",
    )

    parser.add_argument(
        "--recent",
        action="store_true",
        help="Use only most recent survey per location",
    )

    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Filter to specific location (e.g., 'DelMar')",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def get_recent_surveys(df, verbose=False):
    """Get most recent survey per location.

    Parameters
    ----------
    df : pd.DataFrame
        Polygon features with 'location' and 'survey_date' columns.
    verbose : bool
        Print filtering details.

    Returns
    -------
    pd.DataFrame
        Filtered to most recent survey per location.
    """
    if verbose:
        print("\nFiltering to most recent survey per location:")

    # Group by location, get max survey_date
    recent = df.groupby('location')['survey_date'].max().reset_index()
    recent.columns = ['location', 'max_date']

    if verbose:
        for _, row in recent.iterrows():
            loc = row['location']
            date = row['max_date']
            n_before = len(df[df['location'] == loc])
            n_after = len(df[(df['location'] == loc) & (df['survey_date'] == date)])
            print(f"  {loc}: {date} ({n_after:,} polygons, {n_before:,} total surveys)")

    # Merge back and filter
    df = df.merge(recent, on='location')
    df = df[df['survey_date'] == df['max_date']]
    df = df.drop(columns=['max_date'])

    return df


def plot_risk_map_2d(df, location, output_dir, verbose=False):
    """Generate 2D risk heatmap (alongshore × zone).

    This visualization shows:
    - X-axis: Alongshore distance (spatial position along beach)
    - Y-axis: Elevation zone (lower/middle/upper cliff)
    - Color: Risk score (green=low, yellow=medium, red=high)

    Useful for validating if model is learning real patterns vs.
    spurious "upper zone = high risk" correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Polygon features with predictions for this location.
    location : str
        Location name for plot title.
    output_dir : Path
        Output directory.
    verbose : bool
        Print details.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Create pivot table: alongshore_m (rows) × zone_idx (columns) = risk_score
    # Round alongshore_m to nearest meter for cleaner grid
    df['alongshore_bin'] = df['alongshore_m'].round(0)

    pivot = df.pivot_table(
        index='alongshore_bin',
        columns='zone_idx',
        values='risk_score',
        aggfunc='mean',  # Average if multiple polygons at same location
    )

    # Zone labels
    zone_labels = ['Lower', 'Middle', 'Upper']

    fig, ax = plt.subplots(figsize=(14, 5))

    # Custom colormap: green → yellow → red
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green, yellow, red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('risk', colors, N=n_bins)

    # Plot heatmap
    im = ax.imshow(
        pivot.T,  # Transpose so zones are on Y-axis
        aspect='auto',
        cmap=cmap,
        vmin=0, vmax=1,
        interpolation='nearest',
        origin='lower',
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Risk Score (Probability of Failure)', fontsize=11)

    # Y-axis: zone labels
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(zone_labels, fontsize=10)
    ax.set_ylabel('Cliff Zone (Elevation)', fontsize=11)

    # X-axis: alongshore distance
    # Show every ~50m for readability
    n_ticks = min(10, len(pivot))
    tick_indices = np.linspace(0, len(pivot) - 1, n_ticks, dtype=int)
    tick_labels = [f"{int(pivot.index[i])}" for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel('Alongshore Distance (m)', fontsize=11)

    # Title with survey info
    survey_date = df['survey_date'].iloc[0]
    ax.set_title(
        f'Rockfall Risk Map: {location} ({survey_date})',
        fontsize=13,
        pad=10,
    )

    # Stats annotation
    mean_risk = df['risk_score'].mean()
    max_risk = df['risk_score'].max()
    high_risk_pct = (df['risk_score'] > 0.5).mean() * 100

    stats_text = (
        f"Mean: {mean_risk:.3f} | "
        f"Max: {max_risk:.3f} | "
        f">0.5: {high_risk_pct:.1f}%"
    )
    ax.text(
        0.5, -0.15, stats_text,
        transform=ax.transAxes,
        ha='center',
        fontsize=9,
        color='#555',
    )

    plt.tight_layout()
    fig_path = output_dir / f"risk_map_{location}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    if verbose:
        print(f"  Saved: {fig_path}")


def write_summary_stats(df, output_dir, verbose=False):
    """Write summary statistics to text file.

    Parameters
    ----------
    df : pd.DataFrame
        All predictions.
    output_dir : Path
        Output directory.
    verbose : bool
        Print to console.
    """
    summary_path = output_dir / "risk_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RISK PREDICTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Overall stats
        f.write(f"Total polygon-zones: {len(df):,}\n")
        f.write(f"Locations: {df['location'].nunique()}\n")
        f.write(f"Surveys: {df['survey_date'].nunique()}\n\n")

        f.write(f"Risk Score Statistics:\n")
        f.write(f"  Mean:   {df['risk_score'].mean():.4f}\n")
        f.write(f"  Median: {df['risk_score'].median():.4f}\n")
        f.write(f"  Std:    {df['risk_score'].std():.4f}\n")
        f.write(f"  Min:    {df['risk_score'].min():.4f}\n")
        f.write(f"  Max:    {df['risk_score'].max():.4f}\n\n")

        # Risk categories
        low = (df['risk_score'] < 0.3).sum()
        med = ((df['risk_score'] >= 0.3) & (df['risk_score'] < 0.7)).sum()
        high = (df['risk_score'] >= 0.7).sum()

        f.write(f"Risk Categories:\n")
        f.write(f"  Low (<0.3):      {low:6,} ({low/len(df)*100:5.1f}%)\n")
        f.write(f"  Medium (0.3-0.7): {med:6,} ({med/len(df)*100:5.1f}%)\n")
        f.write(f"  High (>=0.7):    {high:6,} ({high/len(df)*100:5.1f}%)\n\n")

        # By location
        f.write("=" * 60 + "\n")
        f.write("BY LOCATION\n")
        f.write("=" * 60 + "\n\n")

        for location in sorted(df['location'].unique()):
            loc_df = df[df['location'] == location]
            f.write(f"{location}:\n")
            f.write(f"  Polygon-zones: {len(loc_df):,}\n")
            f.write(f"  Survey date:   {loc_df['survey_date'].iloc[0]}\n")
            f.write(f"  Mean risk:     {loc_df['risk_score'].mean():.4f}\n")
            f.write(f"  Max risk:      {loc_df['risk_score'].max():.4f}\n")
            f.write(f"  High risk %:   {(loc_df['risk_score'] >= 0.7).mean()*100:.1f}%\n")
            f.write("\n")

        # By zone (validate height hypothesis)
        f.write("=" * 60 + "\n")
        f.write("BY ELEVATION ZONE (Height Hypothesis Test)\n")
        f.write("=" * 60 + "\n\n")
        f.write("If model is spuriously learning 'upper=failure', we'd see:\n")
        f.write("  Lower: low risk, Middle: medium risk, Upper: high risk\n\n")

        zone_names = {0: 'Lower', 1: 'Middle', 2: 'Upper'}
        for zone_idx in [0, 1, 2]:
            zone_df = df[df['zone_idx'] == zone_idx]
            zone_name = zone_names[zone_idx]
            f.write(f"{zone_name} Zone:\n")
            f.write(f"  Mean risk: {zone_df['risk_score'].mean():.4f}\n")
            f.write(f"  Std:       {zone_df['risk_score'].std():.4f}\n")
            f.write(f"  High %:    {(zone_df['risk_score'] >= 0.7).mean()*100:.1f}%\n")
            f.write("\n")

        f.write("=" * 60 + "\n")

    if verbose:
        print(f"\nSummary saved: {summary_path}")
        print("\nPreviewing zone statistics:")
        with open(summary_path, 'r') as f:
            lines = f.readlines()
            # Find and print zone section
            start = None
            for i, line in enumerate(lines):
                if "BY ELEVATION ZONE" in line:
                    start = i
                    break
            if start:
                print(''.join(lines[start:start+15]))


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )
        verbose = True
    else:
        logging.basicConfig(level=logging.WARNING)
        verbose = False

    # Validate inputs
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        return 1

    if not args.features.exists():
        print(f"Error: Features not found: {args.features}", file=sys.stderr)
        return 1

    try:
        # Load model
        if verbose:
            print(f"Loading model: {args.model}")
        model = joblib.load(args.model)

        # Load features
        if verbose:
            print(f"Loading features: {args.features}")
        df = pd.read_csv(args.features)

        if verbose:
            print(f"  Rows: {len(df):,}")
            print(f"  Locations: {df['location'].nunique()}")
            print(f"  Surveys: {df['survey_date'].nunique()}")

        # Filter to recent if requested
        if args.recent:
            df = get_recent_surveys(df, verbose=verbose)

        # Filter to specific location if requested
        if args.location:
            if args.location not in df['location'].values:
                print(
                    f"Error: Location '{args.location}' not found. "
                    f"Available: {sorted(df['location'].unique())}",
                    file=sys.stderr,
                )
                return 1
            df = df[df['location'] == args.location]
            if verbose:
                print(f"\nFiltered to location: {args.location} ({len(df):,} rows)")

        # Get feature columns (same as training)
        feature_cols = get_feature_columns(df)
        if len(feature_cols) == 0:
            print("Error: No feature columns found in data", file=sys.stderr)
            return 1

        if verbose:
            print(f"\nUsing {len(feature_cols)} features")

        X = df[feature_cols]

        # Make predictions
        if verbose:
            print(f"\nGenerating predictions...")

        risk_probs = model.predict_proba(X)

        # Handle both 1D and 2D output from predict_proba
        if len(risk_probs.shape) == 2:
            # Binary classifier: (n_samples, 2) -> get class 1 probabilities
            risk_probs = risk_probs[:, 1]
        # else: already 1D array of probabilities

        # Add predictions to dataframe
        df['risk_score'] = risk_probs

        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Save predictions CSV
        csv_path = args.output / "risk_predictions.csv"
        output_cols = [
            'location', 'survey_date', 'survey_file',
            'polygon_id', 'alongshore_m',
            'zone', 'zone_idx',
            'n_points', 'z_min', 'z_max', 'z_mean',
            'risk_score',
        ]
        df[output_cols].to_csv(csv_path, index=False)

        if verbose:
            print(f"\nPredictions saved: {csv_path}")
            print(f"  Rows: {len(df):,}")
            print(f"  Risk score range: [{df['risk_score'].min():.4f}, {df['risk_score'].max():.4f}]")

        # Generate visualizations per location
        if verbose:
            print(f"\nGenerating risk maps...")

        locations = sorted(df['location'].unique())
        for location in locations:
            location_df = df[df['location'] == location]
            plot_risk_map_2d(location_df, location, args.output, verbose=verbose)

        # Write summary statistics
        write_summary_stats(df, args.output, verbose=verbose)

        if verbose:
            print(f"\n{'='*60}")
            print(f"SUCCESS")
            print(f"{'='*60}")
            print(f"Generated {len(locations)} risk maps")
            print(f"Output directory: {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
