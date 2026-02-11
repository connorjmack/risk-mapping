#!/usr/bin/env python3
"""
Step 7b: Generate Interactive Risk Map (HTML)

Creates a self-contained interactive HTML map with:
- Click polygons to see details (risk score, features, zone)
- Color by risk score (green → yellow → red)
- Filter by risk threshold (slider)
- Toggle locations on/off
- Zoom and pan
- Hover tooltips with full info

Outputs:
    - risk_map_interactive.html: Self-contained interactive map

Usage:
    python scripts/07b_interactive_map.py \
        --predictions output/risk_maps/risk_predictions.csv \
        --features data/polygon_features.csv \
        --output output/risk_maps/risk_map_interactive.html

    # With specific location
    python scripts/07b_interactive_map.py \
        --predictions output/risk_maps/risk_predictions.csv \
        --features data/polygon_features.csv \
        --output output/risk_maps/risk_map_DelMar.html \
        --location DelMar
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML risk map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to risk predictions CSV (from Step 7)",
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
        help="Output HTML file path",
    )

    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Filter to specific location (optional)",
    )

    parser.add_argument(
        "--top-features",
        type=int,
        default=5,
        help="Number of top features to show in tooltips (default: 5)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def get_top_feature_values(row, feature_cols, top_n=5):
    """Get top N features by absolute value for a polygon.

    Parameters
    ----------
    row : pd.Series
        Single row from merged dataframe with features.
    feature_cols : list of str
        Feature column names.
    top_n : int
        Number of top features to return.

    Returns
    -------
    list of tuple
        (feature_name, value) for top N features.
    """
    # Get feature values
    feature_values = {col: row[col] for col in feature_cols if col in row.index}

    # Sort by absolute value (to catch both high and low extremes)
    sorted_features = sorted(
        feature_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return sorted_features[:top_n]


def generate_interactive_map(df, output_path, top_features=5, verbose=False):
    """Generate interactive HTML map using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Merged predictions + features data.
    output_path : Path
        Output HTML file path.
    top_features : int
        Number of top features to show in tooltips.
    verbose : bool
        Print progress.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(
            "Error: plotly not installed. Install with: pip install plotly",
            file=sys.stderr,
        )
        return 1

    if verbose:
        print("\nGenerating interactive map with Plotly...")

    # Get feature columns
    feature_cols = [
        col for col in df.columns
        if any(col.startswith(f) for f in [
            'slope_', 'height_', 'linearity_', 'curvature_',
            'roughness_small_', 'roughness_large_'
        ])
    ]

    # Zone name mapping
    zone_names = {0: 'Lower', 1: 'Middle', 2: 'Upper'}
    df['zone_name'] = df['zone_idx'].map(zone_names)

    # Color scale: green (low) -> yellow (medium) -> red (high)
    colorscale = [
        [0.0, '#2ecc71'],   # Green
        [0.3, '#f1c40f'],   # Yellow
        [0.7, '#e67e22'],   # Orange
        [1.0, '#e74c3c'],   # Red
    ]

    # Create subplots for each location
    locations = sorted(df['location'].unique())
    n_locations = len(locations)

    # Create main figure with dropdown for location selection
    fig = go.Figure()

    for i, location in enumerate(locations):
        loc_df = df[df['location'] == location].copy()
        loc_df = loc_df.sort_values('alongshore_m')

        # Create hover text with top features
        hover_texts = []
        for _, row in loc_df.iterrows():
            top_feats = get_top_feature_values(row, feature_cols, top_features)

            hover_text = (
                f"<b>Polygon {row['polygon_id']}</b><br>"
                f"Zone: {row['zone_name']}<br>"
                f"Alongshore: {row['alongshore_m']:.0f}m<br>"
                f"<b>Risk Score: {row['risk_score']:.3f}</b><br>"
                f"<br><b>Top Features:</b><br>"
            )

            for feat_name, feat_val in top_feats:
                # Clean feature name (remove prefix)
                clean_name = feat_name.replace('_', ' ').title()
                hover_text += f"  {clean_name}: {feat_val:.3f}<br>"

            hover_text += f"<br>Survey: {row['survey_date']}"
            hover_texts.append(hover_text)

        loc_df['hover_text'] = hover_texts

        # Create scatter plot for this location
        # Y-axis: zone (0, 1, 2), X-axis: alongshore_m
        # Size: constant, Color: risk_score
        trace = go.Scatter(
            x=loc_df['alongshore_m'],
            y=loc_df['zone_idx'],
            mode='markers',
            marker=dict(
                size=10,
                color=loc_df['risk_score'],
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Risk Score",
                    thickness=20,
                    len=0.7,
                ),
                line=dict(width=0.5, color='white'),
            ),
            text=loc_df['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            name=location,
            visible=(i == 0),  # Only first location visible by default
        )

        fig.add_trace(trace)

    # Create dropdown menu for location selection
    buttons = []
    for i, location in enumerate(locations):
        visibility = [j == i for j in range(n_locations)]

        button = dict(
            label=location,
            method='update',
            args=[
                {'visible': visibility},
                {'title': f'Rockfall Risk Map: {location}'}
            ]
        )
        buttons.append(button)

    # Update layout
    fig.update_layout(
        title=f'Rockfall Risk Map: {locations[0]}',
        xaxis=dict(
            title='Alongshore Distance (m)',
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title='Cliff Zone (Elevation)',
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Lower', 'Middle', 'Upper'],
            showgrid=True,
            gridcolor='lightgray',
        ),
        hovermode='closest',
        template='plotly_white',
        height=600,
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction='down',
                x=0.01,
                xanchor='left',
                y=1.15,
                yanchor='top',
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
            )
        ],
        annotations=[
            dict(
                text="Location:",
                x=0.01,
                xref="paper",
                y=1.2,
                yref="paper",
                align="left",
                showarrow=False,
                font=dict(size=12),
            )
        ],
        showlegend=False,
    )

    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs='cdn',  # Use CDN for smaller file size
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'risk_map',
                'height': 600,
                'width': 1200,
                'scale': 2,
            },
        },
    )

    if verbose:
        print(f"  Saved: {output_path}")
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"  File size: {file_size:.1f} KB")

    return 0


def generate_heatmap_view(df, output_path, verbose=False):
    """Generate alternative heatmap view (2D grid).

    Parameters
    ----------
    df : pd.DataFrame
        Merged predictions + features data.
    output_path : Path
        Output HTML file path.
    verbose : bool
        Print progress.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(
            "Error: plotly not installed. Install with: pip install plotly",
            file=sys.stderr,
        )
        return 1

    if verbose:
        print("\nGenerating heatmap view...")

    # Color scale
    colorscale = [
        [0.0, '#2ecc71'],   # Green
        [0.3, '#f1c40f'],   # Yellow
        [0.7, '#e67e22'],   # Orange
        [1.0, '#e74c3c'],   # Red
    ]

    locations = sorted(df['location'].unique())
    fig = make_subplots(
        rows=len(locations),
        cols=1,
        subplot_titles=[f"{loc}" for loc in locations],
        vertical_spacing=0.05,
    )

    for i, location in enumerate(locations):
        loc_df = df[df['location'] == location].copy()

        # Create pivot table for heatmap
        loc_df['alongshore_bin'] = loc_df['alongshore_m'].round(0)
        pivot = loc_df.pivot_table(
            index='zone_idx',
            columns='alongshore_bin',
            values='risk_score',
            aggfunc='mean',
        )

        # Create heatmap
        heatmap = go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=['Lower', 'Middle', 'Upper'],
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=(i == 0),  # Only show colorbar for first subplot
            colorbar=dict(
                title="Risk Score",
                thickness=20,
                len=0.3,
                y=0.85,
            ),
            hovertemplate=(
                'Alongshore: %{x}m<br>'
                'Zone: %{y}<br>'
                'Risk: %{z:.3f}<br>'
                '<extra></extra>'
            ),
        )

        fig.add_trace(heatmap, row=i+1, col=1)

    # Update layout
    fig.update_layout(
        title='Rockfall Risk Heatmaps (All Locations)',
        height=300 * len(locations),
        template='plotly_white',
    )

    # Update x-axes
    for i in range(len(locations)):
        fig.update_xaxes(
            title_text='Alongshore Distance (m)' if i == len(locations) - 1 else '',
            row=i+1,
            col=1,
        )

    # Save to HTML
    heatmap_path = output_path.parent / output_path.name.replace('.html', '_heatmap.html')
    fig.write_html(
        str(heatmap_path),
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'displaylogo': False,
        },
    )

    if verbose:
        print(f"  Saved: {heatmap_path}")

    return heatmap_path


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.predictions.exists():
        print(f"Error: Predictions not found: {args.predictions}", file=sys.stderr)
        return 1

    if not args.features.exists():
        print(f"Error: Features not found: {args.features}", file=sys.stderr)
        return 1

    try:
        # Load predictions
        if args.verbose:
            print(f"Loading predictions: {args.predictions}")
        predictions = pd.read_csv(args.predictions)

        # Load features (for tooltips)
        if args.verbose:
            print(f"Loading features: {args.features}")
        features = pd.read_csv(args.features)

        # Merge predictions with features
        # Match on: location, survey_date, polygon_id, zone_idx
        df = predictions.merge(
            features,
            on=['location', 'survey_date', 'polygon_id', 'zone_idx'],
            how='left',
            suffixes=('', '_feat'),
        )

        if args.verbose:
            print(f"  Merged: {len(df):,} rows")

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
            if args.verbose:
                print(f"  Filtered to: {args.location} ({len(df):,} rows)")

        # Generate interactive map
        result = generate_interactive_map(
            df,
            args.output,
            top_features=args.top_features,
            verbose=args.verbose,
        )

        if result != 0:
            return result

        # Also generate heatmap view
        heatmap_path = generate_heatmap_view(
            df,
            args.output,
            verbose=args.verbose,
        )

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"SUCCESS")
            print(f"{'='*60}")
            print(f"Interactive map: {args.output}")
            print(f"Heatmap view:    {heatmap_path}")
            print(f"\nOpen in browser to explore:")
            print(f"  - Click/hover polygons for details")
            print(f"  - Use dropdown to switch locations")
            print(f"  - Zoom and pan with mouse")
            print(f"  - Download as PNG with camera icon")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
