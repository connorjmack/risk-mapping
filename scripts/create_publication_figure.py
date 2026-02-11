#!/usr/bin/env python3
"""
Create Publication-Quality Two-Panel Figure

Panel A: ROC curves for different validation schemes
Panel B: Feature importance grouped by type

Nature journal styling: professional, clean, color-blind friendly
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pc_rai.ml.training_data import get_feature_columns

# Nature journal styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Color-blind friendly palette (Wong 2011)
COLORS = {
    'temporal': '#E69F00',    # Orange
    'spatial': '#56B4E9',     # Sky blue
    'stratified': '#009E73',  # Bluish green
    'random': '#999999',      # Gray
}

# Feature groups with colors
FEATURE_COLORS = {
    'height': '#D55E00',      # Vermillion
    'slope': '#0072B2',       # Blue
    'linearity': '#009E73',   # Bluish green
    'curvature': '#CC79A7',   # Reddish purple
    'roughness_small': '#F0E442',  # Yellow
    'roughness_large': '#E69F00',  # Orange
}


def classify_feature(feature_name):
    """Classify feature into groups."""
    if 'height' in feature_name:
        return 'height', FEATURE_COLORS['height']
    elif 'slope' in feature_name:
        return 'slope', FEATURE_COLORS['slope']
    elif 'linearity' in feature_name:
        return 'linearity', FEATURE_COLORS['linearity']
    elif 'curvature' in feature_name:
        return 'curvature', FEATURE_COLORS['curvature']
    elif 'roughness_small' in feature_name:
        return 'roughness_small', FEATURE_COLORS['roughness_small']
    elif 'roughness_large' in feature_name:
        return 'roughness_large', FEATURE_COLORS['roughness_large']
    else:
        return 'other', '#999999'


def compute_roc_cv(X, y, groups, cv_type='temporal', n_splits=5, random_state=42):
    """
    Compute ROC curves for a given CV scheme.

    Returns:
        mean_fpr: Array of FPR values
        mean_tpr: Array of mean TPR values
        std_tpr: Array of TPR standard deviations
        mean_auc: Mean AUC-ROC
        std_auc: Standard deviation of AUC-ROC
    """
    if cv_type == 'temporal':
        # Leave-one-year-out
        cv = GroupKFold(n_splits=n_splits)
        split_groups = groups
    elif cv_type == 'spatial':
        # Leave-one-beach-out
        cv = GroupKFold(n_splits=n_splits)
        split_groups = groups
    else:  # stratified
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_groups = None

    # Train RF model and collect ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, test_idx in cv.split(X, y, split_groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        # Interpolate to common FPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # Compute AUC
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Compute mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def plot_panel_a_roc(ax, X, y, groups, year_groups):
    """Plot Panel A: ROC curves for different CV schemes."""

    print("Computing ROC curves...")

    # Determine n_splits for CV schemes
    n_years = year_groups.nunique()
    n_locations = groups.nunique()

    # Temporal CV (leave-one-year-out)
    print(f"  - Temporal CV (leave-one-year-out, {n_years} years)...")
    fpr_temp, tpr_temp, std_temp, auc_temp, std_auc_temp = compute_roc_cv(
        X, y, year_groups, cv_type='temporal', n_splits=min(n_years, 5)
    )

    # Spatial CV (leave-one-beach-out)
    print(f"  - Spatial CV (leave-one-beach-out, {n_locations} locations)...")
    fpr_spat, tpr_spat, std_spat, auc_spat, std_auc_spat = compute_roc_cv(
        X, y, groups, cv_type='spatial', n_splits=min(n_locations, 5)
    )

    # Stratified CV (within-sample)
    print("  - Stratified CV (within-sample)...")
    fpr_strat, tpr_strat, std_strat, auc_strat, std_auc_strat = compute_roc_cv(
        X, y, groups, cv_type='stratified', n_splits=5
    )

    # Plot ROC curves
    # Temporal (most important - bold)
    ax.plot(fpr_temp, tpr_temp, color=COLORS['temporal'], linewidth=2.0,
            label=f'Temporal CV (AUC = {auc_temp:.3f} ± {std_auc_temp:.3f})', zorder=3)
    ax.fill_between(fpr_temp, tpr_temp - std_temp, tpr_temp + std_temp,
                     color=COLORS['temporal'], alpha=0.15, zorder=2)

    # Spatial
    ax.plot(fpr_spat, tpr_spat, color=COLORS['spatial'], linewidth=1.5,
            label=f'Spatial CV (AUC = {auc_spat:.3f} ± {std_auc_spat:.3f})', zorder=3)
    ax.fill_between(fpr_spat, tpr_spat - std_spat, tpr_spat + std_spat,
                     color=COLORS['spatial'], alpha=0.15, zorder=2)

    # Stratified (reference)
    ax.plot(fpr_strat, tpr_strat, color=COLORS['stratified'], linewidth=1.5,
            linestyle='--', alpha=0.7,
            label=f'Stratified CV (AUC = {auc_strat:.3f} ± {std_auc_strat:.3f})', zorder=1)

    # Random classifier
    ax.plot([0, 1], [0, 1], color=COLORS['random'], linewidth=1.0,
            linestyle=':', label='Random (AUC = 0.500)', zorder=0)

    # Styling
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='black')

    # Panel label
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


def plot_panel_b_importance(ax, stability_model, top_n=15):
    """Plot Panel B: Feature importance grouped by type."""

    print("Plotting feature importance...")

    # Get importances
    importances_dict = stability_model.feature_importances

    # Sort and take top N
    items = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0] for item in items]
    values = [item[1] for item in items]

    # Classify features and assign colors
    colors = [classify_feature(name)[1] for name in names]

    # Create horizontal bar chart
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)

    # Format feature names (remove prefix, clean up)
    clean_names = []
    for name in names:
        # Split by underscore, capitalize
        parts = name.split('_')
        if len(parts) >= 2:
            feature = parts[0].capitalize()
            stat = parts[-1].upper() if parts[-1] in ['mean', 'std', 'min', 'max'] else parts[-1].capitalize()
            clean_name = f"{feature} ({stat})"
        else:
            clean_name = name.capitalize()
        clean_names.append(clean_name)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.set_xlim([0, max(values) * 1.1])

    # Panel label
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

    # Create custom legend for feature groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FEATURE_COLORS['height'], edgecolor='black', label='Height'),
        Patch(facecolor=FEATURE_COLORS['slope'], edgecolor='black', label='Slope'),
        Patch(facecolor=FEATURE_COLORS['linearity'], edgecolor='black', label='Linearity'),
        Patch(facecolor=FEATURE_COLORS['curvature'], edgecolor='black', label='Curvature'),
        Patch(facecolor=FEATURE_COLORS['roughness_small'], edgecolor='black', label='Roughness (small)'),
        Patch(facecolor=FEATURE_COLORS['roughness_large'], edgecolor='black', label='Roughness (large)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              framealpha=0.95, edgecolor='black', ncol=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Create publication figure')
    parser.add_argument('--model', type=Path, default=Path('models/rf_model.joblib'),
                        help='Path to trained model')
    parser.add_argument('--data', type=Path, default=Path('data/training_data.csv'),
                        help='Path to training data')
    parser.add_argument('--output', type=Path, default=Path('figures/main/'),
                        help='Output directory')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PNG output')
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating Publication-Quality Two-Panel Figure")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading model: {args.model}")
    stability_model = joblib.load(args.model)
    print(f"  Features: {len(stability_model.feature_names)}")

    # Load data
    print(f"\nLoading data: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Samples: {len(df):,}")
    print(f"  Cases: {(df['label'] == 1).sum():,}")
    print(f"  Controls: {(df['label'] == 0).sum():,}")

    # Prepare features and labels
    feature_cols = stability_model.feature_names
    X = df[feature_cols]
    y = df['label']
    groups = df['location']

    # Create year groups for temporal CV
    if 'survey_date' in df.columns:
        # survey_date is stored as integer YYYYMMDD, extract year
        df['year'] = df['survey_date'].astype(str).str[:4].astype(int)
        year_groups = df['year']
        n_years = year_groups.nunique()
        print(f"  Years: {sorted(year_groups.unique().tolist())}")
        print(f"  Unique years: {n_years}")
    else:
        # Fallback to location groups
        print("  Warning: No survey_date column, using location for temporal CV")
        year_groups = groups
        n_years = year_groups.nunique()

    print()

    # Create figure with two panels
    fig = plt.figure(figsize=(7.2, 3.6))  # Nature single column width
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: ROC curves
    plot_panel_a_roc(ax_a, X, y, groups, year_groups)

    # Panel B: Feature importance
    plot_panel_b_importance(ax_b, stability_model, top_n=15)

    # Save figure
    output_png = args.output / 'figure_performance.png'

    plt.savefig(output_png, dpi=args.dpi, bbox_inches='tight', facecolor='white')

    print()
    print("=" * 70)
    print("Figure saved successfully!")
    print("=" * 70)
    print(f"PNG: {output_png}")
    print()
    print("Figure specifications:")
    print(f"  - Size: 7.2 × 3.6 inches (Nature single-column width)")
    print(f"  - Resolution: {args.dpi} DPI")
    print(f"  - Format: High-resolution PNG")
    print(f"  - Color scheme: Color-blind friendly (Wong 2011)")
    print(f"  - Typography: Arial/Helvetica, Nature journal specs")
    print()

    plt.close()


if __name__ == '__main__':
    main()
