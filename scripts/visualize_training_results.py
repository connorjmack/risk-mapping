#!/usr/bin/env python3
"""
Visualize Random Forest training results.

Generates plots from the trained model and training data:
1. CV performance by location (AUC-ROC, AUC-PR bar chart)
2. Feature importances (top 20, colored by feature group)
3. ROC curves per fold
4. Precision-Recall curves per fold
5. Predicted probability distributions (cases vs controls)
6. Confusion matrix heatmap

Usage:
    python scripts/visualize_training_results.py \
        --model models/rf_model.joblib \
        --data data/training_data.csv \
        --output output/training_results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from pc_rai.ml.train import StabilityModel
from pc_rai.ml.training_data import get_feature_columns


# Feature group colors
FEATURE_GROUP_COLORS = {
    "height": "#e74c3c",
    "slope": "#3498db",
    "linearity": "#2ecc71",
    "planarity": "#9b59b6",
    "roughness_large": "#e67e22",
    "roughness_small": "#1abc9c",
    "roughness_ratio": "#f39c12",
    "curvature": "#34495e",
    "sphericity": "#95a5a6",
}

LOCATION_COLORS = {
    "Blacks": "#e74c3c",
    "DelMar": "#3498db",
    "Encinitas": "#2ecc71",
    "Solana": "#e67e22",
    "Torrey": "#9b59b6",
}


def get_feature_group(feature_name):
    """Extract feature group from column name (e.g., 'slope_mean' -> 'slope')."""
    for group in FEATURE_GROUP_COLORS:
        if feature_name.startswith(group):
            return group
    return "other"


def run_cv_predictions(X, y, groups, model, stratified=False):
    """Re-run CV to get per-sample predictions.

    Parameters
    ----------
    stratified : bool
        If True, use StratifiedKFold (random splits mixing beaches).
        If False, use GroupKFold (leave-one-beach-out).
    """
    X_array = np.nan_to_num(X.values, nan=0.0)

    if stratified:
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=model.config.random_state,
        )
        cv_splits = list(cv.split(X_array, y))
    else:
        n_splits = len(np.unique(groups))
        cv = GroupKFold(n_splits=n_splits)
        cv_splits = list(cv.split(X_array, y, groups))

    y_proba = np.zeros(len(y))
    fold_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators=model.config.n_estimators,
            max_depth=model.config.max_depth,
            min_samples_leaf=model.config.min_samples_leaf,
            min_samples_split=model.config.min_samples_split,
            class_weight=model.config.class_weight,
            random_state=model.config.random_state,
            n_jobs=model.config.n_jobs,
        )
        rf.fit(X_train, y_train)

        proba = rf.predict_proba(X_val)[:, 1]
        y_proba[val_idx] = proba

        if stratified:
            label = f"Fold {fold_idx + 1}"
        else:
            val_groups = np.unique(groups[val_idx])
            label = val_groups[0]

        fold_data.append({
            "fold": fold_idx,
            "location": label,
            "y_true": y_val,
            "y_proba": proba,
            "val_idx": val_idx,
        })

    return y_proba, fold_data


def plot_cv_performance(fold_data, output_dir):
    """Bar chart of AUC-ROC and AUC-PR by location."""
    fig, ax = plt.subplots(figsize=(10, 5))

    locations = [f["location"] for f in fold_data]
    auc_rocs = [roc_auc_score(f["y_true"], f["y_proba"]) for f in fold_data]
    auc_prs = [average_precision_score(f["y_true"], f["y_proba"]) for f in fold_data]

    x = np.arange(len(locations))
    width = 0.35

    bars1 = ax.bar(x - width / 2, auc_rocs, width, label="AUC-ROC",
                   color="#3498db", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, auc_prs, width, label="AUC-PR",
                   color="#e67e22", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Leave-One-Beach-Out Cross-Validation Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(locations)
    ax.legend()
    ax.set_ylim(0, max(max(auc_rocs), max(auc_prs)) + 0.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.grid(axis="y", alpha=0.3)

    # Add overall metrics
    overall_roc = np.mean(auc_rocs)
    overall_pr = np.mean(auc_prs)
    ax.text(0.98, 0.02,
            f"Mean AUC-ROC: {overall_roc:.3f}\nMean AUC-PR: {overall_pr:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    out_path = output_dir / "cv_performance_by_location.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_importances(model, output_dir, top_n=20):
    """Horizontal bar chart of top feature importances, colored by group."""
    importances = model.feature_importances
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

    features = [f[0] for f in sorted_features][::-1]
    values = [f[1] for f in sorted_features][::-1]
    colors = [FEATURE_GROUP_COLORS.get(get_feature_group(f), "#95a5a6") for f in features]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(features, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.grid(axis="x", alpha=0.3)

    # Legend for feature groups
    from matplotlib.patches import Patch
    present_groups = set(get_feature_group(f) for f in features)
    legend_elements = [
        Patch(facecolor=FEATURE_GROUP_COLORS[g], label=g)
        for g in sorted(present_groups) if g in FEATURE_GROUP_COLORS
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Feature Group")

    plt.tight_layout()
    out_path = output_dir / "feature_importances.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_roc_curves(fold_data, output_dir):
    """ROC curves per fold overlaid."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for fd in fold_data:
        fpr, tpr, _ = roc_curve(fd["y_true"], fd["y_proba"])
        auc = roc_auc_score(fd["y_true"], fd["y_proba"])
        color = LOCATION_COLORS.get(fd["location"], "gray")
        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f'{fd["location"]} (AUC={auc:.3f})')

    # Overall
    all_y = np.concatenate([f["y_true"] for f in fold_data])
    all_p = np.concatenate([f["y_proba"] for f in fold_data])
    fpr, tpr, _ = roc_curve(all_y, all_p)
    overall_auc = roc_auc_score(all_y, all_p)
    ax.plot(fpr, tpr, color="black", linewidth=2.5, linestyle="--",
            label=f"Overall (AUC={overall_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Leave-One-Beach-Out CV")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "roc_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pr_curves(fold_data, output_dir):
    """Precision-Recall curves per fold overlaid."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for fd in fold_data:
        precision, recall, _ = precision_recall_curve(fd["y_true"], fd["y_proba"])
        ap = average_precision_score(fd["y_true"], fd["y_proba"])
        color = LOCATION_COLORS.get(fd["location"], "gray")
        ax.plot(recall, precision, color=color, linewidth=1.5,
                label=f'{fd["location"]} (AP={ap:.3f})')

    # Overall
    all_y = np.concatenate([f["y_true"] for f in fold_data])
    all_p = np.concatenate([f["y_proba"] for f in fold_data])
    precision, recall, _ = precision_recall_curve(all_y, all_p)
    overall_ap = average_precision_score(all_y, all_p)
    ax.plot(recall, precision, color="black", linewidth=2.5, linestyle="--",
            label=f"Overall (AP={overall_ap:.3f})")

    # Baseline = prevalence
    prevalence = all_y.mean()
    ax.axhline(y=prevalence, color="gray", linestyle=":", alpha=0.5,
               label=f"Baseline ({prevalence:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Leave-One-Beach-Out CV")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "pr_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_probability_distribution(fold_data, output_dir):
    """Histogram of predicted probabilities for cases vs controls."""
    all_y = np.concatenate([f["y_true"] for f in fold_data])
    all_p = np.concatenate([f["y_proba"] for f in fold_data])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(all_p[all_y == 0], bins=bins, alpha=0.6, color="#3498db",
            label="Controls (label=0)", density=True)
    ax.hist(all_p[all_y == 1], bins=bins, alpha=0.6, color="#e74c3c",
            label="Cases (label=1)", density=True)
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Threshold=0.5")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Overall Probability Distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Per-location
    ax = axes[1]
    for fd in fold_data:
        color = LOCATION_COLORS.get(fd["location"], "gray")
        cases = fd["y_proba"][fd["y_true"] == 1]
        controls = fd["y_proba"][fd["y_true"] == 0]
        separation = cases.mean() - controls.mean()
        ax.scatter(fd["location"], separation,
                   color=color, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.text(fd["location"], separation + 0.005, f"{separation:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean Probability Separation\n(cases - controls)")
    ax.set_title("Class Separation by Location")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "probability_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion_matrices(fold_data, output_dir):
    """Confusion matrices per location + overall."""
    n_folds = len(fold_data)
    fig, axes = plt.subplots(1, n_folds + 1, figsize=(4 * (n_folds + 1), 4))

    all_y = np.concatenate([f["y_true"] for f in fold_data])
    all_p = np.concatenate([f["y_proba"] for f in fold_data])

    datasets = fold_data + [{
        "location": "Overall",
        "y_true": all_y,
        "y_proba": all_p,
    }]

    for i, fd in enumerate(datasets):
        ax = axes[i]
        y_pred = (fd["y_proba"] >= 0.5).astype(int)
        cm = confusion_matrix(fd["y_true"], y_pred)

        # Normalize by row (true label)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

        # Add text annotations
        for r in range(2):
            for c in range(2):
                text_color = "white" if cm_pct[r, c] > 60 else "black"
                ax.text(c, r, f"{cm[r, c]:,}\n({cm_pct[r, c]:.0f}%)",
                        ha="center", va="center", fontsize=9, color=text_color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Control", "Case"])
        ax.set_yticklabels(["Control", "Case"])
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("Actual")

        accuracy = (y_pred == fd["y_true"]).mean()
        title = fd["location"]
        if title == "Overall":
            ax.set_title(f"{title}\n(acc={accuracy:.3f})", fontweight="bold")
        else:
            ax.set_title(f"{title}\n(acc={accuracy:.3f})")

    plt.suptitle("Confusion Matrices (threshold=0.5)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = output_dir / "confusion_matrices.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_importance_by_group(model, output_dir):
    """Pie/bar chart of total importance by feature group."""
    importances = model.feature_importances
    group_totals = {}
    for feat, imp in importances.items():
        group = get_feature_group(feat)
        group_totals[group] = group_totals.get(group, 0) + imp

    groups = sorted(group_totals.keys(), key=lambda g: group_totals[g], reverse=True)
    values = [group_totals[g] for g in groups]
    colors = [FEATURE_GROUP_COLORS.get(g, "#95a5a6") for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(groups, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Total Importance")
    ax.set_title("Feature Importance by Group")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "importance_by_group.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Random Forest training results"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/rf_model.joblib"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/training_data.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output/training_results"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="Use StratifiedKFold instead of GroupKFold by location",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"Error: Training data not found: {args.data}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    model = StabilityModel.load(args.model)
    df = pd.read_csv(args.data)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["label"].values
    groups = df["location"].values

    print(f"  {len(df):,} samples, {len(feature_cols)} features, "
          f"{len(np.unique(groups))} locations")

    # Re-run CV to get per-sample predictions
    cv_type = "StratifiedKFold" if args.stratified else "GroupKFold by location"
    print(f"\nRe-running cross-validation ({cv_type}) for prediction curves...")
    y_proba, fold_data = run_cv_predictions(
        X, y, groups, model, stratified=args.stratified,
    )

    # Generate all plots
    print("\nGenerating plots...")
    plot_cv_performance(fold_data, args.output)
    plot_feature_importances(model, args.output)
    plot_importance_by_group(model, args.output)
    plot_roc_curves(fold_data, args.output)
    plot_pr_curves(fold_data, args.output)
    plot_probability_distribution(fold_data, args.output)
    plot_confusion_matrices(fold_data, args.output)

    print(f"\nAll plots saved to: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
