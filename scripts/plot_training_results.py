#!/usr/bin/env python3
"""
Visualize Random Forest Training Results

Creates publication-quality plots:
1. Feature importance (top 20)
2. Cross-validation performance
3. ROC curves (per-fold and overall)
4. Precision-Recall curves
5. Confusion matrix
6. Prediction distribution

Usage:
    python scripts/plot_training_results.py \
        --model models/rf_model.joblib \
        --data data/training_data.csv \
        --output output/training_plots/ \
        --group-by location
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pc_rai.ml.training_data import get_feature_columns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Random Forest training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/rf_model.joblib"),
        help="Path to trained model",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training_data.csv"),
        help="Path to training data CSV",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/training_plots"),
        help="Output directory for plots",
    )

    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Column to group by for CV (e.g., 'location')",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )

    return parser.parse_args()


def plot_feature_importance(stability_model, output_path, top_n=20):
    """Plot feature importance."""
    # Get importances from StabilityModel (it's a dict)
    importances_dict = stability_model.feature_importances

    # Convert to sorted list
    items = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot bars
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_cv_performance(cv_results, output_path):
    """Plot cross-validation performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    folds = range(1, len(cv_results) + 1)
    auc_roc = [r["auc_roc"] for r in cv_results]
    auc_pr = [r["auc_pr"] for r in cv_results]

    # AUC-ROC
    ax1.plot(folds, auc_roc, "o-", linewidth=2, markersize=8, label="Per-fold")
    ax1.axhline(
        np.mean(auc_roc), color="red", linestyle="--", linewidth=2, label="Mean"
    )
    ax1.set_xlabel("Fold", fontsize=12)
    ax1.set_ylabel("AUC-ROC", fontsize=12)
    ax1.set_title("Cross-Validation: AUC-ROC", fontsize=14, fontweight="bold")
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(alpha=0.3)
    ax1.legend()

    # AUC-PR
    ax2.plot(folds, auc_pr, "o-", linewidth=2, markersize=8, label="Per-fold")
    ax2.axhline(np.mean(auc_pr), color="red", linestyle="--", linewidth=2, label="Mean")
    ax2.set_xlabel("Fold", fontsize=12)
    ax2.set_ylabel("AUC-PR", fontsize=12)
    ax2.set_title("Cross-Validation: AUC-PR", fontsize=14, fontweight="bold")
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_roc_curves(cv_results, output_path):
    """Plot ROC curves for all CV folds."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each fold
    for i, result in enumerate(cv_results, 1):
        fpr = result["fpr"]
        tpr = result["tpr"]
        auc_val = result["auc_roc"]
        groups_str = ", ".join(result["test_groups"][:2])  # Show first 2 groups
        if len(result["test_groups"]) > 2:
            groups_str += "..."

        ax.plot(
            fpr,
            tpr,
            linewidth=1.5,
            alpha=0.7,
            label=f"Fold {i} (AUC={auc_val:.3f}, {groups_str})",
        )

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random (AUC=0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (Cross-Validation)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_pr_curves(cv_results, output_path):
    """Plot Precision-Recall curves for all CV folds."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each fold
    for i, result in enumerate(cv_results, 1):
        precision = result["precision"]
        recall = result["recall"]
        auc_val = result["auc_pr"]
        groups_str = ", ".join(result["test_groups"][:2])
        if len(result["test_groups"]) > 2:
            groups_str += "..."

        ax.plot(
            recall,
            precision,
            linewidth=1.5,
            alpha=0.7,
            label=f"Fold {i} (AUC={auc_val:.3f}, {groups_str})",
        )

    # Baseline (random classifier)
    baseline = 0.5  # Assuming balanced classes
    ax.axhline(baseline, color="k", linestyle="--", linewidth=2, label="Random")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (CV)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path, from_cv=False):
    """Plot confusion matrix with percentages.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_path : Path
        Output file path
    from_cv : bool
        If True, indicates predictions are from cross-validation (honest estimate)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (percentage of each true class)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)", rotation=270, labelpad=20)

    # Labels
    classes = ["Control (0)", "Case (1)"]
    title = "Confusion Matrix (Cross-Validation)" if from_cv else "Confusion Matrix (Training Data)"
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations (percentage + count)
    thresh = 50.0  # Midpoint for color threshold
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]:,})"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_pct[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_prediction_distribution(y_true, y_pred_proba, output_path):
    """Plot distribution of predicted probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate by true label
    cases = y_pred_proba[y_true == 1]
    controls = y_pred_proba[y_true == 0]

    # Histograms
    ax.hist(
        controls, bins=50, alpha=0.6, label="Controls (true 0)", color="blue", density=True
    )
    ax.hist(
        cases, bins=50, alpha=0.6, label="Cases (true 1)", color="red", density=True
    )

    ax.set_xlabel("Predicted Probability (Case)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Prediction Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def run_cv_and_collect_results(stability_model, X, y, groups, n_folds=5):
    """Run cross-validation and collect ROC/PR data + CV predictions.

    Returns
    -------
    results : list of dict
        Per-fold metrics and curves
    y_pred_cv : np.ndarray
        Out-of-fold predictions for all samples (for honest confusion matrix)
    """
    from sklearn.ensemble import RandomForestClassifier

    cv = GroupKFold(n_splits=n_folds)
    results = []

    # Get the underlying sklearn model's parameters
    sklearn_model = stability_model.model
    model_params = sklearn_model.get_params()

    # Collect out-of-fold predictions
    y_pred_cv = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train fold model (use sklearn directly, not StabilityModel wrapper)
        fold_model = RandomForestClassifier(**model_params)
        fold_model.fit(X_train, y_train)

        # Predictions
        y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)

        # Store out-of-fold predictions
        y_pred_cv[test_idx] = y_pred_binary

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_roc = auc(fpr, tpr)

        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)

        # Test groups
        test_groups = groups.iloc[test_idx].unique().tolist()

        results.append(
            {
                "fold": fold,
                "fpr": fpr,
                "tpr": tpr,
                "auc_roc": auc_roc,
                "precision": precision,
                "recall": recall,
                "auc_pr": auc_pr,
                "test_groups": test_groups,
            }
        )

    return results, y_pred_cv


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Visualizing Training Results")
    print("=" * 60)
    print()

    # Load model (StabilityModel wrapper)
    print(f"Loading model: {args.model}")
    stability_model = joblib.load(args.model)

    # Print CV metrics from model if available
    if hasattr(stability_model, 'cv_metrics') and stability_model.cv_metrics:
        cv_metrics = stability_model.cv_metrics
        if 'auc_roc' in cv_metrics:
            print(f"  CV AUC-ROC: {cv_metrics['auc_roc']:.3f}")
        if 'auc_pr' in cv_metrics:
            print(f"  CV AUC-PR: {cv_metrics['auc_pr']:.3f}")

    # Load data
    print(f"\nLoading data: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Rows: {len(df):,}")

    # Prepare features and labels
    # Use feature names from the trained model to ensure consistency
    feature_cols = stability_model.feature_names
    X = df[feature_cols]
    y = df["label"]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Cases: {(y == 1).sum():,}")
    print(f"  Controls: {(y == 0).sum():,}")

    # Groups for CV
    groups = df[args.group_by] if args.group_by else None

    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    print()

    # 1. Feature importance
    plot_feature_importance(
        stability_model, args.output / "feature_importance.png", top_n=20
    )

    # 2. Cross-validation curves and honest confusion matrix (if groups provided)
    if groups is not None:
        print(f"\nRunning {args.n_folds}-fold CV to generate ROC/PR curves...")
        cv_results, y_pred_cv = run_cv_and_collect_results(
            stability_model, X, y, groups, n_folds=args.n_folds
        )

        plot_cv_performance(cv_results, args.output / "cv_performance.png")
        plot_roc_curves(cv_results, args.output / "roc_curves.png")
        plot_pr_curves(cv_results, args.output / "pr_curves.png")

        # Confusion matrix from CV predictions (honest estimate)
        print("Computing confusion matrix from CV predictions...")
        plot_confusion_matrix(
            y, y_pred_cv, args.output / "confusion_matrix_cv.png", from_cv=True
        )

        # Get CV probabilities for distribution plot
        print("Computing CV prediction probabilities...")
        y_pred_proba_cv = np.zeros(len(y))
        cv = GroupKFold(n_splits=args.n_folds)
        sklearn_model = stability_model.model
        model_params = sklearn_model.get_params()

        from sklearn.ensemble import RandomForestClassifier
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]

            fold_model = RandomForestClassifier(**model_params)
            fold_model.fit(X_train, y_train)
            y_pred_proba_cv[test_idx] = fold_model.predict_proba(X_test)[:, 1]

        # Distribution plot from CV predictions
        plot_prediction_distribution(
            y, y_pred_proba_cv, args.output / "prediction_distribution_cv.png"
        )

    else:
        print("\nNote: No groups provided. Skipping CV plots.")
        print("Generating plots on training data (optimistic estimates)...")

        # Fall back to training predictions (optimistic)
        y_pred_proba = stability_model.predict_proba(X)
        y_pred = stability_model.predict(X)

        plot_confusion_matrix(
            y, y_pred, args.output / "confusion_matrix_train.png", from_cv=False
        )
        plot_prediction_distribution(
            y, y_pred_proba, args.output / "prediction_distribution_train.png"
        )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Plots saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
