"""
Cumulative feature ablation study for stability prediction.

Trains Random Forest models with progressively more feature groups to show
the marginal contribution of each group to predictive performance. Useful for
understanding which morphometric features drive rockfall prediction and where
diminishing returns set in.

Usage:
    from pc_rai.ml.ablation import run_ablation, plot_ablation

    results = run_ablation(training_df, verbose=True)
    results.to_dataframe().to_csv("ablation_results.csv", index=False)
    plot_ablation(results, "ablation_curve.png")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import MLConfig
from .train import train_model
from .training_data import get_feature_columns


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physically-motivated cumulative ordering:
# geometry -> structure -> texture
DEFAULT_FEATURE_ORDER = [
    "slope",            # fundamental geometry
    "height",           # cliff position / elevation
    "linearity",        # structural edges, joints
    "planarity",        # surface flatness
    "curvature",        # surface variation
    "sphericity",       # point scatter
    "roughness_small",  # fine-scale texture
    "roughness_large",  # broad-scale texture
    "roughness_ratio",  # multi-scale relationship
]

# Colors for each feature group (matches scripts/visualize_training_results.py)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_feature_group(feature_name: str) -> str:
    """Extract feature group from a column name.

    Parameters
    ----------
    feature_name : str
        Column name like ``slope_mean`` or ``roughness_large_p90``.

    Returns
    -------
    str
        Feature group name (e.g., ``slope``, ``roughness_large``).
        Returns ``"other"`` if no known group matches.

    Notes
    -----
    Iteration order of ``FEATURE_GROUP_COLORS`` matters: ``roughness_large``
    must be checked before ``roughness_small`` and ``roughness_ratio`` to
    avoid prefix collisions. This is guaranteed by dict insertion order.
    """
    for group in FEATURE_GROUP_COLORS:
        if feature_name.startswith(group):
            return group
    return "other"


def get_group_columns(all_columns: List[str], group_name: str) -> List[str]:
    """Get columns belonging to a feature group.

    Parameters
    ----------
    all_columns : list of str
        All available feature column names.
    group_name : str
        Feature group name (e.g., ``slope``).

    Returns
    -------
    list of str
        Columns belonging to that group.
    """
    return [c for c in all_columns if get_feature_group(c) == group_name]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AblationStep:
    """Results from a single ablation step.

    Parameters
    ----------
    step : int
        Step index (0-based).
    feature_group : str
        Name of feature group added at this step.
    feature_columns : list of str
        All feature columns used at this step (cumulative).
    n_features : int
        Number of features used at this step.
    auc_roc : float
        Overall cross-validated AUC-ROC.
    auc_pr : float
        Overall cross-validated AUC-PR.
    fold_metrics : list of dict
        Per-fold metrics from train_model().
    """

    step: int
    feature_group: str
    feature_columns: List[str]
    n_features: int
    auc_roc: float
    auc_pr: float
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AblationResults:
    """Full results from cumulative ablation study.

    Parameters
    ----------
    steps : list of AblationStep
        Results for each cumulative step.
    feature_order : list of str
        Ordered list of feature groups used.
    config : MLConfig
        Training configuration used.
    n_samples : int
        Total number of training samples.
    n_positive : int
        Number of positive (case) samples.
    """

    steps: List[AblationStep]
    feature_order: List[str]
    config: MLConfig
    n_samples: int = 0
    n_positive: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a CSV-ready DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per ablation step with columns: step, feature_group,
            n_features, auc_roc, auc_pr, auc_roc_std, auc_pr_std,
            plus per-fold columns.
        """
        rows = []
        for step in self.steps:
            row = {
                "step": step.step,
                "feature_group": step.feature_group,
                "n_features": step.n_features,
                "auc_roc": step.auc_roc,
                "auc_pr": step.auc_pr,
            }

            # Fold-level metrics
            fold_rocs = []
            fold_prs = []
            for fm in step.fold_metrics:
                fold_idx = fm["fold"]
                roc = fm.get("auc_roc", np.nan)
                pr = fm.get("auc_pr", np.nan)
                row[f"auc_roc_fold_{fold_idx}"] = roc
                row[f"auc_pr_fold_{fold_idx}"] = pr
                fold_rocs.append(roc)
                fold_prs.append(pr)

            row["auc_roc_std"] = float(np.nanstd(fold_rocs))
            row["auc_pr_std"] = float(np.nanstd(fold_prs))

            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core ablation function
# ---------------------------------------------------------------------------


def run_ablation(
    df: pd.DataFrame,
    feature_order: Optional[List[str]] = None,
    config: Optional[MLConfig] = None,
    groups: Optional[np.ndarray] = None,
    normalize: bool = False,
    verbose: bool = True,
) -> AblationResults:
    """Run cumulative feature ablation study.

    Trains a Random Forest model at each step, cumulatively adding one
    feature group at a time. Returns metrics at each step to show
    the marginal contribution of each feature group.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with feature columns and ``label`` column.
    feature_order : list of str, optional
        Ordered list of feature group names to add cumulatively.
        Defaults to ``DEFAULT_FEATURE_ORDER``.
    config : MLConfig, optional
        Training configuration. Uses defaults if None.
    groups : np.ndarray, optional
        Group labels for GroupKFold CV (e.g., location values).
    normalize : bool
        Whether to normalize features with StandardScaler.
    verbose : bool
        Print progress.

    Returns
    -------
    AblationResults
        Results for each cumulative step.

    Raises
    ------
    ValueError
        If ``feature_order`` contains groups with no matching columns.
    """
    feature_order = feature_order or list(DEFAULT_FEATURE_ORDER)
    config = config or MLConfig()

    # Get all available feature columns
    all_feature_cols = get_feature_columns(df)
    y = df["label"].values

    # Validate that each group has columns
    for group in feature_order:
        cols = get_group_columns(all_feature_cols, group)
        if len(cols) == 0:
            raise ValueError(
                f"Feature group '{group}' has no matching columns in the data. "
                f"Available groups: {sorted(set(get_feature_group(c) for c in all_feature_cols))}"
            )

    if verbose:
        print(f"Cumulative feature ablation study")
        print(f"  Samples: {len(y):,} ({int(y.sum()):,} positive)")
        print(f"  Feature groups: {len(feature_order)}")
        print(f"  CV strategy: {'GroupKFold' if groups is not None else 'StratifiedKFold'} "
              f"({config.cv_n_splits} folds)")
        print()

    # Run cumulative ablation
    steps = []
    cumulative_cols = []

    for i, group in enumerate(feature_order):
        group_cols = get_group_columns(all_feature_cols, group)
        cumulative_cols = cumulative_cols + group_cols

        X = df[cumulative_cols]

        if verbose:
            print(f"  Step {i}: + {group} ({len(group_cols)} cols, "
                  f"{len(cumulative_cols)} total) ... ", end="", flush=True)

        model = train_model(
            X, y,
            config=config,
            groups=groups,
            normalize=normalize,
            verbose=False,
        )

        cv = model.cv_metrics
        auc_roc = cv.get("auc_roc", np.nan)
        auc_pr = cv.get("auc_pr", np.nan)
        fold_metrics = cv.get("fold_metrics", [])

        step = AblationStep(
            step=i,
            feature_group=group,
            feature_columns=list(cumulative_cols),
            n_features=len(cumulative_cols),
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            fold_metrics=fold_metrics,
        )
        steps.append(step)

        if verbose:
            print(f"AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")

    if verbose:
        print(f"\nDone. Final AUC-ROC={steps[-1].auc_roc:.3f}, "
              f"AUC-PR={steps[-1].auc_pr:.3f}")

    return AblationResults(
        steps=steps,
        feature_order=feature_order,
        config=config,
        n_samples=len(y),
        n_positive=int(y.sum()),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ablation(
    results: AblationResults,
    output_path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> None:
    """Generate manuscript-quality ablation curve figure.

    Shows cumulative AUC-ROC and AUC-PR as feature groups are added,
    with error bars from cross-validation folds.

    Parameters
    ----------
    results : AblationResults
        Output from ``run_ablation()``.
    output_path : str or Path
        Path to save the figure (PNG).
    figsize : tuple of float
        Figure size in inches (width, height).
    dpi : int
        Resolution for saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_steps = len(results.steps)
    x = np.arange(n_steps)

    # Extract metrics and error bars
    auc_rocs = [s.auc_roc for s in results.steps]
    auc_prs = [s.auc_pr for s in results.steps]

    roc_stds = []
    pr_stds = []
    for s in results.steps:
        fold_rocs = [fm.get("auc_roc", np.nan) for fm in s.fold_metrics]
        fold_prs = [fm.get("auc_pr", np.nan) for fm in s.fold_metrics]
        roc_stds.append(np.nanstd(fold_rocs))
        pr_stds.append(np.nanstd(fold_prs))

    # Marker colors by feature group added at each step
    marker_colors = [
        FEATURE_GROUP_COLORS.get(s.feature_group, "#95a5a6")
        for s in results.steps
    ]

    # X-tick labels
    labels = []
    for i, s in enumerate(results.steps):
        if i == 0:
            labels.append(s.feature_group)
        else:
            labels.append(f"+ {s.feature_group}")

    fig, ax = plt.subplots(figsize=figsize)

    # AUC-ROC line
    ax.errorbar(
        x, auc_rocs, yerr=roc_stds,
        fmt="-o", color="#2c3e50", markeredgecolor="white",
        markeredgewidth=0.8, markersize=8, linewidth=2, capsize=4,
        label="AUC-ROC", zorder=3,
    )
    # Color individual markers
    for xi, c in zip(x, marker_colors):
        ax.plot(xi, auc_rocs[xi], "o", color=c, markersize=8,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # AUC-PR line
    ax.errorbar(
        x, auc_prs, yerr=pr_stds,
        fmt="--s", color="#7f8c8d", markeredgecolor="white",
        markeredgewidth=0.8, markersize=7, linewidth=2, capsize=4,
        label="AUC-PR", zorder=3,
    )
    for xi, c in zip(x, marker_colors):
        ax.plot(xi, auc_prs[xi], "s", color=c, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # Baselines
    ax.axhline(0.5, color="#bdc3c7", linestyle=":", linewidth=1, label="Random (ROC)")
    if results.n_samples > 0:
        prevalence = results.n_positive / results.n_samples
        ax.axhline(prevalence, color="#d5dbdb", linestyle=":", linewidth=1,
                    label=f"Prevalence ({prevalence:.2f})")

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC Score", fontsize=11)
    ax.set_xlabel("Cumulative Feature Groups", fontsize=11)
    ax.set_title("Cumulative Feature Ablation", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Y-axis range: give some padding below the minimum
    all_vals = auc_rocs + auc_prs
    y_min = max(0.4, min(all_vals) - 0.05)
    y_max = min(1.0, max(all_vals) + 0.05)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
