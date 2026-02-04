"""
Model training for stability prediction.

Trains Random Forest models on transect-level features to predict rockfall probability.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupKFold
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    raise ImportError(
        "scikit-learn and joblib are required. "
        "Install with: pip install scikit-learn joblib"
    )

from .config import MLConfig


@dataclass
class StabilityModel:
    """Trained stability prediction model.

    Attributes
    ----------
    model : RandomForestClassifier or RandomForestRegressor
        Trained sklearn model.
    feature_names : list of str
        Names of features used in training.
    feature_importances : dict
        Feature importance scores.
    scaler : StandardScaler, optional
        Feature scaler if normalization was used.
    config : MLConfig
        Training configuration.
    train_metadata : dict
        Training metadata (dates, beaches, etc.).
    cv_metrics : dict
        Cross-validation metrics.
    """

    model: Any
    feature_names: List[str]
    feature_importances: Dict[str, float]
    scaler: Optional[StandardScaler]
    config: MLConfig
    train_metadata: Dict[str, Any] = field(default_factory=dict)
    cv_metrics: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of rockfall for each transect.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        np.ndarray
            (N,) array of probabilities.
        """
        # Ensure features are in correct order
        X_ordered = X[self.feature_names].values

        # Apply scaling if used
        if self.scaler is not None:
            X_ordered = self.scaler.transform(X_ordered)

        # Handle NaN values
        X_ordered = np.nan_to_num(X_ordered, nan=0.0)

        return self.model.predict_proba(X_ordered)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        threshold : float
            Probability threshold for positive class.

        Returns
        -------
        np.ndarray
            (N,) array of binary predictions.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str or Path
            Output path (will save as .joblib).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as joblib
        joblib.dump(self, path)
        print(f"Model saved to {path}")

        # Also save metadata as JSON for inspection
        metadata_path = path.with_suffix(".json")
        metadata = {
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "train_metadata": {
                k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v
                for k, v in self.train_metadata.items()
            },
            "cv_metrics": self.cv_metrics,
            "config": {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "min_samples_leaf": self.config.min_samples_leaf,
                "class_weight": self.config.class_weight,
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StabilityModel":
        """Load model from disk.

        Parameters
        ----------
        path : str or Path
            Path to saved model.

        Returns
        -------
        StabilityModel
            Loaded model.
        """
        return joblib.load(path)


def prepare_training_data(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    label_column: str = "has_event",
    merge_on: List[str] = ["transect_id"],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Prepare training data by merging features and labels.

    Parameters
    ----------
    features_df : pd.DataFrame
        Transect-level features.
    labels_df : pd.DataFrame
        Transect-level labels.
    feature_columns : list of str, optional
        Columns to use as features. If None, auto-detect numeric columns.
    label_column : str
        Column to use as label.
    merge_on : list of str
        Columns to merge features and labels on.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Label vector.
    merged_df : pd.DataFrame
        Full merged dataframe.
    """
    # Merge features and labels
    merged = features_df.merge(labels_df, on=merge_on, how="inner")

    # Auto-detect feature columns if not specified
    if feature_columns is None:
        # Exclude metadata columns
        exclude_cols = [
            "transect_id",
            "alongshore_dist",
            "tr_id",
            "scan_date",
            "source_file",
            "event_count",
            "has_event",
            "total_volume",
            "max_volume",
            "point_count",
        ]
        # Also exclude any columns ending with _x or _y from merge
        exclude_cols += [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]

        feature_columns = [
            c
            for c in merged.columns
            if c not in exclude_cols and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

    # Extract features and labels
    X = merged[feature_columns].copy()
    y = merged[label_column].values

    print(f"Prepared {len(X):,} samples with {len(feature_columns)} features")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"Class balance: {y.mean()*100:.1f}% positive")

    return X, y, merged


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    config: Optional[MLConfig] = None,
    groups: Optional[np.ndarray] = None,
    normalize: bool = False,
    verbose: bool = True,
) -> StabilityModel:
    """Train a Random Forest model for stability prediction.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Label vector.
    config : MLConfig, optional
        Training configuration.
    groups : np.ndarray, optional
        Group labels for GroupKFold cross-validation (e.g., beach IDs).
    normalize : bool
        Whether to normalize features.
    verbose : bool
        Print training progress.

    Returns
    -------
    StabilityModel
        Trained model with cross-validation metrics.
    """
    config = config or MLConfig()

    feature_names = list(X.columns)
    X_array = X.values.copy()

    # Handle NaN values
    X_array = np.nan_to_num(X_array, nan=0.0)

    # Optional normalization
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        min_samples_split=config.min_samples_split,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    # Cross-validation
    if verbose:
        print("\nPerforming cross-validation...")

    if groups is not None:
        # GroupKFold for leave-one-group-out style validation
        cv = GroupKFold(n_splits=min(config.cv_n_splits, len(np.unique(groups))))
        cv_splits = list(cv.split(X_array, y, groups))
    else:
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=config.cv_n_splits, shuffle=True, random_state=config.random_state)
        cv_splits = list(cv.split(X_array, y))

    # Get cross-validated predictions
    y_proba_cv = np.zeros(len(y))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train on fold
        model_fold = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            min_samples_split=config.min_samples_split,
            class_weight=config.class_weight,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        model_fold.fit(X_train, y_train)

        # Predict on validation set
        y_proba_fold = model_fold.predict_proba(X_val)[:, 1]
        y_proba_cv[val_idx] = y_proba_fold

        # Calculate fold metrics
        if len(np.unique(y_val)) > 1:
            fold_auc_roc = roc_auc_score(y_val, y_proba_fold)
            fold_auc_pr = average_precision_score(y_val, y_proba_fold)
        else:
            fold_auc_roc = np.nan
            fold_auc_pr = np.nan

        fold_metrics.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_positive_val": y_val.sum(),
            "auc_roc": fold_auc_roc,
            "auc_pr": fold_auc_pr,
        })

        if verbose:
            group_info = ""
            if groups is not None:
                val_groups = np.unique(groups[val_idx])
                group_info = f" (groups: {val_groups})"
            print(
                f"  Fold {fold_idx + 1}: AUC-ROC={fold_auc_roc:.3f}, "
                f"AUC-PR={fold_auc_pr:.3f}{group_info}"
            )

    # Calculate overall CV metrics
    cv_metrics = {
        "fold_metrics": fold_metrics,
    }

    if len(np.unique(y)) > 1:
        cv_metrics["auc_roc"] = roc_auc_score(y, y_proba_cv)
        cv_metrics["auc_pr"] = average_precision_score(y, y_proba_cv)

        if verbose:
            print(f"\nOverall CV metrics:")
            print(f"  AUC-ROC: {cv_metrics['auc_roc']:.3f}")
            print(f"  AUC-PR: {cv_metrics['auc_pr']:.3f}")

    # Train final model on all data
    if verbose:
        print("\nTraining final model on all data...")

    model.fit(X_array, y)

    # Get feature importances
    feature_importances = dict(zip(feature_names, model.feature_importances_))

    # Sort by importance
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    if verbose:
        print("\nTop 10 feature importances:")
        for i, (feat, imp) in enumerate(list(feature_importances.items())[:10]):
            print(f"  {i+1}. {feat}: {imp:.4f}")

    # Create model object
    stability_model = StabilityModel(
        model=model,
        feature_names=feature_names,
        feature_importances=feature_importances,
        scaler=scaler,
        config=config,
        train_metadata={
            "train_date": datetime.now(),
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_features": len(feature_names),
        },
        cv_metrics=cv_metrics,
    )

    return stability_model


def evaluate_model(
    model: StabilityModel,
    X: pd.DataFrame,
    y: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained model on a test set.

    Parameters
    ----------
    model : StabilityModel
        Trained model.
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        True labels.
    verbose : bool
        Print evaluation results.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)

    metrics = {}

    if len(np.unique(y)) > 1:
        metrics["auc_roc"] = roc_auc_score(y, y_proba)
        metrics["auc_pr"] = average_precision_score(y, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["accuracy"] = (y_pred == y).mean()

    # Precision, recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    metrics["precision_at_thresholds"] = precision.tolist()
    metrics["recall_at_thresholds"] = recall.tolist()

    if verbose:
        print("\nEvaluation Results:")
        print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.3f}")
        print(f"  AUC-PR: {metrics.get('auc_pr', 'N/A'):.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=["No Event", "Event"]))

    return metrics
