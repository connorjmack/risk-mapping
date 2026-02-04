#!/usr/bin/env python3
"""
Prepare temporally-aligned training data for Del Mar beach using 1m polygons.

This script implements a case-control study design where:
- Cases: pre-failure morphology features (from scans before events)
- Controls: features from polygons without subsequent events

Uses 1m polygon shapefiles for precise spatial matching.
Polygon IDs correspond directly to alongshore meter positions.

Usage:
    python scripts/prepare_delmar_training_temporal.py
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from pc_rai.ml.config import MLConfig
from pc_rai.ml.data_prep import load_events, filter_events, print_event_summary
from pc_rai.ml.temporal import create_temporal_training_data
from pc_rai.ml.train import train_model


# =============================================================================
# Configuration
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
EVENTS_PATH = BASE_DIR / "utiliies/events/DelMar_events_qc_20260203_085647.csv"
POLYGON_SHAPEFILE = BASE_DIR / "utiliies/polygons_1m/DelMarPolygons595to620at1m/DelMarPolygons595to620at1m"
POINT_CLOUD_DIR = BASE_DIR / "output/lessClasses/rai"
OUTPUT_DIR = BASE_DIR / "models"

# ML Configuration
CONFIG = MLConfig(
    min_volume=5.0,  # Only events >= 5 m³
    qc_flags_include=["real", "unreviewed"],  # Include unreviewed as real
    qc_flags_exclude=["construction", "noise"],
    transect_half_width=5.0,  # Not used for polygons, but kept for compatibility
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
)

# Temporal alignment settings
LOOKFORWARD_DAYS = 365  # Look for events up to 1 year after scan
MIN_DAYS_BEFORE = 7  # Scan must be at least 7 days before event
CONTROL_RATIO = 1.0  # Equal number of controls and cases


def main():
    print("=" * 70)
    print("DEL MAR TEMPORALLY-ALIGNED TRAINING (1m Polygons)")
    print("=" * 70)
    print("\nThis training uses a case-control study design:")
    print("- Cases: pre-failure morphology (scans taken BEFORE events)")
    print("- Controls: polygons without subsequent events")
    print("- Spatial resolution: 1m polygons (IDs = alongshore meters)")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load and filter events
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Load and filter events")
    print("=" * 70)

    events = load_events(EVENTS_PATH)

    # Filter events
    events_filtered = filter_events(
        events,
        config=CONFIG,
        verbose=True,
    )

    print_event_summary(events_filtered)

    # =========================================================================
    # Step 2: Create temporally-aligned dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Create temporally-aligned training dataset")
    print("=" * 70)

    print(f"\nSettings:")
    print(f"  Polygon shapefile: {POLYGON_SHAPEFILE.name}")
    print(f"  Lookforward window: {LOOKFORWARD_DAYS} days")
    print(f"  Min days before event: {MIN_DAYS_BEFORE} days")
    print(f"  Control ratio: {CONTROL_RATIO}")

    dataset, aligner = create_temporal_training_data(
        events=events_filtered,
        point_cloud_dir=POINT_CLOUD_DIR,
        polygon_shapefile=POLYGON_SHAPEFILE,
        lookforward_days=LOOKFORWARD_DAYS,
        min_days_before=MIN_DAYS_BEFORE,
        control_ratio=CONTROL_RATIO,
        scan_pattern="*_rai.laz",
        verbose=True,
    )

    if len(dataset) == 0:
        print("\nERROR: No training samples generated!")
        print("Check that events overlap with polygon extent and scan dates.")
        return

    # =========================================================================
    # Step 3: Prepare features and labels
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Prepare training features")
    print("=" * 70)

    # Auto-detect feature columns
    exclude_cols = [
        "polygon_id", "scan_date", "point_count",
        "label", "event_date", "days_to_event", "event_volume", "event_idx",
    ]

    feature_columns = [
        c for c in dataset.columns
        if c not in exclude_cols
        and dataset[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    print(f"\nFeature columns ({len(feature_columns)}):")
    for col in feature_columns:
        print(f"  - {col}")

    X = dataset[feature_columns].copy()
    y = dataset["label"].values

    # Handle NaN values
    X = X.fillna(0)

    print(f"\nTraining data shape: {X.shape}")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"Class balance: {y.mean()*100:.1f}% positive")

    # =========================================================================
    # Step 4: Train model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Train Random Forest model")
    print("=" * 70)

    model = train_model(
        X=X,
        y=y,
        config=CONFIG,
        verbose=True,
    )

    # =========================================================================
    # Step 5: Save model and dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Save model and dataset")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = OUTPUT_DIR / "delmar_stability_rf_temporal_1m.joblib"
    model.save(model_path)

    # Save training dataset for inspection
    dataset_path = OUTPUT_DIR / "delmar_training_temporal_1m.csv"
    dataset.to_csv(dataset_path, index=False)
    print(f"Training dataset saved to {dataset_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Events processed: {len(events_filtered):,}")
    print(f"  Training samples: {len(X):,}")
    print(f"    - Cases (pre-failure): {(y == 1).sum():,}")
    print(f"    - Controls: {(y == 0).sum():,}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Model saved to: {model_path}")

    if model.cv_metrics.get("auc_roc"):
        print(f"\nCross-validation performance:")
        print(f"  AUC-ROC: {model.cv_metrics['auc_roc']:.3f}")
        print(f"  AUC-PR: {model.cv_metrics['auc_pr']:.3f}")

    # Show top features
    if model.feature_importances:
        print("\nTop 10 features by importance:")
        sorted_features = sorted(
            model.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for name, importance in sorted_features:
            print(f"  {name}: {importance:.4f}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
