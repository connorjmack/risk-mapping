#!/usr/bin/env python3
"""
Prepare temporally-aligned training data for Del Mar beach.

This script implements a case-control study design where:
- Cases: pre-failure morphology features (from scans before events)
- Controls: features from transects without subsequent events

This ensures we're training on predictive features (pre-failure state)
rather than descriptive features (post-failure state).

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
from pc_rai.ml.train import prepare_training_data, train_model


# =============================================================================
# Configuration
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
EVENTS_PATH = BASE_DIR / "utiliies/events/DelMar_events_qc_20260203_085647.csv"
TRANSECTS_PATH = BASE_DIR / "utiliies/transects_10m/transect_lines.shp"
POINT_CLOUD_DIR = BASE_DIR / "output/lessClasses/rai"
OUTPUT_DIR = BASE_DIR / "models"

# ML Configuration
CONFIG = MLConfig(
    min_volume=5.0,  # Only events >= 5 m³
    qc_flags_include=["real", "unreviewed"],  # Include unreviewed as real
    qc_flags_exclude=["construction", "noise"],
    transect_half_width=5.0,  # 10m total corridor width
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
)

# Temporal alignment settings
LOOKFORWARD_DAYS = 365  # Look for events up to 1 year after scan
MIN_DAYS_BEFORE = 7  # Scan must be at least 7 days before event
CONTROL_RATIO = 1.0  # Equal number of controls and cases


def find_delmar_point_clouds(point_cloud_dir: Path) -> list:
    """Find point cloud files that cover Del Mar.

    Looks for files with 'DelMar' in the name or with MOP ranges
    that overlap Del Mar (approximately 590-636).
    """
    all_files = list(point_cloud_dir.glob("*_rai.laz"))

    delmar_files = []
    for f in all_files:
        name = f.name.lower()
        # Check if filename contains 'delmar'
        if "delmar" in name:
            delmar_files.append(f)
        # Also check MOP range in filename (format: YYYYMMDD_XXXXX_XXXXX)
        # where XXXXX are MOP numbers * 100 (e.g., 00590 = MOP 5.90)
        else:
            parts = f.stem.split("_")
            if len(parts) >= 3:
                try:
                    mop_start = int(parts[1]) / 100  # Convert to MOP
                    mop_end = int(parts[2]) / 100
                    # Check if range overlaps Del Mar
                    if mop_start <= 6.36 and mop_end >= 5.90:
                        delmar_files.append(f)
                except ValueError:
                    pass

    # Sort by date (filename starts with YYYYMMDD)
    delmar_files.sort(key=lambda f: f.name[:8])

    return delmar_files


def main():
    print("=" * 70)
    print("DEL MAR TEMPORALLY-ALIGNED TRAINING")
    print("=" * 70)
    print("\nThis training uses a case-control study design:")
    print("- Cases: pre-failure morphology (scans taken BEFORE events)")
    print("- Controls: transects without subsequent events")
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
    # Step 2: Discover available scans
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Discover Del Mar point cloud scans")
    print("=" * 70)

    point_cloud_files = find_delmar_point_clouds(POINT_CLOUD_DIR)

    if not point_cloud_files:
        print("ERROR: No point cloud files found for Del Mar!")
        print(f"Searched in: {POINT_CLOUD_DIR}")
        return

    print(f"\nFound {len(point_cloud_files)} point cloud files for Del Mar:")
    for f in point_cloud_files[:5]:
        print(f"  - {f.name}")
    if len(point_cloud_files) > 5:
        print(f"  ... and {len(point_cloud_files) - 5} more")

    # Parse scan dates
    scan_dates = []
    for f in point_cloud_files:
        try:
            date_str = f.stem[:8]
            scan_date = pd.to_datetime(date_str, format="%Y%m%d")
            scan_dates.append(scan_date)
        except Exception:
            pass

    if scan_dates:
        print(f"\nScan date range: {min(scan_dates).strftime('%Y-%m-%d')} to "
              f"{max(scan_dates).strftime('%Y-%m-%d')}")

    # =========================================================================
    # Step 3: Create temporally-aligned dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Create temporally-aligned training dataset")
    print("=" * 70)

    print(f"\nSettings:")
    print(f"  Lookforward window: {LOOKFORWARD_DAYS} days")
    print(f"  Min days before event: {MIN_DAYS_BEFORE} days")
    print(f"  Control ratio: {CONTROL_RATIO}")

    dataset, aligner = create_temporal_training_data(
        events=events_filtered,
        point_cloud_dir=POINT_CLOUD_DIR,
        transects_path=TRANSECTS_PATH,
        half_width=CONFIG.transect_half_width,
        lookforward_days=LOOKFORWARD_DAYS,
        min_days_before=MIN_DAYS_BEFORE,
        control_ratio=CONTROL_RATIO,
        scan_pattern="*_rai.laz",
        verbose=True,
    )

    # =========================================================================
    # Step 4: Prepare features and labels
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Prepare training features")
    print("=" * 70)

    # Auto-detect feature columns
    exclude_cols = [
        "transect_id", "alongshore_dist", "tr_id", "scan_date", "source_file",
        "event_count", "has_event", "total_volume", "max_volume", "point_count",
        "label", "event_date", "days_to_event", "event_volume", "event_idx",
    ]
    exclude_cols += [c for c in dataset.columns if c.endswith("_x") or c.endswith("_y")]

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
    # Step 5: Train model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Train Random Forest model")
    print("=" * 70)

    model = train_model(
        X=X,
        y=y,
        config=CONFIG,
        verbose=True,
    )

    # =========================================================================
    # Step 6: Save model and dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Save model and dataset")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = OUTPUT_DIR / "delmar_stability_rf_temporal.joblib"
    model.save(model_path)

    # Save training dataset for inspection
    dataset_path = OUTPUT_DIR / "delmar_training_temporal.csv"
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

    # Compare to baseline
    print("\n" + "-" * 40)
    print("COMPARISON TO BASELINE (non-temporal):")
    print("-" * 40)
    print("  Baseline AUC-ROC: 0.602")
    print("  Baseline AUC-PR: 0.694")
    if model.cv_metrics.get("auc_roc"):
        delta_roc = model.cv_metrics["auc_roc"] - 0.602
        delta_pr = model.cv_metrics["auc_pr"] - 0.694
        print(f"  Temporal AUC-ROC: {model.cv_metrics['auc_roc']:.3f} ({delta_roc:+.3f})")
        print(f"  Temporal AUC-PR: {model.cv_metrics['auc_pr']:.3f} ({delta_pr:+.3f})")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
