#!/usr/bin/env python3
"""
Prepare training data for Del Mar beach.

This script demonstrates the ML training pipeline:
1. Load and filter rockfall events
2. Map events to transects
3. Extract features from point clouds
4. Combine features and labels
5. Train Random Forest model

Usage:
    python scripts/prepare_delmar_training.py
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from pc_rai.ml.config import MLConfig
from pc_rai.ml.data_prep import load_events, filter_events, print_event_summary
from pc_rai.ml.labels import TransectLabeler
from pc_rai.ml.features import TransectFeatureExtractor, extract_features_from_multiple_scans
from pc_rai.ml.train import prepare_training_data, train_model, evaluate_model


# =============================================================================
# Configuration
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
EVENTS_PATH = BASE_DIR / "utiliies/events/DelMar_events_qc_20260203_085647.csv"
TRANSECTS_PATH = BASE_DIR / "utiliies/transects_10m/transect_lines.shp"
POINT_CLOUD_DIR = BASE_DIR / "output/lessClasses/rai"
OUTPUT_DIR = BASE_DIR / "models"

# Del Mar alongshore range (approximate, from the event data)
# These are the alongshore positions in the transect coordinate system
DELMAR_ALONGSHORE_MIN = 590.0  # MOP range start
DELMAR_ALONGSHORE_MAX = 636.0  # MOP range end

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
    print("DEL MAR TRAINING DATA PREPARATION")
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
    # Step 2: Load transects and map events
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Map events to transects")
    print("=" * 70)

    labeler = TransectLabeler(TRANSECTS_PATH, transect_spacing=10.0)

    # Map events to transects
    labels = labeler.map_events_to_transects(events_filtered)

    print(f"\nLabel statistics:")
    print(f"  Total transects: {len(labels):,}")
    print(f"  Transects with events: {(labels['has_event'] == 1).sum():,}")
    print(f"  Transects without events: {(labels['has_event'] == 0).sum():,}")
    print(f"  Total event-transect overlaps: {labels['event_count'].sum():,}")

    # =========================================================================
    # Step 3: Find and load point clouds
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Extract features from point clouds")
    print("=" * 70)

    # Find Del Mar point clouds
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

    # For this example, use just a few files to keep it fast
    # In production, you'd use all files
    files_to_process = point_cloud_files[:3]  # Limit for demo
    print(f"\nProcessing {len(files_to_process)} files for demo...")

    # Extract features
    features = extract_features_from_multiple_scans(
        las_paths=files_to_process,
        labeler=labeler,
        half_width=CONFIG.transect_half_width,
        verbose=True,
    )

    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")

    # =========================================================================
    # Step 4: Prepare training data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Prepare training data")
    print("=" * 70)

    # For this simple example, we'll use all events as labels
    # (not considering temporal alignment between scans and events)
    # In production, you'd want to match each scan to events that occurred
    # AFTER that scan date

    X, y, merged = prepare_training_data(
        features_df=features,
        labels_df=labels,
        label_column="has_event",
        merge_on=["transect_id"],
    )

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
    # Step 6: Save model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Save model")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "delmar_stability_rf.joblib"
    model.save(model_path)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Events processed: {len(events_filtered):,}")
    print(f"  Transects: {len(labels):,}")
    print(f"  Training samples: {len(X):,}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Positive class rate: {y.mean()*100:.1f}%")
    print(f"  Model saved to: {model_path}")

    if model.cv_metrics.get("auc_roc"):
        print(f"\nCross-validation performance:")
        print(f"  AUC-ROC: {model.cv_metrics['auc_roc']:.3f}")
        print(f"  AUC-PR: {model.cv_metrics['auc_pr']:.3f}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
