# ML Training Pipeline Outline
## Rockfall Stability Score Prediction

### Overview

Train a Random Forest classifier to predict rockfall probability using pre-failure cliff morphology. Training uses 1m polygon resolution; inference aggregates to 10m chunks.

**Key principle:** Compute features on entire subsampled cloud FIRST, then aggregate into polygons AFTER.

---

## Pipeline Steps

### Step 1: Identify Pre-Event Surveys

**Input:**
- `all_noveg_files.csv` — list of all available noveg LAS files
- `DelMar_events_qc_*.csv` — event catalog with geometry

**Process:**
```
For each event >= 5m³ (qc_flag in ['real', 'unreviewed']):
    Find the most recent survey BEFORE event start_date
    (with minimum gap of 7 days)
```

**Output:** `pre_event_surveys.csv`
| survey_date | survey_file | event_date | event_id | days_before | event_volume |

---

### Step 2: Subsample & Extract Point-Level Features

**Input:**
- Noveg LAS files (from Step 1)

**Process:**
```
For each survey_file in pre_event_surveys:
    1. Load full LAS point cloud
    2. Subsample to 50cm voxel grid (entire cloud)
    3. Compute features at EVERY subsampled point:
       - slope (from normal vector)
       - roughness_small (0.5m neighborhood)
       - roughness_large (2.0m neighborhood)
       - roughness_ratio (small/large)
       - height (Z relative to local base)
    4. Save subsampled cloud with features (full extent)
```

**Output:** `subsampled/{survey_date}_subsampled_features.laz`

Each point has: X, Y, Z, slope, roughness_small, roughness_large, r_ratio, height

---

### Step 3: Assign Points to Polygons (UTM → Polygon ID)

**Input:**
- Subsampled point clouds with features (from Step 2)
- `DelMarPolygons595to620at1m.shp` — 1m polygon shapefile

**Process:**
```
# Derive mapping from polygon shapefile
For each polygon in shapefile:
    y_centroid = polygon.geometry.centroid.y
    polygon_id = polygon['Id']

# Build lookup: polygon_id = f(UTM_Y)
# Likely: polygon_id = round((Y - y_offset) / scale)

# Apply to point cloud
For each point in subsampled cloud:
    point.polygon_id = utm_y_to_polygon_id(point.Y)
```

**Output:** Point cloud now has `polygon_id` attribute for each point

---

### Step 4: Aggregate Features by Polygon (Upper/Lower Split)

**Input:**
- Subsampled point clouds with features + polygon_id (from Step 3)

**Process:**
```
For each unique polygon_id:
    points = all points with this polygon_id
    z_median = median(points.Z)

    lower_points = points where Z < z_median
    upper_points = points where Z >= z_median

    For zone in [lower, upper]:
        For feature in [slope, roughness_small, roughness_large, r_ratio, height]:
            compute: mean, max, std, p90

# Result: 5 features × 4 stats × 2 zones = 40 features per polygon
```

**Output:** `polygon_features/{survey_date}_polygon_features.csv`
| survey_date | polygon_id | slope_mean_lower | slope_max_lower | ... | height_p90_upper |

---

### Step 5: Label Polygons from Events

**Input:**
- Polygon features (from Step 4)
- Events catalog
- Pre-event survey mapping (from Step 1)

**Process:**
```
For each (survey, event) pair from Step 1:
    # Get polygons affected by this event
    alongshore_start = event.alongshore_start_m
    alongshore_end = event.alongshore_end_m
    affected_polygon_ids = range(floor(alongshore_start), ceil(alongshore_end) + 1)

    # Label polygons for this survey
    For each polygon_id in survey's polygon_features:
        if polygon_id in affected_polygon_ids:
            label = 1  # POSITIVE (pre-failure morphology)
        else:
            label = 0  # NEGATIVE (no event observed)
```

**Output:** `training_data.csv`
| survey_date | polygon_id | [40 features] | label | days_to_event | event_volume |

---

### Step 6: Train Random Forest

**Input:**
- `training_data.csv`

**Process:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

# Prepare data
X = df[feature_columns]  # 40 features
y = df['label']

# Train
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)

# Cross-validation (leave-one-year-out)
years = df['survey_date'].dt.year
for train_idx, test_idx in GroupKFold(n_splits=n_years).split(X, y, groups=years):
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    evaluate(rf, X.iloc[test_idx], y.iloc[test_idx])
```

**Metrics:**
- AUC-ROC
- AUC-PR (better for imbalanced data)
- Feature importance

**Output:** `models/rf_stability_score.pkl`

---

### Step 7: Inference (10m Aggregation)

**Input:**
- New survey point cloud (never seen before)
- Trained model

**Process:**
```
1. Subsample & extract features (same as Step 2)
2. Assign polygon IDs (same as Step 3)
3. Aggregate by polygon (same as Step 4)
4. Predict probability for each 1m polygon:
   probs = rf.predict_proba(X)[:, 1]

5. Aggregate to 10m chunks:
   For each 10m chunk:
       chunk_polygons = polygons in this 10m range
       chunk_prob = mean(polygon_probs)  # or max for conservative
```

**Output:** `predictions/{survey_date}_risk_10m.csv`
| chunk_start_m | chunk_end_m | mean_probability | max_probability | risk_class |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                              │
└─────────────────────────────────────────────────────────────────────┘

[Events CSV] + [all_noveg_files.csv]
        │
        ▼
┌─────────────────────┐
│ Step 1: Identify    │
│ pre-event surveys   │
└─────────────────────┘
        │
        ▼
[List of LAS files to process]
        │
        ▼
┌─────────────────────┐
│ Step 2: Subsample   │    ← Process ENTIRE cloud
│ + Extract Features  │    ← Features at EVERY point
└─────────────────────┘
        │
        ▼
[Subsampled cloud: X, Y, Z, slope, roughness, height, ...]
        │
        ▼
┌─────────────────────┐
│ Step 3: Assign      │    ← Simple UTM Y lookup
│ Polygon IDs         │    ← polygon_id = f(Y)
└─────────────────────┘
        │
        ▼
[Cloud with polygon_id per point]
        │
        ▼
┌─────────────────────┐
│ Step 4: Aggregate   │    ← Group by polygon_id
│ by Polygon          │    ← Upper/lower split
└─────────────────────┘    ← mean, max, std, p90
        │
        ▼
[40 features per polygon]
        │
        ▼
┌─────────────────────┐
│ Step 5: Label       │    ← Match to events
│ Polygons            │    ← 1 = pre-failure, 0 = no event
└─────────────────────┘
        │
        ▼
[training_data.csv]
        │
        ▼
┌─────────────────────┐
│ Step 6: Train RF    │
└─────────────────────┘
        │
        ▼
[Trained Model]


┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                              │
└─────────────────────────────────────────────────────────────────────┘

[New LAS file]
        │
        ▼
Steps 2-4 (same pipeline)
        │
        ▼
[40 features per polygon]
        │
        ▼
┌─────────────────────┐
│ Step 7: Predict     │    ← Apply trained model
│ + Aggregate 10m     │    ← Average predictions
└─────────────────────┘
        │
        ▼
[10m risk scores]
```

---

## Module Structure

```
pc_rai/
├── ml/
│   ├── survey_selection.py    # Step 1: find pre-event surveys
│   ├── feature_extraction.py  # Step 2: subsample + compute features
│   ├── polygon_assignment.py  # Step 3: UTM → polygon ID
│   ├── aggregation.py         # Step 4: aggregate by polygon
│   ├── labeling.py            # Step 5: attach labels
│   ├── training.py            # Step 6: train RF
│   └── inference.py           # Step 7: predict + aggregate

scripts/
├── 01_identify_surveys.py     # Run Step 1
├── 02_extract_features.py     # Run Step 2
├── 03_build_training_data.py  # Run Steps 3-5
├── 04_train_model.py          # Run Step 6
└── 05_predict.py              # Run Step 7

data/
├── pre_event_surveys.csv      # Step 1 output
├── subsampled/                # Step 2 output (LAZ files)
├── polygon_features/          # Step 4 output
├── training_data.csv          # Step 5 output
└── models/                    # Step 6 output
```

---

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_volume` | 5.0 m³ | Filter noise, focus on significant events |
| `min_days_before` | 7 days | Ensure scan captures pre-failure state |
| `voxel_size` | 0.5 m | Balance resolution vs computation |
| `roughness_small` | 0.5 m | Local surface texture |
| `roughness_large` | 2.0 m | Broader morphology |
| `elevation_split` | median Z per polygon | Adaptive upper/lower zones |
| `class_weight` | balanced | Handle class imbalance |
| `cv_strategy` | leave-one-year-out | Test temporal generalization |
| `inference_aggregation` | 10 m | Practical output resolution |

---

## Feature Summary

**Per-point features (Step 2):**
- slope
- roughness_small
- roughness_large
- roughness_ratio
- height

**Per-polygon features (Step 4):**

| Feature | Lower Zone | Upper Zone |
|---------|------------|------------|
| slope | mean, max, std, p90 | mean, max, std, p90 |
| roughness_small | mean, max, std, p90 | mean, max, std, p90 |
| roughness_large | mean, max, std, p90 | mean, max, std, p90 |
| r_ratio | mean, max, std, p90 | mean, max, std, p90 |
| height | mean, max, std, p90 | mean, max, std, p90 |

**Total: 5 × 4 × 2 = 40 features per polygon**

---

## Next Steps

1. [ ] Implement Step 1: Survey selection
2. [ ] Implement Step 2: Feature extraction (subsample + features)
3. [ ] Implement Step 3: Polygon assignment (derive UTM→polygon mapping)
4. [ ] Implement Steps 4-5: Aggregation + labeling
5. [ ] Implement Step 6: Training
6. [ ] Evaluate and iterate
