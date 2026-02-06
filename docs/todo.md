# PC-RAI Development Todo

> **Instructions for Claude Code**: Work through these tasks sequentially. Each task has clear acceptance criteria. Mark tasks complete with `[x]` as you finish them. Run the specified tests before moving to the next task. Reference `docs/prd.md` for detailed specifications.

---

## Project Status

- **Current Phase**: v1.0 Complete, v2.x ML Pipeline — Full-Scale Training
- **Last Completed Task**: Prototype RF trained on test subset (20 surveys, 5 beaches)
- **Next Up**: **Full-scale pipeline — process all cropped LAS files, train on all data**
- **Tests Passing**: 228 (7 polygon indexing + 18 polygon features output tests)
- **Blocking Issues**: None — prototype pipeline validated, ready to scale up

### Prototype Results (Test Subset — 20 surveys, 5 beaches)
- StratifiedKFold CV: AUC-ROC=0.855 — **inflated due to spatial data leakage**
- **Temporal CV (leave-one-year-out): AUC-ROC=0.696, AUC-PR=0.652** — honest metric
- GroupKFold by location: AUC-ROC=0.616 — tests spatial generalization

### Full-Scale Training Strategy (NEXT)
- **Volume threshold**: Raise from 5 m³ → **10 m³** (cleaner signal, 2,604 events)
- **Data**: Process ALL cropped LAS files across 6 beaches (2017-2025)
- **Hold-out**: Reserve **2025 data as true test set** (never seen during training/CV)
- **CV**: Leave-one-year-out on 2017-2024 for hyperparameter tuning
- **Goal**: Better model from more training data + cleaner labels + honest evaluation

### v2.x ML Pipeline Progress

**Key Design**:
1. Compute features on subsampled point cloud
2. Aggregate into 1m alongshore polygons × elevation zones (lower/middle/upper)
3. Case-control labeling from pre-event survey matches
4. Train Random Forest for rockfall probability

**Elevation Zones**: Each 1m polygon is split into relative thirds (lower/middle/upper) to capture different cliff behavior at different heights.

---

#### Step 1: Identify Pre-Event Surveys ✅

**Module**: `pc_rai/ml/survey_selection.py`
**Script**: `scripts/01_identify_surveys.py`

**Subtasks**:
- [x] **1.1** Load `all_noveg_files.csv` and parse dates from filenames (YYYYMMDD prefix)
  - Test: `assert df['survey_date'].notna().all()`
- [x] **1.2** Load event CSV and filter to events >= 5 m³ with `qc_flag` in `['real', 'unreviewed']`
  - Test: `assert (filtered_events['volume'] >= 5.0).all()`
- [x] **1.3** For each event, find most recent survey BEFORE `start_date` (min 7-day gap)
  - Test: `assert (pairs['days_before'] >= 7).all()`
- [x] **1.4** Handle edge cases: events with no valid pre-event survey → skip with warning
  - Test: script logs warning count, returns valid pairs only
- [x] **1.5** Output deduplicated list of surveys to `data/pre_event_surveys.csv`
  - Columns: `survey_date, survey_file, event_date, event_id, days_before, event_volume`
  - Test: `assert Path('data/pre_event_surveys.csv').exists()`

**Verify**:
```bash
python scripts/01_identify_surveys.py \
    --events utiliies/events/DelMar_events_qc_*.csv \
    --surveys utiliies/events/all_noveg_files.csv \
    --output data/pre_event_surveys.csv
# Expected: CSV created with N survey-event pairs
```

---

#### Step 2: Subsample & Extract Point-Level Features ✅

**Module**: `pc_rai/ml/feature_extraction.py`
**Script**: `scripts/02_extract_features.py`

**Subtasks**:
- [x] **2.1** Implement voxel grid subsampling (50cm default)
  - Test: 10M points → ~400K points (97% reduction) ✓
- [x] **2.2** Compute normals via CloudComPy MST (preferred) or PCA fallback
  - Use `--subsample-only` then run `compute_normals_mst.py` separately
  - Test: output has `NormalX`, `NormalY`, `NormalZ` dims ✓
- [x] **2.3** Compute slope at every subsampled point
  - Test: slope range 0-180° ✓
- [x] **2.4** Compute roughness_small (r=1.0m) and roughness_large (r=2.5m)
  - Radii increased to ensure sufficient neighbors at 0.5m voxel spacing
- [x] **2.5** Compute roughness_ratio = roughness_small / roughness_large
  - Test: division by zero handled (set to NaN) ✓
- [x] **2.6** Compute relative height (Z - local Z_min within 5m horizontal)
  - Test: height >= 0 ✓
- [x] **2.7** Compute eigenvalue features at large scale (r=2.5m):
  - `planarity`: (λ2 - λ3) / λ1 - high for flat surfaces
  - `linearity`: (λ1 - λ2) / λ1 - high for edges/ridges
  - `sphericity`: λ3 / λ1 - high for scattered/rough areas
  - `curvature`: λ3 / (λ1 + λ2 + λ3) - surface variation
- [x] **2.8** Save as LAZ with extra dims (9 features total)
  - Output: `data/test_features/*_subsampled_features.laz` ✓

**Verify**:
```bash
# Step 2a: Subsample only
python scripts/02_extract_features.py \
    --input-dir data/test_data/no_veg/ \
    --output-dir data/test_subsampled/ \
    --subsample-only

# Step 2b: Compute normals (CloudComPy environment)
python scripts/compute_normals_mst.py data/test_subsampled/ \
    --output-dir data/test_subsampled_normals/

# Step 2c: Extract features
python scripts/02_extract_features.py \
    --input-dir data/test_subsampled_normals/ \
    --output-dir data/test_features/
```

---

#### Step 3: Aggregate Features to Polygon-Zones ✅ (re-run needed)

**Module**: `pc_rai/ml/polygon_aggregation.py`
**Script**: `scripts/03_aggregate_polygons.py`

**Subtasks**:
- [x] **3.1** Bin points into 1m alongshore polygons (auto-detect alongshore axis)
  - Test: `assert n_bins == ceil(alongshore_extent)`
- [x] **3.2** Split each polygon into lower/middle/upper thirds by relative elevation
  - Test: all three zones have points (unless polygon is very flat)
- [x] **3.3** Compute 7 statistics for each feature in each zone
  - Features: `slope, roughness_small, roughness_large, roughness_ratio, height, planarity, linearity, sphericity, curvature`
  - Stats: `mean, std, min, max, p10, p50, p90`
  - Zones: `lower, middle, upper`
  - Total: 9 features × 7 stats = 63 feature columns per zone
- [x] **3.4** Handle polygon-zones with <5 points → skip
- [x] **3.5** Save combined output to `data/polygon_features.csv`
  - Columns: `survey_date, survey_file, location, polygon_id, alongshore_m, zone, [features]`
- [x] **3.6** Fix polygon indexing: local alongshore_m framework
  - DelMar: `alongshore_m = Id - min(Id)` (0-based from shapefile Id field)
  - Other beaches: `alongshore_m = centroid_y - min(centroid_y)` (local meters)
  - Tests: `tests/test_polygon_indexing.py` (7 tests passing)
  - **Needs re-run** to regenerate `polygon_features.csv` with corrected coordinates

**Verify**:
```bash
python scripts/03_aggregate_polygons.py \
    --input-dir data/test_features/ \
    --output data/polygon_features.csv
# Expected: CSV with ~3 rows per polygon (one per zone) per survey
```

---

#### Step 4: Assemble Training Data (Case-Control) ✅

**Module**: `pc_rai/ml/training_data.py`
**Script**: `scripts/04_assemble_training_data.py`

**Subtasks**:
- [x] **4.1** Load polygon features and pre-event survey matches
  - Match surveys by date + location key
- [x] **4.2** Label polygon-zones as cases or controls
  - Case (label=1): Polygon alongshore range overlaps with event extent
  - Control (label=0): No subsequent event at this polygon
- [x] **4.3** Optionally consider elevation overlap (event elevation vs zone z_range)
- [x] **4.4** Balance controls with cases (default 1:1 ratio, configurable)
- [x] **4.5** Save to `data/training_data.csv`
  - Columns: `[metadata], [features], label, event_volume, event_id, days_before_event`

**Verify**:
```bash
python scripts/04_assemble_training_data.py \
    --features data/polygon_features.csv \
    --surveys data/test_pre_event_surveys.csv \
    --output data/training_data.csv \
    --min-volume 5.0 \
    --control-ratio 1.0
# Expected: Balanced dataset with ~50% cases, ~50% controls
```

---

#### Step 5: Train Random Forest ✅

**Module**: `pc_rai/ml/train.py`
**Script**: `scripts/05_train_model.py`

**Subtasks**:
- [x] **5.1** Load `training_data.csv` and extract feature columns automatically
  - 47,778 rows, 63 features, balanced (23,889 cases / 23,889 controls)
- [x] **5.2** Handle class imbalance with `class_weight='balanced'`
- [x] **5.3** Train RandomForestClassifier (n_estimators=100, max_depth=None)
- [x] **5.4** Compute evaluation metrics: AUC-ROC, AUC-PR via cross-validation
  - StratifiedKFold (5-fold) CV: **AUC-ROC=0.855, AUC-PR=0.858**
  - Leave-one-beach-out CV: AUC-ROC=0.853-0.857 per fold (very tight variance)
  - Accuracy=77.5%, Sensitivity=81%, Specificity=74% at threshold=0.5
  - Previous GroupKFold result: AUC-ROC=0.616 (before stratified approach)
- [x] **5.5** Extract and rank feature importances
  - Top individual: height_p10 (0.034), height_min (0.030), slope_min (0.029)
  - Top groups: height (0.166), slope (0.132), linearity (0.128), planarity (0.112)
  - All 9 feature groups contribute (range 0.085-0.166), no single feature dominates
- [x] **5.6** Save trained model to `models/rf_model.joblib`
  - Model + metadata JSON saved
- [x] **5.7** Generate diagnostic plots
  - Confusion matrices (per fold + overall)
  - ROC curves with AUC per fold
  - PR curves with AP per fold
  - Feature importances (top 20 + by group)
  - Probability distributions (cases vs controls, class separation ~0.27)
  - CV performance bar chart by location
  - Output: `output/training_results/stratified/`

**Key Results**:
- Model generalizes well across beaches (fold AUC spread < 0.004)
- Physically interpretable features: height and slope dominate, eigenvalue features add value
- Probability distributions show good separation (cases peak ~0.7-0.8, controls ~0.2-0.3)
- 26% FPR may include polygons that would fail given longer observation window

**Verify**:
```bash
python scripts/05_train_model.py \
    --input data/training_data.csv \
    --output models/rf_model.joblib \
    --group-by location -v
# Expected: Model saved with CV metrics printed
```

---

#### Step 6: Full-Scale Pipeline Processing 🔜 **NEXT PRIORITY**

**Goal**: Process all available cropped LAS files through the pipeline and retrain on the full dataset with revised parameters.

**Data Inventory** (52,365 events across 6 beaches, 2017-2025):

| Location | Total Events | Events ≥10 m³ | Date Range |
|----------|-------------|--------------|------------|
| DelMar | 11,476 | 570 | 2017-03 to 2025-11 |
| SanElijo | 7,342 | 285 | 2017-12 to 2025-11 |
| Solana | 5,850 | 406 | 2017-12 to 2025-11 |
| Torrey | 6,087 | 562 | 2017-11 to 2025-11 |
| Encinitas | 12,726 | 488 | 2017-12 to 2025-11 |
| Blacks | 8,884 | 293 | 2017-11 to 2025-10 |

**Subtasks**:
- [ ] **6.1** Inventory all cropped LAS files on network drive (`/Volumes/group/.../results/{Location}/cropped/`)
  - Count files per location, verify date coverage
- [ ] **6.2** Update `scripts/01_identify_surveys.py` to work with all 6 beaches
  - Raise volume threshold to 10 m³
  - Include all QC'd event files from `utiliies/events/qc_ed/`
  - Output: comprehensive `data/full_pre_event_surveys.csv`
- [ ] **6.3** Subsample all identified survey LAS files (50cm voxel grid)
  - Script: `scripts/02_extract_features.py --subsample-only`
  - Output: `data/full_subsampled/`
- [ ] **6.4** Compute normals via CloudComPy (separate conda env, batch)
  - Script: `scripts/compute_normals_mst.py`
  - Output: `data/full_subsampled_normals/`
  - **Bottleneck**: This is the slowest step — may need overnight/weekend run
- [ ] **6.5** Extract features (slope, roughness, eigenvalues)
  - Script: `scripts/02_extract_features.py`
  - Output: `data/full_features/`
- [ ] **6.6** Aggregate to polygon-zones (1m × elevation thirds)
  - Script: `scripts/03_aggregate_polygons.py`
  - Ensure polygon shapefiles exist for all 6 beaches in `utiliies/polygons_1m/`
  - Output: `data/full_polygon_features.csv`
- [ ] **6.7** Assemble training data with 10 m³ threshold, hold out 2025
  - Script: `scripts/04_assemble_training_data.py --min-volume 10.0`
  - Split: 2017-2024 for train/CV, 2025 for held-out test
  - Output: `data/full_training_data.csv`, `data/full_test_data_2025.csv`
- [ ] **6.8** Train and evaluate on full dataset
  - Temporal CV (leave-one-year-out) on 2017-2024
  - Final evaluation on held-out 2025 data
  - Script: `scripts/05_train_model.py --group-by year`
  - Output: `models/rf_model_full.joblib`, `output/training_results/full_scale/`
- [ ] **6.9** Compare full-scale vs prototype results
  - Document improvement from more data + cleaner threshold

**Verify**:
```bash
# Steps 6.1-6.5 require network drive access + CloudComPy environment
# Steps 6.6-6.8 run in normal Python environment:
python scripts/03_aggregate_polygons.py \
    --input-dir data/full_features/ \
    --output data/full_polygon_features.csv

python scripts/04_assemble_training_data.py \
    --features data/full_polygon_features.csv \
    --surveys data/full_pre_event_surveys.csv \
    --output data/full_training_data.csv \
    --min-volume 10.0

python scripts/05_train_model.py \
    --input data/full_training_data.csv \
    --output models/rf_model_full.joblib \
    --group-by year -v
```

---

#### Step 7: Cross-Validation (Leave-One-Year-Out) ⏳

**Module**: `pc_rai/ml/training.py` (extend)
**Script**: `scripts/04_train_model.py --cv leave-one-year-out`

**Subtasks**:
- [ ] **7.1** Extract year from `survey_date` for grouping
  - Test: years span expected range (e.g., 2016-2024)
- [ ] **7.2** Implement leave-one-year-out CV using GroupKFold
  - Test: each fold excludes exactly one year
- [ ] **7.3** Compute metrics per fold: AUC-ROC, AUC-PR
  - Test: results dict has entry per year
- [ ] **7.4** Summarize CV results: mean ± std across folds
  - Test: summary table printed/saved
- [ ] **7.5** Visualize CV performance (optional: boxplot of AUC per year)
  - Output: `figures/cv_results.png`

**Verify**:
```bash
python scripts/04_train_model.py \
    --training-data data/training_data.csv \
    --output models/rf_stability_score.joblib \
    --cv leave-one-year-out
# Expected: Per-year metrics + summary stats
```

---

#### Step 8: Inference Pipeline (10m Aggregation) ⏳

**Module**: `pc_rai/ml/inference.py`
**Script**: `scripts/05_predict.py`

**Subtasks**:
- [ ] **8.1** Load trained model from `.joblib`
  - Test: model loads, has `predict_proba` method
- [ ] **8.2** Reuse Steps 2-4 pipeline for new survey (subsample → features → polygon agg)
  - Test: output has same 40 features as training
- [ ] **8.3** Predict probability for each 1m polygon
  - Test: `probs.shape[0] == n_polygons`, `0 <= probs.all() <= 1`
- [ ] **8.4** Aggregate 1m predictions to 10m chunks (mean and max)
  - Test: output has `chunk_start_m, chunk_end_m, mean_prob, max_prob`
- [ ] **8.5** Classify risk level: Low (<0.3), Medium (0.3-0.6), High (>0.6)
  - Test: `risk_class.isin(['Low', 'Medium', 'High']).all()`
- [ ] **8.6** Save predictions to `predictions/{survey_date}_risk_10m.csv`
  - Test: CSV exists with expected columns

**Verify**:
```bash
python scripts/05_predict.py \
    --model models/rf_stability_score.joblib \
    --input new_survey.las \
    --polygon-shapefile utiliies/polygons_1m/DelMarPolygons*.shp \
    --output predictions/
# Expected: risk_10m.csv with Low/Medium/High classifications
```

---

#### Pipeline Summary Checklist

| Step | Module | Script | Subtasks | Status |
|------|--------|--------|----------|--------|
| 1 | `survey_selection.py` | `01_identify_surveys.py` | 5 | ✅ (prototype) |
| 2 | `feature_extraction.py` | `02_extract_features.py` | 8 | ✅ (prototype) |
| 3 | `polygon_aggregation.py` | `03_aggregate_polygons.py` | 6 | ✅ (prototype) |
| 4 | `training_data.py` | `04_assemble_training_data.py` | 5 | ✅ (prototype) |
| 5 | `train.py` | `05_train_model.py` | 7 | ✅ (prototype, temporal AUC-ROC=0.696) |
| **6** | **All modules** | **All scripts** | **9** | **🔜 NEXT — Full-scale processing** |
| 7 | `train.py` | `05_train_model.py --cv` | 5 | ⏳ (subsumed by Step 6) |
| 8 | `inference.py` | `05_predict.py` | 6 | ⏳ |

**Total: 52 subtasks**

**Note**: Steps 1-5 were validated on a test subset (20 surveys, 5 beaches). Step 6 scales up to ALL available data with revised parameters (10 m³ threshold, 2025 hold-out).

---

## Additional Features (Post-v1.0)

### Task 8.1: PCA-Based Unsupervised Classification
**Goal**: Add data-driven classification that discovers natural groupings.

**Implemented**:
- [x] `pc_rai/classification/pca_classifier.py` module
- [x] StandardScaler + PCA + K-means pipeline
- [x] Auto-detection of optimal cluster count (3-12) using silhouette score
- [x] Cluster interpretation based on feature statistics
- [x] RAI comparison utilities (confusion matrix, purity metrics)
- [x] Integration with `RAIClassifier.process()` via `run_pca` parameter
- [x] CLI `--pca` flag
- [x] `pca_cluster` field saved to output LAZ files
- [x] Tests in `tests/test_pca_classifier.py`

### Task 8.2: Output Directory Restructuring
**Goal**: Organize outputs more cleanly.

**Implemented**:
- [x] LAZ files saved to `output/rai/`
- [x] Reports saved to `output/rai/`
- [x] Figures saved to `output/figures/<date>/`

### Task 8.3: Visualization Improvements
**Goal**: Better default rendering for point cloud visualizations.

**Implemented**:
- [x] Fixed zoom level (was too zoomed out)
- [x] Proper axis limits based on data extent

### Task 8.4: CloudComPy Normal Computation
**Goal**: Script for computing normals using CloudComPy with MST orientation.

**Implemented**:
- [x] `scripts/compute_normals_mst.py`
- [x] Westward (-X) normal orientation bias for cliff-face scans
- [x] Batch processing support

### Task 8.5: Classification Threshold Adjustments
**Goal**: Adapt thresholds for California coastal bluffs.

**Implemented**:
- [x] Default `thresh_overhang` changed from 90° to 80° (steep, not necessarily overhanging)
- [x] CLI `--steep-threshold` argument to override the default
- [x] CLI `--subsample` argument for voxel grid subsampling (optional, off by default)
- [x] Graceful batch processing with `--replace` flag and resume capability

**COMPLETED - Decision Tree Revision**:
> Simplified from 8-class to 5-class scheme adapted for California coastal bluffs:
> - Overhang threshold lowered from 90° to 80° for steep coastal faces
> - Three discontinuous classes (Df, Dc, Dw) merged into single Discontinuous class
> - Overhang classes (Os, Oc) merged into Steep/Overhang
> - Added Structure class for detecting seawalls/engineered surfaces (steep + very smooth)
> - See `pc_rai/classification/decision_tree.py` for implementation

---

## Phase 9: ML-Based Stability Prediction (v2.x) - REDESIGNED

> **Prerequisites**: v1.x must be complete. Point-level features (slope, roughness) and RAI classification must be functional.
> **Reference**: See `docs/prd.md` Section 3.10 and `docs/training_pipeline_outline.md` for detailed specifications.

### Key Design Decisions (NEW APPROACH)

1. **Feature-First Pipeline**
   - Subsample entire cloud to 50cm first
   - Compute features at EVERY subsampled point
   - THEN assign points to polygons (no expensive point-in-polygon)
   - THEN aggregate features per polygon

2. **Upper/Lower Elevation Split**
   - Each polygon split at median Z elevation
   - Captures vertical structure (toe vs crest)
   - 5 features × 4 stats × 2 zones = 40 features per polygon

3. **Binary Classification**
   - Event polygons = positive (label=1)
   - Non-event polygons = negative (label=0, treated as negative)
   - Use `class_weight='balanced'` for imbalance

4. **Training Resolution vs Inference Resolution**
   - Training: 1m polygon resolution
   - Inference output: 10m aggregated chunks

5. **Event Filtering**
   - Events >= 5 m³
   - Include "real" and "unreviewed" QC flags
   - Exclude "construction" and "noise"

---

### Task 9.1: Identify Pre-Event Surveys ⏳ PENDING
**Goal**: Find surveys from `all_noveg_files.csv` that occurred before large events.

**File**: `pc_rai/ml/survey_selection.py`

**Implement**:
```python
def load_survey_catalog(csv_path: Path) -> pd.DataFrame:
    """Load all_noveg_files.csv and parse dates from filenames."""

def find_pre_event_surveys(
    surveys: pd.DataFrame,
    events: pd.DataFrame,
    min_days_before: int = 7,
) -> pd.DataFrame:
    """
    For each event, find the most recent survey BEFORE event start_date.

    Returns DataFrame with columns:
        survey_date, survey_file, event_date, event_id, days_before, event_volume
    """

def deduplicate_surveys(survey_event_pairs: pd.DataFrame) -> List[str]:
    """Get unique list of survey files that need feature extraction."""
```

**Data Source**: `utiliies/events/all_noveg_files.csv`

**Acceptance Criteria**:
- [ ] Parse survey dates from filenames (YYYYMMDD prefix)
- [ ] Match events to pre-event surveys (at least 7 days before)
- [ ] Return deduplicated list of surveys to process
- [ ] Handle events with no valid pre-event survey

---

### Task 9.2: Subsample & Extract Point-Level Features ⏳ PENDING
**Goal**: Subsample point clouds and compute features at every point.

**File**: `pc_rai/ml/feature_extraction.py`

**Implement**:
```python
def subsample_point_cloud(las_path: Path, voxel_size: float = 0.5) -> np.ndarray:
    """
    Subsample point cloud to voxel grid.

    Returns (N, 3) array of XYZ coordinates.
    """

def compute_point_features(
    xyz: np.ndarray,
    normals: np.ndarray,
    r_small: float = 0.5,
    r_large: float = 2.0,
) -> pd.DataFrame:
    """
    Compute features at every point.

    Returns DataFrame with columns:
        x, y, z, slope, roughness_small, roughness_large, r_ratio, height
    """

def process_survey(
    las_path: Path,
    output_dir: Path,
    voxel_size: float = 0.5,
) -> Path:
    """
    Full processing for one survey:
    1. Load LAS
    2. Subsample to voxel grid
    3. Compute normals (if needed)
    4. Compute features at every point
    5. Save subsampled cloud with features

    Returns path to output file.
    """
```

**Output**: `subsampled/{survey_date}_subsampled_features.laz`

**Acceptance Criteria**:
- [ ] Subsample to 50cm voxel grid
- [ ] Compute slope from normal vectors
- [ ] Compute roughness at two scales
- [ ] Compute height relative to local base
- [ ] Save with all features as extra dimensions
- [ ] ~10M points → ~400K points per scan

---

### Task 9.3: UTM → Polygon ID Mapping ⏳ PENDING
**Goal**: Derive mapping from UTM Y coordinate to polygon ID.

**File**: `pc_rai/ml/polygon_assignment.py`

**Implement**:
```python
def derive_utm_to_polygon_mapping(shapefile_path: Path) -> Callable:
    """
    Analyze polygon shapefile to derive Y → polygon_id mapping.

    Returns function: polygon_id = f(utm_y)
    """

def assign_polygon_ids(
    points_df: pd.DataFrame,
    mapping_func: Callable,
) -> pd.DataFrame:
    """
    Add polygon_id column to points DataFrame based on Y coordinate.

    Simple integer lookup - no expensive point-in-polygon tests!
    """
```

**Key Insight**: UTM Y corresponds to alongshore position. Polygon IDs ARE alongshore meter values.

**Acceptance Criteria**:
- [ ] Extract Y centroids from polygon geometries
- [ ] Fit linear mapping: polygon_id = round(Y * scale + offset)
- [ ] Assign polygon_id to every point in subsampled cloud
- [ ] Handle points outside polygon coverage

---

### Task 9.4: Aggregate Features by Polygon (Upper/Lower Split) ⏳ PENDING
**Goal**: Aggregate point features to polygon level with vertical stratification.

**File**: `pc_rai/ml/aggregation.py`

**Implement**:
```python
def aggregate_by_polygon(
    points_df: pd.DataFrame,
    features: List[str] = ['slope', 'roughness_small', 'roughness_large', 'r_ratio', 'height'],
    stats: List[str] = ['mean', 'max', 'std', 'p90'],
) -> pd.DataFrame:
    """
    Aggregate point features to polygon level with upper/lower split.

    For each polygon:
        1. Find all points with that polygon_id
        2. Split at median Z elevation
        3. Compute stats for lower zone and upper zone

    Returns DataFrame with 40 feature columns per polygon:
        slope_mean_lower, slope_max_lower, ..., height_p90_upper
    """

def compute_percentile_90(x: np.ndarray) -> float:
    """Helper for p90 aggregation."""
    return np.percentile(x, 90)
```

**Output**: `polygon_features/{survey_date}_polygon_features.csv`

**Acceptance Criteria**:
- [ ] Split each polygon at median Z
- [ ] Compute 4 stats for 5 features in 2 zones = 40 features
- [ ] Handle polygons with few points gracefully
- [ ] Output one row per polygon

---

### Task 9.5: Label Polygons from Events ⏳ PENDING
**Goal**: Attach labels to polygon features based on events.

**File**: `pc_rai/ml/labeling.py`

**Implement**:
```python
def label_polygons(
    polygon_features: pd.DataFrame,
    survey_event_pairs: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label polygons based on events.

    For each (survey, event) pair:
        - Get affected polygon IDs from event alongshore extent
        - Mark those polygons as label=1
        - All other polygons get label=0

    Returns training_data DataFrame with:
        survey_date, polygon_id, [40 features], label, days_to_event, event_volume
    """
```

**Output**: `training_data.csv`

**Acceptance Criteria**:
- [ ] Map event alongshore extent to polygon IDs
- [ ] Label=1 for event polygons, Label=0 for others
- [ ] Track metadata: days_to_event, event_volume
- [ ] Handle multiple events affecting same polygon

---

### Task 9.6: Train Random Forest ⏳ PENDING
**Goal**: Train Random Forest model with class weighting.

**File**: `pc_rai/ml/training.py`

**Implement**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

def train_model(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    n_estimators: int = 100,
    max_depth: int = 15,
    class_weight: str = 'balanced',
) -> RandomForestClassifier:
    """
    Train Random Forest with balanced class weights.
    """

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute AUC-ROC, AUC-PR, and other metrics.
    """

def get_feature_importances(
    model: RandomForestClassifier,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Return sorted feature importances.
    """
```

**Acceptance Criteria**:
- [ ] Train with `class_weight='balanced'`
- [ ] Compute AUC-ROC and AUC-PR
- [ ] Extract feature importances
- [ ] Save model to disk

---

### Task 9.7: Cross-Validation (Leave-One-Year-Out) ⏳ PENDING
**Goal**: Validate temporal generalization.

**File**: `pc_rai/ml/training.py` (extend)

**Implement**:
```python
def leave_one_year_out_cv(
    training_data: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[int, Dict[str, float]]:
    """
    Leave-one-year-out cross-validation.

    Returns dict mapping year → {'auc_roc': float, 'auc_pr': float, ...}
    """

def summarize_cv_results(results: Dict) -> pd.DataFrame:
    """Create summary table with mean/std across folds."""
```

**Acceptance Criteria**:
- [ ] Hold out each year for testing
- [ ] Compute metrics per fold
- [ ] Summarize mean/std across folds
- [ ] Visualize results

---

### Task 9.8: Inference Pipeline ⏳ PENDING
**Goal**: Apply trained model to new point clouds with 10m output aggregation.

**File**: `pc_rai/ml/inference.py`

**Implement**:
```python
def predict_survey(
    las_path: Path,
    model: RandomForestClassifier,
    polygon_mapping: Callable,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Run full inference pipeline on a new survey:
    1. Subsample & extract features (same as training Step 2)
    2. Assign polygon IDs (same as training Step 3)
    3. Aggregate by polygon (same as training Step 4)
    4. Predict probability for each 1m polygon

    Returns DataFrame with polygon_id, probability
    """

def aggregate_to_10m(
    polygon_predictions: pd.DataFrame,
    method: str = 'mean',  # or 'max' for conservative
) -> pd.DataFrame:
    """
    Aggregate 1m polygon predictions to 10m chunks.

    Returns DataFrame with chunk_start_m, chunk_end_m, probability
    """

def classify_risk_level(
    probability: float,
    thresholds: Tuple[float, float] = (0.3, 0.6)
) -> str:
    """Convert probability to Low/Medium/High risk."""
```

**Output**: `predictions/{survey_date}_risk_10m.csv`

**Acceptance Criteria**:
- [ ] Same preprocessing as training pipeline
- [ ] Predictions for all polygons with sufficient points
- [ ] Aggregate to 10m chunks
- [ ] Output risk level classification

---

### Task 9.9: CLI Scripts ⏳ PENDING
**Goal**: Command-line scripts for running the full pipeline.

**Scripts to create**:

```bash
# Step 1: Identify pre-event surveys
scripts/01_identify_surveys.py \
    --events utiliies/events/DelMar_events_qc_*.csv \
    --surveys utiliies/events/all_noveg_files.csv \
    --output data/pre_event_surveys.csv

# Step 2: Extract features (run on HPC)
scripts/02_extract_features.py \
    --survey-list data/pre_event_surveys.csv \
    --output-dir data/subsampled/

# Step 3: Build training data
scripts/03_build_training_data.py \
    --subsampled-dir data/subsampled/ \
    --polygon-shapefile utiliies/polygons_1m/DelMarPolygons*.shp \
    --survey-events data/pre_event_surveys.csv \
    --output data/training_data.csv

# Step 4: Train model
scripts/04_train_model.py \
    --training-data data/training_data.csv \
    --output models/stability_rf.joblib \
    --cv leave-one-year-out

# Step 5: Predict on new surveys
scripts/05_predict.py \
    --model models/stability_rf.joblib \
    --input new_survey.las \
    --output predictions/
```

**Acceptance Criteria**:
- [ ] Each script runs independently
- [ ] Clear `--help` documentation
- [ ] Exit codes: 0 success, 1 error
- [ ] Progress bars for long operations

---

## Phase 0: Project Setup

### Task 0.1: Create Project Structure
**Goal**: Set up the basic directory structure and package files.

**Create these files/directories**:
```
pc_rai/
├── __init__.py
├── __main__.py
├── cli.py
├── config.py
├── io/
│   └── __init__.py
├── normals/
│   └── __init__.py
├── features/
│   └── __init__.py
├── classification/
│   └── __init__.py
├── visualization/
│   └── __init__.py
├── reporting/
│   └── __init__.py
└── utils/
    └── __init__.py
tests/
├── __init__.py
├── conftest.py
└── test_data/
pyproject.toml
README.md
```

**`__init__.py` contents**: Each should be empty or contain `"""Module docstring."""`

**`__main__.py` contents**:
```python
"""Entry point for python -m pc_rai."""
from pc_rai.cli import main

if __name__ == "__main__":
    main()
```

**`pyproject.toml`**: Include dependencies:
- numpy>=1.21
- scipy>=1.7
- laspy[lazrs]>=2.4
- matplotlib>=3.5
- open3d>=0.17
- tqdm>=4.62
- pyyaml>=6.0
- pytest>=7.0 (dev dependency)

**Acceptance Criteria**:
- [x] All directories exist
- [x] `pip install -e .` succeeds
- [x] `python -c "import pc_rai"` succeeds

---

### Task 0.2: Create Configuration Module
**Goal**: Define configuration dataclasses with sensible defaults.

**File**: `pc_rai/config.py`

**Implement**:
```python
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

@dataclass
class RAIConfig:
    """Configuration for RAI processing."""
    # Normal computation
    compute_normals: bool = True
    cloudcompare_path: str = "CloudCompare"
    normal_radius: float = 0.1
    mst_neighbors: int = 10
    
    # Slope
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    # Roughness - radius method
    radius_small: float = 0.175
    radius_large: float = 0.425
    
    # Roughness - knn method  
    k_small: int = 30
    k_large: int = 100
    
    # Shared roughness settings
    min_neighbors: int = 5
    methods: List[str] = field(default_factory=lambda: ["radius", "knn"])
    
    # Classification thresholds (adapted from Markus et al. 2023)
    thresh_overhang: float = 80.0
    thresh_talus_slope: float = 42.0
    thresh_r_small_low: float = 6.0
    thresh_r_small_mid: float = 11.0
    thresh_r_large: float = 12.0
    thresh_structure_roughness: float = 4.0
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    compress_output: bool = True
    visualization_dpi: int = 300
    visualization_views: List[str] = field(default_factory=lambda: ["front", "oblique"])

# RAI class definitions - Simplified 5-class scheme
RAI_CLASS_NAMES = {
    0: "Unclassified",
    1: "Talus",
    2: "Intact",
    3: "Discontinuous",
    4: "Steep/Overhang",
    5: "Structure",
}

RAI_CLASS_ABBREV = {
    0: "U", 1: "T", 2: "I", 3: "D", 4: "O", 5: "St"
}

RAI_CLASS_COLORS = {
    0: "#9E9E9E",  # Gray
    1: "#C8A2C8",  # Light Purple
    2: "#4CAF50",  # Green
    3: "#2196F3",  # Blue
    4: "#FF9800",  # Orange
    5: "#795548",  # Brown
}

def load_config(yaml_path: Path) -> RAIConfig:
    """Load configuration from YAML file."""
    # Implementation: parse YAML and return RAIConfig
    pass

def save_config(config: RAIConfig, yaml_path: Path) -> None:
    """Save configuration to YAML file."""
    pass
```

**Test File**: `tests/test_config.py`
```python
def test_default_config():
    from pc_rai.config import RAIConfig
    config = RAIConfig()
    assert config.radius_small == 0.175
    assert config.thresh_talus_slope == 42.0
    assert "radius" in config.methods

def test_rai_class_names():
    from pc_rai.config import RAI_CLASS_NAMES
    assert len(RAI_CLASS_NAMES) == 6
    assert RAI_CLASS_NAMES[1] == "Talus"
    assert RAI_CLASS_NAMES[5] == "Structure"
```

**Acceptance Criteria**:
- [x] `RAIConfig` instantiates with defaults
- [x] All threshold values match PRD specifications
- [x] `pytest tests/test_config.py` passes

---

## Phase 1: Data I/O

### Task 1.1: LAS Reader
**Goal**: Read LAS/LAZ files and extract XYZ + optional normals.

**File**: `pc_rai/io/las_reader.py`

**Implement**:
```python
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import laspy

@dataclass
class PointCloud:
    """Container for point cloud data."""
    xyz: np.ndarray  # (N, 3) float64
    normals: Optional[np.ndarray] = None  # (N, 3) float32
    source_file: Optional[Path] = None
    
    # Store original LAS data for preservation
    _las_data: Optional[laspy.LasData] = None
    
    @property
    def n_points(self) -> int:
        return len(self.xyz)
    
    @property
    def has_normals(self) -> bool:
        return self.normals is not None
    
    @property
    def bounds(self) -> Dict[str, tuple]:
        """Return min/max for each dimension."""
        return {
            'x': (self.xyz[:, 0].min(), self.xyz[:, 0].max()),
            'y': (self.xyz[:, 1].min(), self.xyz[:, 1].max()),
            'z': (self.xyz[:, 2].min(), self.xyz[:, 2].max()),
        }

def load_point_cloud(filepath: Path) -> PointCloud:
    """
    Load a LAS/LAZ file into a PointCloud object.
    
    Extracts XYZ coordinates and normal vectors if present.
    Preserves original LAS data for later output.
    """
    # Implementation here
    pass

def has_valid_normals(las: laspy.LasData) -> bool:
    """Check if LAS file has valid normal vectors."""
    # Check for NormalX, NormalY, NormalZ extra dimensions
    pass
```

**Test File**: `tests/test_las_reader.py`
```python
import numpy as np
import pytest
from pathlib import Path

def test_load_synthetic_las(tmp_path):
    """Create a minimal LAS file and load it."""
    import laspy
    
    # Create synthetic data
    n_points = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n_points)
    las.y = np.random.uniform(0, 10, n_points)
    las.z = np.random.uniform(0, 5, n_points)
    
    # Save
    filepath = tmp_path / "test.las"
    las.write(filepath)
    
    # Load and verify
    from pc_rai.io.las_reader import load_point_cloud
    cloud = load_point_cloud(filepath)
    
    assert cloud.n_points == n_points
    assert cloud.xyz.shape == (n_points, 3)
    assert not cloud.has_normals

def test_point_cloud_bounds():
    from pc_rai.io.las_reader import PointCloud
    xyz = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float64)
    cloud = PointCloud(xyz=xyz)
    bounds = cloud.bounds
    assert bounds['x'] == (0, 10)
    assert bounds['z'] == (0, 30)
```

**Acceptance Criteria**:
- [x] Can load LAS file and access XYZ as numpy array
- [x] `PointCloud.n_points` returns correct count
- [x] `PointCloud.bounds` returns correct min/max
- [x] Handles files without normals gracefully
- [x] `pytest tests/test_las_reader.py` passes

---

### Task 1.2: LAS Writer with Extra Dimensions
**Goal**: Write LAS files with custom extra dimensions for RAI attributes.

**File**: `pc_rai/io/las_writer.py`

**Implement**:
```python
import numpy as np
from pathlib import Path
from typing import Dict
import laspy
from pc_rai.io.las_reader import PointCloud

# Extra dimensions to add
RAI_EXTRA_DIMS = {
    'slope_deg': ('f4', 'Slope angle in degrees'),
    'roughness_small_radius': ('f4', 'Small-scale roughness (radius method)'),
    'roughness_large_radius': ('f4', 'Large-scale roughness (radius method)'),
    'roughness_small_knn': ('f4', 'Small-scale roughness (k-NN method)'),
    'roughness_large_knn': ('f4', 'Large-scale roughness (k-NN method)'),
    'rai_class_radius': ('u1', 'RAI class (radius method)'),
    'rai_class_knn': ('u1', 'RAI class (k-NN method)'),
    'neighbor_count_small': ('u2', 'Neighbor count at small scale'),
    'neighbor_count_large': ('u2', 'Neighbor count at large scale'),
}

def save_point_cloud(
    cloud: PointCloud,
    attributes: Dict[str, np.ndarray],
    output_path: Path,
    compress: bool = True
) -> None:
    """
    Save point cloud with RAI attributes as extra dimensions.
    
    Parameters
    ----------
    cloud : PointCloud
        Original point cloud (preserves all original attributes)
    attributes : dict
        Dictionary mapping attribute names to numpy arrays
        Keys should match RAI_EXTRA_DIMS
    output_path : Path
        Output file path (.las or .laz)
    compress : bool
        If True, save as LAZ
    """
    pass
```

**Test File**: `tests/test_las_writer.py`
```python
import numpy as np
import pytest
from pathlib import Path

def test_write_with_extra_dims(tmp_path):
    """Write a LAS file with extra dimensions and read it back."""
    import laspy
    from pc_rai.io.las_reader import PointCloud, load_point_cloud
    from pc_rai.io.las_writer import save_point_cloud
    
    # Create synthetic cloud
    n_points = 50
    xyz = np.random.uniform(0, 10, (n_points, 3))
    cloud = PointCloud(xyz=xyz)
    
    # Create attributes
    attributes = {
        'slope_deg': np.random.uniform(0, 90, n_points).astype(np.float32),
        'rai_class_radius': np.random.randint(0, 8, n_points).astype(np.uint8),
    }
    
    # Write
    output_path = tmp_path / "output.las"
    save_point_cloud(cloud, attributes, output_path, compress=False)
    
    # Verify file exists and has extra dims
    assert output_path.exists()
    las = laspy.read(output_path)
    assert 'slope_deg' in [dim.name for dim in las.point_format.extra_dimensions]
    assert np.allclose(las['slope_deg'], attributes['slope_deg'])
```

**Acceptance Criteria**:
- [x] Can write LAS file with custom extra dimensions
- [x] Extra dimensions are readable with laspy
- [x] Original point attributes are preserved
- [x] LAZ compression works when enabled
- [x] `pytest tests/test_las_writer.py` passes

---

## Phase 2: Core Algorithms

### Task 2.1: Spatial Utilities (KD-Tree)
**Goal**: Efficient neighbor queries for roughness calculation.

**File**: `pc_rai/utils/spatial.py`

**Implement**:
```python
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List
from tqdm import tqdm

class SpatialIndex:
    """Wrapper around scipy cKDTree for neighbor queries."""
    
    def __init__(self, points: np.ndarray):
        """
        Build KD-tree from points.
        
        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of XYZ coordinates
        """
        self.points = points
        self.tree = cKDTree(points)
        self.n_points = len(points)
    
    def query_radius(
        self, 
        radius: float,
        return_counts: bool = False,
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Query all neighbors within radius for each point.
        
        Returns
        -------
        neighbors : list of np.ndarray
            List of neighbor indices for each point
        counts : np.ndarray (optional)
            Number of neighbors per point
        """
        pass
    
    def query_knn(
        self,
        k: int,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k nearest neighbors for each point.
        
        Returns
        -------
        distances : np.ndarray
            (N, k) distances to neighbors
        indices : np.ndarray
            (N, k) indices of neighbors
        """
        pass
```

**Test File**: `tests/test_spatial.py`
```python
import numpy as np
import pytest

def test_spatial_index_creation():
    from pc_rai.utils.spatial import SpatialIndex
    points = np.random.uniform(0, 10, (1000, 3))
    index = SpatialIndex(points)
    assert index.n_points == 1000

def test_radius_query():
    from pc_rai.utils.spatial import SpatialIndex
    # Create grid of points with known spacing
    x = np.linspace(0, 1, 10)
    points = np.array([[i, 0, 0] for i in x])
    index = SpatialIndex(points)
    
    neighbors, counts = index.query_radius(0.15, return_counts=True, show_progress=False)
    # Each point should have ~1-2 neighbors within 0.15 (spacing is 0.111)
    assert all(c >= 1 for c in counts)

def test_knn_query():
    from pc_rai.utils.spatial import SpatialIndex
    points = np.random.uniform(0, 10, (100, 3))
    index = SpatialIndex(points)
    
    distances, indices = index.query_knn(k=5, show_progress=False)
    assert distances.shape == (100, 5)
    assert indices.shape == (100, 5)
    # First neighbor should be self (distance 0)
    assert np.allclose(distances[:, 0], 0)
```

**Acceptance Criteria**:
- [x] KD-tree builds successfully
- [x] Radius query returns correct neighbor counts
- [x] K-NN query returns correct shape
- [x] Progress bar displays during long operations
- [x] `pytest tests/test_spatial.py` passes

---

### Task 2.2: Slope Calculation
**Goal**: Calculate slope angle from normal vectors.

**File**: `pc_rai/features/slope.py`

**Implement**:
```python
import numpy as np
from typing import Tuple

def calculate_slope(
    normals: np.ndarray,
    up_vector: Tuple[float, float, float] = (0, 0, 1)
) -> np.ndarray:
    """
    Calculate slope angle from normal vectors.
    
    Slope is the angle between the surface normal and the up vector.
    - 0° = horizontal surface (normal points up)
    - 90° = vertical surface
    - >90° = overhanging surface
    - 180° = inverted surface (normal points down)
    
    Parameters
    ----------
    normals : np.ndarray
        (N, 3) array of unit normal vectors
    up_vector : tuple
        Direction considered "up" (default: +Z)
        
    Returns
    -------
    slope_deg : np.ndarray
        (N,) array of slope angles in degrees
    """
    pass

def identify_overhangs(slope_deg: np.ndarray, threshold: float = 90.0) -> np.ndarray:
    """Return boolean mask of overhanging points."""
    return slope_deg > threshold
```

**Test File**: `tests/test_slope.py`
```python
import numpy as np
import pytest

def test_horizontal_surface():
    """Normal pointing up = 0° slope."""
    from pc_rai.features.slope import calculate_slope
    normals = np.array([[0, 0, 1]], dtype=np.float32)  # pointing up
    slope = calculate_slope(normals)
    assert np.isclose(slope[0], 0, atol=0.01)

def test_vertical_surface():
    """Normal pointing sideways = 90° slope."""
    from pc_rai.features.slope import calculate_slope
    normals = np.array([[1, 0, 0]], dtype=np.float32)  # pointing in X
    slope = calculate_slope(normals)
    assert np.isclose(slope[0], 90, atol=0.01)

def test_overhang():
    """Normal pointing down-ish = >90° slope."""
    from pc_rai.features.slope import calculate_slope
    # 45° below horizontal
    normals = np.array([[0.707, 0, -0.707]], dtype=np.float32)
    slope = calculate_slope(normals)
    assert slope[0] > 90

def test_inverted():
    """Normal pointing straight down = 180° slope."""
    from pc_rai.features.slope import calculate_slope
    normals = np.array([[0, 0, -1]], dtype=np.float32)
    slope = calculate_slope(normals)
    assert np.isclose(slope[0], 180, atol=0.01)

def test_identify_overhangs():
    from pc_rai.features.slope import identify_overhangs
    slopes = np.array([45, 89, 91, 120, 180])
    mask = identify_overhangs(slopes)
    assert list(mask) == [False, False, True, True, True]
```

**Acceptance Criteria**:
- [x] Horizontal surface → 0°
- [x] Vertical surface → 90°
- [x] Overhang → >90°
- [x] Inverted surface → 180°
- [x] Works with custom up vector
- [x] `pytest tests/test_slope.py` passes

---

### Task 2.3: Roughness Calculation
**Goal**: Calculate multi-scale roughness using both radius and k-NN methods.

**File**: `pc_rai/features/roughness.py`

**Implement**:
```python
import numpy as np
from typing import Tuple, Optional
from pc_rai.utils.spatial import SpatialIndex

def calculate_roughness_radius(
    slope_deg: np.ndarray,
    spatial_index: SpatialIndex,
    radius: float,
    min_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate roughness as std dev of slope within radius.
    
    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees
    spatial_index : SpatialIndex
        Pre-built spatial index
    radius : float
        Search radius in same units as points
    min_neighbors : int
        Minimum neighbors for valid calculation
        
    Returns
    -------
    roughness : np.ndarray
        (N,) roughness values in degrees (NaN where insufficient neighbors)
    neighbor_counts : np.ndarray
        (N,) number of neighbors found
    """
    pass

def calculate_roughness_knn(
    slope_deg: np.ndarray,
    spatial_index: SpatialIndex,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate roughness as std dev of slope for k nearest neighbors.
    
    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees
    spatial_index : SpatialIndex
        Pre-built spatial index
    k : int
        Number of neighbors
        
    Returns
    -------
    roughness : np.ndarray
        (N,) roughness values in degrees
    neighbor_counts : np.ndarray
        (N,) all values equal to k
    """
    pass

def calculate_all_roughness(
    slope_deg: np.ndarray,
    spatial_index: SpatialIndex,
    radius_small: float = 0.175,
    radius_large: float = 0.425,
    k_small: int = 30,
    k_large: int = 100,
    min_neighbors: int = 5,
    methods: list = ["radius", "knn"]
) -> dict:
    """
    Calculate all roughness metrics.
    
    Returns dictionary with keys:
    - roughness_small_radius
    - roughness_large_radius
    - roughness_small_knn
    - roughness_large_knn
    - neighbor_count_small
    - neighbor_count_large
    """
    pass
```

**Test File**: `tests/test_roughness.py`
```python
import numpy as np
import pytest

def test_uniform_slope_zero_roughness():
    """Points with identical slopes should have ~0 roughness."""
    from pc_rai.features.roughness import calculate_roughness_knn
    from pc_rai.utils.spatial import SpatialIndex
    
    # Create grid of points
    points = np.random.uniform(0, 1, (100, 3))
    # All have same slope
    slopes = np.full(100, 45.0)
    
    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_knn(slopes, index, k=10)
    
    assert np.allclose(roughness, 0, atol=0.01)

def test_variable_slope_nonzero_roughness():
    """Points with varying slopes should have positive roughness."""
    from pc_rai.features.roughness import calculate_roughness_knn
    from pc_rai.utils.spatial import SpatialIndex
    
    points = np.random.uniform(0, 1, (100, 3))
    # Varying slopes
    slopes = np.random.uniform(30, 60, 100)
    
    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_knn(slopes, index, k=10)
    
    assert roughness.mean() > 0

def test_insufficient_neighbors():
    """Sparse points should return NaN for radius method."""
    from pc_rai.features.roughness import calculate_roughness_radius
    from pc_rai.utils.spatial import SpatialIndex
    
    # Very spread out points
    points = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], dtype=np.float64)
    slopes = np.array([45, 45, 45])
    
    index = SpatialIndex(points)
    roughness, counts = calculate_roughness_radius(slopes, index, radius=0.5, min_neighbors=2)
    
    # Should be NaN due to insufficient neighbors
    assert np.all(np.isnan(roughness))
```

**Acceptance Criteria**:
- [x] Uniform slopes → zero roughness
- [x] Variable slopes → positive roughness
- [x] Insufficient neighbors → NaN (radius method)
- [x] K-NN always returns valid values
- [x] Both scales (small/large) calculated correctly
- [x] `pytest tests/test_roughness.py` passes

---

### Task 2.4: Classification Decision Tree
**Goal**: Implement the simplified RAI 5-class decision tree.

**File**: `pc_rai/classification/decision_tree.py`

**Implement**:
```python
import numpy as np
from dataclasses import dataclass
from pc_rai.config import RAIConfig

@dataclass
class ClassificationThresholds:
    """Thresholds for simplified 5-class RAI decision tree."""
    overhang: float = 80.0
    talus_slope: float = 42.0
    r_small_low: float = 6.0
    r_small_mid: float = 11.0
    r_large: float = 12.0
    structure_roughness: float = 4.0
    
    @classmethod
    def from_config(cls, config: RAIConfig) -> 'ClassificationThresholds':
        return cls(
            overhang=config.thresh_overhang,
            talus_slope=config.thresh_talus_slope,
            r_small_low=config.thresh_r_small_low,
            r_small_mid=config.thresh_r_small_mid,
            r_large=config.thresh_r_large,
            structure_roughness=config.thresh_structure_roughness,
        )

def classify_points(
    slope_deg: np.ndarray,
    r_small: np.ndarray,
    r_large: np.ndarray,
    thresholds: ClassificationThresholds = None
) -> np.ndarray:
    """
    Classify points using simplified 5-class RAI decision tree.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees
    r_small : np.ndarray
        (N,) small-scale roughness in degrees
    r_large : np.ndarray
        (N,) large-scale roughness in degrees
    thresholds : ClassificationThresholds
        Classification thresholds (uses defaults if None)

    Returns
    -------
    classes : np.ndarray
        (N,) uint8 array of class codes 0-5

    Class Codes:
        0 = Unclassified (invalid data)
        1 = Talus (T)
        2 = Intact (I)
        3 = Discontinuous (D)
        4 = Steep/Overhang (O)
        5 = Structure (St)
    """
    pass

def get_class_statistics(classes: np.ndarray) -> dict:
    """
    Calculate classification statistics.
    
    Returns dict with count and percentage for each class.
    """
    pass
```

**Test File**: `tests/test_classification.py`
```python
import numpy as np
import pytest
from pc_rai.classification.decision_tree import classify_points, ClassificationThresholds

@pytest.fixture
def thresholds():
    return ClassificationThresholds()

def test_talus(thresholds):
    """Low slope + low roughness = Talus."""
    slope = np.array([30.0])  # < 42°
    r_small = np.array([3.0])  # < 6°
    r_large = np.array([5.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 1  # Talus

def test_intact_steep(thresholds):
    """Steep slope + low roughness = Intact."""
    slope = np.array([60.0])  # > 42°
    r_small = np.array([3.0])  # < 6°
    r_large = np.array([5.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 2  # Intact

def test_discontinuous_high_r_small(thresholds):
    """High r_small = Discontinuous."""
    slope = np.array([60.0])
    r_small = np.array([15.0])  # > 11°
    r_large = np.array([25.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 3  # Discontinuous

def test_discontinuous_high_r_large(thresholds):
    """Moderate r_small + high r_large = Discontinuous."""
    slope = np.array([60.0])
    r_small = np.array([8.0])  # 6° < x < 11°
    r_large = np.array([15.0])  # > 12°
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 3  # Discontinuous

def test_intact_intermediate(thresholds):
    """Intermediate r_small + low r_large = Intact."""
    slope = np.array([60.0])
    r_small = np.array([8.0])  # 6° < x < 11°
    r_large = np.array([8.0])  # < 12°
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 2  # Intact

def test_steep_overhang(thresholds):
    """Slope > 80° with roughness = Steep/Overhang."""
    slope = np.array([85.0])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 4  # Steep/Overhang

def test_structure(thresholds):
    """Slope > 80° with very low roughness = Structure."""
    slope = np.array([85.0])
    r_small = np.array([2.0])  # < 4°
    r_large = np.array([2.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 5  # Structure

def test_unclassified_nan(thresholds):
    """NaN roughness = Unclassified."""
    slope = np.array([60.0])
    r_small = np.array([np.nan])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 0  # Unclassified

def test_vectorized(thresholds):
    """Test with multiple points."""
    n = 1000
    slope = np.random.uniform(0, 180, n)
    r_small = np.random.uniform(0, 30, n)
    r_large = np.random.uniform(0, 30, n)
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert len(classes) == n
    assert classes.dtype == np.uint8
    assert np.all((classes >= 0) & (classes <= 5))
```

**Acceptance Criteria**:
- [x] All 5 classes correctly assigned based on decision tree
- [x] NaN roughness → Unclassified
- [x] Vectorized implementation handles 1M+ points efficiently
- [x] All boundary conditions tested
- [x] `pytest tests/test_classification.py` passes

---

## Phase 3: CloudCompare Integration

### Task 3.1: CloudCompare CLI Wrapper
**Goal**: Compute normals using CloudCompare command line.

**File**: `pc_rai/normals/cloudcompare.py`

**Implement**:
```python
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from typing import Optional
import laspy

def find_cloudcompare() -> Optional[Path]:
    """Find CloudCompare executable on system."""
    # Check common locations:
    # - Linux: CloudCompare, cloudcompare
    # - macOS: /Applications/CloudCompare.app/Contents/MacOS/CloudCompare
    # - Windows: C:\Program Files\CloudCompare\CloudCompare.exe
    pass

def compute_normals_cloudcompare(
    input_path: Path,
    output_path: Path,
    radius: float = 0.1,
    mst_neighbors: int = 10,
    cloudcompare_path: str = "CloudCompare"
) -> bool:
    """
    Compute normals using CloudCompare CLI.
    
    Parameters
    ----------
    input_path : Path
        Input LAS/LAZ file
    output_path : Path
        Output path for file with normals
    radius : float
        Local radius for normal estimation
    mst_neighbors : int
        Number of neighbors for MST orientation
    cloudcompare_path : str
        Path to CloudCompare executable
        
    Returns
    -------
    success : bool
        True if normals were computed successfully
    """
    # Build command
    cmd = [
        cloudcompare_path,
        "-SILENT",
        "-O", str(input_path),
        "-OCTREE_NORMALS", str(radius),
        "-ORIENT_NORMS_MST", str(mst_neighbors),
        "-C_EXPORT_FMT", "LAS",
        "-SAVE_CLOUDS", "FILE", str(output_path)
    ]
    
    # Execute and handle errors
    pass

def extract_normals_from_las(las_path: Path) -> Optional[np.ndarray]:
    """
    Extract normal vectors from LAS file.
    
    CloudCompare saves normals as NormalX, NormalY, NormalZ extra dimensions.
    """
    pass
```

**Test File**: `tests/test_cloudcompare.py`
```python
import pytest
from pathlib import Path
from pc_rai.normals.cloudcompare import find_cloudcompare

def test_find_cloudcompare():
    """Test CloudCompare detection (may skip if not installed)."""
    cc_path = find_cloudcompare()
    if cc_path is None:
        pytest.skip("CloudCompare not found on system")
    assert cc_path.exists() or cc_path.name == "CloudCompare"

# Integration test (requires CloudCompare)
@pytest.mark.skipif(
    find_cloudcompare() is None,
    reason="CloudCompare not installed"
)
def test_compute_normals(tmp_path):
    """Full integration test with CloudCompare."""
    import laspy
    import numpy as np
    from pc_rai.normals.cloudcompare import compute_normals_cloudcompare
    
    # Create simple test file
    n = 100
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = np.random.uniform(0, 10, n)
    las.y = np.random.uniform(0, 10, n)
    las.z = np.random.uniform(0, 1, n)  # Mostly flat
    
    input_path = tmp_path / "input.las"
    output_path = tmp_path / "output.las"
    las.write(input_path)
    
    # Compute normals
    success = compute_normals_cloudcompare(input_path, output_path)
    assert success
    assert output_path.exists()
```

**Acceptance Criteria**:
- [x] Detects CloudCompare on system (or reports not found)
- [x] Successfully calls CloudCompare CLI
- [x] Extracts computed normals from output file
- [x] Handles errors gracefully (CloudCompare not found, failed computation)
- [x] `pytest tests/test_cloudcompare.py` passes (or skips if CC not installed)

---

## Phase 4: Visualization

### Task 4.1: 3D Point Cloud Rendering
**Goal**: Render classified point clouds to images.

**File**: `pc_rai/visualization/render_3d.py`

**Implement**:
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple
import open3d as o3d
from pc_rai.config import RAI_CLASS_COLORS, RAI_CLASS_NAMES

def create_rai_colormap():
    """Create matplotlib colormap for RAI classes."""
    colors = [RAI_CLASS_COLORS[i] for i in range(6)]
    return ListedColormap(colors)

def render_classification(
    xyz: np.ndarray,
    classes: np.ndarray,
    view: str = "front",
    title: str = "RAI Classification",
    figsize: Tuple[int, int] = (12, 10),
    point_size: float = 1.0,
    dpi: int = 300,
    show_legend: bool = True,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Render classified point cloud to matplotlib figure.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates
    classes : np.ndarray
        (N,) class codes 0-5
    view : str
        "front", "oblique", "top"
    title : str
        Figure title
    output_path : str, optional
        If provided, save figure to this path
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    pass

def render_continuous(
    xyz: np.ndarray,
    values: np.ndarray,
    view: str = "front",
    title: str = "",
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
    colorbar_label: str = "",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Render point cloud with continuous colormap (for slope, roughness)."""
    pass

def get_view_params(view: str, bounds: dict) -> dict:
    """Get camera parameters for different views."""
    # Return elev, azim, and optionally zoom for matplotlib 3D
    pass
```

**Test File**: `tests/test_visualization.py`
```python
import numpy as np
import pytest
from pathlib import Path

def test_render_classification(tmp_path):
    from pc_rai.visualization.render_3d import render_classification
    
    # Create synthetic data
    n = 1000
    xyz = np.random.uniform(0, 10, (n, 3))
    classes = np.random.randint(0, 6, n).astype(np.uint8)
    
    output_path = tmp_path / "test_render.png"
    fig = render_classification(xyz, classes, output_path=str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0  # File has content
    plt.close(fig)

def test_render_continuous(tmp_path):
    from pc_rai.visualization.render_3d import render_continuous
    
    n = 1000
    xyz = np.random.uniform(0, 10, (n, 3))
    values = np.random.uniform(0, 90, n)
    
    output_path = tmp_path / "test_slope.png"
    fig = render_continuous(xyz, values, title="Slope", output_path=str(output_path))
    
    assert output_path.exists()
    plt.close(fig)
```

**Acceptance Criteria**:
- [x] Renders classification with correct colors
- [x] Renders continuous values with colorbar
- [x] Multiple view angles work (front, oblique, top)
- [x] Legend shows all class names
- [x] Saves to PNG at specified DPI
- [x] `pytest tests/test_visualization.py` passes

---

### Task 4.2: Multi-Panel Figures
**Goal**: Create comparison and summary figures.

**File**: `pc_rai/visualization/figures.py`

**Implement**:
```python
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from pc_rai.visualization.render_3d import render_classification, render_continuous

def create_comparison_figure(
    xyz: np.ndarray,
    classes_radius: np.ndarray,
    classes_knn: np.ndarray,
    output_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """Side-by-side comparison of radius vs k-NN classification."""
    pass

def create_summary_figure(
    xyz: np.ndarray,
    slope: np.ndarray,
    roughness_small: np.ndarray,
    roughness_large: np.ndarray,
    classes: np.ndarray,
    output_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """4-panel summary: slope, r_small, r_large, classification."""
    pass

def create_histogram_figure(
    classes: np.ndarray,
    output_path: str = None
) -> plt.Figure:
    """Bar chart of class distribution."""
    pass
```

**Acceptance Criteria**:
- [x] Comparison figure shows both methods side-by-side
- [x] Summary figure shows all computed features
- [x] Histogram shows class distribution with colors
- [x] All figures save successfully
- [x] `pytest tests/test_figures.py` passes

---

## Phase 5: Reporting

### Task 5.1: Statistics and Report Generation
**Goal**: Calculate summary statistics and generate reports.

**File**: `pc_rai/reporting/statistics.py`

**Implement**:
```python
import numpy as np
from typing import Dict
from pc_rai.config import RAI_CLASS_NAMES

def calculate_classification_stats(classes: np.ndarray) -> Dict:
    """
    Calculate classification statistics.
    
    Returns dict with per-class counts and percentages.
    """
    pass

def calculate_feature_stats(values: np.ndarray, name: str) -> Dict:
    """Calculate mean, std, min, max for a feature array."""
    pass

def calculate_method_agreement(
    classes_radius: np.ndarray,
    classes_knn: np.ndarray
) -> Dict:
    """
    Calculate agreement between radius and k-NN methods.
    
    Returns:
        agreement_pct: Percentage of points classified identically
        cohens_kappa: Cohen's kappa coefficient
        confusion_matrix: Confusion matrix
    """
    pass
```

**File**: `pc_rai/reporting/report_writer.py`

**Implement**:
```python
from pathlib import Path
from typing import Dict
import json
from datetime import datetime

def write_markdown_report(
    stats: Dict,
    output_path: Path,
    input_file: str,
    config_summary: Dict
) -> None:
    """Write summary report as Markdown file."""
    pass

def write_json_report(
    stats: Dict,
    output_path: Path,
    input_file: str,
    config_summary: Dict
) -> None:
    """Write summary report as JSON file."""
    pass
```

**Acceptance Criteria**:
- [x] Class statistics correctly count all points
- [x] Feature statistics match numpy calculations
- [x] Cohen's kappa calculated correctly
- [x] Markdown report is valid and readable
- [x] JSON report parses correctly
- [x] `pytest tests/test_reporting.py` passes

---

## Phase 6: Main Pipeline & CLI

### Task 6.1: RAI Classifier Class
**Goal**: Unified interface for the complete processing pipeline.

**File**: `pc_rai/classifier.py`

**Implement**:
```python
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm

from pc_rai.config import RAIConfig
from pc_rai.io.las_reader import PointCloud, load_point_cloud
from pc_rai.io.las_writer import save_point_cloud
from pc_rai.normals.cloudcompare import compute_normals_cloudcompare
from pc_rai.features.slope import calculate_slope
from pc_rai.features.roughness import calculate_all_roughness
from pc_rai.classification.decision_tree import classify_points
from pc_rai.utils.spatial import SpatialIndex

@dataclass
class RAIResult:
    """Container for RAI processing results."""
    source_file: str
    n_points: int
    
    # Features
    slope_deg: np.ndarray
    roughness_small_radius: Optional[np.ndarray]
    roughness_large_radius: Optional[np.ndarray]
    roughness_small_knn: Optional[np.ndarray]
    roughness_large_knn: Optional[np.ndarray]
    neighbor_count_small: np.ndarray
    neighbor_count_large: np.ndarray
    
    # Classifications
    rai_class_radius: Optional[np.ndarray]
    rai_class_knn: Optional[np.ndarray]
    
    # Statistics
    statistics: Dict
    
    # Timing
    timing: Dict

class RAIClassifier:
    """Main class for RAI point cloud classification."""
    
    def __init__(self, config: RAIConfig = None):
        self.config = config or RAIConfig()
    
    def process(
        self,
        cloud: PointCloud,
        compute_normals: bool = True
    ) -> RAIResult:
        """
        Run full RAI processing pipeline.
        
        Steps:
        1. Compute normals (if needed)
        2. Calculate slope
        3. Build spatial index
        4. Calculate roughness (both methods if configured)
        5. Classify points
        6. Compute statistics
        """
        pass
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        generate_visualizations: bool = True,
        generate_report: bool = True
    ) -> RAIResult:
        """
        Process a single file end-to-end.
        
        Loads input, processes, saves output with attributes,
        generates visualizations and report.
        """
        pass
    
    def process_batch(
        self,
        input_paths: list,
        output_dir: Path,
        **kwargs
    ) -> list:
        """Process multiple files."""
        pass
```

**Test File**: `tests/test_classifier.py`
```python
import numpy as np
import pytest
from pathlib import Path

def test_classifier_synthetic(tmp_path):
    """Test full pipeline on synthetic data."""
    from pc_rai.classifier import RAIClassifier
    from pc_rai.io.las_reader import PointCloud
    from pc_rai.config import RAIConfig
    
    # Create synthetic point cloud with normals
    n = 1000
    xyz = np.random.uniform(0, 10, (n, 3))
    # Normals pointing mostly up (horizontal surface)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # All pointing up
    
    cloud = PointCloud(xyz=xyz, normals=normals)
    
    config = RAIConfig(methods=["radius", "knn"])
    classifier = RAIClassifier(config)
    
    result = classifier.process(cloud, compute_normals=False)
    
    assert result.n_points == n
    assert len(result.slope_deg) == n
    assert len(result.rai_class_radius) == n
    assert len(result.rai_class_knn) == n
    assert 'classification_radius' in result.statistics
```

**Acceptance Criteria**:
- [x] Full pipeline runs on synthetic data
- [x] All result fields populated correctly
- [x] Statistics computed
- [x] Timing information captured
- [x] `pytest tests/test_classifier.py` passes

---

### Task 6.2: Command-Line Interface
**Goal**: Fully functional CLI for all operations.

**File**: `pc_rai/cli.py`

**Implement**:
```python
import argparse
import sys
from pathlib import Path
from pc_rai.config import RAIConfig, load_config
from pc_rai.classifier import RAIClassifier

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='pc-rai',
        description='Point Cloud Rockfall Activity Index Classification'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process point cloud(s)')
    process_parser.add_argument('input', type=Path, help='Input LAS/LAZ file or directory')
    process_parser.add_argument('-o', '--output', type=Path, required=True, help='Output directory')
    process_parser.add_argument('-c', '--config', type=Path, help='Configuration YAML file')
    process_parser.add_argument('--batch', action='store_true', help='Process all files in directory')
    process_parser.add_argument('--skip-normals', action='store_true', help='Skip normal computation')
    process_parser.add_argument('--methods', choices=['radius', 'knn', 'both'], default='both')
    process_parser.add_argument('--no-visualize', action='store_true', help='Skip visualizations')
    process_parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    process_parser.add_argument('-v', '--verbose', action='store_true')
    
    # Visualize command (for already-processed files)
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('input', type=Path, help='Processed LAS file with RAI attributes')
    viz_parser.add_argument('-o', '--output', type=Path, required=True, help='Output directory')
    
    return parser

def main(args=None):
    """Main entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    if parsed.command is None:
        parser.print_help()
        return 1
    
    if parsed.command == 'process':
        return run_process(parsed)
    elif parsed.command == 'visualize':
        return run_visualize(parsed)
    
    return 0

def run_process(args) -> int:
    """Run processing command."""
    pass

def run_visualize(args) -> int:
    """Run visualization command."""
    pass

if __name__ == '__main__':
    sys.exit(main())
```

**Test File**: `tests/test_cli.py`
```python
import pytest
from pc_rai.cli import create_parser, main

def test_parser_process():
    parser = create_parser()
    args = parser.parse_args(['process', 'input.las', '-o', 'output/'])
    assert args.command == 'process'
    assert args.input.name == 'input.las'
    assert args.output.name == 'output'

def test_parser_help():
    parser = create_parser()
    # Should not raise
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    assert exc.value.code == 0

def test_main_no_command(capsys):
    result = main([])
    assert result == 1
```

**Acceptance Criteria**:
- [x] `python -m pc_rai --help` shows usage
- [x] `python -m pc_rai process input.las -o output/` runs pipeline
- [x] `--batch` processes multiple files
- [x] `--config` loads custom YAML
- [x] `--skip-normals` skips normal computation
- [x] `--no-visualize` and `--no-report` work
- [x] `pytest tests/test_cli.py` passes

---

## Phase 7: Integration & Documentation

### Task 7.1: End-to-End Integration Test
**Goal**: Verify complete pipeline works correctly.

**File**: `tests/test_integration.py`

**Implement**:
```python
import numpy as np
import pytest
from pathlib import Path
import laspy

@pytest.fixture
def synthetic_cliff_las(tmp_path):
    """Create a synthetic cliff point cloud."""
    n = 5000
    
    # Create points along a "cliff" profile
    # Bottom third: talus (low slope)
    # Middle: vertical cliff face
    # Top: some overhangs
    
    x = np.random.uniform(0, 10, n)
    y = np.random.uniform(0, 5, n)
    z = np.zeros(n)
    
    # Assign heights based on y (distance from cliff)
    # This creates a cliff-like structure
    talus_mask = y < 1.5
    cliff_mask = (y >= 1.5) & (y < 3.5)
    top_mask = y >= 3.5
    
    z[talus_mask] = y[talus_mask] * 0.5  # Gentle slope
    z[cliff_mask] = 0.75 + (y[cliff_mask] - 1.5) * 5  # Steep
    z[top_mask] = 10 + np.random.uniform(-0.5, 0.5, top_mask.sum())
    
    # Save as LAS
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = x
    las.y = y
    las.z = z
    
    filepath = tmp_path / "synthetic_cliff.las"
    las.write(filepath)
    return filepath

def test_full_pipeline_synthetic(synthetic_cliff_las, tmp_path):
    """Test complete pipeline on synthetic data."""
    from pc_rai.classifier import RAIClassifier
    from pc_rai.config import RAIConfig
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    config = RAIConfig(
        compute_normals=True,
        methods=["radius", "knn"]
    )
    
    classifier = RAIClassifier(config)
    result = classifier.process_file(
        synthetic_cliff_las,
        output_dir,
        generate_visualizations=True,
        generate_report=True
    )
    
    # Check outputs exist
    assert (output_dir / "synthetic_cliff_rai.las").exists()
    assert (output_dir / "synthetic_cliff_classification_radius.png").exists()
    assert (output_dir / "synthetic_cliff_report.md").exists()
    
    # Check result validity
    assert result.n_points > 0
    assert not np.all(np.isnan(result.slope_deg))
    
    # Check we got multiple classes
    unique_classes = np.unique(result.rai_class_radius)
    assert len(unique_classes) > 1

def test_cli_integration(synthetic_cliff_las, tmp_path):
    """Test CLI end-to-end."""
    from pc_rai.cli import main
    
    output_dir = tmp_path / "cli_output"
    
    result = main([
        'process',
        str(synthetic_cliff_las),
        '-o', str(output_dir),
        '--methods', 'both'
    ])
    
    assert result == 0
    assert output_dir.exists()
```

**Acceptance Criteria**:
- [x] Full pipeline runs without errors
- [x] All output files generated
- [x] Multiple RAI classes detected in output
- [x] CLI integration works
- [x] `pytest tests/test_integration.py` passes

---

### Task 7.2: Documentation
**Goal**: Complete README and usage documentation.

**File**: `README.md`

**Contents**:
- Project description
- Installation instructions
- Quick start example
- CLI usage
- Python API usage
- Configuration options
- Output file descriptions
- References to original papers

**Acceptance Criteria**:
- [x] README has installation instructions
- [x] README has CLI usage examples
- [x] README has Python API examples
- [x] README explains output formats
- [x] README references Dunham et al. 2017 and Markus et al. 2023

---

## Task Checklist Summary

### Phase 0: Setup
- [x] 0.1 Project structure
- [x] 0.2 Configuration module

### Phase 1: Data I/O
- [x] 1.1 LAS reader
- [x] 1.2 LAS writer with extra dims

### Phase 2: Core Algorithms
- [x] 2.1 Spatial utilities (KD-tree)
- [x] 2.2 Slope calculation
- [x] 2.3 Roughness calculation
- [x] 2.4 Classification decision tree

### Phase 3: CloudCompare
- [x] 3.1 CloudCompare CLI wrapper

### Phase 4: Visualization
- [x] 4.1 3D rendering
- [x] 4.2 Multi-panel figures

### Phase 5: Reporting
- [x] 5.1 Statistics and reports

### Phase 6: Pipeline & CLI
- [x] 6.1 RAI Classifier class
- [x] 6.2 Command-line interface

### Phase 7: Integration
- [x] 7.1 Integration tests
- [x] 7.2 Documentation

### Phase 8: Extensions (Post-v1.0)
- [x] 8.1 PCA-based unsupervised classification
- [x] 8.2 Output directory restructuring
- [x] 8.3 Visualization improvements
- [x] 8.4 CloudComPy normal computation script

### Phase 9: ML-Based Stability Prediction (v2.x) - REDESIGNED
- [x] 9.1 Identify pre-event surveys (`survey_selection.py`)
- [x] 9.2 Subsample & extract point-level features (`feature_extraction.py`)
- [x] 9.3 Polygon assignment & aggregation (`polygon_aggregation.py`)
- [x] 9.4 Assemble training data - case-control (`training_data.py`)
- [x] 9.5 Train Random Forest prototype (`train.py`) — temporal AUC-ROC=0.696
- [ ] **9.6 Full-scale pipeline: process all data, retrain (10 m³, 2025 hold-out)** 🔜
- [ ] 9.7 Inference pipeline with 10m aggregation (`inference.py`)
- [ ] 9.8 CLI scripts for full pipeline

---

## Notes for Claude Code Agent

1. **Run tests after each task**: Use `pytest tests/test_<module>.py -v` to verify implementation
2. **Reference the PRD**: See `docs/prd.md` for detailed specifications (especially Appendix D for v2.x)
3. **Handle missing dependencies**: If a test fails due to missing packages, install them
4. **CloudCompare optional**: Tasks that require CloudCompare can be skipped if not installed - mark as such
5. **Incremental commits**: Each task should result in working, tested code
6. **Update this file**: Mark tasks `[x]` as complete and update "Project Status" section

## Future Work

### Task 9.1: Cliff Top Detection and Noise Removal
**Goal**: Automatically identify the cliff top boundary and exclude points above/beyond it (e.g., vegetation, buildings, terrain behind the cliff edge) that introduce noise into the RAI classification.

**Approach Options**:
- [ ] Integrate with [CoastSeg/CliffDelineaTool 2.0](https://github.com/SatelliteShorelines/CoastSeg) to delineate cliff top from satellite imagery, then use that boundary to clip the point cloud
- [ ] Computer vision-based cliff top identifier that works directly on the point cloud geometry (e.g., detect the sharp slope break at the cliff edge using curvature/slope gradient analysis)

**Acceptance Criteria**:
- [ ] Points above/beyond the cliff top are identified and masked or removed before RAI classification
- [ ] Works across different coastal bluff morphologies (vertical cliffs, sloped bluffs, etc.)
- [ ] Optional — users can disable cliff top clipping if not needed
- [ ] Visualizations show the detected cliff top boundary for QA/QC

---

## Known Issues

### Mini Ranger Surveys
Mini Ranger LiDAR surveys may produce anomalous RAI classification results. The point cloud characteristics (density, noise, scan geometry) differ from standard TLS surveys and can cause unexpected class distributions. Consider excluding Mini Ranger data from batch processing or flagging it for manual review until the issue is better understood.

### Transect Rendering on Risk Map Figures
Transects are not rendering correctly on the risk map figures for some locations. Needs investigation and fix.

---

*Last updated: February 5, 2026 (v1.0 + Extensions, v2.x ML prototype complete, full-scale pipeline next — 10 m³ threshold, all 6 beaches, 2025 hold-out)*
