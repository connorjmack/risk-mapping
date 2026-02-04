# PC-RAI Development Todo

> **Instructions for Claude Code**: Work through these tasks sequentially. Each task has clear acceptance criteria. Mark tasks complete with `[x]` as you finish them. Run the specified tests before moving to the next task. Reference `docs/prd.md` for detailed specifications.

---

## Project Status

- **Current Phase**: v1.0 Complete, v2.x ML Pipeline In Progress
- **Last Completed Task**: 9.4 Temporal Alignment Module
- **Next Up**: 9.5 Optimize Polygon Feature Extraction
- **Tests Passing**: 225 (1 flaky CloudCompare integration test)
- **Blocking Issues**: Feature extraction too slow for all polygons (see Task 9.5)

### v2.x ML Pipeline Progress

| Task | Status | Notes |
|------|--------|-------|
| 9.1 ML Package Setup | ✅ Complete | `pc_rai/ml/` with config, data_prep |
| 9.2 Event Label Loading | ✅ Complete | CSV-based with QC filtering |
| 9.3 Polygon Spatial Matching | ✅ Complete | 1m polygons (ID = alongshore_m) |
| 9.4 Temporal Alignment | ✅ Complete | Case-control design |
| 9.5 Optimize Feature Extraction | 🔄 In Progress | Need to extract only needed polygons |
| 9.6 Random Forest Training | ⏳ Pending | `train.py` exists |
| 9.7 Cross-Validation | ⏳ Pending | Leave-one-beach-out |

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

## Phase 9: ML-Based Stability Prediction (v2.x)

> **Prerequisites**: v1.x must be complete. Point-level features (slope, roughness) and RAI classification must be functional.
> **Reference**: See `docs/prd.md` Section 3.10 for detailed specifications.

### Key Design Decisions (Implemented)

1. **1m Polygon Resolution** (not 10m transects)
   - Polygon IDs = alongshore meter positions
   - Precise spatial alignment with event alongshore positions

2. **Temporal Alignment** (case-control study design)
   - Cases: pre-failure morphology (scan BEFORE event)
   - Controls: polygons without subsequent events
   - Trains on predictive features, not post-failure descriptions

3. **Event Filtering**
   - Events >= 5 m³
   - Include "real" and "unreviewed" QC flags
   - Exclude "construction" and "noise"

---

### Task 9.1: ML Package Setup & Dependencies ✅ COMPLETE
**Goal**: Create the ML module structure and install required dependencies.

**Implemented Files**:
```
pc_rai/ml/
├── __init__.py          # Module exports
├── config.py            # MLConfig dataclass
├── data_prep.py         # Event CSV loading and filtering
├── polygons.py          # 1m polygon spatial matching
├── temporal.py          # Temporal alignment for case-control training
├── train.py             # Random Forest training
│
└── (legacy - 10m transects)
    ├── labels.py        # Transect-based labeling
    └── features.py      # Transect-based feature extraction
```

**MLConfig** (in `pc_rai/ml/config.py`):
```python
@dataclass
class MLConfig:
    min_volume: float = 5.0                    # m³ threshold for events
    qc_flags_include: List[str] = ["real", "unreviewed"]
    qc_flags_exclude: List[str] = ["construction", "noise"]
    transect_half_width: float = 5.0           # meters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_leaf: int = 5
    class_weight: str = "balanced"
    random_state: int = 42
```

**Acceptance Criteria**:
- [x] `pc_rai/ml/` directory exists with all module files
- [x] `MLConfig` dataclass in `pc_rai/ml/config.py`
- [x] `python -c "from pc_rai.ml import *"` succeeds

---

### Task 9.2: Event Label Loading ✅ COMPLETE
**Goal**: Load rockfall events from CSV and filter by QC flags and volume.

**File**: `pc_rai/ml/data_prep.py`

**Implemented**:
```python
def load_events(csv_path: Path) -> pd.DataFrame:
    """Load event CSV with date parsing."""

def filter_events(
    events: pd.DataFrame,
    config: MLConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """Filter events by QC flags and minimum volume."""

def print_event_summary(events: pd.DataFrame) -> None:
    """Print summary statistics of filtered events."""
```

**Data Source**: `utiliies/events/<Beach>_events_qc_*.csv`

**Event CSV Columns**:
- `mid_date`, `start_date`, `end_date` (temporal)
- `alongshore_centroid_m`, `alongshore_start_m`, `alongshore_end_m` (spatial)
- `volume`, `height`, `width` (geometry)
- `qc_flag` (real, unreviewed, construction, noise)

**Acceptance Criteria**:
- [x] Can load event CSV with date parsing
- [x] Filters by QC flags (include real/unreviewed, exclude construction/noise)
- [x] Filters by minimum volume (>= 5 m³)
- [x] Prints summary statistics

---

### Task 9.3: Polygon Spatial Matching ✅ COMPLETE
**Goal**: Match events to 1m polygons and extract features using vectorized point-in-polygon tests.

**File**: `pc_rai/ml/polygons.py`

**Implemented**:
```python
@dataclass
class Polygon:
    """A single 1m alongshore polygon."""
    polygon_id: int          # Equals alongshore meter position
    vertices: np.ndarray
    x_min, x_max, y_min, y_max: float
    _path: MplPath           # For vectorized contains_points

    def points_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized point-in-polygon test using matplotlib.path."""

class PolygonLabeler:
    """Loads and manages 1m polygon shapefiles."""

    def find_polygons_for_event(
        self, alongshore_start: float, alongshore_end: float
    ) -> List[int]:
        """Find polygon IDs that overlap with an event's alongshore extent."""
```

**Data Source**: `utiliies/polygons_1m/<Beach>Polygons*/`

**Key Insight**: Polygon ID = alongshore meter position (e.g., polygon 626 = 626m alongshore)

**Acceptance Criteria**:
- [x] Load polygon shapefiles with pyshp
- [x] Vectorized point-in-polygon using matplotlib.path
- [x] Match event alongshore range to polygon IDs
- [x] Bounding box pre-filter for efficiency

---

### Task 9.4: Temporal Alignment ✅ COMPLETE
**Goal**: Implement case-control study design with pre-failure morphology.

**File**: `pc_rai/ml/temporal.py`

**Implemented**:
```python
class TemporalAligner:
    """Aligns events to pre-failure scans for case-control training."""

    def discover_scans(self, pattern: str = "*_rai.laz") -> int:
        """Find available point cloud scans."""

    def find_pre_event_scan(self, event_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Find the most recent scan before an event (at least min_days_before)."""

    def load_scan_features(self, scan_date: pd.Timestamp) -> pd.DataFrame:
        """Load point cloud once and extract all polygon features."""

    def create_case_control_dataset(
        self, events: pd.DataFrame, control_ratio: float = 1.0
    ) -> pd.DataFrame:
        """Create temporally-aligned case-control dataset."""

def create_temporal_training_data(
    events: pd.DataFrame,
    point_cloud_dir: Path,
    polygon_shapefile: Path,
    min_days_before: int = 7,
    control_ratio: float = 1.0,
) -> Tuple[pd.DataFrame, TemporalAligner]:
    """Convenience function to create temporally-aligned training data."""
```

**Training Script**: `scripts/prepare_delmar_training_temporal.py`

**Acceptance Criteria**:
- [x] Groups events by pre-event scan date
- [x] Loads each scan once (not per-polygon)
- [x] Extracts features for event polygons (cases)
- [x] Samples control polygons from same scans
- [x] Tracks temporal metadata (scan_date, event_date, days_to_event)

---

### Task 9.5: Optimize Polygon Feature Extraction 🔄 IN PROGRESS
**Goal**: Reduce processing time by only extracting features for needed polygons.

**Problem**: Extracting features for all 2285 polygons per scan is too slow (hours).

**Solution**: Only extract features for:
1. Event polygons (cases) - determined by event alongshore extent
2. Sampled control polygons (not all 2000+)

**Implementation Plan**:
```python
# In create_case_control_dataset():
# 1. First pass: identify all needed polygon IDs across all events
needed_polygon_ids = set()
for event in events:
    polygon_ids = labeler.find_polygons_for_event(event.alongshore_start, event.alongshore_end)
    needed_polygon_ids.update(polygon_ids)

# 2. Add sampled control polygon IDs (e.g., 2x number of case polygons)
control_polygon_ids = sample_controls(all_polygon_ids - needed_polygon_ids, n=len(needed_polygon_ids) * 2)
needed_polygon_ids.update(control_polygon_ids)

# 3. Extract features only for needed polygons
features_df = extract_all_polygon_features(las_path, labeler, polygon_ids=list(needed_polygon_ids))
```

**Acceptance Criteria**:
- [ ] Identify needed polygon IDs before loading point clouds
- [ ] Extract features only for needed polygons
- [ ] Processing time < 10 minutes for Del Mar (18 scans × ~200 polygons)
- [ ] Results equivalent to full extraction

---

### Task 9.6: Random Forest Training ✅ COMPLETE
**Goal**: Train Random Forest model with class weighting for imbalanced data.

**File**: `pc_rai/ml/train.py`

**Implemented**:
```python
@dataclass
class StabilityModel:
    """Trained stability prediction model."""
    model: RandomForestClassifier
    feature_names: List[str]
    feature_importances: Dict[str, float]
    cv_metrics: Dict[str, float]      # AUC-ROC, AUC-PR from cross-validation
    hyperparameters: Dict[str, Any]
    train_date: datetime

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict stability score (0-1) for polygons."""

    def save(self, path: Path) -> None:
        """Save model + metadata as .joblib and .json files."""

    @classmethod
    def load(cls, path: Path) -> "StabilityModel":
        """Load model from disk."""

def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    config: MLConfig,
    verbose: bool = True
) -> StabilityModel:
    """Train RF with stratified k-fold CV and class balancing."""
```

**Cross-Validation**: Uses `StratifiedKFold(n_splits=5)` with balanced class weights.

**Metrics Computed**:
- AUC-ROC (area under ROC curve)
- AUC-PR (area under precision-recall curve)
- Feature importances

**Acceptance Criteria**:
- [x] Model trains with balanced class weights
- [x] Cross-validation computes AUC-ROC and AUC-PR
- [x] Feature importances extracted
- [x] Model saves/loads with metadata
- [x] Predictions return probabilities in [0, 1]

---

### Task 9.7: Leave-One-Beach-Out Cross-Validation ⏳ PENDING
**Goal**: Implement spatial generalization validation.

**File**: `pc_rai/ml/train.py` (extend existing)

**Plan**:
```python
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import numpy as np
from typing import Dict
from pc_rai.ml.train import train_stability_model, StabilityModel
from pc_rai.config import MLConfig

def leave_one_beach_out_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    beach_ids: np.ndarray,
    config: MLConfig
) -> Dict[str, Dict[str, float]]:
    """
    Leave-one-beach-out cross-validation.

    Returns
    -------
    Dict mapping beach_id → {'auc_roc': float, 'auc_pr': float, 'n_samples': int, 'n_events': int}
    """
    pass

def leave_one_year_out_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray,
    config: MLConfig
) -> Dict[int, Dict[str, float]]:
    """
    Leave-one-year-out temporal validation.

    Returns
    -------
    Dict mapping year → {'auc_roc': float, 'auc_pr': float, ...}
    """
    pass

def summarize_cv_results(results: Dict) -> pd.DataFrame:
    """Create summary table of CV results."""
    pass
```

**Test File**: `tests/test_ml_validate.py`

**Acceptance Criteria**:
- [ ] Leave-one-beach-out correctly holds out each beach
- [ ] Leave-one-year-out correctly holds out each year
- [ ] Metrics computed for each fold
- [ ] Summary table has mean/std across folds
- [ ] `pytest tests/test_ml_validate.py` passes

---

### Task 9.7: Metrics & Evaluation
**Goal**: Compute AUC-ROC, AUC-PR, confusion matrices, and comparison with rule-based RAI.

**File**: `pc_rai/ml/metrics.py`

**Implement**:
```python
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import numpy as np
import pandas as pd
from typing import Dict

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Returns dict with:
        auc_roc, auc_pr, precision, recall, f1, accuracy
    """
    pass

def compare_with_rai(
    rai_classes: np.ndarray,
    y_true: np.ndarray,
    ml_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compare ML predictions against rule-based RAI.

    Uses RAI classes 3 (Discontinuous) and 4 (Steep/Overhang) as "high risk".
    Returns AUC improvement and other comparison metrics.
    """
    pass

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: Path) -> None:
    """Generate and save ROC curve plot."""
    pass

def plot_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: Path) -> None:
    """Generate and save Precision-Recall curve plot."""
    pass

def plot_feature_importances(importances: Dict[str, float], output_path: Path) -> None:
    """Generate and save feature importance bar chart."""
    pass
```

**Test File**: `tests/test_ml_metrics.py`

**Acceptance Criteria**:
- [ ] AUC-ROC computed correctly (compare with sklearn)
- [ ] AUC-PR computed correctly
- [ ] Confusion matrix has correct shape
- [ ] RAI comparison identifies improvement over rule-based
- [ ] All plots generate without errors
- [ ] `pytest tests/test_ml_metrics.py` passes

---

### Task 9.8: Inference Pipeline
**Goal**: Apply trained model to new point clouds.

**File**: `pc_rai/ml/predict.py`

**Implement**:
```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import geopandas as gpd
from pc_rai.ml.train import StabilityModel
from pc_rai.features.aggregation import aggregate_laz_to_transects

def predict_transect_stability(
    laz_dir: Path,
    transects_path: Path,
    model: StabilityModel,
) -> pd.DataFrame:
    """
    Predict stability scores for all transects from LAZ files.

    Returns DataFrame with columns:
        transect_id, stability_score, scan_file
    """
    pass

def classify_risk_level(
    stability_score: float,
    thresholds: tuple = (0.3, 0.6)
) -> str:
    """
    Convert continuous score to discrete risk level.

    Returns: 'Low', 'Medium', or 'High'
    """
    pass

def export_predictions_to_shapefile(
    predictions: pd.DataFrame,
    transects: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """Export predictions as shapefile for GIS visualization."""
    pass
```

**Test File**: `tests/test_ml_predict.py`

**Acceptance Criteria**:
- [ ] Predictions generated for all transects with valid features
- [ ] Stability scores in [0, 1] range
- [ ] Risk level classification works
- [ ] Shapefile export includes geometry and scores
- [ ] `pytest tests/test_ml_predict.py` passes

---

### Task 9.9: CLI Scripts for Training/Prediction
**Goal**: Command-line scripts for model training and inference.

**File**: `scripts/train_rf_model.py`

**Implement**:
```python
"""
Train Random Forest stability model.

Usage:
    python scripts/train_rf_model.py \
        --features data/transect_features.parquet \
        --labels data/transect_labels.parquet \
        --output models/stability_rf.joblib \
        --cv leave-one-beach-out
"""
# See docs/prd.md Appendix D for full implementation
```

**File**: `scripts/predict_stability.py`

**Implement**:
```python
"""
Predict stability scores for new point clouds.

Usage:
    python scripts/predict_stability.py \
        --model models/stability_rf.joblib \
        --laz-dir output/rai/ \
        --transects data/transects.shp \
        --output predictions/stability_scores.csv
"""
# See docs/prd.md Appendix D for full implementation
```

**File**: `scripts/prepare_training_data.py`

**Implement**:
```python
"""
Prepare training dataset from LAZ files, transects, and event polygons.

Usage:
    python scripts/prepare_training_data.py \
        --laz-dir output/rai/ \
        --transects data/transects.shp \
        --events data/rockfall_events.shp \
        --scan-metadata data/scan_dates.csv \
        --output data/training/
"""
```

**Acceptance Criteria**:
- [ ] `train_rf_model.py` trains and saves model
- [ ] `predict_stability.py` loads model and generates predictions
- [ ] `prepare_training_data.py` creates training parquet files
- [ ] All scripts have `--help` documentation
- [ ] Exit codes: 0 for success, non-zero for errors

---

### Task 9.10: Risk Map Integration
**Goal**: Integrate ML stability scores with existing risk map visualization.

**File**: `scripts/risk_map_regional.py` (modify existing)

**Changes**:
```python
# Add argument:
parser.add_argument('--scores', type=Path, help='ML stability scores CSV')

# In render_regional_risk_map():
def load_stability_scores(scores_path: Path) -> Dict[int, float]:
    """Load ML stability scores by transect ID."""
    df = pd.read_csv(scores_path)
    return dict(zip(df['transect_id'], df['stability_score']))

# When --scores provided, use stability_score instead of energy_sum for coloring
```

**Acceptance Criteria**:
- [ ] `--scores` argument added to CLI
- [ ] When provided, map colors by ML stability score
- [ ] Colorbar label updates to "Stability Score" or "Failure Probability"
- [ ] Works with existing energy-based mode when --scores not provided
- [ ] Map renders correctly with both modes

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

### Phase 9: ML-Based Stability Prediction (v2.x)
- [ ] 9.1 ML package setup & dependencies
- [ ] 9.2 Event label loading
- [ ] 9.3 Transect feature aggregation
- [ ] 9.4 Training data preparation
- [ ] 9.5 Random Forest training
- [ ] 9.6 Cross-validation framework
- [ ] 9.7 Metrics & evaluation
- [ ] 9.8 Inference pipeline
- [ ] 9.9 CLI scripts for training/prediction
- [ ] 9.10 Risk map integration

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

*Last updated: February 2025 (v1.0 + Extensions, v2.x ML tasks added)*
