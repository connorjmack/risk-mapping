# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PC-RAI** is a Python tool for classifying LiDAR point clouds into rockfall hazard categories using the Rockfall Activity Index (RAI) methodology. It adapts the grid-based RAI algorithm to work directly on point clouds.

## Key Documentation

| File | Purpose |
|------|---------|
| `README.md` | User-facing documentation, installation, CLI usage |
| `docs/prd.md` | Product requirements, specifications, data structures |
| `docs/todo.md` | Task list with acceptance criteria (v1.0 complete + extensions) |
| `docs/pipeline_data_guide.md` | Data paths, formats, and schemas for ML pipeline |
| `docs/training_pipeline_outline.md` | ML training workflow overview |
| `CLAUDE.md` | This file - agent instructions |

**Project Status**: v1.0 complete with extensions (PCA classification, improved visualizations).
v2.x ML pipeline steps 1-6 complete (survey matching, feature extraction, aggregation, training, ablation study).
266 tests total; ~51 pre-existing failures in v1 modules (viz, energy, classification, integration) unrelated to ML pipeline work.
Trained models available with temporal CV: AUC-ROC=0.701, AUC-PR=0.667 on test dataset (72,782 samples).

## Dependencies

Core: numpy, scipy, laspy, matplotlib, scikit-learn, joblib, pyshp (shapefile)
**Not available:** geopandas, fiona — use `pyshp` (the `shapefile` module) for reading shapefiles.

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_slope.py -v

# Run with coverage
pytest tests/ --cov=pc_rai --cov-report=term-missing

# Type checking (if mypy installed)
mypy pc_rai/

# Format code
black pc_rai/ tests/

# Lint
ruff check pc_rai/ tests/
```

## Project Structure

```
pc_rai/
├── __init__.py          # Package init, version
├── __main__.py          # Entry: python -m pc_rai
├── cli.py               # Argparse CLI
├── config.py            # RAIConfig dataclass, constants
├── classifier.py        # Main RAIClassifier class
├── io/                  # LAS file I/O
├── normals/             # CloudCompare integration
├── features/            # Slope, roughness calculation
├── classification/      # Decision tree + PCA classifier
│   ├── decision_tree.py # RAI decision tree logic
│   └── pca_classifier.py # PCA + K-means unsupervised classification
├── visualization/       # Rendering, figures
├── reporting/           # Statistics, reports
├── utils/               # Spatial index, timing, logging
└── ml/                  # v2.x ML pipeline for rockfall prediction
    ├── config.py            # MLConfig dataclass
    ├── data_prep.py         # Load and filter event CSVs
    ├── features.py          # Feature definitions
    ├── feature_extraction.py # Subsample + compute point features
    ├── labels.py            # Event label handling
    ├── polygons.py          # Polygon geometry operations
    ├── polygon_aggregation.py # Aggregate to 1m polygon-zones
    ├── survey_selection.py  # Survey matching logic
    ├── temporal.py          # Temporal alignment utilities
    ├── training_data.py     # Case-control dataset assembly
    ├── train.py             # Random Forest training
    └── ablation.py          # Cumulative feature ablation study

scripts/
├── 01_identify_surveys.py        # Match surveys to events
├── 02_extract_features.py        # Subsample + feature extraction
├── 03_aggregate_polygons.py      # Aggregate to polygon-zones
├── 04_assemble_training_data.py  # Create case-control dataset
├── 05_train_model.py             # Train Random Forest model
├── 06_ablation_study.py          # Cumulative feature ablation
├── compute_normals_mst.py        # CloudComPy normal computation
├── compute_features.py           # Batch feature computation
├── subsample_all.py              # Batch subsampling
├── validate_labels.py            # Label validation checks
├── plot_training_results.py      # Visualize training metrics (v2.x)
├── generate_test_data.py         # Generate synthetic test data
├── prepare_delmar_training.py    # Del Mar site-specific prep
├── prepare_delmar_training_temporal.py  # Temporal training prep
├── plot_decision_tree.py         # Visualize decision tree
├── risk_map_regional.py          # County-wide risk map
└── setup_linux_env.sh            # Linux environment setup
```

## Coding Standards

### Style
- **Formatting**: black with default settings
- **Line length**: 88 characters (black default)
- **Imports**: stdlib → third-party → local, alphabetized within groups
- **Docstrings**: NumPy style

### Type Hints
Always use type hints for function signatures:
```python
def calculate_slope(
    normals: np.ndarray,
    up_vector: tuple[float, float, float] = (0, 0, 1)
) -> np.ndarray:
    """Calculate slope from normals."""
    ...
```

### Error Handling
- Use specific exceptions, not bare `except:`
- Always provide informative error messages

### Testing
- Every module needs a corresponding test file
- Use pytest fixtures for shared setup (see `conftest.py`)
- Test edge cases: empty arrays, NaN values, single points
- Use `tmp_path` fixture for file I/O tests

## Key Implementation Details

### NumPy Array Conventions
| Data | Shape | Dtype | Notes |
|------|-------|-------|-------|
| XYZ coordinates | (N, 3) | float64 | Required |
| Normal vectors | (N, 3) | float32 | Unit vectors |
| Slope | (N,) | float32 | Degrees, 0-180 |
| Roughness | (N,) | float32 | Degrees, NaN for invalid |
| Classification | (N,) | uint8 | Codes 0-5 |

### RAI Class Codes (Simplified 5-class scheme)
```python
0 = Unclassified (invalid/insufficient data)
1 = Talus (T)
2 = Intact (I)
3 = Discontinuous (D) - potential rockfall source
4 = Steep/Overhang (O) - high risk steep faces (slope >80°)
5 = Structure (St) - seawalls, engineered surfaces
```

### Decision Tree Logic (Critical)
```
if slope > 80°:
    if r_small < 6° AND r_large < 6° → Structure (St)  # Dual-scale smoothness check
    else → Steep/Overhang (O)
elif r_small < 6°:
    if slope < 42° → Talus (T)
    else → Intact (I)
elif r_small > 15° → Discontinuous (D)
elif r_large > 15° → Discontinuous (D)
else → Intact (I)
```

### Classification Smoothing
After decision tree classification, a spatial majority-vote smoothing is applied:
```python
# For each point, find k=25 nearest neighbors
# Assign the most common class among neighbors
# This reduces noise while preserving class boundaries
```

### Default Parameters (adapted from Markus et al. 2023)
```python
radius_small = 1.0    # meters (tuned for 50cm point spacing)
radius_large = 2.5    # meters
k_small = 40          # neighbors
k_large = 120         # neighbors
thresh_overhang = 80.0         # degrees (80° for coastal bluffs)
thresh_talus_slope = 42.0      # degrees
thresh_structure_roughness = 6.0  # degrees (dual-scale check)
thresh_r_small_mid = 15.0      # degrees (roughness threshold for Discontinuous)
thresh_r_large = 15.0          # degrees (large-scale roughness threshold)
smoothing_k = 25               # neighbors for majority-vote smoothing
```

### PCA-Based Classification
The PCA classifier (`pc_rai/classification/pca_classifier.py`) provides unsupervised classification:
```python
from pc_rai.classification.pca_classifier import classify_pca, get_cluster_interpretation

# Uses same features: slope, roughness_small, roughness_large
result = classify_pca(slope_deg, roughness_small, roughness_large)

# Auto-detects optimal clusters (3-12) using silhouette score
print(f"Found {result.n_clusters} clusters")
print(f"Silhouette score: {result.silhouette_avg:.3f}")

# Get human-readable interpretations
interpretations = get_cluster_interpretation(result)
for cluster_id, interp in interpretations.items():
    print(f"  Cluster {cluster_id}: {interp}")
```

Labels are stored as `pca_cluster` in output LAZ files (-1 for invalid points).

### Energy Calculation
Per-point rockfall energy is computed using Dunham et al. (2017) methodology:
```python
E_kj = 0.5 * ρ * A * D * g * H / 1000

# Where:
ρ = 2400 kg/m³     # Rock density
A = 0.01 m²        # Point area (10cm spacing)
g = 9.81 m/s²      # Gravity
H = point elevation (relative)

# D = class-specific failure depth × instability rate:
#   Talus (T):         0.0 m × 0.00 = 0.0 m
#   Intact (I):        0.05 m × 0.03 = 0.0015 m
#   Discontinuous (D): 0.5 m × 0.10 = 0.05 m
#   Steep/Overhang (O):1.0 m × 0.50 = 0.5 m
#   Structure (St):    0.0 m × 0.00 = 0.0 m
```

Energy is stored as `energy_kj_knn` in output LAZ files.

### Output Directory Structure
```
output/
├── rai/                    # LAZ files (flat, for bulk processing)
│   └── *_rai.laz
├── reports/<LOCATION>/     # Reports organized by location
│   ├── *_report.md
│   └── *_report.json
├── panels/<LOCATION>/      # 4-panel summary figures by location
│   └── *_panels.png        # Slope, classification, roughness, histogram
└── heatmap/<LOCATION>/     # Transect risk heatmaps by location
    └── *_risk_map_3d.png
```

Location is extracted from filename patterns:
- `YYYYMMDD_LOCATION_...` → LOCATION (e.g., `20241215_TORP_subsamp1` → `TORP`)
- `LOCATION_YYYYMMDD_...` → LOCATION
- Falls back to `misc/` if no pattern matches

## Common Pitfalls

### 1. Slope Calculation
The slope angle is measured FROM the up vector, not the horizontal:
```python
# CORRECT: arccos of Z component of unit normal
slope = np.degrees(np.arccos(normals[:, 2]))

# WRONG: arctan or angle from horizontal
```

### 2. KD-Tree Queries
Use `scipy.spatial.cKDTree` (C implementation), not `KDTree`:
```python
# CORRECT
from scipy.spatial import cKDTree
tree = cKDTree(points)

# SLOWER
from scipy.spatial import KDTree
```

### 3. LAS Extra Dimensions
laspy requires `ExtraBytesParams` for adding custom dimensions:
```python
las.add_extra_dim(laspy.ExtraBytesParams(name="slope_deg", type=np.float32))
las["slope_deg"] = slope_values
```

### 4. NaN Handling in Classification
Points with NaN roughness should be classified as 0 (Unclassified):
```python
invalid = np.isnan(r_small) | np.isnan(r_large)
classes[invalid] = 0
```

### 5. Polygon Indexing (v2.x ML)
DelMar shapefile has `Id` field (475-2759), no `MOP_ID`. Use `alongshore_m = Id - min(Id)`.
Other beaches have `MOP_ID` (e.g., 708.001). Use `alongshore_m = centroid_y - min(centroid_y)`.
Events use local 0-based alongshore meters. 1-5m alignment error is acceptable (events span 5-30m+).

### 6. CloudCompare Output Location
CloudCompare may save files with modified names. Check for:
- `{filename}_WITH_NORMALS.las`
- Files in unexpected directories
- Use explicit `-SAVE_CLOUDS FILE {path}` argument

## Testing Without CloudCompare

CloudCompare is an external dependency that may not be installed. Design tests to work without it:

```python
@pytest.fixture
def cloud_with_normals():
    """Create point cloud with synthetic normals (no CloudCompare needed)."""
    n = 1000
    xyz = np.random.uniform(0, 10, (n, 3))
    # Synthetic normals - mostly pointing up with some variation
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    # Add noise
    normals += np.random.normal(0, 0.1, (n, 3))
    # Renormalize
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return PointCloud(xyz=xyz, normals=normals)
```

## Performance Considerations

### Large Point Clouds (>10M points)
- Build KD-tree once, reuse for all roughness scales
- Use vectorized numpy operations
- Consider chunked processing for very large files
- Monitor memory usage

### Visualization
- Subsample for preview (every Nth point)
- Full resolution only for final output
- Use `matplotlib` Agg backend for headless rendering

## Updating Progress

After completing each task:

1. Run tests: `pytest tests/test_<module>.py -v`
2. Update `todo.md`:
   - Change `[ ]` to `[x]` for completed items
   - Update "Project Status" section at top
3. Commit with message: `feat(module): implement <feature>`

## Getting Unstuck

If tests fail unexpectedly:
1. Check array shapes with `print(arr.shape, arr.dtype)`
2. Check for NaN: `print(np.isnan(arr).sum())`
3. Verify ranges: `print(arr.min(), arr.max())`
4. Run single test with `-v --tb=long` for full traceback

If unclear on requirements:
1. Check `docs/prd.md` for specifications
2. Check Appendix A (decision tree pseudocode)
3. Check Appendix B (CloudCompare commands)
4. Check Appendix D (v2.x ML implementation guide)

## External Documentation

### CloudComPy

Local CloudComPy docs and test examples are available at:
```
/Users/cjmack/Tools/CloudComPy311/github-docs
```
Reference this directory for CloudComPy API usage, Python bindings, and test patterns.

#### Activating CloudComPy Environment

To compute normals using CloudComPy, activate the conda environment first:

```bash
# Activate CloudComPy environment
. /Users/cjmack/Tools/CloudComPy311/bin/condaCloud.zsh activate cloud-compy

# Compute normals for test data
python scripts/compute_normals_mst.py tests/test_data --output-dir output/normals --radius 1.0

# Deactivate when done
conda deactivate
```

Then process with PC-RAI using `--skip-normals`:
```bash
source .venv/bin/activate
pc-rai process output/normals/*.las -o output/rai --skip-normals -v
```

---

## v2.x ML Pipeline: Temporal Alignment with 1m Polygons

The v2.x ML pipeline uses a **case-control study design** with **1m polygon bins** and **elevation zones** for predicting rockfall probability from pre-failure cliff morphology.

### Key Design Decisions

1. **1m Polygon Resolution with Elevation Zones**
   - Alongshore bins at 1m resolution
   - Each polygon split into **lower/middle/upper thirds** (relative elevation)
   - Captures different failure behavior at different cliff heights

2. **Temporal Alignment** (case-control design)
   - **Cases**: Pre-failure morphology (features from scans taken BEFORE events)
   - **Controls**: Features from polygons that did NOT have subsequent events
   - This ensures we're training on **predictive** features, not post-failure descriptions

3. **Event Filtering**
   - Include events >= 5 m³ (significant failures)
   - Include "real" and "unreviewed" QC flags
   - Exclude "construction" and "noise" events

4. **Features Computed**
   - Slope (from normals)
   - Roughness at small (1m) and large (2.5m) scales
   - Eigenvalue features: planarity, linearity, sphericity, curvature
   - Relative height
   - Aggregated per polygon-zone: mean, std, min, max, p10, p50, p90

### ML Pipeline Scripts (Test Data)

```bash
# Step 1: Find pre-event surveys (match surveys to future events)
python scripts/01_identify_surveys.py \
    --events utiliies/events/qc_ed/DelMar_events_qc_*.csv \
    --surveys utiliies/file_lists/all_noveg_files.csv \
    --output data/test_pre_event_surveys.csv \
    --location DelMar \
    --min-volume 5.0 -v

# Step 2: Extract features from point clouds
python scripts/02_extract_features.py \
    --input-dir data/test_data/no_veg/ \
    --output-dir data/test_subsampled/ \
    --subsample-only

# Compute normals with CloudComPy (separate environment)
. /Users/cjmack/Tools/CloudComPy311/bin/condaCloud.zsh activate cloud-compy
python scripts/compute_normals_mst.py data/test_subsampled/ \
    --output-dir data/test_subsampled_normals/
conda deactivate

# Extract features (slope, roughness, eigenvalues)
python scripts/02_extract_features.py \
    --input-dir data/test_subsampled_normals/ \
    --output-dir data/test_features/

# Step 3: Aggregate to polygon-zones
python scripts/03_aggregate_polygons.py \
    --input-dir data/test_features/ \
    --output data/polygon_features.csv

# Step 4: Assemble training data (case-control)
python scripts/04_assemble_training_data.py \
    --features data/polygon_features.csv \
    --surveys data/test_pre_event_surveys.csv \
    --output data/training_data.csv

# Step 5: Train model
python scripts/05_train_model.py \
    --input data/training_data.csv \
    --output models/rf_model.joblib \
    --group-by location -v

# Step 5b: Cumulative feature ablation study
python scripts/06_ablation_study.py \
    --input data/training_data.csv \
    --output output/ablation/ -v

# With leave-one-beach-out CV (honest generalization)
python scripts/06_ablation_study.py \
    --input data/training_data.csv \
    --output output/ablation_by_location/ \
    --group-by location -v
```

### Full Dataset Production Workflow

For production training on all locations, process QC'd events in batch:

```bash
# Step 1: Process all locations (QC'd event files have timestamps)
for location in DelMar SanElijo Encinitas Solana Torrey Blacks; do
    echo "Processing $location..."

    # Find most recent QC file for this location
    event_file=$(ls -t utiliies/events/qc_ed/${location}_events_qc_*.csv 2>/dev/null | head -1)

    if [ -z "$event_file" ]; then
        echo "  Warning: No QC file found for $location, skipping..."
        continue
    fi

    echo "  Using: $event_file"

    python scripts/01_identify_surveys.py \
        --events "$event_file" \
        --surveys utiliies/file_lists/all_noveg_files.csv \
        --output data/pre_event_surveys_${location}.csv \
        --location $location \
        --min-volume 5.0 -v
done

# Combine all location results
head -1 data/pre_event_surveys_DelMar.csv > data/pre_event_surveys_full.csv
tail -n +2 -q data/pre_event_surveys_*.csv >> data/pre_event_surveys_full.csv

# Step 2-3: Features already extracted and aggregated to data/polygon_features.csv

# Step 4: Assemble training data (full dataset)
python scripts/04_assemble_training_data.py \
    --features data/polygon_features.csv \
    --surveys data/pre_event_surveys_full.csv \
    --output data/training_data.csv -v

# Step 5: Train model (with leave-one-beach-out CV)
python scripts/05_train_model.py \
    --input data/training_data.csv \
    --output models/rf_model_full.joblib \
    --group-by location -v
```

**Important Notes:**
- QC'd event files have timestamps: `{Location}_events_qc_{timestamp}.csv`
- Use `ls -t` to find the most recent version for each location
- The `--location` flag is required to filter surveys and prevent cross-location mismatches
- Step 2 (feature extraction) is assumed done; features are pre-computed on the network drive

### Data Locations

**Note:** The `utiliies/` directory is misspelled in the repo (missing a 't'). Do not rename it.

| Data | Path |
|------|------|
| Raw event CSVs | `utiliies/events/raw/<Beach>_events.csv` (15 cols, includes MOP coords) |
| QC'd event CSVs | `utiliies/events/qc_ed/<Beach>_events_qc_*.csv` (12 cols, no MOP coords) |
| 1m polygon shapefiles | `utiliies/polygons_1m/` |
| Pre-event survey matches | `data/test_pre_event_surveys.csv` |
| Raw test data | `data/test_data/no_veg/*.las` |
| Subsampled | `data/test_subsampled/*.laz` |
| With normals | `data/test_subsampled_normals/*.laz` |
| With features | `data/test_features/*.laz` |
| Polygon features | `data/polygon_features.csv` |
| Training data | `data/training_data.csv` |
| Trained Models | `models/` |
| Ablation results | `output/ablation/ablation_results.csv`, `output/ablation/ablation_curve.png` |

### Training Data Schema

Each row in `training_data.csv` represents one polygon-zone:

```
Metadata columns:
- survey_date, survey_file, location
- polygon_id, alongshore_m
- zone (lower/middle/upper), zone_idx (0/1/2)
- n_points, z_min, z_max, z_mean, z_range

Feature columns (for each base feature):
- {feature}_mean, {feature}_std, {feature}_min, {feature}_max
- {feature}_p10, {feature}_p50, {feature}_p90

Label columns:
- label (0=control, 1=case)
- event_volume, event_id, days_before_event
```

---

## LiDAR Processing Data (External Network Drive)

The v2.x ML pipeline accesses processed LiDAR data from the shared network drive.

### Base Path
```
macOS:  /Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/
Linux:  /project/group/LiDAR/LidarProcessing/LidarProcessingCliffs/
```

macOS requires the network drive mounted (VPN if off-campus). Check with `ls /Volumes/group/LiDAR/` or `ls /project/group/LiDAR/`.

### Key Paths

| Data | Path (relative to LIDAR_BASE) |
|------|------|
| Cropped point clouds (ML input) | `results/<Location>/cropped/*_cropped.las` |
| Survey metadata | `survey_lists/surveys_<Location>.csv` |
| Event labels | `results/event_lists/erosion/<Location>_events.csv` |
| Transect shapefiles (local repo) | `utiliies/transects_10m/transect_lines.shp` |

**Locations:** DelMar, SanElijo, Encinitas, Cardiff, Solana, Torrey

The `cropped/` directory is the primary ML input — raw cliff-face point clouds after spatial cropping but before beach/vegetation removal. File pattern: `YYYYMMDD_<CODE>_*_cropped.las` (codes: DM, SE, EN, CF, SB, TP).

---

## References

- Dunham et al. (2017) - Original RAI methodology
- Markus et al. (2023) - Updated parameters, 5-year inventory study
- laspy docs: https://laspy.readthedocs.io/
- Open3D docs: https://www.open3d.org/docs/
- scipy.spatial: https://docs.scipy.org/doc/scipy/reference/spatial.html
- scikit-learn: https://scikit-learn.org/stable/ (PCA, K-means, silhouette score)
