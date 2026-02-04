# Product Requirements Document: PC-RAI

## Point Cloud-Native Rockfall Activity Index Tool

> **For Claude Code**: This PRD defines a Python tool for classifying LiDAR point clouds into rockfall hazard categories. Use this document to guide implementation. Key sections for coding: [Module Structure](#51-module-structure), [Data Flow](#52-data-flow), [Decision Tree](#appendix-a-decision-tree-pseudocode), and [Functional Requirements](#3-functional-requirements).

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 2.0 |
| Date | February 2026 |
| Status | Active Development |
| Primary Language | Python 3.9+ |
| Input Format | LAS/LAZ |
| Output Format | LAS/LAZ + PNG + Markdown/JSON |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [User Stories](#2-user-stories)
3. [Functional Requirements](#3-functional-requirements)
4. [Technical Requirements](#4-technical-requirements)
5. [System Architecture](#5-system-architecture)
6. [Interface Specifications](#6-interface-specifications)
7. [Output Specifications](#7-output-specifications)
8. [Testing Requirements](#8-testing-requirements)
9. [Acceptance Criteria](#9-acceptance-criteria)
10. [Future Considerations](#10-future-considerations)
11. [Glossary](#11-glossary)
12. [References](#12-references)
13. [Appendices](#appendix-a-decision-tree-pseudocode)

---

## 1. Executive Summary

### 1.1 Purpose

Develop a Python-based tool that implements a point cloud-native adaptation of the Rockfall Activity Index (RAI) methodology for coastal cliff hazard assessment. The tool has two operational modes:

1. **Rule-Based Classification (v1.x)**: Classifies LiDAR point clouds into five morphological hazard classes based on slope angle and multi-scale surface roughness using a static decision tree.

2. **ML-Based Stability Score (v2.x)**: Uses supervised learning (Random Forest) trained on 7+ years of rockfall event labels to predict failure probability, outputting a continuous stability score at the transect level.

### 1.2 Background

The RAI methodology (Dunham et al. 2017, Markus et al. 2023) was originally developed using a grid-based approach for rock slope hazard assessment along transportation corridors. This tool adapts the core algorithm to work directly on point clouds, eliminating grid-related distortions and simplifying the processing pipeline.

**v2.0 Enhancement**: The static decision tree approach uses expert-defined thresholds that may not generalize across diverse geological settings. To address this, v2.0 introduces a supervised learning approach that learns optimal feature combinations from actual rockfall event data, enabling:

- Data-driven threshold optimization
- Statewide scalability without site-specific tuning
- Continuous probability outputs instead of discrete classes
- Direct validation against historical events

### 1.3 Scope

**In Scope (v1.x - Rule-Based):**

- Normal vector computation via CloudCompare CLI
- Slope angle calculation from normals
- Multi-scale roughness calculation (both radius and k-NN methods)
- Five-class morphological classification
- Classification smoothing via majority voting
- RAI kinetic energy scoring (from Dunham et al. 2017)
- LAS output with classification attributes
- Static visualization products
- Batch processing capability

**In Scope (v2.x - ML-Based):**

- Supervised learning pipeline using Random Forest
- Training on historical rockfall event labels (polygon-based, 1m resolution)
- Transect-level feature aggregation (10m transects)
- Continuous stability score prediction (0-1 probability)
- Cross-validation framework (leave-one-beach-out, leave-one-year-out)
- Feature importance analysis
- Model persistence and deployment
- Comparison metrics with rule-based approach

**Out of Scope (v1.x/v2.x):**

- Change detection integration
- Temporal analysis between epochs
- Interactive visualization/GUI
- Real-time processing
- Web interface

**Future Scope (v3.x - Transformer Model):**

- Integration of environmental forcing data (waves, rainfall)
- Temporal sequence modeling
- Transformer-based architecture for multi-modal prediction

### 1.4 Core Algorithm Summary

#### v1.x Rule-Based Pipeline

```
INPUT:  Point cloud (XYZ)
    ↓
STEP 1: Compute normal vectors (via CloudCompare MST)
    ↓
STEP 2: Calculate slope angle per point: slope = arccos(Nz)
    ↓
STEP 3: Calculate roughness at two scales:
        R_small = std_dev(slope) within small neighborhood
        R_large = std_dev(slope) within large neighborhood
    ↓
STEP 4: Apply decision tree classification → 5 RAI classes
    ↓
STEP 5: Smooth classification via majority voting (reduces noise)
    ↓
STEP 6: Calculate per-point energy contribution (optional)
    ↓
OUTPUT: Classified point cloud + visualizations + report
```

#### v2.x ML-Based Pipeline (Temporal Alignment with 1m Polygons)

```
TRAINING PHASE (Case-Control Study Design):
    INPUT:  Point clouds + Event CSV (alongshore positions) + 1m polygon shapefiles
        ↓
    STEP 1: Filter events (>5m³, exclude construction/noise)
        ↓
    STEP 2: For each event, find pre-event scan (most recent scan before event)
        ↓
    STEP 3: Match event alongshore extent to polygon IDs (polygon_id = alongshore_m)
        ↓
    STEP 4: Extract features from pre-event scan at event polygon locations (CASES)
            - Load point cloud once per scan
            - Vectorized point-in-polygon tests
            - Aggregate: slope, r_small, r_large, r_ratio, height (mean, max, p90, std)
        ↓
    STEP 5: Sample control polygons from same scans (CONTROLS)
            - Polygons that did NOT have events in the lookforward window
        ↓
    STEP 6: Train Random Forest with balanced class weights
        ↓
    STEP 7: Validate (leave-one-beach-out + leave-one-year-out)
        ↓
    OUTPUT: Trained model + feature importances + AUC-ROC/AUC-PR metrics

INFERENCE PHASE:
    INPUT:  New point cloud + 1m polygon definitions
        ↓
    STEP 1: Load point cloud, extract all polygon features in single pass
        ↓
    STEP 2: Apply trained RF model to each polygon
        ↓
    OUTPUT: Stability score per polygon (0-1) + alongshore risk profile
```

---

## 2. User Stories

### 2.1 Primary User

Coastal geomorphology researcher processing multi-temporal LiDAR surveys of cliff faces to identify and monitor erosion-prone areas.

### 2.2 Stories

| ID | Story | Priority |
|----|-------|----------|
| US-1 | As a researcher, I want to classify a cliff point cloud into morphological hazard zones so that I can identify areas with different erosion susceptibilities. | Must Have |
| US-2 | As a researcher, I want to compare radius-based and k-NN roughness methods so that I can determine which better captures morphological variation at my sites. | Must Have |
| US-3 | As a researcher, I want to batch process multiple epochs of LiDAR data so that I can analyze temporal changes in morphological classification. | Should Have |
| US-4 | As a researcher, I want publication-ready visualizations so that I can include results in manuscripts and presentations. | Must Have |
| US-5 | As a researcher, I want the classification attributes saved to the point cloud so that I can perform additional analysis in other software. | Must Have |
| US-6 | As a researcher, I want to train a predictive model on historical rockfall events so that I can predict failure probability on new sites. | Must Have (v2.x) |
| US-7 | As a researcher, I want to validate model performance using leave-one-beach-out cross-validation so that I can assess spatial generalization. | Must Have (v2.x) |
| US-8 | As a researcher, I want to compare RF model predictions against the rule-based approach so that I can quantify improvement from supervised learning. | Should Have (v2.x) |
| US-9 | As a researcher, I want feature importance rankings so that I can understand which morphological features best predict rockfall. | Should Have (v2.x) |
| US-10 | As a researcher, I want the model to scale to statewide data without site-specific tuning so that I can apply it to unseen beaches. | Must Have (v2.x) |

---

## 3. Functional Requirements

### 3.1 Input Handling

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Accept single LAS/LAZ file as input | Must Have |
| FR-1.2 | Accept directory of LAS/LAZ files for batch processing | Must Have |
| FR-1.3 | Validate input files exist and are readable | Must Have |
| FR-1.4 | Report point count and spatial extent on load | Should Have |
| FR-1.5 | Handle point clouds with or without existing normal vectors | Must Have |
| FR-1.6 | Support compressed LAZ format | Should Have |

### 3.2 Normal Vector Computation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Integrate with CloudCompare CLI for normal computation | Must Have |
| FR-2.2 | Use Minimum Spanning Tree orientation method | Must Have |
| FR-2.3 | Allow configuration of normal estimation radius/neighbor count | Should Have |
| FR-2.4 | Verify consistent outward-facing normal orientation | Must Have |
| FR-2.5 | Skip normal computation if valid normals already exist | Should Have |
| FR-2.6 | Report warning if normal computation fails for subset of points | Must Have |

### 3.3 Slope Calculation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Calculate slope angle from normal vector for each point | Must Have |
| FR-3.2 | Support configurable "up" direction (default: +Z) | Should Have |
| FR-3.3 | Output slope in degrees (0-180° range) | Must Have |
| FR-3.4 | Correctly identify overhangs as slope > 90° | Must Have |

**Implementation Note:**

```python
# Slope calculation from normal vector
# Assuming Z-up convention and normalized normal vectors
slope_radians = np.arccos(normals[:, 2])  # Nz component
slope_degrees = np.degrees(slope_radians)
# Result: 0° = horizontal, 90° = vertical, >90° = overhang
```

### 3.4 Roughness Calculation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Calculate small-scale roughness (R_small) as std dev of neighbor slopes | Must Have |
| FR-4.2 | Calculate large-scale roughness (R_large) as std dev of neighbor slopes | Must Have |
| FR-4.3 | Implement radius-based neighbor search method | Must Have |
| FR-4.4 | Implement k-NN neighbor search method | Must Have |
| FR-4.5 | Default radius values: 1.0m (small), 2.5m (large) | Must Have |
| FR-4.6 | Default k-NN values: k=40 (small), k=120 (large) | Must Have |
| FR-4.7 | Allow user override of all radius/k-NN parameters | Should Have |
| FR-4.8 | Flag points with insufficient neighbors (< minimum threshold) | Must Have |
| FR-4.9 | Store roughness values from both methods when both are computed | Must Have |

**Implementation Note:**

```python
# Roughness calculation pseudocode
def calculate_roughness(points, slopes, kdtree, radius):
    roughness = np.zeros(len(points))
    for i, point in enumerate(points):
        neighbor_indices = kdtree.query_ball_point(point, radius)
        if len(neighbor_indices) >= MIN_NEIGHBORS:
            roughness[i] = np.std(slopes[neighbor_indices])
        else:
            roughness[i] = np.nan  # Flag insufficient neighbors
    return roughness
```

### 3.5 Classification

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Implement five-class RAI decision tree | Must Have |
| FR-5.2 | Use updated thresholds from Markus et al. 2023 (42° talus threshold) | Must Have |
| FR-5.3 | Allow user override of classification thresholds via config file | Should Have |
| FR-5.4 | Assign classification independently using radius and k-NN roughness | Must Have |
| FR-5.5 | Report classification distribution statistics | Should Have |
| FR-5.6 | Handle edge cases (insufficient neighbors) with "Unclassified" label | Must Have |
| FR-5.7 | Apply majority voting smoothing to reduce classification noise | Must Have |
| FR-5.8 | Allow configuration of smoothing neighborhood size (default k=25) | Should Have |

**Classification Smoothing:**

After initial classification, a majority voting filter is applied to reduce "salt-and-pepper" noise:

```python
def smooth_classification(classes, xyz, k=25):
    """Smooth classification using k-NN majority voting.

    For each point, find k nearest neighbors and assign the
    most common class among valid (non-unclassified) neighbors.
    """
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k)
    for i in range(n_points):
        neighbor_classes = classes[indices[i]]
        valid = neighbor_classes[neighbor_classes > 0]
        if len(valid) > 0:
            smoothed[i] = mode(valid)
    return smoothed
```

**Classification Scheme (Simplified 5-class):**

| Code | Class Name | Abbreviation | Description |
|------|------------|--------------|-------------|
| 0 | Unclassified | U | Insufficient data for classification |
| 1 | Talus | T | Debris accumulation, low slope (<42°), smooth |
| 2 | Intact | I | Smooth rock face, few discontinuities |
| 3 | Discontinuous | D | Potential rockfall source (merged Df, Dc, Dw) |
| 4 | Steep/Overhang | O | High risk steep faces (slope >80°) |
| 5 | Structure | St | Seawalls, riprap, engineered surfaces |

**Classification Thresholds (adapted from Markus et al. 2023, tuned for California coastal bluffs):**

| Parameter | Threshold | Units | Notes |
|-----------|-----------|-------|-------|
| Steep/Overhang detection | 80 | degrees | Lowered from 90° for coastal bluffs |
| Talus maximum slope | 42 | degrees | From Markus et al. 2023 |
| R_small low threshold | 6 | degrees | Below this = smooth surface |
| R_small mid threshold | 15 | degrees | Raised from 11° to keep more points as Intact |
| R_large threshold | 15 | degrees | Raised from 12° to keep more points as Intact |
| Structure roughness (max) | 2 | degrees | Below this on steep faces = engineered |

### 3.6 Output - LAS File

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | Output LAS 1.4 format file | Must Have |
| FR-6.2 | Preserve all original point attributes | Must Have |
| FR-6.3 | Add extra dimension: `slope_deg` (float32) | Must Have |
| FR-6.4 | Add extra dimension: `roughness_small_radius` (float32) | Must Have |
| FR-6.5 | Add extra dimension: `roughness_large_radius` (float32) | Must Have |
| FR-6.6 | Add extra dimension: `roughness_small_knn` (float32) | Must Have |
| FR-6.7 | Add extra dimension: `roughness_large_knn` (float32) | Must Have |
| FR-6.8 | Add extra dimension: `rai_class_radius` (uint8) | Must Have |
| FR-6.9 | Add extra dimension: `rai_class_knn` (uint8) | Must Have |
| FR-6.10 | Add extra dimension: `neighbor_count_small` (uint16) | Should Have |
| FR-6.11 | Add extra dimension: `neighbor_count_large` (uint16) | Should Have |
| FR-6.12 | Support LAZ compression for output | Should Have |

### 3.7 Output - Visualization

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-7.1 | Generate 3D visualization of classified point cloud | Must Have |
| FR-7.2 | Use consistent color scheme across all outputs | Must Have |
| FR-7.3 | Output PNG images at configurable resolution (default: 300 DPI) | Must Have |
| FR-7.4 | Generate visualization for radius-based classification | Must Have |
| FR-7.5 | Generate visualization for k-NN-based classification | Must Have |
| FR-7.6 | Generate comparison visualization (side-by-side) | Should Have |
| FR-7.7 | Include legend with class names and colors | Must Have |
| FR-7.8 | Include scale bar | Should Have |
| FR-7.9 | Support multiple view angles (front, oblique) | Should Have |
| FR-7.10 | Generate slope angle visualization (continuous colormap) | Should Have |
| FR-7.11 | Generate roughness visualizations (continuous colormap) | Should Have |

**Color Scheme:**

```python
RAI_COLORS = {
    0: "#9E9E9E",  # Unclassified - Gray
    1: "#C8A2C8",  # Talus - Light Purple
    2: "#4CAF50",  # Intact - Green
    3: "#2196F3",  # Discontinuous - Blue
    4: "#FF9800",  # Steep/Overhang - Orange
    5: "#795548",  # Structure - Brown
}

RAI_NAMES = {
    0: "Unclassified",
    1: "Talus",
    2: "Intact",
    3: "Discontinuous",
    4: "Steep/Overhang",
    5: "Structure",
}
```

### 3.8 Reporting

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-8.1 | Generate summary statistics report (Markdown) | Must Have |
| FR-8.2 | Report point counts per class | Must Have |
| FR-8.3 | Report percentage of total per class | Must Have |
| FR-8.4 | Report mean/std/min/max for slope and roughness | Should Have |
| FR-8.5 | Report comparison metrics between radius and k-NN | Should Have |
| FR-8.6 | Report processing time per stage | Should Have |
| FR-8.7 | Output report as JSON for programmatic access | Should Have |

### 3.9 Energy Calculation (Optional)

Implements RAI kinetic energy scoring from Dunham et al. (2017). Each point receives an annual energy contribution based on its class, height above base, and class-specific failure parameters.

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-9.1 | Calculate per-point energy contribution (kJ) | Should Have |
| FR-9.2 | Use class-specific failure depths from Dunham et al. (2017) Table 1 | Should Have |
| FR-9.3 | Use class-specific instability rates from Dunham et al. (2017) Table 1 | Should Have |
| FR-9.4 | Support configurable base elevation (default: min Z) | Should Have |
| FR-9.5 | Report total energy and per-class energy contributions | Should Have |

**Energy Calculation Formula:**

```
E = ρ × A × d × g × h × r

Where:
  ρ = rock density (2600 kg/m³)
  A = cell area (0.0025 m² for 5cm grid)
  d = failure depth (m, class-specific)
  g = gravity (9.8 m/s²)
  h = height above base (m)
  r = instability rate (fraction/year, class-specific)
```

**Default Parameters (adapted from Dunham et al. 2017, simplified 5-class):**

| Class | Failure Depth (m) | Instability Rate | Notes |
|-------|-------------------|------------------|-------|
| 0 - Unclassified | 0.0 | 0% | No contribution |
| 1 - Talus | 0.025 | 0% | Stable, no rockfall source |
| 2 - Intact | 0.05 | 0.1% | Minimal rockfall potential |
| 3 - Discontinuous | 0.2 | 0.4% | Average of Df, Dc, Dw |
| 4 - Steep/Overhang | 0.625 | 2% | Average of Os, Oc |
| 5 - Structure | 0.0 | 0% | Engineered, no natural rockfall |

### 3.10 ML-Based Stability Score (v2.x)

Implements a supervised learning approach using Random Forest to predict rockfall probability from point cloud morphology features. Trained on 7+ years of historical rockfall event labels across 5 beaches.

**Key Design: Temporal Alignment with 1m Polygons**

The v2.x pipeline uses a **case-control study design** with **1m polygon shapefiles** for precise spatial matching:

- **Cases**: Pre-failure morphology features (from scans taken BEFORE events)
- **Controls**: Features from polygons that did NOT have subsequent events
- **Spatial Resolution**: 1m polygons where **polygon ID = alongshore meter position**

This ensures we're training on **predictive** features (pre-failure state) rather than descriptive features (post-failure state).

#### 3.10.1 Training Data Preparation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10.1 | Load event labels from CSV files with alongshore positions | Must Have |
| FR-10.2 | Match events to 1m polygons using alongshore_start_m / alongshore_end_m | Must Have |
| FR-10.3 | Find pre-event scan (most recent scan at least N days before event) | Must Have |
| FR-10.4 | Extract features from pre-event scan at event polygon locations (cases) | Must Have |
| FR-10.5 | Sample control polygons from same scans (no subsequent events) | Must Have |
| FR-10.6 | Track temporal metadata (scan date, event date, days_to_event, beach ID) | Must Have |

#### 3.10.2 Feature Engineering

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10.7 | Compute point-level features: slope, r_small, r_large, height | Must Have |
| FR-10.8 | Aggregate features to 1m polygon level using multiple statistics | Must Have |
| FR-10.9 | Support configurable aggregation functions (mean, max, std, percentiles) | Should Have |
| FR-10.10 | Compute derived features: r_ratio = r_small / r_large | Should Have |
| FR-10.11 | Handle missing values (insufficient neighbors) gracefully | Must Have |
| FR-10.12 | Use vectorized point-in-polygon tests for efficient feature extraction | Must Have |

**1m Polygon Feature Table:**

| Feature | Aggregations | Description |
|---------|--------------|-------------|
| `slope` | mean, max, p90, std | Slope angle statistics within polygon |
| `r_small` | mean, max, p90, std | Small-scale roughness |
| `r_large` | mean, max, p90, std | Large-scale roughness |
| `r_ratio` | mean, max, p90, std | Scale-invariant texture (r_small / r_large) |
| `height` | mean, max, p90, std | Elevation statistics (Z coordinate) |
| `point_count` | - | Number of points in polygon |

#### 3.10.3 Model Training

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10.13 | Train Random Forest classifier/regressor on transect features | Must Have |
| FR-10.14 | Apply class weighting to handle severe imbalance (most transects have no events) | Must Have |
| FR-10.15 | Support hyperparameter tuning (n_estimators, max_depth, min_samples_leaf) | Should Have |
| FR-10.16 | Output feature importance rankings | Must Have |
| FR-10.17 | Persist trained model to disk (joblib/pickle) | Must Have |
| FR-10.18 | Log training metadata (features used, hyperparameters, training size) | Should Have |

#### 3.10.4 Validation Strategy

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10.19 | Implement leave-one-beach-out cross-validation | Must Have |
| FR-10.20 | Implement leave-one-year-out temporal validation | Must Have |
| FR-10.21 | Report AUC-ROC for overall discrimination | Must Have |
| FR-10.22 | Report AUC-PR for imbalanced data evaluation | Must Have |
| FR-10.23 | Report recall at fixed precision thresholds | Should Have |
| FR-10.24 | Generate confusion matrices per fold | Should Have |
| FR-10.25 | Compare performance against rule-based RAI classification | Should Have |

**Validation Framework:**

```
Spatial Generalization (Leave-One-Beach-Out):
    Fold 1: Train on beaches 2,3,4,5 → Test on beach 1
    Fold 2: Train on beaches 1,3,4,5 → Test on beach 2
    ...
    Fold 5: Train on beaches 1,2,3,4 → Test on beach 5

Temporal Generalization (Leave-One-Year-Out):
    Train on years 1-6 → Test on year 7
    (or k-fold temporal CV)
```

#### 3.10.5 Inference

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10.26 | Load trained model from disk | Must Have |
| FR-10.27 | Apply model to new point cloud + transect definitions | Must Have |
| FR-10.28 | Output continuous stability score (0-1) per transect | Must Have |
| FR-10.29 | Support optional thresholding to discrete risk categories | Should Have |
| FR-10.30 | Generate transect-level risk maps compatible with existing visualization | Must Have |

---

## 4. Technical Requirements

### 4.1 Performance Targets

| Metric | Target |
|--------|--------|
| 10M point cloud processing time | < 10 minutes |
| Memory usage (50M points) | < 16 GB RAM |
| KD-tree construction (10M points) | < 30 seconds |
| Neighbor queries (10M points) | < 5 minutes |

### 4.2 Dependencies

**Required Python Packages (v1.x):**

```
python>=3.9
numpy>=1.21
scipy>=1.7
laspy>=2.0
matplotlib>=3.5
open3d>=0.15
tqdm>=4.62
pyyaml>=6.0
```

**Additional Required Packages (v2.x ML):**

```
scikit-learn>=1.0   # Random Forest, metrics, cross-validation
pandas>=1.4         # DataFrame operations for feature tables
geopandas>=0.10     # Shapefile handling for event polygons
shapely>=1.8        # Geometric operations (transect-polygon intersection)
joblib>=1.1         # Model persistence
```

**External Software:**

```
CloudCompare>=2.12  # For normal computation via CLI
```

**Optional (for enhanced functionality):**

```
lazrs>=0.5          # LAZ compression
numba>=0.55         # JIT compilation for speedup
pyshp>=2.3          # Shapefile reading (alternative to geopandas)
```

### 4.3 Platform Support

| Platform | Support Level |
|----------|---------------|
| Linux (Ubuntu 20.04+) | Primary |
| macOS (11+) | Secondary |
| Windows 10/11 | Secondary |

### 4.4 Code Quality Standards

- Type hints for all public functions
- NumPy-style docstrings
- Unit tests with pytest (≥80% coverage for core modules)
- Code formatting with black
- Linting with ruff

---

## 5. System Architecture

### 5.1 Module Structure

```
pc_rai/
├── __init__.py
├── __main__.py             # Entry point for `python -m pc_rai`
├── cli.py                  # Command-line interface (argparse)
├── config.py               # Configuration dataclasses and defaults
│
├── io/
│   ├── __init__.py
│   ├── las_reader.py       # LAS/LAZ input using laspy
│   ├── las_writer.py       # LAS/LAZ output with extra dimensions
│   └── validators.py       # Input validation utilities
│
├── normals/
│   ├── __init__.py
│   ├── cloudcompare.py     # CloudCompare CLI integration
│   └── orientation.py      # Normal orientation verification
│
├── features/
│   ├── __init__.py
│   ├── slope.py            # Slope calculation from normals
│   ├── roughness.py        # Roughness (radius + k-NN methods)
│   └── aggregation.py      # Transect-level feature aggregation (v2.x)
│
├── classification/
│   ├── __init__.py
│   ├── decision_tree.py    # RAI classification logic (v1.x)
│   ├── thresholds.py       # Configurable threshold parameters
│   └── energy.py           # Rockfall energy calculation
│
├── ml/                     # Machine learning modules (v2.x)
│   ├── __init__.py
│   ├── config.py           # MLConfig dataclass with hyperparameters
│   ├── data_prep.py        # Load and filter event CSVs
│   ├── polygons.py         # 1m polygon spatial matching (polygon ID = alongshore_m)
│   ├── temporal.py         # Temporal alignment for case-control training
│   ├── train.py            # Model training (Random Forest) + cross-validation
│   ├── predict.py          # Inference on new data
│   │
│   └── (legacy - 10m transect-based)
│       ├── labels.py       # Event → transect label mapping
│       └── features.py     # Transect-level feature extraction
│
├── visualization/
│   ├── __init__.py
│   ├── colors.py           # Color schemes and colormaps
│   ├── render_3d.py        # 3D point cloud rendering (Open3D)
│   ├── figures.py          # Multi-panel figure generation
│   └── risk_map.py         # Transect-level risk map visualization
│
├── reporting/
│   ├── __init__.py
│   ├── statistics.py       # Summary statistics computation
│   └── report_writer.py    # Markdown and JSON output
│
└── utils/
    ├── __init__.py
    ├── spatial.py          # KD-tree construction, neighbor queries
    ├── timing.py           # Performance timing utilities
    └── logging.py          # Logging configuration

scripts/                    # Standalone scripts
├── prepare_delmar_training_temporal.py  # Generate temporally-aligned training data for Del Mar
├── compute_normals_mst.py               # CloudComPy normal computation
├── risk_map_regional.py                 # Generate regional risk maps
└── predict_stability.py                 # Apply trained model to new data
```

### 5.1.1 Performance Note: Polygon Feature Extraction

Extracting features for all 2285 polygons (1m resolution over ~2.3km coastline) from large point clouds is computationally expensive. The current implementation:

1. Loads each scan once (not per-polygon)
2. Uses vectorized point-in-polygon tests (matplotlib.path)
3. Caches features per (scan_date, polygon_id)

**Planned Optimization**: Only extract features for polygons that are actually needed:
- Event polygons (cases)
- Sampled control polygons (not all 2000+ polygons per scan)

This should reduce processing time from hours to minutes.

### 5.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│                     LAS/LAZ File(s)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. LOAD & VALIDATE                           │
│  Module: io/las_reader.py                                       │
│  • Read LAS file with laspy                                     │
│  • Extract XYZ coordinates as numpy array                       │
│  • Check for existing normal vectors (Nx, Ny, Nz)               │
│  • Report point count, spatial extent                           │
│  Output: PointCloud object with xyz, optional normals           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. NORMAL COMPUTATION                           │
│  Module: normals/cloudcompare.py                                │
│  • Skip if valid normals already exist                          │
│  • Export to temporary PLY file                                 │
│  • Call CloudCompare CLI:                                       │
│    CloudCompare -SILENT -O temp.ply \                           │
│      -OCTREE_NORMALS <radius> \                                 │
│      -ORIENT_NORMS_MST <k> \                                    │
│      -SAVE_CLOUDS FILE temp_normals.ply                         │
│  • Import computed normals                                      │
│  • Verify outward orientation (majority Nz > 0 or consistent)   │
│  Output: normals array (N, 3)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  3. SLOPE CALCULATION                           │
│  Module: features/slope.py                                      │
│  • slope = arccos(Nz) for each point                            │
│  • Convert to degrees                                           │
│  • Range: 0° (horizontal) to 180° (inverted)                    │
│  Output: slope array (N,)                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 4. BUILD SPATIAL INDEX                          │
│  Module: utils/spatial.py                                       │
│  • Construct scipy.spatial.cKDTree on XYZ                       │
│  Output: KDTree object                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│  5a. ROUGHNESS (RADIUS)     │ │   5b. ROUGHNESS (K-NN)      │
│  Module: features/          │ │   Module: features/         │
│          roughness.py       │ │           roughness.py      │
│                             │ │                             │
│  For each point:            │ │   For each point:           │
│  • query_ball_point(r)      │ │   • query(k) neighbors      │
│  • R = std(slope[neighbors])│ │   • R = std(slope[neighbors])│
│                             │ │                             │
│  Scales:                    │ │   Scales:                   │
│  • r_small = 0.175m         │ │   • k_small = 30            │
│  • r_large = 0.425m         │ │   • k_large = 100           │
│                             │ │                             │
│  Output:                    │ │   Output:                   │
│  • roughness_small_radius   │ │   • roughness_small_knn     │
│  • roughness_large_radius   │ │   • roughness_large_knn     │
│  • neighbor_count_small     │ │   • neighbor_count_small    │
│  • neighbor_count_large     │ │   • neighbor_count_large    │
└─────────────────────────────┘ └─────────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ 6a. CLASSIFY (RADIUS)       │ │  6b. CLASSIFY (K-NN)        │
│ Module: classification/     │ │  Module: classification/    │
│         decision_tree.py    │ │          decision_tree.py   │
│                             │ │                             │
│ Apply decision tree:        │ │  Apply decision tree:       │
│ • Input: slope, R_small,    │ │  • Input: slope, R_small,   │
│          R_large            │ │           R_large           │
│ • Output: class code (0-5)  │ │  • Output: class code (0-5) │
│                             │ │                             │
│ Output: rai_class_radius    │ │  Output: rai_class_knn      │
└─────────────────────────────┘ └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   7. AGGREGATE RESULTS                          │
│  Module: reporting/statistics.py                                │
│  • Combine all computed attributes                              │
│  • Calculate summary statistics per class                       │
│  • Compare radius vs k-NN (agreement %, Cohen's kappa)          │
│  Output: RAIResult dataclass with all arrays and stats          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
│  8a. LAS OUTPUT   │ │ 8b. VISUALIZE │ │   8c. REPORT      │
│  Module: io/      │ │ Module: viz/  │ │ Module: reporting/│
│  las_writer.py    │ │ render_3d.py  │ │ report_writer.py  │
│                   │ │ figures.py    │ │                   │
│  • Copy original  │ │               │ │ • Markdown        │
│  • Add extra dims │ │ • 3D renders  │ │ • JSON            │
│  • Write LAS 1.4  │ │ • Legends     │ │ • Statistics      │
│                   │ │ • Multi-view  │ │ • Timing          │
└───────────────────┘ └───────────────┘ └───────────────────┘
```

### 5.3 Key Data Structures

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PointCloud:
    """Container for point cloud data."""
    xyz: np.ndarray              # (N, 3) float64
    normals: Optional[np.ndarray] = None  # (N, 3) float32
    colors: Optional[np.ndarray] = None   # (N, 3) uint8
    intensity: Optional[np.ndarray] = None  # (N,) uint16
    # ... other LAS attributes preserved
    
    @property
    def n_points(self) -> int:
        return len(self.xyz)

@dataclass
class RAIResult:
    """Container for RAI classification results."""
    # Input reference
    source_file: str
    n_points: int
    
    # Computed features
    slope_deg: np.ndarray           # (N,) float32
    
    # Radius-based roughness
    roughness_small_radius: np.ndarray  # (N,) float32
    roughness_large_radius: np.ndarray  # (N,) float32
    
    # K-NN-based roughness
    roughness_small_knn: np.ndarray     # (N,) float32
    roughness_large_knn: np.ndarray     # (N,) float32
    
    # Neighbor counts (for QC)
    neighbor_count_small: np.ndarray    # (N,) uint16
    neighbor_count_large: np.ndarray    # (N,) uint16
    
    # Classifications
    rai_class_radius: np.ndarray        # (N,) uint8
    rai_class_knn: np.ndarray           # (N,) uint8
    
    # Statistics
    statistics: dict

@dataclass
class RAIConfig:
    """Configuration parameters for RAI processing."""
    # Normal computation
    compute_normals: bool = True
    cloudcompare_path: str = "CloudCompare"
    normal_radius: float = 1.0  # 1.0m for stable plane fitting
    mst_neighbors: int = 10

    # Slope
    up_vector: tuple = (0, 0, 1)

    # Roughness - radius method (tuned for 50cm point spacing)
    radius_small: float = 1.0   # ~12 points neighborhood
    radius_large: float = 2.5   # ~80 points neighborhood

    # Roughness - knn method (preferred for uniform spacing)
    k_small: int = 40           # ~1.8m neighborhood
    k_large: int = 120          # ~3.0m neighborhood

    # Roughness - shared
    min_neighbors: int = 4
    methods: list = ("knn",)    # Default to k-NN only

    # Classification smoothing
    classification_smoothing_k: int = 25  # Neighbors for majority voting

    # Classification thresholds (tuned for California coastal bluffs)
    thresh_overhang: float = 80.0
    thresh_cantilever: float = 150.0
    thresh_talus_slope: float = 42.0
    thresh_r_small_low: float = 6.0
    thresh_r_small_mid: float = 15.0   # Raised from 11°
    thresh_r_small_high: float = 18.0
    thresh_r_large: float = 15.0       # Raised from 12°
    thresh_structure_roughness: float = 2.0  # Lowered from 4°

    # Output
    output_dir: str = "./output"
    compress_output: bool = True
    visualization_dpi: int = 300
    visualization_views: list = ("front", "oblique")


# ============================================================
# v2.x ML Data Structures
# ============================================================

@dataclass
class TransectFeatures:
    """Aggregated features for a single transect (v2.x)."""
    transect_id: int
    beach_id: str
    scan_date: datetime

    # Slope statistics
    slope_mean: float
    slope_max: float
    slope_p90: float
    slope_std: float

    # Small-scale roughness
    r_small_mean: float
    r_small_max: float
    r_small_std: float

    # Large-scale roughness
    r_large_mean: float
    r_large_max: float
    r_large_std: float

    # Derived features
    r_ratio_mean: float          # r_small / r_large

    # Height statistics
    height_mean: float
    height_max: float
    height_range: float

    # Metadata
    point_count: int
    valid_point_count: int       # Points with valid roughness


@dataclass
class TransectLabel:
    """Event label for a single transect (v2.x)."""
    transect_id: int
    beach_id: str
    time_window_start: datetime
    time_window_end: datetime

    # Label options
    event_count: int             # Number of events in window
    has_event: bool              # Binary: any event occurred
    total_event_area: float      # m² of polygon overlap

    # Temporal metadata
    days_to_next_event: Optional[int]  # For survival analysis


@dataclass
class TrainingDataset:
    """Complete training dataset for ML model (v2.x)."""
    features: pd.DataFrame       # Transect features (one row per transect-scan)
    labels: pd.DataFrame         # Corresponding labels

    # Metadata
    beaches: List[str]           # List of beach IDs
    date_range: Tuple[datetime, datetime]
    n_transects: int
    n_events: int

    # Train/test indices for cross-validation
    cv_folds: Dict[str, Dict[str, np.ndarray]]  # fold_name → {train_idx, test_idx}


@dataclass
class StabilityModel:
    """Trained stability prediction model (v2.x)."""
    model: Any                   # sklearn RandomForestClassifier/Regressor
    feature_names: List[str]     # Features used in training
    feature_importances: Dict[str, float]

    # Training metadata
    train_date: datetime
    train_beaches: List[str]
    train_date_range: Tuple[datetime, datetime]
    hyperparameters: Dict[str, Any]

    # Validation metrics
    cv_metrics: Dict[str, Dict[str, float]]  # fold → {auc_roc, auc_pr, ...}

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict stability score (0-1) for new transects."""
        return self.model.predict_proba(features[self.feature_names])[:, 1]

    def save(self, path: Path) -> None:
        """Persist model to disk."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "StabilityModel":
        """Load model from disk."""
        return joblib.load(path)


@dataclass
class MLConfig:
    """Configuration for ML training pipeline (v2.x)."""
    # Feature aggregation
    aggregation_functions: List[str] = ("mean", "max", "std", "p90")
    transect_half_width: float = 5.0  # meters (10m total corridor)

    # Label configuration
    label_type: str = "binary"        # "binary" or "count"
    time_window_days: int = 365       # Events within this window of scan

    # Model hyperparameters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_leaf: int = 5
    class_weight: str = "balanced"    # Handle imbalance
    random_state: int = 42

    # Validation
    cv_strategy: str = "leave_one_beach_out"  # or "leave_one_year_out"

    # Output
    model_output_path: str = "./models/stability_rf.joblib"
```

---

## 6. Interface Specifications

### 6.1 Command-Line Interface

**Main Commands:**

```bash
# Full processing pipeline
pc-rai process input.las -o output/

# Batch processing
pc-rai process ./data/*.las -o output/ --batch

# With custom config
pc-rai process input.las -o output/ --config config.yaml

# Skip normal computation (use existing)
pc-rai process input.las -o output/ --skip-normals

# Only one roughness method
pc-rai process input.las -o output/ --methods radius

# Generate visualizations from already-processed file
pc-rai visualize processed.las -o figures/
```

**Full CLI Specification:**

```
usage: pc-rai process [-h] -o OUTPUT [-c CONFIG] [--batch]
                      [--skip-normals] [--methods {radius,knn,both}]
                      [--no-visualize] [--no-report]
                      [-v] [-q] [--log LOG]
                      input

Process point cloud(s) with RAI classification.

positional arguments:
  input                 Input LAS/LAZ file or directory

required arguments:
  -o, --output OUTPUT   Output directory

optional arguments:
  -h, --help            Show this help message and exit
  -c, --config CONFIG   Configuration YAML file
  --batch               Process all LAS/LAZ files in directory
  --skip-normals        Skip normal computation (use existing)
  --methods {radius,knn,both}
                        Roughness method(s) to use (default: both)
  --no-visualize        Skip visualization generation
  --no-report           Skip report generation
  -v, --verbose         Verbose output
  -q, --quiet           Suppress non-error output
  --log LOG             Write log to file
```

### 6.2 Python API

```python
"""
Example usage of PC-RAI Python API.
"""

from pc_rai import RAIClassifier, load_point_cloud, save_point_cloud
from pc_rai.config import RAIConfig
from pc_rai.visualization import render_classification, render_slope

# Load point cloud
cloud = load_point_cloud("input.las")
print(f"Loaded {cloud.n_points:,} points")

# Configure classifier
config = RAIConfig(
    radius_small=0.175,
    radius_large=0.425,
    k_small=30,
    k_large=100,
    methods=["radius", "knn"],
    thresh_talus_slope=42.0,
)

# Initialize and run classifier
classifier = RAIClassifier(config)
result = classifier.process(cloud, compute_normals=True)

# Access results
print(f"Slope range: {result.slope_deg.min():.1f}° - {result.slope_deg.max():.1f}°")
print(f"Classification agreement: {result.statistics['agreement_pct']:.1f}%")

# Class distribution
for code, name in RAI_NAMES.items():
    count = (result.rai_class_radius == code).sum()
    pct = 100 * count / result.n_points
    print(f"  {name}: {count:,} ({pct:.1f}%)")

# Save classified point cloud
save_point_cloud(cloud, result, "output.las", compress=True)

# Generate visualizations
fig = render_classification(
    cloud.xyz,
    result.rai_class_radius,
    view="oblique",
    title="RAI Classification (Radius Method)",
)
fig.savefig("classification_radius.png", dpi=300)

fig = render_slope(
    cloud.xyz,
    result.slope_deg,
    view="front",
    title="Slope Angle",
)
fig.savefig("slope.png", dpi=300)
```

### 6.3 Configuration File Format

```yaml
# pc_rai_config.yaml
# Configuration file for PC-RAI processing

# Input handling
input:
  file_pattern: "*.las"  # For batch processing

# Normal vector computation
normals:
  compute: true
  cloudcompare_path: "CloudCompare"  # Or full path to executable
  method: "mst"                      # Minimum spanning tree orientation
  estimation_radius: 1.0             # Radius for local plane fitting (1m for 50cm spacing)
  mst_neighbors: 10                  # Neighbors for MST orientation

# Slope calculation
slope:
  up_vector: [0, 0, 1]  # Z-up coordinate system

# Roughness calculation
roughness:
  methods: ["knn"]        # k-NN preferred for uniform spacing
  min_neighbors: 4        # Minimum for valid calculation

  radius:
    small: 1.0   # meters (~12 points at 50cm spacing)
    large: 2.5   # meters (~80 points at 50cm spacing)

  knn:
    small: 40    # neighbor count for small-scale roughness (~1.8m)
    large: 120   # neighbor count for large-scale roughness (~3.0m)

# Classification smoothing
classification_smoothing:
  k: 25  # Neighbors for majority voting (reduces salt-and-pepper noise)

# Classification thresholds
# Tuned for California coastal bluffs (adapted from Markus et al. 2023)
classification:
  thresholds:
    overhang: 80             # degrees - slope above this is Steep/Overhang
    cantilever: 150          # degrees - threshold for cantilevered overhang
    talus_slope: 42          # degrees - max slope for talus classification
    r_small_low: 6           # degrees - below this is "smooth"
    r_small_mid: 15          # degrees - above this is Discontinuous (raised from 11°)
    r_small_high: 18         # degrees - very high roughness threshold
    r_large: 15              # degrees - large-scale roughness threshold (raised from 12°)
    structure_roughness: 2   # degrees - steep + below this = Structure (lowered from 4°)

# Output settings
output:
  las:
    format: "1.4"          # LAS version
    compress: true         # Output as LAZ
    preserve_attributes: true
  
  visualization:
    enabled: true
    dpi: 300
    views: ["front", "oblique"]
    formats: ["png"]
    
  report:
    enabled: true
    formats: ["markdown", "json"]

# Processing settings
processing:
  chunk_size: 1000000  # Points per chunk for memory management
  n_jobs: -1           # Parallel jobs (-1 = all cores)
  verbose: true
```

---

## 7. Output Specifications

### 7.1 LAS Extra Dimensions

| Dimension Name | Data Type | Units | Description |
|----------------|-----------|-------|-------------|
| `slope_deg` | float32 | degrees | Slope angle (0-180) |
| `roughness_small_radius` | float32 | degrees | Std dev of slope, small radius |
| `roughness_large_radius` | float32 | degrees | Std dev of slope, large radius |
| `roughness_small_knn` | float32 | degrees | Std dev of slope, small k-NN |
| `roughness_large_knn` | float32 | degrees | Std dev of slope, large k-NN |
| `rai_class_radius` | uint8 | code | Classification (radius method) |
| `rai_class_knn` | uint8 | code | Classification (k-NN method) |
| `neighbor_count_small` | uint16 | count | Neighbors at small scale |
| `neighbor_count_large` | uint16 | count | Neighbors at large scale |

### 7.2 Visualization Files

| Filename Pattern | Content |
|------------------|---------|
| `{basename}_classification_radius.png` | 3D view, radius-based classification |
| `{basename}_classification_knn.png` | 3D view, k-NN-based classification |
| `{basename}_slope.png` | 3D view, continuous slope colormap |
| `{basename}_roughness_small.png` | 3D view, small-scale roughness |
| `{basename}_roughness_large.png` | 3D view, large-scale roughness |
| `{basename}_comparison.png` | Side-by-side radius vs k-NN |

### 7.3 Report Structure

**Markdown Report (`{basename}_report.md`):**

```markdown
# RAI Classification Report

## Input
- **File**: example_cliff.las
- **Points**: 5,234,567
- **Extent**: X[234.5, 456.7], Y[123.4, 345.6], Z[0.0, 45.2]
- **Processing Date**: 2025-01-22 14:30:00

## Configuration
- Roughness Methods: radius, knn
- Radius (small/large): 0.175m / 0.425m
- K-NN (small/large): 30 / 100
- Talus Slope Threshold: 42°

## Classification Results (Radius Method)

| Class | Count | Percentage |
|-------|-------|------------|
| Talus (T) | 523,456 | 10.0% |
| Intact (I) | 2,093,826 | 40.0% |
| Discontinuous (D) | 1,308,642 | 25.0% |
| Steep/Overhang (O) | 785,184 | 15.0% |
| Structure (St) | 0 | 0.0% |
| Unclassified (U) | 523,459 | 10.0% |

## Classification Results (K-NN Method)

[Similar table]

## Method Comparison
- **Agreement**: 87.3% of points classified identically
- **Cohen's Kappa**: 0.84

## Feature Statistics

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| Slope (°) | 62.3 | 24.5 | 0.1 | 178.2 |
| R_small_radius (°) | 8.7 | 5.2 | 0.2 | 45.3 |
| R_large_radius (°) | 10.2 | 6.1 | 0.3 | 52.1 |

## Processing Time
- Normal Computation: 45.2s
- KD-Tree Construction: 12.3s
- Roughness Calculation: 156.4s
- Classification: 1.2s
- **Total**: 215.1s
```

**JSON Report (`{basename}_report.json`):**

```json
{
  "input": {
    "file": "example_cliff.las",
    "n_points": 5234567,
    "extent": {
      "x": [234.5, 456.7],
      "y": [123.4, 345.6],
      "z": [0.0, 45.2]
    }
  },
  "config": {
    "roughness_methods": ["radius", "knn"],
    "radius_small": 0.175,
    "radius_large": 0.425,
    "k_small": 30,
    "k_large": 100,
    "thresh_talus_slope": 42.0
  },
  "classification_radius": {
    "0": {"name": "Unclassified", "count": 523459, "percent": 10.0},
    "1": {"name": "Talus", "count": 523456, "percent": 10.0},
    "2": {"name": "Intact", "count": 2093826, "percent": 40.0},
    "3": {"name": "Discontinuous", "count": 1308642, "percent": 25.0},
    "4": {"name": "Steep/Overhang", "count": 785184, "percent": 15.0},
    "5": {"name": "Structure", "count": 0, "percent": 0.0}
  },
  "classification_knn": {},
  "comparison": {
    "agreement_pct": 87.3,
    "cohens_kappa": 0.84
  },
  "statistics": {
    "slope_deg": {"mean": 62.3, "std": 24.5, "min": 0.1, "max": 178.2}
  },
  "timing": {
    "normals_sec": 45.2,
    "kdtree_sec": 12.3,
    "roughness_sec": 156.4,
    "classification_sec": 1.2,
    "total_sec": 215.1
  }
}
```

---

## 8. Testing Requirements

### 8.1 Unit Tests

| Module | Test Cases |
|--------|------------|
| `io/las_reader` | Load LAS, load LAZ, missing file, corrupt file, extract normals |
| `io/las_writer` | Write extra dims, preserve attributes, compression |
| `features/slope` | Horizontal surface (0°), vertical (90°), overhang (>90°), inverted (180°) |
| `features/roughness` | Uniform slope (R=0), variable slope, insufficient neighbors, edge cases |
| `classification/decision_tree` | All 5 classes, boundary conditions, invalid input handling |
| `utils/spatial` | KD-tree construction, radius query, k-NN query |

### 8.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_full_pipeline_small` | Complete processing on 10K point synthetic cloud |
| `test_full_pipeline_medium` | Complete processing on 1M point test cloud |
| `test_batch_processing` | Multiple file processing |
| `test_config_override` | Custom configuration handling |
| `test_skip_normals` | Processing with pre-computed normals |
| `test_visualization_output` | All visualization files generated correctly |

### 8.3 Test Data

Create synthetic point clouds for testing:

```python
# test_data_generator.py

def create_horizontal_plane(n_points=10000):
    """Flat surface - expect Talus or Intact based on roughness."""
    xy = np.random.uniform(0, 10, (n_points, 2))
    z = np.zeros(n_points) + np.random.normal(0, 0.01, n_points)
    return np.column_stack([xy, z])

def create_vertical_cliff(n_points=10000):
    """Vertical surface - expect Intact or Discontinuous based on roughness."""
    xz = np.random.uniform(0, 10, (n_points, 2))
    y = np.zeros(n_points) + np.random.normal(0, 0.01, n_points)
    return np.column_stack([xz[:, 0], y, xz[:, 1]])

def create_overhang(n_points=10000, angle_deg=120):
    """Overhanging surface at specified angle."""
    # Implementation for tilted plane
    pass

def create_rough_surface(n_points=10000, roughness_amplitude=0.1):
    """Surface with controlled roughness for testing thresholds."""
    pass
```

---

## 9. Acceptance Criteria

### 9.1 Minimum Viable Product (MVP)

- [x] Loads LAS files (with or without existing normals)
- [x] Computes normals via CloudCompare CLI
- [x] Calculates slope angle for all points
- [x] Calculates roughness using at least one method (k-NN)
- [x] Classifies all points into 5 RAI classes + unclassified
- [x] Applies classification smoothing via majority voting
- [x] Outputs LAS file with slope, roughness, and classification
- [x] Generates at least one 3D visualization

### 9.2 Full Release (v1.0)

All MVP criteria plus:

- [x] Both roughness methods (radius and k-NN) implemented
- [x] Comparison metrics between methods
- [ ] Batch processing capability
- [x] Complete visualization suite (slope, roughness, classification, comparison)
- [x] Markdown and JSON reports
- [x] YAML configuration file support
- [x] Energy calculation module (Dunham et al. 2017)
- [ ] Full CLI functionality
- [ ] Python API documented and functional
- [ ] Unit test coverage ≥ 80% for core modules
- [ ] README with installation and usage instructions

### 9.3 ML Release (v2.0)

All v1.0 criteria plus:

- [ ] Training data preparation pipeline (event polygons → transect labels)
- [ ] Transect-level feature aggregation module
- [ ] Random Forest training with class weighting
- [ ] Leave-one-beach-out cross-validation
- [ ] Leave-one-year-out temporal validation
- [ ] Feature importance analysis and reporting
- [ ] Model persistence (save/load trained models)
- [ ] Inference pipeline for new point clouds
- [ ] Stability score output (continuous 0-1)
- [ ] Integration with existing risk map visualization
- [ ] Comparison metrics vs rule-based approach (AUC, PR curves)
- [ ] Documentation of ML pipeline usage

### 9.4 Success Metrics (v2.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-ROC (leave-one-beach-out) | > 0.75 | Demonstrates spatial generalization |
| AUC-PR (imbalanced) | > 0.30 | Meaningful precision-recall tradeoff |
| Improvement over rule-based | > 10% AUC | Justifies ML complexity |
| Inference time (1000 transects) | < 1 second | Practical for statewide deployment |

---

## 10. Future Considerations

### 10.1 Version 2.1 (Near-term)

- Hyperparameter optimization via grid search / Bayesian optimization
- Ensemble methods (XGBoost, LightGBM) comparison
- Uncertainty quantification (prediction intervals)
- Active learning for efficient label acquisition
- Integration with M3C2 change detection outputs

### 10.2 Version 3.0 - Transformer Model (Medium-term)

The Random Forest model (v2.x) serves as the **baseline** for a more sophisticated transformer-based model that integrates environmental forcing data.

**Rationale**: The RF model captures what is predictable from point cloud morphology alone. A transformer model can potentially capture additional predictive signal from:

- Temporal wave forcing (significant wave height, wave period)
- Rainfall and groundwater saturation
- Seasonal patterns and antecedent conditions

**Planned Architecture**:

```
Transformer Input (per transect, per timestep):

Static Features (from RF pipeline):
    ├── slope_mean, slope_max, slope_p90, slope_std
    ├── r_small_mean, r_small_max, r_small_std
    ├── r_large_mean, r_large_max, r_large_std
    ├── r_ratio_mean
    ├── height_mean, height_max, height_range
    └── point_count

Temporal Features (new):
    ├── wave_height[t-7d : t]      # 7-day wave history
    ├── wave_period[t-7d : t]
    ├── rainfall[t-7d : t]
    ├── cumulative_rainfall[season]
    └── days_since_last_event

Transformer Architecture:
    ├── Static feature embedding
    ├── Temporal encoder (attention over wave/rain history)
    ├── Cross-attention (static × temporal)
    └── Classification head → P(failure in next T days)
```

**Scientific Questions Addressed**:

| Question | How Answered |
|----------|--------------|
| How much does wave forcing add beyond morphology? | Compare RF AUC vs Transformer AUC |
| Which temporal patterns trigger failures? | Attention weight analysis |
| How far in advance can failures be predicted? | Vary prediction horizon T |
| Do different beaches have different triggering mechanisms? | Per-beach attention patterns |

**Validation Continuity**: Same leave-one-beach-out + leave-one-year-out framework as RF, enabling direct comparison.

### 10.3 Version 4.0 (Long-term)

- Real-time monitoring integration
- GUI application for practitioners
- API deployment for statewide hazard assessment
- Integration with early warning systems
- CloudCompare plugin for interactive analysis

---

## 11. Glossary

| Term | Definition |
|------|------------|
| RAI | Rockfall Activity Index |
| TLS | Terrestrial Laser Scanning |
| LAS/LAZ | LiDAR data formats (LAZ = compressed) |
| MST | Minimum Spanning Tree (normal orientation method) |
| k-NN | k-Nearest Neighbors algorithm |
| KD-tree | K-dimensional tree for spatial indexing |
| RF | Random Forest (ensemble learning algorithm) |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve |
| AUC-PR | Area Under the Precision-Recall Curve |
| Transect | 10m-wide corridor perpendicular to coastline for spatial aggregation |
| Stability Score | Continuous 0-1 probability of rockfall (1 = high risk) |
| Leave-One-Out CV | Cross-validation where one group (beach/year) is held out for testing |
| Feature Importance | Measure of how much each input variable contributes to predictions |
| Class Imbalance | When one class (non-failure) vastly outnumbers another (failure) |

---

## 12. References

1. Dunham, L., Wartman, J., Olsen, M.J., O'Banion, M., Cunningham, K. (2017). Rockfall Activity Index (RAI): A lidar-derived, morphology-based method for hazard assessment. *Engineering Geology*, 221, 184-192.

2. Markus, S.J., Wartman, J., Olsen, M., Darrow, M.M. (2023). Lidar-Derived Rockfall Inventory—An Analysis of the Geomorphic Evolution of Rock Slopes and Modifying the Rockfall Activity Index (RAI). *Remote Sensing*, 15, 4223.

---

## Appendix A: Decision Tree Pseudocode

```python
def classify_point(
    slope: float,
    r_small: float,
    r_large: float,
    thresholds: dict
) -> int:
    """
    Classify a single point using the simplified 5-class RAI decision tree.
    
    Parameters
    ----------
    slope : float
        Slope angle in degrees (0-180)
    r_small : float
        Small-scale roughness in degrees
    r_large : float
        Large-scale roughness in degrees
    thresholds : dict
        Classification thresholds
        
    Returns
    -------
    int
        RAI class code (0-5)
    """
    # Handle invalid roughness (insufficient neighbors)
    if np.isnan(r_small) or np.isnan(r_large):
        return 0  # Unclassified
    
    # Level 1: Steep faces (slope > 80°)
    if slope > thresholds['overhang']:  # 80°
        if r_small < thresholds['structure_roughness']:  # 2°
            return 5  # Structure (seawall/engineered)
        else:
            return 4  # Steep/Overhang

    # Check for smooth surfaces (low small-scale roughness)
    if r_small < thresholds['r_small_low']:  # 6°
        if slope < thresholds['talus_slope']:  # 42°
            return 1  # Talus (T)
        else:
            return 2  # Intact (I)

    # Higher roughness: check for Discontinuous
    if r_small > thresholds['r_small_mid']:  # 15°
        return 3  # Discontinuous
    if r_large > thresholds['r_large']:  # 15°
        return 3  # Discontinuous

    # Moderate roughness (6-15° r_small), low r_large (≤15°)
    return 2  # Intact (I)


def classify_point_cloud(
    slopes: np.ndarray,
    r_small: np.ndarray,
    r_large: np.ndarray,
    thresholds: dict
) -> np.ndarray:
    """
    Vectorized classification for entire point cloud.
    
    Returns array of class codes (0-5) for each point.
    """
    n_points = len(slopes)
    classes = np.zeros(n_points, dtype=np.uint8)
    
    # Unclassified: invalid roughness
    invalid = np.isnan(r_small) | np.isnan(r_large)
    
    # Level 1: Steep faces (slope > 80°)
    steep = slopes > thresholds['overhang']
    structure = steep & (r_small < thresholds['structure_roughness'])
    classes[structure & ~invalid] = 5  # Structure
    classes[steep & ~structure & ~invalid] = 4  # Steep/Overhang
    
    # Level 2: Non-steep terrain
    non_steep = ~steep & ~invalid
    
    # Smooth surfaces (low r_small)
    smooth = non_steep & (r_small < thresholds['r_small_low'])
    classes[smooth & (slopes < thresholds['talus_slope'])] = 1  # Talus
    classes[smooth & (slopes >= thresholds['talus_slope'])] = 2  # Intact
    
    # Higher roughness
    rough = non_steep & ~smooth

    # Discontinuous: high r_small OR high r_large
    discontinuous = rough & (
        (r_small > thresholds['r_small_mid']) |
        (r_large > thresholds['r_large'])
    )
    classes[discontinuous] = 3  # Discontinuous

    # Moderate roughness with low r_large → Intact
    classes[rough & ~discontinuous] = 2  # Intact
    
    return classes
```

---

## Appendix B: CloudCompare CLI Reference

```bash
# Compute normals with MST orientation
CloudCompare -SILENT \
    -O input.las \
    -OCTREE_NORMALS 0.1 \
    -ORIENT_NORMS_MST 10 \
    -C_EXPORT_FMT LAS \
    -SAVE_CLOUDS FILE output_with_normals.las

# Parameters explained:
# -SILENT              : No GUI, command-line only
# -O input.las         : Open input file
# -OCTREE_NORMALS 0.1  : Compute normals with 0.1m local radius
# -ORIENT_NORMS_MST 10 : Orient using MST with 10 neighbors
# -C_EXPORT_FMT LAS    : Set export format to LAS
# -SAVE_CLOUDS FILE x  : Save to specified file

# Alternative: use neighbor count instead of radius
CloudCompare -SILENT \
    -O input.las \
    -OCTREE_NORMALS -KNEAREST 10 \
    -ORIENT_NORMS_MST 10 \
    -C_EXPORT_FMT LAS \
    -SAVE_CLOUDS FILE output_with_normals.las
```

---

## Appendix C: Quick Start Implementation Guide

**For Claude Code: Follow this sequence to implement the tool.**

### Step 1: Project Setup

```bash
mkdir pc_rai
cd pc_rai
python -m venv venv
source venv/bin/activate
pip install numpy scipy laspy matplotlib open3d tqdm pyyaml
```

### Step 2: Implement Core Modules (Priority Order)

1. `pc_rai/config.py` - Configuration dataclass
2. `pc_rai/io/las_reader.py` - Load LAS files with laspy
3. `pc_rai/utils/spatial.py` - KD-tree wrapper
4. `pc_rai/features/slope.py` - Slope from normals
5. `pc_rai/features/roughness.py` - Both methods
6. `pc_rai/classification/decision_tree.py` - Classification logic
7. `pc_rai/io/las_writer.py` - Write output with extra dims
8. `pc_rai/normals/cloudcompare.py` - CLI integration
9. `pc_rai/visualization/render_3d.py` - 3D rendering
10. `pc_rai/cli.py` - Command-line interface

### Step 3: Key Implementation Notes

- Use `scipy.spatial.cKDTree` for neighbor queries (faster than `KDTree`)
- Use `laspy.ExtraBytesParams` for adding extra dimensions
- Use `open3d.visualization` for 3D rendering to images
- Handle memory for large clouds by processing in chunks if needed
- Always preserve original LAS attributes when writing output

---

## Appendix D: v2.x ML Pipeline Implementation Guide

**For Claude Code: Follow this sequence to implement the ML-based stability prediction system.**

### Prerequisites

Before starting v2.x, ensure v1.x is complete:
- [x] Point-level features (slope, roughness) working
- [x] RAI classification pipeline functional
- [x] Output LAZ files with all attributes

### Step 1: Install Additional Dependencies

```bash
pip install scikit-learn>=1.0 pandas>=1.4 geopandas>=0.10 shapely>=1.8 joblib>=1.1
```

### Step 2: Implement ML Modules (Priority Order)

| Order | File | Purpose | Depends On |
|-------|------|---------|------------|
| 1 | `pc_rai/ml/__init__.py` | Package init | - |
| 2 | `pc_rai/ml/labels.py` | Event polygon → transect labels | geopandas |
| 3 | `pc_rai/features/aggregation.py` | Point → transect feature aggregation | v1.x features |
| 4 | `pc_rai/ml/data_prep.py` | Combine features + labels into training set | 2, 3 |
| 5 | `pc_rai/ml/train.py` | Random Forest training | 4 |
| 6 | `pc_rai/ml/validate.py` | Cross-validation framework | 5 |
| 7 | `pc_rai/ml/metrics.py` | AUC-ROC, AUC-PR, confusion matrices | 6 |
| 8 | `pc_rai/ml/predict.py` | Inference on new data | 5 |
| 9 | `scripts/train_rf_model.py` | CLI for training | 5, 6, 7 |
| 10 | `scripts/predict_stability.py` | CLI for inference | 8 |

### Step 3: Key Data Structures

```python
# pc_rai/ml/labels.py
@dataclass
class TransectLabel:
    transect_id: int
    beach_id: str
    scan_date: datetime
    event_count: int      # Number of rockfall events
    has_event: bool       # Binary label
    event_area_m2: float  # Total polygon overlap area

def load_event_polygons(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load rockfall event polygons from shapefile."""
    pass

def assign_labels_to_transects(
    transects: gpd.GeoDataFrame,
    events: gpd.GeoDataFrame,
    time_window_days: int = 365
) -> pd.DataFrame:
    """
    Intersect event polygons with transect corridors.

    Returns DataFrame with columns:
        transect_id, beach_id, scan_date, event_count, has_event, event_area_m2
    """
    pass
```

```python
# pc_rai/features/aggregation.py
AGGREGATION_STATS = {
    'slope': ['mean', 'max', 'std', lambda x: np.percentile(x, 90)],
    'r_small': ['mean', 'max', 'std'],
    'r_large': ['mean', 'max', 'std'],
    'height': ['mean', 'max', lambda x: x.max() - x.min()],
}

def aggregate_to_transect(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    transect_polygon: Polygon,
) -> Dict[str, float]:
    """
    Aggregate point-level features to transect statistics.

    Returns dict like:
        {'slope_mean': 45.2, 'slope_max': 78.1, 'slope_std': 12.3, ...}
    """
    pass

def compute_derived_features(agg_features: Dict[str, float]) -> Dict[str, float]:
    """Add derived features like r_ratio = r_small_mean / r_large_mean."""
    pass
```

```python
# pc_rai/ml/train.py
from sklearn.ensemble import RandomForestClassifier

def train_stability_model(
    X: pd.DataFrame,
    y: np.ndarray,
    config: MLConfig
) -> StabilityModel:
    """
    Train Random Forest with class weighting.

    Parameters
    ----------
    X : DataFrame
        Transect features (one row per transect-scan)
    y : array
        Binary labels (0=no event, 1=event)
    config : MLConfig
        Hyperparameters

    Returns
    -------
    StabilityModel with trained RF and metadata
    """
    rf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,  # 'balanced' handles imbalance
        random_state=config.random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    return StabilityModel(
        model=rf,
        feature_names=list(X.columns),
        feature_importances=dict(zip(X.columns, rf.feature_importances_)),
        # ... metadata
    )
```

### Step 4: Cross-Validation Implementation

```python
# pc_rai/ml/validate.py
from sklearn.model_selection import LeaveOneGroupOut

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
    Dict mapping beach_id → {'auc_roc': float, 'auc_pr': float, ...}
    """
    logo = LeaveOneGroupOut()
    results = {}

    for train_idx, test_idx in logo.split(X, y, groups=beach_ids):
        beach = beach_ids[test_idx[0]]

        # Train on all other beaches
        model = train_stability_model(X.iloc[train_idx], y[train_idx], config)

        # Predict on held-out beach
        y_pred_proba = model.predict(X.iloc[test_idx])

        # Compute metrics
        results[beach] = {
            'auc_roc': roc_auc_score(y[test_idx], y_pred_proba),
            'auc_pr': average_precision_score(y[test_idx], y_pred_proba),
            'n_samples': len(test_idx),
            'n_events': y[test_idx].sum(),
        }

    return results

def leave_one_year_out_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray,
    config: MLConfig
) -> Dict[int, Dict[str, float]]:
    """Leave-one-year-out temporal validation."""
    pass
```

### Step 5: Test Fixtures for ML

```python
# tests/conftest.py (add to existing)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

@pytest.fixture
def synthetic_transects():
    """Create synthetic transect corridors."""
    transects = []
    for i in range(100):
        # 10m wide corridors along coast
        geom = box(i * 10, 0, (i + 1) * 10, 50)
        transects.append({'transect_id': i, 'beach_id': f'BEACH_{i // 20}', 'geometry': geom})
    return gpd.GeoDataFrame(transects, crs='EPSG:32611')

@pytest.fixture
def synthetic_events():
    """Create synthetic rockfall event polygons."""
    events = []
    np.random.seed(42)
    for i in range(20):
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(10, 40)
        geom = box(x, y, x + 5, y + 5)  # Small event polygons
        events.append({
            'event_id': i,
            'event_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'geometry': geom
        })
    return gpd.GeoDataFrame(events, crs='EPSG:32611')

@pytest.fixture
def synthetic_training_data(synthetic_transects, synthetic_events):
    """Create synthetic ML training dataset."""
    # Features: random but correlated with labels
    n = len(synthetic_transects)
    X = pd.DataFrame({
        'slope_mean': np.random.uniform(40, 70, n),
        'slope_max': np.random.uniform(60, 90, n),
        'r_small_mean': np.random.uniform(5, 20, n),
        'r_large_mean': np.random.uniform(8, 25, n),
    })

    # Labels: 10% event rate (imbalanced)
    y = np.random.binomial(1, 0.1, n)

    return X, y, synthetic_transects['beach_id'].values
```

### Step 6: Test Cases for ML Modules

```python
# tests/test_ml_labels.py
def test_load_event_polygons(tmp_path, synthetic_events):
    # Save and reload
    path = tmp_path / "events.shp"
    synthetic_events.to_file(path)

    from pc_rai.ml.labels import load_event_polygons
    loaded = load_event_polygons(path)
    assert len(loaded) == len(synthetic_events)

def test_assign_labels(synthetic_transects, synthetic_events):
    from pc_rai.ml.labels import assign_labels_to_transects
    labels = assign_labels_to_transects(synthetic_transects, synthetic_events)

    assert 'transect_id' in labels.columns
    assert 'has_event' in labels.columns
    assert labels['has_event'].dtype == bool

# tests/test_ml_train.py
def test_train_rf(synthetic_training_data):
    X, y, _ = synthetic_training_data
    from pc_rai.ml.train import train_stability_model
    from pc_rai.config import MLConfig

    model = train_stability_model(X, y, MLConfig())

    assert model.model is not None
    assert len(model.feature_importances) == len(X.columns)
    assert all(0 <= v <= 1 for v in model.feature_importances.values())

def test_model_persistence(synthetic_training_data, tmp_path):
    X, y, _ = synthetic_training_data
    from pc_rai.ml.train import train_stability_model
    from pc_rai.config import MLConfig

    model = train_stability_model(X, y, MLConfig())

    # Save and reload
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = StabilityModel.load(path)

    # Predictions should match
    assert np.allclose(model.predict(X), loaded.predict(X))

# tests/test_ml_validate.py
def test_leave_one_beach_out(synthetic_training_data):
    X, y, beach_ids = synthetic_training_data
    from pc_rai.ml.validate import leave_one_beach_out_cv
    from pc_rai.config import MLConfig

    results = leave_one_beach_out_cv(X, y, beach_ids, MLConfig())

    # Should have one result per unique beach
    assert len(results) == len(np.unique(beach_ids))

    # Each result should have metrics
    for beach, metrics in results.items():
        assert 'auc_roc' in metrics
        assert 0 <= metrics['auc_roc'] <= 1
```

### Step 7: CLI Scripts

```python
# scripts/train_rf_model.py
"""
Train Random Forest stability model.

Usage:
    python scripts/train_rf_model.py \
        --features data/transect_features.parquet \
        --labels data/transect_labels.parquet \
        --output models/stability_rf.joblib \
        --cv leave-one-beach-out
"""
import argparse
from pathlib import Path
import pandas as pd
from pc_rai.ml.train import train_stability_model
from pc_rai.ml.validate import leave_one_beach_out_cv
from pc_rai.config import MLConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cv', choices=['leave-one-beach-out', 'leave-one-year-out', 'none'],
                        default='leave-one-beach-out')
    args = parser.parse_args()

    # Load data
    X = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    y = labels['has_event'].values

    config = MLConfig()

    # Cross-validation
    if args.cv == 'leave-one-beach-out':
        cv_results = leave_one_beach_out_cv(X, y, labels['beach_id'].values, config)
        print("Cross-validation results:")
        for beach, metrics in cv_results.items():
            print(f"  {beach}: AUC-ROC={metrics['auc_roc']:.3f}, AUC-PR={metrics['auc_pr']:.3f}")

    # Train final model on all data
    model = train_stability_model(X, y, config)
    model.save(args.output)
    print(f"Model saved to {args.output}")

    # Feature importances
    print("\nFeature importances:")
    for feat, imp in sorted(model.feature_importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

if __name__ == '__main__':
    main()
```

```python
# scripts/predict_stability.py
"""
Predict stability scores for new point clouds.

Usage:
    python scripts/predict_stability.py \
        --model models/stability_rf.joblib \
        --laz-dir output/rai/ \
        --transects data/transects.shp \
        --output predictions/stability_scores.csv
"""
import argparse
from pathlib import Path
import pandas as pd
from pc_rai.ml.predict import predict_transect_stability
from pc_rai.ml.train import StabilityModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--laz-dir', type=Path, required=True)
    parser.add_argument('--transects', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    model = StabilityModel.load(args.model)

    # Process each LAZ file
    results = predict_transect_stability(
        laz_dir=args.laz_dir,
        transects_path=args.transects,
        model=model
    )

    results.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
```

### Step 8: Integration with Risk Map

The `scripts/risk_map_regional.py` already aggregates energy to transects. Extend it to support ML stability scores:

```python
# Add to risk_map_regional.py

def load_stability_scores(scores_path: Path) -> Dict[int, float]:
    """Load ML stability scores by transect ID."""
    df = pd.read_csv(scores_path)
    return dict(zip(df['transect_id'], df['stability_score']))

# In render_regional_risk_map(), add option:
#   --scores predictions/stability_scores.csv
#
# If provided, use stability_score instead of energy_sum for coloring
```

### Key Implementation Notes

1. **Class imbalance**: Most transects have no events (~90%). Use `class_weight='balanced'` and evaluate with AUC-PR, not accuracy.

2. **Feature normalization**: Apply per-scan percentile normalization before aggregation to handle different LiDAR systems.

3. **Temporal leakage**: When doing leave-one-year-out CV, ensure the held-out year is strictly after training years (not random).

4. **Shapely operations**: Use `geopandas.sjoin` for efficient spatial joins, not loops.

5. **Missing data**: Some transects may have few points after clipping to cliff face. Filter transects with `point_count < 100`.

---

*End of PRD*
