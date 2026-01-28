# Product Requirements Document: PC-RAI

## Point Cloud-Native Rockfall Activity Index Tool

> **For Claude Code**: This PRD defines a Python tool for classifying LiDAR point clouds into rockfall hazard categories. Use this document to guide implementation. Key sections for coding: [Module Structure](#51-module-structure), [Data Flow](#52-data-flow), [Decision Tree](#appendix-a-decision-tree-pseudocode), and [Functional Requirements](#3-functional-requirements).

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 Draft |
| Date | January 2025 |
| Status | Planning |
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

Develop a Python-based tool that implements a point cloud-native adaptation of the Rockfall Activity Index (RAI) methodology for coastal cliff hazard assessment. The tool classifies LiDAR point clouds into five morphological hazard classes based on slope angle and multi-scale surface roughness.

### 1.2 Background

The RAI methodology (Dunham et al. 2017, Markus et al. 2023) was originally developed using a grid-based approach for rock slope hazard assessment along transportation corridors. This tool adapts the core algorithm to work directly on point clouds, eliminating grid-related distortions and simplifying the processing pipeline.

### 1.3 Scope

**In Scope:**

- Normal vector computation via CloudCompare CLI
- Slope angle calculation from normals
- Multi-scale roughness calculation (both radius and k-NN methods)
- Five-class morphological classification
- LAS output with classification attributes
- Static visualization products
- Batch processing capability

**Out of Scope (v1.0):**

- Full RAI kinetic energy scoring
- Change detection integration
- Interactive visualization/GUI
- Real-time processing
- Web interface

### 1.4 Core Algorithm Summary

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
OUTPUT: Classified point cloud + visualizations + report
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
| FR-4.5 | Default radius values: 0.175m (small), 0.425m (large) | Must Have |
| FR-4.6 | Default k-NN values: k=30 (small), k=100 (large) | Must Have |
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

**Classification Scheme (Simplified 5-class):**

| Code | Class Name | Abbreviation | Description |
|------|------------|--------------|-------------|
| 0 | Unclassified | U | Insufficient data for classification |
| 1 | Talus | T | Debris accumulation, low slope (<42°), smooth |
| 2 | Intact | I | Smooth rock face, few discontinuities |
| 3 | Discontinuous | D | Potential rockfall source (merged Df, Dc, Dw) |
| 4 | Steep/Overhang | O | High risk steep faces (slope >80°) |
| 5 | Structure | St | Seawalls, riprap, engineered surfaces |

**Classification Thresholds (adapted from Markus et al. 2023):**

| Parameter | Threshold | Units |
|-----------|-----------|-------|
| Steep/Overhang detection | 80 | degrees |
| Talus maximum slope | 42 | degrees |
| R_small low threshold | 6 | degrees |
| R_small mid threshold | 11 | degrees |
| R_large threshold | 12 | degrees |
| Structure roughness (max) | 4 | degrees |

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

**Required Python Packages:**

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

**External Software:**

```
CloudCompare>=2.12  # For normal computation via CLI
```

**Optional (for enhanced functionality):**

```
lazrs>=0.5          # LAZ compression
numba>=0.55         # JIT compilation for speedup
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
│   └── roughness.py        # Roughness (radius + k-NN methods)
│
├── classification/
│   ├── __init__.py
│   ├── decision_tree.py    # RAI classification logic
│   └── thresholds.py       # Configurable threshold parameters
│
├── visualization/
│   ├── __init__.py
│   ├── colors.py           # Color schemes and colormaps
│   ├── render_3d.py        # 3D point cloud rendering (Open3D)
│   └── figures.py          # Multi-panel figure generation
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
```

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
    normal_radius: float = 0.1
    mst_neighbors: int = 10
    
    # Slope
    up_vector: tuple = (0, 0, 1)
    
    # Roughness - radius method
    radius_small: float = 0.175
    radius_large: float = 0.425
    
    # Roughness - knn method
    k_small: int = 30
    k_large: int = 100
    
    # Roughness - shared
    min_neighbors: int = 5
    methods: list = ("radius", "knn")
    
    # Classification thresholds (adapted from Markus et al. 2023)
    thresh_overhang: float = 80.0
    thresh_talus_slope: float = 42.0
    thresh_r_small_low: float = 6.0
    thresh_r_small_mid: float = 11.0
    thresh_r_large: float = 12.0
    thresh_structure_roughness: float = 4.0
    
    # Output
    output_dir: str = "./output"
    compress_output: bool = True
    visualization_dpi: int = 300
    visualization_views: list = ("front", "oblique")
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
  estimation_radius: 0.1             # Radius for local plane fitting
  mst_neighbors: 10                  # Neighbors for MST orientation

# Slope calculation
slope:
  up_vector: [0, 0, 1]  # Z-up coordinate system

# Roughness calculation
roughness:
  methods: ["radius", "knn"]  # Compute both methods
  min_neighbors: 5            # Minimum for valid calculation
  
  radius:
    small: 0.175  # meters (corresponds to 35cm window diameter)
    large: 0.425  # meters (corresponds to 85cm window diameter)
  
  knn:
    small: 30   # neighbor count for small-scale roughness
    large: 100  # neighbor count for large-scale roughness

# Classification thresholds
# Adapted from Markus et al. 2023 for California coastal bluffs
classification:
  thresholds:
    overhang: 80           # degrees - slope above this is Steep/Overhang
    talus_slope: 42        # degrees - max slope for talus classification
    r_small_low: 6         # degrees - below this is "smooth"
    r_small_mid: 11        # degrees - above this is Discontinuous
    r_large: 12            # degrees - large-scale roughness threshold for Discontinuous
    structure_roughness: 4  # degrees - steep + below this = Structure

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
    """Vertical surface - expect I, Df, Dc, or Dw based on roughness."""
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

- [ ] Loads LAS files (with or without existing normals)
- [ ] Computes normals via CloudCompare CLI
- [ ] Calculates slope angle for all points
- [ ] Calculates roughness using at least one method (radius)
- [ ] Classifies all points into 7 RAI classes + unclassified
- [ ] Outputs LAS file with slope, roughness, and classification
- [ ] Generates at least one 3D visualization

### 9.2 Full Release (v1.0)

All MVP criteria plus:

- [ ] Both roughness methods (radius and k-NN) implemented
- [ ] Comparison metrics between methods
- [ ] Batch processing capability
- [ ] Complete visualization suite (slope, roughness, classification, comparison)
- [ ] Markdown and JSON reports
- [ ] YAML configuration file support
- [ ] Full CLI functionality
- [ ] Python API documented and functional
- [ ] Unit test coverage ≥ 80% for core modules
- [ ] README with installation and usage instructions

---

## 10. Future Considerations

### 10.1 Version 1.1

- RAI kinetic energy scoring (requires height-above-base)
- Integration with M3C2 change detection
- Temporal analysis between epochs

### 10.2 Version 2.0

- GUI application
- Machine learning-enhanced classification
- Automatic parameter tuning
- CloudCompare plugin

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
        if r_small < thresholds['structure_roughness']:  # 4°
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
    if r_small > thresholds['r_small_mid']:  # 11°
        return 3  # Discontinuous
    if r_large > thresholds['r_large']:  # 12°
        return 3  # Discontinuous

    # Moderate roughness, low r_large
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

*End of PRD*
