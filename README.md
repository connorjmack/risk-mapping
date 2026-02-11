# PC-RAI: Point Cloud Rockfall Activity Index

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-266%20passed-brightgreen.svg)]()

A Python tool for classifying LiDAR point clouds into rockfall hazard categories using a point cloud-native adaptation of the Rockfall Activity Index (RAI) methodology.

## Overview

PC-RAI adapts the grid-based RAI algorithm (Dunham et al. 2017, Markus et al. 2023) to work directly on point clouds. It classifies each point into one of five morphological hazard classes based on slope angle and multi-scale surface roughness.

**Key Features:**
- Decision tree-based RAI classification (5 morphological classes)
- k-NN roughness calculation at multiple scales
- Classification smoothing via spatial majority voting
- Per-point rockfall energy calculation
- PCA-based unsupervised classification with automatic cluster detection
- Location-organized output structure
- Publication-ready visualizations (4-panel summaries, transect heatmaps)
- Batch processing support
- CloudCompare integration for normal computation

### RAI Classes (Simplified 5-class scheme)

| Code | Class | Description |
|------|-------|-------------|
| U | Unclassified | Insufficient neighbors or invalid data |
| T | Talus | Debris accumulation, low slope (<42°), smooth |
| I | Intact | Smooth rock face, few discontinuities |
| D | Discontinuous | Potential rockfall source (rough surfaces) |
| O | Steep/Overhang | High risk steep faces (slope >80°) |
| St | Structure | Seawalls, riprap, engineered surfaces |

## Installation

```bash
# Clone repository
git clone https://github.com/connorjmack/pc-rai.git
cd pc-rai

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import pc_rai; print(f'PC-RAI v{pc_rai.__version__}')"
```

### Requirements

- Python 3.9+
- CloudCompare or CloudComPy (optional, for normal computation)

**Core dependencies** (installed automatically):
- numpy, scipy - Numerical computing and spatial indexing
- laspy[lazrs] - LAS/LAZ file I/O
- matplotlib - Visualization
- scikit-learn - PCA and clustering
- tqdm - Progress bars
- pyyaml - Configuration files

See `pyproject.toml` for full dependency list.

### Installing CloudCompare (Optional)

CloudCompare is only needed if your point clouds don't have pre-computed normals.

#### Option A: CloudComPy (Recommended for batch processing)

CloudComPy provides Python bindings for CloudCompare, enabling scripted normal computation with MST orientation.

```bash
# Activate CloudComPy environment
. /Users/cjmack/Tools/CloudComPy311/bin/condaCloud.zsh activate cloud-compy

# Compute normals for all files in a directory
python scripts/compute_normals_mst.py tests/test_data --output-dir output/normals --radius 1.0

# Deactivate when done
conda deactivate
```

#### Option B: CloudCompare CLI

- **macOS**: `brew install --cask cloudcompare`
- **Ubuntu**: `sudo apt install cloudcompare`
- **Windows**: Download from [cloudcompare.org](https://www.cloudcompare.org/release/)

## Quick Start

### Command Line

```bash
# Process a single point cloud
pc-rai process input.las -o output/

# Process with existing normals (skip CloudCompare)
pc-rai process input.las -o output/ --skip-normals

# Use only radius-based roughness method
pc-rai process input.las -o output/ --methods radius

# Use only k-NN roughness method
pc-rai process input.las -o output/ --methods knn

# Batch process a directory
pc-rai process ./data/ -o output/ --batch

# Process without visualizations or reports
pc-rai process input.las -o output/ --no-visualize --no-report

# Run PCA-based unsupervised classification (auto-detects clusters)
pc-rai process input.las -o output/ --pca -v

# Batch process with PCA classification
pc-rai process ./normals/ -o output/ --batch --pca --skip-normals -v

# Generate visualizations from a processed file
pc-rai visualize output/rai/input_rai.laz -o output/
```

### Python API

```python
import numpy as np
from pc_rai import RAIClassifier, RAIConfig, load_point_cloud, save_point_cloud

# Load point cloud
cloud = load_point_cloud("input.las")
print(f"Loaded {cloud.n_points:,} points")

# Create classifier with custom config
config = RAIConfig(
    methods=["radius", "knn"],
    radius_small=0.175,
    radius_large=0.425,
    k_small=30,
    k_large=100,
)
classifier = RAIClassifier(config)

# Process (use compute_normals=False if normals already exist)
result = classifier.process(cloud, compute_normals=False)

# Access results
print(f"Slope range: {result.slope_deg.min():.1f}° - {result.slope_deg.max():.1f}°")
print(f"Unique classes (radius): {np.unique(result.rai_class_radius)}")
print(f"Unique classes (k-NN): {np.unique(result.rai_class_knn)}")

# View statistics
for class_code, class_name in enumerate(["U", "T", "I", "D", "O", "St"]):
    count = (result.rai_class_radius == class_code).sum()
    pct = 100 * count / result.n_points
    print(f"  {class_name}: {count:,} ({pct:.1f}%)")

# Access timing info
print(f"Total processing time: {result.timing['total']:.2f}s")

# Run with PCA-based unsupervised classification
result_pca = classifier.process(cloud, compute_normals=False, run_pca=True)

# Access PCA results
if result_pca.pca_result is not None:
    pca = result_pca.pca_result
    print(f"PCA clusters found: {pca.n_clusters}")
    print(f"Silhouette score: {pca.silhouette_avg:.3f}")
    print(f"Variance explained: {sum(pca.explained_variance_ratio)*100:.1f}%")
```

#### File-Based Processing

```python
from pathlib import Path
from pc_rai import RAIClassifier, RAIConfig

config = RAIConfig(methods=["radius", "knn"])
classifier = RAIClassifier(config)

# Process single file with all outputs
result = classifier.process_file(
    Path("input.las"),
    Path("output/"),
    compute_normals=False,
    generate_visualizations=True,
    generate_report=True,
    verbose=True,
)

# Batch process multiple files
input_files = list(Path("data/").glob("*.las"))
results = classifier.process_batch(
    input_files,
    Path("output/"),
    compute_normals=False,
)
```

## CLI Reference

```
pc-rai process <input> -o <output> [options]

Positional:
  input               Input LAS/LAZ file or directory

Options:
  -o, --output        Output directory (required)
  -c, --config        Configuration YAML file
  --batch             Process all LAS/LAZ files in directory
  --skip-normals      Skip normal computation (use existing)
  --methods           Roughness method: radius, knn, or both (default: knn)
  --pca               Run PCA-based unsupervised classification
  --no-visualize      Skip visualization generation
  --no-report         Skip report generation
  -v, --verbose       Verbose output

pc-rai visualize <input> -o <output> [options]

Positional:
  input               Processed LAS file with RAI attributes

Options:
  -o, --output        Output directory (required)
  --dpi               Output image resolution (default: 300)
  --views             View angles: front, oblique, top, side (default: front oblique)
```

## Output Files

PC-RAI produces outputs in an organized directory structure:

```
output/
├── rai/                          # Classified point clouds (flat)
│   └── input_rai.laz
├── reports/<LOCATION>/           # Reports organized by location
│   ├── input_report.md
│   └── input_report.json
├── panels/<LOCATION>/            # 4-panel summary figures by location
│   └── input_panels.png          # Slope, classification, roughness, histogram
└── heatmap/<LOCATION>/           # Transect risk heatmaps by location
    └── input_risk_map_3d.png
```

Location is automatically extracted from filename patterns (e.g., `20241215_TORP_subsamp1` → `TORP/`).

### 1. Classified LAS/LAZ File (`output/rai/*_rai.laz`)

Extra dimensions added to the point cloud:

| Dimension | Type | Description |
|-----------|------|-------------|
| `slope_deg` | float32 | Slope angle in degrees (0-180°) |
| `roughness_small_knn` | float32 | Small-scale roughness (k-NN method) |
| `roughness_large_knn` | float32 | Large-scale roughness (k-NN method) |
| `rai_class_knn` | uint8 | RAI class code (k-NN method, smoothed) |
| `energy_kj_knn` | float32 | Rockfall energy per point (kJ) |
| `pca_cluster` | int32 | PCA cluster label (if `--pca` used, -1 for invalid) |

### 2. Visualizations

**Panels** (`output/panels/<LOCATION>/`):
| File | Description |
|------|-------------|
| `*_panels.png` | 4-panel summary: slope, classification, roughness, histogram |

**Heatmaps** (`output/heatmap/<LOCATION>/`):
| File | Description |
|------|-------------|
| `*_risk_map_3d.png` | 3D transect risk visualization with satellite basemap |

### 3. Reports (`output/reports/<LOCATION>/`)

| File | Description |
|------|-------------|
| `*_report.md` | Human-readable Markdown report |
| `*_report.json` | Machine-readable JSON with all statistics |

## Configuration

### YAML Configuration File

Create a `config.yaml` to customize parameters:

```yaml
# Roughness calculation parameters
radius_small: 1.0        # Small-scale radius (meters)
radius_large: 2.5        # Large-scale radius (meters)
k_small: 40              # Small-scale k-NN neighbors
k_large: 120             # Large-scale k-NN neighbors
min_neighbors: 5         # Minimum neighbors for valid roughness

# Classification thresholds (adapted from Markus et al. 2023)
thresh_talus_slope: 42.0          # Max slope for Talus class (degrees)
thresh_overhang: 80.0             # Min slope for Steep/Overhang
thresh_structure_roughness: 6.0   # Max roughness for Structure (dual-scale check)

# Roughness thresholds (degrees)
thresh_r_small_low: 6.0           # Below this = smooth (Talus or Intact)
thresh_r_small_mid: 15.0          # Above this = Discontinuous
thresh_r_large: 15.0              # Large-scale threshold for Discontinuous

# Classification smoothing
smoothing_k: 25                   # Neighbors for majority-vote smoothing

# Processing options
methods:
  - knn

# Output options
compress_output: true           # Output as .laz instead of .las
visualization_dpi: 300
```

Use with: `pc-rai process input.las -o output/ --config config.yaml`

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius_small` | 1.0 | Small-scale roughness radius (m) |
| `radius_large` | 2.5 | Large-scale roughness radius (m) |
| `k_small` | 40 | Small-scale k-NN neighbors |
| `k_large` | 120 | Large-scale k-NN neighbors |
| `min_neighbors` | 5 | Minimum neighbors for valid roughness |
| `thresh_talus_slope` | 42.0 | Maximum slope for Talus (°) |
| `thresh_overhang` | 80.0 | Minimum slope for Steep/Overhang (°) |
| `thresh_r_small_mid` | 15.0 | Roughness threshold for Discontinuous (°) |
| `thresh_r_large` | 15.0 | Large-scale roughness threshold (°) |
| `smoothing_k` | 25 | Neighbors for classification smoothing |
| `compress_output` | true | Compress output as LAZ |

## Algorithm

### Decision Tree

The classification follows a simplified 5-class decision tree adapted from Markus et al. (2023) for California coastal bluffs:

```
IF slope > 80°:
    IF roughness_small < 6° AND roughness_large < 6° → Structure (St)
    ELSE → Steep/Overhang (O)
ELIF roughness_small < 6°:
    IF slope < 42° → Talus (T)
    ELSE → Intact (I)
ELIF roughness_small > 15° → Discontinuous (D)
ELIF roughness_large > 15° → Discontinuous (D)
ELSE → Intact (I)
```

After classification, spatial smoothing is applied using majority voting (k=25 neighbors) to reduce noise.

### Roughness Calculation

Surface roughness is computed as the angular deviation of point normals from the local mean normal:

1. For each point, find neighbors (by radius or k-NN)
2. Compute the mean normal vector of neighbors
3. Calculate angle between each normal and the mean
4. Roughness = standard deviation of these angles

Two methods are available:
- **Radius method**: Fixed spatial radius (0.175m, 0.425m)
- **k-NN method**: Fixed neighbor count (30, 100)

### PCA-Based Unsupervised Classification

In addition to the rule-based RAI decision tree, PC-RAI offers an unsupervised PCA-based classification mode that discovers natural groupings in the data:

1. **Features**: Uses the same attributes (slope, small-scale roughness, large-scale roughness)
2. **Standardization**: Features are standardized using z-score normalization
3. **PCA**: Principal Component Analysis reduces dimensionality (typically keeping all 3 components)
4. **Clustering**: K-means clustering groups points into morphologically similar categories
5. **Auto-detection**: Optimal cluster count (3-12) is determined using silhouette score

This approach is useful for:
- Exploring data without predefined class boundaries
- Validating RAI classification results
- Discovering site-specific morphological patterns

Each cluster is automatically interpreted based on its feature statistics (e.g., "steep, rough (small-scale), fragmented (large-scale)").

### Energy Calculation

Per-point rockfall energy is computed using the methodology from Dunham et al. (2017):

```
E (kJ) = 0.5 × ρ × A × D × g × H / 1000
```

Where:
- ρ = 2400 kg/m³ (rock density)
- A = 0.01 m² (point area for 10cm spacing)
- D = effective failure depth (class-specific depth × instability rate)
- g = 9.81 m/s² (gravity)
- H = point elevation (relative)

Class-specific effective depths:
| Class | Failure Depth | Instability Rate | Effective D |
|-------|---------------|------------------|-------------|
| Talus | 0.0 m | 0.00 | 0.0 m |
| Intact | 0.05 m | 0.03 | 0.0015 m |
| Discontinuous | 0.5 m | 0.10 | 0.05 m |
| Steep/Overhang | 1.0 m | 0.50 | 0.5 m |
| Structure | 0.0 m | 0.00 | 0.0 m |

## Project Structure

```
pc_rai/
├── __init__.py          # Package init, public API
├── __main__.py          # Entry point: python -m pc_rai
├── cli.py               # Command-line interface
├── config.py            # RAIConfig, constants, thresholds
├── classifier.py        # Main RAIClassifier class
├── io/                  # LAS file I/O
│   ├── las_reader.py    # PointCloud class, load functions
│   └── las_writer.py    # Save with extra dimensions
├── normals/             # Normal computation
│   └── cloudcompare.py  # CloudCompare CLI wrapper
├── features/            # Feature extraction
│   ├── slope.py         # Slope from normals
│   └── roughness.py     # Multi-scale roughness
├── classification/      # Classification
│   ├── decision_tree.py # RAI decision tree logic
│   └── pca_classifier.py # PCA-based unsupervised classification
├── visualization/       # Output generation
│   ├── render_3d.py     # 3D point cloud rendering
│   └── figures.py       # Multi-panel figures
├── reporting/           # Statistics and reports
│   ├── statistics.py    # Class stats, method agreement
│   └── report_writer.py # Markdown/JSON reports
└── utils/               # Utilities
    └── spatial.py       # KD-tree spatial index

scripts/
├── compute_normals_mst.py  # CloudComPy normal computation with westward bias
└── risk_map_regional.py    # Generate county-wide risk map from multiple surveys
```

## Development

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_classifier.py -v

# Run with coverage
pytest tests/ --cov=pc_rai --cov-report=term-missing

# Format code
black pc_rai/ tests/

# Lint
ruff check pc_rai/ tests/
```

## References

1. Dunham, L., Wartman, J., Olsen, M.J., O'Banion, M., Cunningham, K. (2017). Rockfall Activity Index (RAI): A lidar-derived, morphology-based method for hazard assessment. *Engineering Geology*, 221, 184-192. https://doi.org/10.1016/j.enggeo.2017.03.006

2. Markus, S.J., Wartman, J., Olsen, M., Darrow, M.M. (2023). Lidar-Derived Rockfall Inventory—An Analysis of the Geomorphic Evolution of Rock Slopes and Modifying the Rockfall Activity Index (RAI). *Remote Sensing*, 15, 4223. https://doi.org/10.3390/rs15174223

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Connor Mack (connorjmack)
