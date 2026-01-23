# PC-RAI: Point Cloud Rockfall Activity Index

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for classifying LiDAR point clouds into rockfall hazard categories using a point cloud-native adaptation of the Rockfall Activity Index (RAI) methodology.

## Overview

PC-RAI adapts the grid-based RAI algorithm (Dunham et al. 2017, Markus et al. 2023) to work directly on point clouds. It classifies each point into one of seven morphological hazard classes based on slope angle and multi-scale surface roughness.

### RAI Classes

| Code | Class | Description |
|------|-------|-------------|
| U | Unclassified | Insufficient neighbors or invalid data |
| T | Talus | Debris accumulation, low slope (<42°), smooth |
| I | Intact | Smooth rock face, few discontinuities |
| Df | Fragmented Discontinuous | Closely fractured, large-scale roughness |
| Dc | Closely Spaced Discontinuous | Moderate fracturing (11-18° roughness) |
| Dw | Widely Spaced Discontinuous | Widely fractured (>18° roughness) |
| Os | Shallow Overhang | Overhanging (90-150° slope) |
| Oc | Cantilevered Overhang | Severely overhanging (>150°) |

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
- CloudCompare (optional, for normal computation)
- See `pyproject.toml` for Python dependencies

### Installing CloudCompare (Optional)

CloudCompare is only needed if your point clouds don't have pre-computed normals:

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

# Verbose output
pc-rai process input.las -o output/ -v

# Generate visualizations from a processed file
pc-rai visualize output/input_rai.las -o figures/
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
for class_code, class_name in enumerate(["U", "T", "I", "Df", "Dc", "Dw", "Os", "Oc"]):
    count = (result.rai_class_radius == class_code).sum()
    pct = 100 * count / result.n_points
    print(f"  {class_name}: {count:,} ({pct:.1f}%)")

# Access timing info
print(f"Total processing time: {result.timing['total']:.2f}s")
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
  --methods           Roughness method: radius, knn, or both (default: both)
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

PC-RAI produces the following outputs:

### 1. Classified LAS File (`*_rai.las` or `*_rai.laz`)

Extra dimensions added to the point cloud:

| Dimension | Type | Description |
|-----------|------|-------------|
| `slope_deg` | float32 | Slope angle in degrees (0-180°) |
| `roughness_small_radius` | float32 | Small-scale roughness (radius method) |
| `roughness_large_radius` | float32 | Large-scale roughness (radius method) |
| `roughness_small_knn` | float32 | Small-scale roughness (k-NN method) |
| `roughness_large_knn` | float32 | Large-scale roughness (k-NN method) |
| `neighbor_count_small` | uint16 | Neighbor count at small scale |
| `neighbor_count_large` | uint16 | Neighbor count at large scale |
| `rai_class_radius` | uint8 | RAI class code (radius method) |
| `rai_class_knn` | uint8 | RAI class code (k-NN method) |

### 2. Visualizations (PNG)

| File | Description |
|------|-------------|
| `*_classification_radius_*.png` | 3D classification map (radius method) |
| `*_classification_knn_*.png` | 3D classification map (k-NN method) |
| `*_slope.png` | Slope angle visualization |
| `*_roughness_small_*.png` | Small-scale roughness map |
| `*_roughness_large_*.png` | Large-scale roughness map |
| `*_comparison.png` | Side-by-side method comparison |
| `*_summary.png` | 4-panel summary figure |
| `*_histogram.png` | Class distribution bar chart |

### 3. Reports

| File | Description |
|------|-------------|
| `*_report.md` | Human-readable Markdown report |
| `*_report.json` | Machine-readable JSON with all statistics |

## Configuration

### YAML Configuration File

Create a `config.yaml` to customize parameters:

```yaml
# Roughness calculation parameters
radius_small: 0.175      # Small-scale radius (meters)
radius_large: 0.425      # Large-scale radius (meters)
k_small: 30              # Small-scale k-NN neighbors
k_large: 100             # Large-scale k-NN neighbors
min_neighbors: 5         # Minimum neighbors for valid roughness

# Classification thresholds
thresh_talus_slope: 42.0  # Max slope for Talus class (degrees)
thresh_overhang: 90.0     # Min slope for overhang classes
thresh_cantilever: 150.0  # Min slope for cantilevered overhang

# Roughness thresholds (degrees)
thresh_rough_small_intact: 6.0    # Max roughness for Intact
thresh_rough_small_dc: 11.0       # Max for Closely Spaced
thresh_rough_small_dw: 18.0       # Max for Widely Spaced
thresh_rough_large_df: 12.0       # Min large-scale for Fragmented

# Processing options
methods:
  - radius
  - knn

# Output options
compress_output: true           # Output as .laz instead of .las
visualization_dpi: 300
visualization_views:
  - front
  - oblique
```

Use with: `pc-rai process input.las -o output/ --config config.yaml`

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius_small` | 0.175 | Small-scale roughness radius (m) |
| `radius_large` | 0.425 | Large-scale roughness radius (m) |
| `k_small` | 30 | Small-scale k-NN neighbors |
| `k_large` | 100 | Large-scale k-NN neighbors |
| `min_neighbors` | 5 | Minimum neighbors for valid roughness |
| `thresh_talus_slope` | 42.0 | Maximum slope for Talus (°) |
| `thresh_overhang` | 90.0 | Minimum slope for overhangs (°) |
| `thresh_cantilever` | 150.0 | Minimum slope for cantilevered (°) |
| `compress_output` | true | Compress output as LAZ |

## Algorithm

### Decision Tree

The classification follows this decision tree:

```
IF slope > 90°:
    IF slope > 150° → Cantilevered Overhang (Oc)
    ELSE → Shallow Overhang (Os)
ELIF roughness_small < 6°:
    IF slope < 42° → Talus (T)
    ELSE → Intact (I)
ELIF roughness_small > 18° → Widely Spaced Discontinuous (Dw)
ELIF roughness_small > 11° → Closely Spaced Discontinuous (Dc)
ELIF roughness_large > 12° → Fragmented Discontinuous (Df)
ELSE → Intact (I)
```

### Roughness Calculation

Surface roughness is computed as the angular deviation of point normals from the local mean normal:

1. For each point, find neighbors (by radius or k-NN)
2. Compute the mean normal vector of neighbors
3. Calculate angle between each normal and the mean
4. Roughness = standard deviation of these angles

Two methods are available:
- **Radius method**: Fixed spatial radius (0.175m, 0.425m)
- **k-NN method**: Fixed neighbor count (30, 100)

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
│   └── decision_tree.py # RAI decision tree logic
├── visualization/       # Output generation
│   ├── render_3d.py     # 3D point cloud rendering
│   └── figures.py       # Multi-panel figures
├── reporting/           # Statistics and reports
│   ├── statistics.py    # Class stats, method agreement
│   └── report_writer.py # Markdown/JSON reports
└── utils/               # Utilities
    └── spatial.py       # KD-tree spatial index
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
