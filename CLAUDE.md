# CLAUDE.md - Project Instructions for Claude Code

> This file provides context and instructions for Claude Code agents working on this project.

## Project Overview

**PC-RAI** is a Python tool for classifying LiDAR point clouds into rockfall hazard categories using the Rockfall Activity Index (RAI) methodology. It adapts the grid-based RAI algorithm to work directly on point clouds.

## Key Documentation

| File | Purpose |
|------|---------|
| `README.md` | User-facing documentation, installation, CLI usage |
| `prd.md` | Product requirements, specifications, data structures |
| `todo.md` | Task list with acceptance criteria (v1.0 complete + extensions) |
| `CLAUDE.md` | This file - agent instructions |

**Project Status**: v1.0 complete with extensions (PCA classification, improved visualizations).
225 tests passing (1 flaky CloudCompare integration test).

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
└── utils/               # Spatial index, timing, logging

scripts/
└── compute_normals_mst.py  # CloudComPy normal computation
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
- Create custom exceptions in `pc_rai/exceptions.py` if needed
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
    if r_small < 4° → Structure (St)
    else → Steep/Overhang (O)
elif r_small < 6°:
    if slope < 42° → Talus (T)
    else → Intact (I)
elif r_small > 11° → Discontinuous (D)
elif r_large > 12° → Discontinuous (D)
else → Intact (I)
```

### Default Parameters (adapted from Markus et al. 2023)
```python
radius_small = 1.0    # meters (tuned for 50cm point spacing)
radius_large = 2.5    # meters
k_small = 40          # neighbors
k_large = 120         # neighbors
thresh_overhang = 80.0         # degrees (80° for coastal bluffs)
thresh_talus_slope = 42.0      # degrees
thresh_structure_roughness = 2.0  # degrees
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

### Output Directory Structure
```
output/
├── rai/               # LAZ files and reports
│   ├── *_rai.laz
│   ├── *_report.md
│   └── *_report.json
└── figures/<date>/    # Visualizations organized by date
    ├── *_classification_*.png
    ├── *_histogram_*.png
    └── *_slope.png
```

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

### 5. CloudCompare Output Location
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
1. Check `prd.md` for specifications
2. Check Appendix A (decision tree pseudocode)
3. Check Appendix B (CloudCompare commands)

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

## References

- Dunham et al. (2017) - Original RAI methodology
- Markus et al. (2023) - Updated parameters, 5-year inventory study
- laspy docs: https://laspy.readthedocs.io/
- Open3D docs: https://www.open3d.org/docs/
- scipy.spatial: https://docs.scipy.org/doc/scipy/reference/spatial.html
- scikit-learn: https://scikit-learn.org/stable/ (PCA, K-means, silhouette score)
