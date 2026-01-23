# CLAUDE.md - Project Instructions for Claude Code

> This file provides context and instructions for Claude Code agents working on this project.

## Project Overview

**PC-RAI** is a Python tool for classifying LiDAR point clouds into rockfall hazard categories using the Rockfall Activity Index (RAI) methodology. It adapts the grid-based RAI algorithm to work directly on point clouds.

## Key Documentation

| File | Purpose |
|------|---------|
| `prd.md` | Full product requirements, specifications, data structures |
| `todo.md` | Sequential task list with acceptance criteria |
| `CLAUDE.md` | This file - agent instructions |

**Always read `todo.md` first** to understand current progress and next task.

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
├── classification/      # Decision tree logic
├── visualization/       # Rendering, figures
├── reporting/           # Statistics, reports
└── utils/               # Spatial index, timing, logging
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
| Classification | (N,) | uint8 | Codes 0-7 |

### RAI Class Codes
```python
0 = Unclassified (invalid/insufficient data)
1 = Talus (T)
2 = Intact (I)
3 = Fragmented Discontinuous (Df)
4 = Closely Spaced Discontinuous (Dc)
5 = Widely Spaced Discontinuous (Dw)
6 = Shallow Overhang (Os)
7 = Cantilevered Overhang (Oc)
```

### Decision Tree Logic (Critical)
```
if slope > 90°:
    if slope > 150° → Oc
    else → Os
elif r_small < 6°:
    if slope < 42° → T
    else → I
elif r_small > 18° → Dw
elif r_small > 11° → Dc
elif r_large > 12° → Df
else → I
```

### Default Parameters (Markus et al. 2023)
```python
radius_small = 0.175  # meters
radius_large = 0.425  # meters
k_small = 30          # neighbors
k_large = 100         # neighbors
thresh_talus_slope = 42.0  # degrees (updated from 35°)
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

## References

- Dunham et al. (2017) - Original RAI methodology
- Markus et al. (2023) - Updated parameters, 5-year inventory study
- laspy docs: https://laspy.readthedocs.io/
- Open3D docs: https://www.open3d.org/docs/
- scipy.spatial: https://docs.scipy.org/doc/scipy/reference/spatial.html
