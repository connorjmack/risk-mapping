# PC-RAI Development Todo

> **Instructions for Claude Code**: Work through these tasks sequentially. Each task has clear acceptance criteria. Mark tasks complete with `[x]` as you finish them. Run the specified tests before moving to the next task. Reference `pc_rai_prd.md` for detailed specifications.

---

## Project Status

- **Current Phase**: Phase 6 (Pipeline & CLI)
- **Last Completed Task**: 5.1 Statistics and Reports
- **Blocking Issues**: None

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
    
    # Classification thresholds (Markus et al. 2023)
    thresh_overhang: float = 90.0
    thresh_cantilever: float = 150.0
    thresh_talus_slope: float = 42.0
    thresh_r_small_low: float = 6.0
    thresh_r_small_mid: float = 11.0
    thresh_r_small_high: float = 18.0
    thresh_r_large: float = 12.0
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    compress_output: bool = True
    visualization_dpi: int = 300
    visualization_views: List[str] = field(default_factory=lambda: ["front", "oblique"])

# RAI class definitions
RAI_CLASS_NAMES = {
    0: "Unclassified",
    1: "Talus",
    2: "Intact", 
    3: "Fragmented Discontinuous",
    4: "Closely Spaced Discontinuous",
    5: "Widely Spaced Discontinuous",
    6: "Shallow Overhang",
    7: "Cantilevered Overhang",
}

RAI_CLASS_ABBREV = {
    0: "U", 1: "T", 2: "I", 3: "Df", 4: "Dc", 5: "Dw", 6: "Os", 7: "Oc"
}

RAI_CLASS_COLORS = {
    0: "#9E9E9E",  # Gray
    1: "#C8A2C8",  # Light Purple
    2: "#4CAF50",  # Green
    3: "#81D4FA",  # Light Blue
    4: "#2196F3",  # Blue
    5: "#1565C0",  # Dark Blue
    6: "#FFEB3B",  # Yellow
    7: "#F44336",  # Red
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
    assert len(RAI_CLASS_NAMES) == 8
    assert RAI_CLASS_NAMES[1] == "Talus"
    assert RAI_CLASS_NAMES[7] == "Cantilevered Overhang"
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
**Goal**: Implement the RAI 7-class decision tree.

**File**: `pc_rai/classification/decision_tree.py`

**Implement**:
```python
import numpy as np
from dataclasses import dataclass
from pc_rai.config import RAIConfig

@dataclass
class ClassificationThresholds:
    """Thresholds for RAI decision tree."""
    overhang: float = 90.0
    cantilever: float = 150.0
    talus_slope: float = 42.0
    r_small_low: float = 6.0
    r_small_mid: float = 11.0
    r_small_high: float = 18.0
    r_large: float = 12.0
    
    @classmethod
    def from_config(cls, config: RAIConfig) -> 'ClassificationThresholds':
        return cls(
            overhang=config.thresh_overhang,
            cantilever=config.thresh_cantilever,
            talus_slope=config.thresh_talus_slope,
            r_small_low=config.thresh_r_small_low,
            r_small_mid=config.thresh_r_small_mid,
            r_small_high=config.thresh_r_small_high,
            r_large=config.thresh_r_large,
        )

def classify_points(
    slope_deg: np.ndarray,
    r_small: np.ndarray,
    r_large: np.ndarray,
    thresholds: ClassificationThresholds = None
) -> np.ndarray:
    """
    Classify points using RAI decision tree.
    
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
        (N,) uint8 array of class codes 0-7
        
    Class Codes:
        0 = Unclassified (invalid data)
        1 = Talus (T)
        2 = Intact (I)
        3 = Fragmented Discontinuous (Df)
        4 = Closely Spaced Discontinuous (Dc)
        5 = Widely Spaced Discontinuous (Dw)
        6 = Shallow Overhang (Os)
        7 = Cantilevered Overhang (Oc)
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

def test_widely_spaced_discontinuous(thresholds):
    """High r_small = Dw."""
    slope = np.array([60.0])
    r_small = np.array([20.0])  # > 18°
    r_large = np.array([25.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 5  # Dw

def test_closely_spaced_discontinuous(thresholds):
    """Moderate r_small = Dc."""
    slope = np.array([60.0])
    r_small = np.array([15.0])  # 11° < x < 18°
    r_large = np.array([20.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 4  # Dc

def test_fragmented_discontinuous(thresholds):
    """Intermediate r_small + high r_large = Df."""
    slope = np.array([60.0])
    r_small = np.array([8.0])  # 6° < x < 11°
    r_large = np.array([15.0])  # > 12°
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 3  # Df

def test_intact_intermediate(thresholds):
    """Intermediate r_small + low r_large = Intact."""
    slope = np.array([60.0])
    r_small = np.array([8.0])  # 6° < x < 11°
    r_large = np.array([8.0])  # < 12°
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 2  # Intact

def test_shallow_overhang(thresholds):
    """Slope > 90° but < 150° = Os."""
    slope = np.array([120.0])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 6  # Os

def test_cantilevered_overhang(thresholds):
    """Slope > 150° = Oc."""
    slope = np.array([160.0])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 7  # Oc

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
    assert np.all((classes >= 0) & (classes <= 7))
```

**Acceptance Criteria**:
- [x] All 7 classes correctly assigned based on decision tree
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
    colors = [RAI_CLASS_COLORS[i] for i in range(8)]
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
        (N,) class codes 0-7
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
    classes = np.random.randint(0, 8, n).astype(np.uint8)
    
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
- [ ] Full pipeline runs on synthetic data
- [ ] All result fields populated correctly
- [ ] Statistics computed
- [ ] Timing information captured
- [ ] `pytest tests/test_classifier.py` passes

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
- [ ] `python -m pc_rai --help` shows usage
- [ ] `python -m pc_rai process input.las -o output/` runs pipeline
- [ ] `--batch` processes multiple files
- [ ] `--config` loads custom YAML
- [ ] `--skip-normals` skips normal computation
- [ ] `--no-visualize` and `--no-report` work
- [ ] `pytest tests/test_cli.py` passes

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
- [ ] Full pipeline runs without errors
- [ ] All output files generated
- [ ] Multiple RAI classes detected in output
- [ ] CLI integration works
- [ ] `pytest tests/test_integration.py` passes

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
- [ ] README has installation instructions
- [ ] README has CLI usage examples
- [ ] README has Python API examples
- [ ] README explains output formats
- [ ] README references Dunham et al. 2017 and Markus et al. 2023

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
- [ ] 6.1 RAI Classifier class
- [ ] 6.2 Command-line interface

### Phase 7: Integration
- [ ] 7.1 Integration tests
- [ ] 7.2 Documentation

---

## Notes for Claude Code Agent

1. **Run tests after each task**: Use `pytest tests/test_<module>.py -v` to verify implementation
2. **Reference the PRD**: See `pc_rai_prd.md` for detailed specifications
3. **Handle missing dependencies**: If a test fails due to missing packages, install them
4. **CloudCompare optional**: Tasks that require CloudCompare can be skipped if not installed - mark as such
5. **Incremental commits**: Each task should result in working, tested code
6. **Update this file**: Mark tasks `[x]` as complete and update "Project Status" section

---

*Last updated: January 2025*
