"""
Shared pytest fixtures for PC-RAI tests.

These fixtures provide consistent test data across all test modules.
"""

import numpy as np
import pytest
from pathlib import Path
from typing import Tuple
import tempfile


# =============================================================================
# Synthetic Point Cloud Fixtures
# =============================================================================

@pytest.fixture
def simple_xyz() -> np.ndarray:
    """Simple 100-point random point cloud."""
    np.random.seed(42)  # Reproducible
    return np.random.uniform(0, 10, (100, 3)).astype(np.float64)


@pytest.fixture
def simple_normals() -> np.ndarray:
    """Simple normals pointing mostly up (horizontal surface)."""
    n = 100
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # All pointing up
    return normals


@pytest.fixture
def varied_normals() -> np.ndarray:
    """Normals with variation for testing roughness."""
    np.random.seed(42)
    n = 100
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    # Add random variation
    normals += np.random.normal(0, 0.3, (n, 3)).astype(np.float32)
    # Renormalize to unit vectors
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    return normals


@pytest.fixture
def large_point_cloud() -> Tuple[np.ndarray, np.ndarray]:
    """Larger point cloud (10K points) for performance testing."""
    np.random.seed(42)
    n = 10000
    xyz = np.random.uniform(0, 100, (n, 3)).astype(np.float64)
    
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    normals += np.random.normal(0, 0.2, (n, 3)).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    return xyz, normals


# =============================================================================
# Specific Morphology Fixtures
# =============================================================================

@pytest.fixture
def horizontal_surface() -> Tuple[np.ndarray, np.ndarray]:
    """Flat horizontal surface (slope ≈ 0°)."""
    np.random.seed(42)
    n = 500
    x = np.random.uniform(0, 10, n)
    y = np.random.uniform(0, 10, n)
    z = np.zeros(n) + np.random.normal(0, 0.01, n)  # Nearly flat
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # Pointing straight up
    
    return xyz, normals


@pytest.fixture
def vertical_surface() -> Tuple[np.ndarray, np.ndarray]:
    """Vertical cliff face (slope ≈ 90°)."""
    np.random.seed(42)
    n = 500
    x = np.zeros(n) + np.random.normal(0, 0.01, n)  # Flat in X
    y = np.random.uniform(0, 10, n)
    z = np.random.uniform(0, 10, n)
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 0] = 1.0  # Pointing in +X direction (horizontal)
    
    return xyz, normals


@pytest.fixture
def overhang_surface() -> Tuple[np.ndarray, np.ndarray]:
    """Overhanging surface (slope > 90°)."""
    np.random.seed(42)
    n = 500
    y = np.random.uniform(0, 10, n)
    z = np.random.uniform(0, 10, n)
    # X increases as Z increases (overhang leans out)
    x = z * 0.5 + np.random.normal(0, 0.01, n)
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    # Normal pointing down and outward
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 0] = 0.5   # Outward
    normals[:, 2] = -0.866  # Downward (120° from up)
    
    return xyz, normals


@pytest.fixture
def mixed_morphology() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Point cloud with multiple morphological zones.
    
    Returns xyz, normals, and expected_zones (approximate class hints).
    """
    np.random.seed(42)
    n_per_zone = 200
    
    # Zone 1: Talus (low slope, smooth) - expect class 1
    x1 = np.random.uniform(0, 10, n_per_zone)
    y1 = np.random.uniform(0, 5, n_per_zone)
    z1 = y1 * 0.3  # ~17° slope
    n1 = np.tile([0, -0.3, 0.954], (n_per_zone, 1))  # Pointing mostly up
    
    # Zone 2: Vertical cliff (steep, intact) - expect class 2
    x2 = np.random.uniform(0, 10, n_per_zone)
    y2 = np.full(n_per_zone, 5.0) + np.random.normal(0, 0.02, n_per_zone)
    z2 = np.random.uniform(1, 8, n_per_zone)
    n2 = np.tile([0, 1, 0], (n_per_zone, 1))  # Pointing in +Y
    
    # Zone 3: Rough vertical (discontinuous) - expect class 4 or 5
    x3 = np.random.uniform(0, 10, n_per_zone)
    y3 = np.full(n_per_zone, 5.5) + np.random.normal(0, 0.1, n_per_zone)
    z3 = np.random.uniform(1, 8, n_per_zone)
    # Varied normals for roughness
    n3 = np.random.normal(0, 0.4, (n_per_zone, 3)).astype(np.float32)
    n3[:, 1] += 1.0  # Bias toward +Y
    n3 = n3 / np.linalg.norm(n3, axis=1, keepdims=True)
    
    # Zone 4: Overhang - expect class 6
    x4 = np.random.uniform(0, 10, n_per_zone)
    y4 = np.random.uniform(6, 8, n_per_zone)
    z4 = np.random.uniform(7, 9, n_per_zone)
    n4 = np.tile([0, 0.5, -0.866], (n_per_zone, 1))  # 120° from up
    
    xyz = np.vstack([
        np.column_stack([x1, y1, z1]),
        np.column_stack([x2, y2, z2]),
        np.column_stack([x3, y3, z3]),
        np.column_stack([x4, y4, z4]),
    ]).astype(np.float64)
    
    normals = np.vstack([n1, n2, n3, n4]).astype(np.float32)
    
    # Zone labels (not exact classes, just for reference)
    zones = np.concatenate([
        np.full(n_per_zone, 1),  # Talus zone
        np.full(n_per_zone, 2),  # Intact zone
        np.full(n_per_zone, 3),  # Rough zone
        np.full(n_per_zone, 4),  # Overhang zone
    ])
    
    return xyz, normals, zones


# =============================================================================
# LAS File Fixtures
# =============================================================================

@pytest.fixture
def temp_las_file(tmp_path, simple_xyz) -> Path:
    """Create a temporary LAS file for I/O testing."""
    try:
        import laspy
    except ImportError:
        pytest.skip("laspy not installed")
    
    filepath = tmp_path / "test_cloud.las"
    
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = simple_xyz[:, 0]
    las.y = simple_xyz[:, 1]
    las.z = simple_xyz[:, 2]
    las.write(filepath)
    
    return filepath


@pytest.fixture
def temp_las_with_normals(tmp_path, simple_xyz, varied_normals) -> Path:
    """Create a temporary LAS file with normal vectors."""
    try:
        import laspy
    except ImportError:
        pytest.skip("laspy not installed")
    
    filepath = tmp_path / "test_cloud_normals.las"
    
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = simple_xyz[:, 0]
    las.y = simple_xyz[:, 1]
    las.z = simple_xyz[:, 2]
    
    # Add normals as extra dimensions
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
    
    las["NormalX"] = varied_normals[:, 0]
    las["NormalY"] = varied_normals[:, 1]
    las["NormalZ"] = varied_normals[:, 2]
    
    las.write(filepath)
    
    return filepath


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default RAI configuration."""
    from pc_rai.config import RAIConfig
    return RAIConfig()


@pytest.fixture
def custom_config():
    """Custom configuration for testing overrides."""
    from pc_rai.config import RAIConfig
    return RAIConfig(
        radius_small=0.2,
        radius_large=0.5,
        k_small=20,
        k_large=80,
        thresh_talus_slope=35.0,  # Original threshold
        methods=["radius"],  # Only radius method
    )


# =============================================================================
# Classification Test Fixtures
# =============================================================================

@pytest.fixture
def classification_test_cases():
    """
    Test cases for decision tree classification.
    
    Returns list of (slope, r_small, r_large, expected_class) tuples.
    """
    return [
        # (slope, r_small, r_large, expected_class)
        (30.0, 3.0, 5.0, 1),    # Talus: low slope, low roughness
        (60.0, 3.0, 5.0, 2),    # Intact: steep, low roughness
        (60.0, 20.0, 25.0, 5),  # Dw: high r_small
        (60.0, 15.0, 20.0, 4),  # Dc: moderate r_small
        (60.0, 8.0, 15.0, 3),   # Df: intermediate r_small, high r_large
        (60.0, 8.0, 8.0, 2),    # Intact: intermediate r_small, low r_large
        (120.0, 10.0, 10.0, 6), # Os: shallow overhang
        (160.0, 10.0, 10.0, 7), # Oc: cantilevered overhang
    ]


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# =============================================================================
# Skip Conditions
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_cloudcompare: mark test as requiring CloudCompare"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture
def cloudcompare_available() -> bool:
    """Check if CloudCompare is available on the system."""
    import shutil
    
    # Check common executable names
    for name in ["CloudCompare", "cloudcompare", "CloudCompare.exe"]:
        if shutil.which(name):
            return True
    
    # Check common installation paths
    common_paths = [
        "/usr/bin/CloudCompare",
        "/usr/local/bin/CloudCompare",
        "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare",
        "C:\\Program Files\\CloudCompare\\CloudCompare.exe",
    ]
    for path in common_paths:
        if Path(path).exists():
            return True
    
    return False


@pytest.fixture
def skip_without_cloudcompare(cloudcompare_available):
    """Skip test if CloudCompare is not available."""
    if not cloudcompare_available:
        pytest.skip("CloudCompare not installed")
