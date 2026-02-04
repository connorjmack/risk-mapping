"""
Configuration for ML training pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class MLConfig:
    """Configuration for ML training pipeline.

    Attributes
    ----------
    min_volume : float
        Minimum event volume (m³) to include in training.
    qc_flags_include : list of str
        QC flags to include as positive labels.
    qc_flags_exclude : list of str
        QC flags to exclude entirely.
    transect_half_width : float
        Half-width of transect corridors in meters.

    Feature Aggregation
    -------------------
    aggregation_stats : list of str
        Statistics to compute for each feature.

    Model Hyperparameters
    ---------------------
    n_estimators : int
        Number of trees in Random Forest.
    max_depth : int or None
        Maximum tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    class_weight : str
        How to handle class imbalance.
    random_state : int
        Random seed for reproducibility.
    """

    # Event filtering
    min_volume: float = 5.0  # m³
    qc_flags_include: List[str] = field(
        default_factory=lambda: ["real", "unreviewed"]
    )
    qc_flags_exclude: List[str] = field(
        default_factory=lambda: ["construction", "noise"]
    )

    # Transect configuration
    transect_half_width: float = 5.0  # meters (10m total corridor)

    # Feature aggregation
    aggregation_stats: List[str] = field(
        default_factory=lambda: ["mean", "max", "std", "p90"]
    )

    # Features to extract from point cloud
    point_features: List[str] = field(
        default_factory=lambda: [
            "slope_deg",
            "roughness_small_knn",
            "roughness_large_knn",
        ]
    )

    # Additional derived features
    include_height_features: bool = True
    include_roughness_ratio: bool = True

    # Model hyperparameters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_leaf: int = 5
    min_samples_split: int = 10
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1  # Use all cores

    # Label configuration
    label_type: str = "binary"  # "binary" or "count"

    # Validation
    cv_n_splits: int = 5

    # Output paths
    model_output_dir: Path = field(default_factory=lambda: Path("models"))

    def __post_init__(self):
        """Convert paths to Path objects."""
        if isinstance(self.model_output_dir, str):
            self.model_output_dir = Path(self.model_output_dir)


@dataclass
class BeachConfig:
    """Configuration for a single beach site.

    Attributes
    ----------
    name : str
        Beach identifier (e.g., "DelMar", "Solana").
    events_path : Path
        Path to events CSV file.
    point_cloud_dir : Path
        Directory containing processed point cloud files (*_rai.laz).
    transects_path : Path
        Path to transects shapefile.
    alongshore_range : tuple of float
        (min, max) alongshore position for this beach in the transect coordinate system.
    """

    name: str
    events_path: Path
    point_cloud_dir: Path
    transects_path: Path
    alongshore_range: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """Convert paths to Path objects."""
        if isinstance(self.events_path, str):
            self.events_path = Path(self.events_path)
        if isinstance(self.point_cloud_dir, str):
            self.point_cloud_dir = Path(self.point_cloud_dir)
        if isinstance(self.transects_path, str):
            self.transects_path = Path(self.transects_path)
