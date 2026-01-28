"""
Configuration module for PC-RAI.

Contains the RAIConfig dataclass with all processing parameters and
class definitions for the simplified 5-class RAI classification scheme.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class RAIConfig:
    """Configuration for RAI processing.

    Parameters
    ----------
    compute_normals : bool
        Whether to compute normals using CloudCompare.
    cloudcompare_path : str
        Path to CloudCompare executable.
    normal_radius : float
        Local radius for normal estimation (meters).
    mst_neighbors : int
        Number of neighbors for MST orientation.
    up_vector : tuple
        Direction considered "up" for slope calculation.
    radius_small : float
        Small-scale radius for roughness (meters). Default 0.175 from Markus et al. 2023.
    radius_large : float
        Large-scale radius for roughness (meters). Default 0.425 from Markus et al. 2023.
    k_small : int
        Number of neighbors for small-scale k-NN roughness.
    k_large : int
        Number of neighbors for large-scale k-NN roughness.
    min_neighbors : int
        Minimum neighbors for valid roughness calculation.
    methods : list
        Roughness methods to use: ["radius"], ["knn"], or ["radius", "knn"].
    classification_smoothing_k : int
        Number of neighbors for majority voting in classification smoothing.
        Higher values produce smoother (less noisy) classification results.
    thresh_overhang : float
        Slope threshold for overhang classification (degrees).
    thresh_cantilever : float
        Slope threshold for cantilevered overhang (degrees).
    thresh_talus_slope : float
        Slope threshold for talus classification (degrees). Default 42 from Markus et al. 2023.
    thresh_r_small_low : float
        Low threshold for small-scale roughness (degrees).
    thresh_r_small_mid : float
        Mid threshold for small-scale roughness (degrees).
    thresh_r_small_high : float
        High threshold for small-scale roughness (degrees).
    thresh_r_large : float
        Threshold for large-scale roughness (degrees).
    output_dir : Path
        Default output directory.
    compress_output : bool
        Whether to compress output as LAZ.
    visualization_dpi : int
        DPI for visualization output.
    visualization_views : list
        List of view angles for visualization.
    """

    # Normal computation
    compute_normals: bool = True
    cloudcompare_path: str = "CloudCompare"
    normal_radius: float = 1.0  # 1.0m for stable plane fitting at 50cm spacing
    mst_neighbors: int = 10

    # Slope
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Roughness - radius method
    # Tuned for 50cm point spacing (~4 pts/m²):
    # - small: 1.0m radius → ~12 points (meter-scale roughness)
    # - large: 2.5m radius → ~80 points (multi-meter roughness)
    radius_small: float = 1.0
    radius_large: float = 2.5

    # Roughness - knn method (preferred for uniform 50cm spacing)
    # k_small=40 ≈ 1.8m neighborhood, k_large=120 ≈ 3.0m neighborhood
    # Larger neighborhoods provide more stable roughness estimates and reduce classification noise
    k_small: int = 40
    k_large: int = 120

    # Shared roughness settings
    min_neighbors: int = 4
    methods: List[str] = field(default_factory=lambda: ["knn"])

    # Classification smoothing
    # k neighbors for majority voting to reduce salt-and-pepper classification noise
    classification_smoothing_k: int = 25

    # Classification thresholds
    # Note: thresh_overhang changed from 90° (Markus et al. 2023) to 80° for
    # California coastal bluffs where steep (not necessarily overhanging) surfaces
    # are mechanically significant. See todo.md for decision tree revision notes.
    thresh_overhang: float = 80.0
    thresh_cantilever: float = 150.0
    thresh_talus_slope: float = 42.0
    thresh_r_small_low: float = 6.0
    thresh_r_small_mid: float = 15.0  # Increased from 11° - more points stay Intact
    thresh_r_small_high: float = 18.0
    thresh_r_large: float = 15.0  # Increased from 12° - more points stay Intact
    # Structure detection: steep slopes with very low roughness are likely seawalls/engineered
    thresh_structure_roughness: float = 2.0  # Below this = structure (very smooth engineered surfaces)

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    compress_output: bool = True
    visualization_dpi: int = 300
    visualization_views: List[str] = field(default_factory=lambda: ["front", "oblique"])


# RAI class definitions - Simplified 5-class scheme
# Merged: Df+Dc+Dw → Discontinuous, Sc+Oc → Steep/Overhang
RAI_CLASS_NAMES: Dict[int, str] = {
    0: "Unclassified",
    1: "Talus",
    2: "Intact",
    3: "Discontinuous",  # Merged Df, Dc, Dw - potential rockfall source
    4: "Steep/Overhang",  # Merged Sc, Oc - high risk steep faces
    5: "Structure",  # Seawalls, riprap, engineered surfaces
}

RAI_CLASS_ABBREV: Dict[int, str] = {
    0: "U",
    1: "T",
    2: "I",
    3: "D",  # Discontinuous
    4: "O",  # Steep/Overhang
    5: "St",  # Structure
}

RAI_CLASS_COLORS: Dict[int, str] = {
    0: "#9E9E9E",  # Gray - Unclassified
    1: "#C8A2C8",  # Light Purple - Talus
    2: "#4CAF50",  # Green - Intact
    3: "#2196F3",  # Blue - Discontinuous
    4: "#FF9800",  # Orange - Steep/Overhang (high risk)
    5: "#795548",  # Brown - Structure
}


def load_config(yaml_path: Path) -> RAIConfig:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML configuration file.

    Returns
    -------
    RAIConfig
        Configuration object with values from file.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file contains invalid configuration.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return RAIConfig()

    # Flatten nested structure if present
    config_dict = _flatten_config(data)

    # Convert output_dir to Path if present
    if "output_dir" in config_dict:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    # Convert up_vector to tuple if present
    if "up_vector" in config_dict:
        config_dict["up_vector"] = tuple(config_dict["up_vector"])

    return RAIConfig(**config_dict)


def save_config(config: RAIConfig, yaml_path: Path) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : RAIConfig
        Configuration object to save.
    yaml_path : Path
        Path to output YAML file.
    """
    # Convert to nested dictionary structure for readability
    data = _unflatten_config(config)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML structure to flat config dict."""
    result = {}

    # Handle nested sections
    section_mappings = {
        "normals": ["compute_normals", "cloudcompare_path", "normal_radius", "mst_neighbors"],
        "slope": ["up_vector"],
        "roughness": {
            "radius": ["radius_small", "radius_large"],
            "knn": ["k_small", "k_large"],
            "shared": ["min_neighbors", "methods"],
        },
        "classification": {
            "thresholds": [
                "thresh_overhang",
                "thresh_cantilever",
                "thresh_talus_slope",
                "thresh_r_small_low",
                "thresh_r_small_mid",
                "thresh_r_small_high",
                "thresh_r_large",
            ],
        },
        "output": ["output_dir", "compress_output", "visualization_dpi", "visualization_views"],
    }

    # Process nested roughness section
    if "roughness" in data:
        roughness = data["roughness"]
        if "radius" in roughness:
            if "small" in roughness["radius"]:
                result["radius_small"] = roughness["radius"]["small"]
            if "large" in roughness["radius"]:
                result["radius_large"] = roughness["radius"]["large"]
        if "knn" in roughness:
            if "small" in roughness["knn"]:
                result["k_small"] = roughness["knn"]["small"]
            if "large" in roughness["knn"]:
                result["k_large"] = roughness["knn"]["large"]
        if "min_neighbors" in roughness:
            result["min_neighbors"] = roughness["min_neighbors"]
        if "methods" in roughness:
            result["methods"] = roughness["methods"]

    # Process nested classification section
    if "classification" in data:
        classification = data["classification"]
        if "thresholds" in classification:
            thresholds = classification["thresholds"]
            for key, value in thresholds.items():
                # Convert key names like "talus_slope" to "thresh_talus_slope"
                if not key.startswith("thresh_"):
                    result[f"thresh_{key}"] = value
                else:
                    result[key] = value

    # Process other top-level keys
    for key, value in data.items():
        if key not in ["roughness", "classification"]:
            if isinstance(value, dict):
                # Flatten simple nested dicts
                for k, v in value.items():
                    result[k] = v
            else:
                result[key] = value

    return result


def _unflatten_config(config: RAIConfig) -> Dict[str, Any]:
    """Convert flat config to nested structure for YAML output."""
    return {
        "normals": {
            "compute_normals": config.compute_normals,
            "cloudcompare_path": config.cloudcompare_path,
            "normal_radius": config.normal_radius,
            "mst_neighbors": config.mst_neighbors,
        },
        "slope": {
            "up_vector": list(config.up_vector),
        },
        "roughness": {
            "radius": {
                "small": config.radius_small,
                "large": config.radius_large,
            },
            "knn": {
                "small": config.k_small,
                "large": config.k_large,
            },
            "min_neighbors": config.min_neighbors,
            "methods": config.methods,
        },
        "classification": {
            "thresholds": {
                "overhang": config.thresh_overhang,
                "cantilever": config.thresh_cantilever,
                "talus_slope": config.thresh_talus_slope,
                "r_small_low": config.thresh_r_small_low,
                "r_small_mid": config.thresh_r_small_mid,
                "r_small_high": config.thresh_r_small_high,
                "r_large": config.thresh_r_large,
            },
        },
        "output": {
            "output_dir": str(config.output_dir),
            "compress_output": config.compress_output,
            "visualization_dpi": config.visualization_dpi,
            "visualization_views": config.visualization_views,
        },
    }
