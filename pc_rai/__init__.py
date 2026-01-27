"""
PC-RAI: Point Cloud Rockfall Activity Index Classification.

A Python tool for classifying LiDAR point clouds into rockfall hazard
categories using a point cloud-native adaptation of the Rockfall Activity
Index (RAI) methodology.
"""

__version__ = "0.1.0"
__author__ = "connorjmack"

# Import public API
from pc_rai.config import RAIConfig, RAI_CLASS_NAMES, RAI_CLASS_ABBREV, RAI_CLASS_COLORS
from pc_rai.io import PointCloud, load_point_cloud, save_point_cloud
from pc_rai.classifier import RAIClassifier, RAIResult
from pc_rai.classification.energy import RAIEnergyParams, calculate_point_energy

__all__ = [
    "__version__",
    # Config
    "RAIConfig",
    "RAI_CLASS_NAMES",
    "RAI_CLASS_ABBREV",
    "RAI_CLASS_COLORS",
    # I/O
    "PointCloud",
    "load_point_cloud",
    "save_point_cloud",
    # Classifier
    "RAIClassifier",
    "RAIResult",
    # Energy
    "RAIEnergyParams",
    "calculate_point_energy",
]
