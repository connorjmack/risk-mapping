"""
Machine Learning module for PC-RAI stability prediction.

This module provides supervised learning capabilities for predicting
rockfall probability based on point cloud morphology features.

Key components:
- data_prep: Load and filter event data
- labels: Map events to transect-level labels
- features: Aggregate point cloud features to transects
- train: Random Forest model training
- predict: Apply trained model to new data
"""

from .config import MLConfig
from .data_prep import load_events, filter_events
from .labels import TransectLabeler
from .features import TransectFeatureExtractor
from .train import StabilityModel, train_model

__all__ = [
    "MLConfig",
    "load_events",
    "filter_events",
    "TransectLabeler",
    "TransectFeatureExtractor",
    "StabilityModel",
    "train_model",
]
