"""
Machine Learning module for PC-RAI stability prediction.

This module provides supervised learning capabilities for predicting
rockfall probability based on point cloud morphology features.

Key components:
- data_prep: Load and filter event data
- polygons: 1m polygon-based spatial matching (polygon IDs = alongshore meters)
- temporal: Temporal alignment for case-control training
- train: Random Forest model training
- predict: Apply trained model to new data

Legacy (10m transects):
- labels: Map events to transect-level labels
- features: Aggregate point cloud features to transects
"""

from .config import MLConfig
from .data_prep import load_events, filter_events
from .polygons import PolygonLabeler, Polygon
from .temporal import TemporalAligner, create_temporal_training_data
from .train import StabilityModel, train_model
from .survey_selection import (
    load_survey_catalog,
    find_pre_event_surveys,
    deduplicate_surveys,
    create_pre_event_survey_dataset,
)
from .feature_extraction import (
    voxel_subsample,
    extract_features,
    compute_eigenvalue_features,
    process_survey,
    process_survey_list,
)
from .polygon_aggregation import (
    aggregate_survey,
    aggregate_survey_batch,
    extract_location,
    extract_all_locations,
    extract_mop_range,
    get_overlapping_locations,
    load_polygons,
    assign_points_to_polygons,
)
from .training_data import (
    assemble_training_data,
    get_feature_columns,
    load_polygon_features,
    load_pre_event_surveys,
    balance_controls,
)

# Legacy transect-based (kept for backwards compatibility)
from .labels import TransectLabeler
from .features import TransectFeatureExtractor

__all__ = [
    "MLConfig",
    "load_events",
    "filter_events",
    "PolygonLabeler",
    "Polygon",
    "TemporalAligner",
    "create_temporal_training_data",
    "StabilityModel",
    "train_model",
    # Survey selection (Step 1)
    "load_survey_catalog",
    "find_pre_event_surveys",
    "deduplicate_surveys",
    "create_pre_event_survey_dataset",
    # Polygon aggregation (Step 3)
    "aggregate_survey",
    "aggregate_survey_batch",
    "extract_location",
    "extract_all_locations",
    "extract_mop_range",
    "get_overlapping_locations",
    "load_polygons",
    "assign_points_to_polygons",
    # Training data assembly (Step 4)
    "assemble_training_data",
    "get_feature_columns",
    "load_polygon_features",
    "load_pre_event_surveys",
    "balance_controls",
    # Legacy
    "TransectLabeler",
    "TransectFeatureExtractor",
]
