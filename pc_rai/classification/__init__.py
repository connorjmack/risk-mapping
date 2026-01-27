"""Classification module for RAI decision tree, energy calculation, and PCA-based clustering."""

from pc_rai.classification.decision_tree import (
    ClassificationThresholds,
    classify_points,
    get_class_statistics,
    get_class_distribution,
    compare_classifications,
)

from pc_rai.classification.energy import (
    RAIEnergyParams,
    calculate_point_energy,
    calculate_velocity,
    calculate_mass,
    get_energy_statistics,
)

from pc_rai.classification.pca_classifier import (
    PCAClassificationResult,
    classify_pca,
    compare_with_rai,
    get_cluster_interpretation,
)

__all__ = [
    # Decision tree
    "ClassificationThresholds",
    "classify_points",
    "get_class_statistics",
    "get_class_distribution",
    "compare_classifications",
    # Energy calculation
    "RAIEnergyParams",
    "calculate_point_energy",
    "calculate_velocity",
    "calculate_mass",
    "get_energy_statistics",
    # PCA-based
    "PCAClassificationResult",
    "classify_pca",
    "compare_with_rai",
    "get_cluster_interpretation",
]
