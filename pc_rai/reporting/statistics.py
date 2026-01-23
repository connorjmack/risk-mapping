"""
Statistics computation for RAI classification results.

Calculates classification statistics, feature statistics, and method agreement.
"""

import numpy as np
from typing import Dict, Optional

from pc_rai.config import RAI_CLASS_NAMES, RAI_CLASS_ABBREV


def calculate_classification_stats(classes: np.ndarray) -> Dict:
    """
    Calculate classification statistics.

    Parameters
    ----------
    classes : np.ndarray
        (N,) array of class codes 0-7.

    Returns
    -------
    dict
        Dictionary with per-class statistics:
        - 'total': Total number of points
        - 'by_class': Dict mapping class code to {'count', 'percent', 'name', 'abbrev'}
    """
    n_total = len(classes)

    stats = {
        "total": n_total,
        "by_class": {},
    }

    for class_code in range(8):
        count = int((classes == class_code).sum())
        percent = 100 * count / n_total if n_total > 0 else 0.0

        stats["by_class"][class_code] = {
            "count": count,
            "percent": round(percent, 2),
            "name": RAI_CLASS_NAMES[class_code],
            "abbrev": RAI_CLASS_ABBREV[class_code],
        }

    return stats


def calculate_feature_stats(
    values: np.ndarray,
    name: str,
    percentiles: tuple = (5, 25, 50, 75, 95),
) -> Dict:
    """
    Calculate summary statistics for a feature array.

    Parameters
    ----------
    values : np.ndarray
        (N,) array of values.
    name : str
        Name of the feature.
    percentiles : tuple
        Percentiles to calculate.

    Returns
    -------
    dict
        Dictionary with statistics:
        - 'name': Feature name
        - 'count': Number of values (excluding NaN)
        - 'nan_count': Number of NaN values
        - 'mean', 'std', 'min', 'max': Basic statistics
        - 'percentiles': Dict mapping percentile to value
    """
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    n_valid = len(valid_values)
    n_nan = len(values) - n_valid

    if n_valid == 0:
        return {
            "name": name,
            "count": 0,
            "nan_count": n_nan,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {p: None for p in percentiles},
        }

    return {
        "name": name,
        "count": n_valid,
        "nan_count": n_nan,
        "mean": round(float(np.mean(valid_values)), 4),
        "std": round(float(np.std(valid_values)), 4),
        "min": round(float(np.min(valid_values)), 4),
        "max": round(float(np.max(valid_values)), 4),
        "percentiles": {
            p: round(float(np.percentile(valid_values, p)), 4) for p in percentiles
        },
    }


def calculate_method_agreement(
    classes_radius: np.ndarray,
    classes_knn: np.ndarray,
) -> Dict:
    """
    Calculate agreement between radius and k-NN classification methods.

    Parameters
    ----------
    classes_radius : np.ndarray
        (N,) array of class codes from radius method.
    classes_knn : np.ndarray
        (N,) array of class codes from k-NN method.

    Returns
    -------
    dict
        Dictionary with agreement statistics:
        - 'agreement_count': Number of points with identical classification
        - 'agreement_pct': Percentage of agreement
        - 'cohens_kappa': Cohen's kappa coefficient
        - 'confusion_matrix': 8x8 confusion matrix (radius x knn)
        - 'per_class_agreement': Per-class agreement rates
    """
    n_total = len(classes_radius)
    if n_total == 0:
        return {
            "agreement_count": 0,
            "agreement_pct": 0.0,
            "cohens_kappa": 0.0,
            "confusion_matrix": np.zeros((8, 8), dtype=np.int64).tolist(),
            "per_class_agreement": {},
        }

    # Count agreements
    agreement_mask = classes_radius == classes_knn
    agreement_count = int(agreement_mask.sum())
    agreement_pct = 100 * agreement_count / n_total

    # Build confusion matrix
    confusion = np.zeros((8, 8), dtype=np.int64)
    for i in range(n_total):
        confusion[classes_radius[i], classes_knn[i]] += 1

    # Calculate Cohen's kappa
    kappa = _calculate_cohens_kappa(confusion)

    # Per-class agreement
    per_class = {}
    for class_code in range(8):
        mask_radius = classes_radius == class_code
        mask_knn = classes_knn == class_code
        n_either = int((mask_radius | mask_knn).sum())
        n_both = int((mask_radius & mask_knn).sum())
        if n_either > 0:
            per_class[class_code] = {
                "name": RAI_CLASS_NAMES[class_code],
                "agreement_pct": round(100 * n_both / n_either, 2),
                "radius_count": int(mask_radius.sum()),
                "knn_count": int(mask_knn.sum()),
            }

    return {
        "agreement_count": agreement_count,
        "agreement_pct": round(agreement_pct, 2),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": confusion.tolist(),
        "per_class_agreement": per_class,
    }


def _calculate_cohens_kappa(confusion: np.ndarray) -> float:
    """
    Calculate Cohen's kappa coefficient from confusion matrix.

    Parameters
    ----------
    confusion : np.ndarray
        (8, 8) confusion matrix.

    Returns
    -------
    float
        Cohen's kappa coefficient (-1 to 1).
    """
    n_total = confusion.sum()
    if n_total == 0:
        return 0.0

    # Observed agreement (diagonal)
    p_o = np.trace(confusion) / n_total

    # Expected agreement by chance
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n_total * n_total)

    # Kappa
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


def calculate_all_statistics(
    slope_deg: np.ndarray,
    roughness_small_radius: Optional[np.ndarray] = None,
    roughness_large_radius: Optional[np.ndarray] = None,
    roughness_small_knn: Optional[np.ndarray] = None,
    roughness_large_knn: Optional[np.ndarray] = None,
    classes_radius: Optional[np.ndarray] = None,
    classes_knn: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calculate all statistics for RAI processing results.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    roughness_small_radius : np.ndarray, optional
        (N,) small-scale roughness from radius method.
    roughness_large_radius : np.ndarray, optional
        (N,) large-scale roughness from radius method.
    roughness_small_knn : np.ndarray, optional
        (N,) small-scale roughness from k-NN method.
    roughness_large_knn : np.ndarray, optional
        (N,) large-scale roughness from k-NN method.
    classes_radius : np.ndarray, optional
        (N,) classification from radius method.
    classes_knn : np.ndarray, optional
        (N,) classification from k-NN method.

    Returns
    -------
    dict
        Dictionary with all statistics.
    """
    stats = {
        "n_points": len(slope_deg),
        "features": {},
        "classification_radius": None,
        "classification_knn": None,
        "method_agreement": None,
    }

    # Feature statistics
    stats["features"]["slope_deg"] = calculate_feature_stats(slope_deg, "slope_deg")

    if roughness_small_radius is not None:
        stats["features"]["roughness_small_radius"] = calculate_feature_stats(
            roughness_small_radius, "roughness_small_radius"
        )
    if roughness_large_radius is not None:
        stats["features"]["roughness_large_radius"] = calculate_feature_stats(
            roughness_large_radius, "roughness_large_radius"
        )
    if roughness_small_knn is not None:
        stats["features"]["roughness_small_knn"] = calculate_feature_stats(
            roughness_small_knn, "roughness_small_knn"
        )
    if roughness_large_knn is not None:
        stats["features"]["roughness_large_knn"] = calculate_feature_stats(
            roughness_large_knn, "roughness_large_knn"
        )

    # Classification statistics
    if classes_radius is not None:
        stats["classification_radius"] = calculate_classification_stats(classes_radius)
    if classes_knn is not None:
        stats["classification_knn"] = calculate_classification_stats(classes_knn)

    # Method agreement
    if classes_radius is not None and classes_knn is not None:
        stats["method_agreement"] = calculate_method_agreement(
            classes_radius, classes_knn
        )

    return stats
