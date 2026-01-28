"""
RAI classification decision tree for PC-RAI.

Implements a simplified 5-class Rockfall Activity Index classification based on
slope angle and multi-scale roughness. Adapted from Markus et al. (2023).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pc_rai.config import RAIConfig, RAI_CLASS_NAMES


@dataclass
class ClassificationThresholds:
    """Thresholds for RAI decision tree.

    All angles in degrees.

    Attributes
    ----------
    overhang : float
        Slope threshold for steep/overhang classification (default 80° for coastal bluffs).
    talus_slope : float
        Slope threshold for talus vs steep terrain (default 42° from Markus et al. 2023).
    r_small_low : float
        Low threshold for small-scale roughness (default 6°).
    r_small_mid : float
        Mid threshold for small-scale roughness (default 15°).
        Points with r_small > this go to Discontinuous regardless of r_large.
    r_large : float
        Threshold for large-scale roughness (default 15°).
        Used for moderate r_small (6-15°) to distinguish Intact vs Discontinuous.
    structure_roughness : float
        Maximum roughness for structure detection (default 4°).
        Steep slopes with roughness below this are classified as Structure.
    """

    overhang: float = 80.0
    talus_slope: float = 42.0
    r_small_low: float = 6.0
    r_small_mid: float = 11.0
    r_large: float = 12.0
    structure_roughness: float = 4.0

    @classmethod
    def from_config(cls, config: RAIConfig) -> "ClassificationThresholds":
        """Create thresholds from RAIConfig."""
        return cls(
            overhang=config.thresh_overhang,
            talus_slope=config.thresh_talus_slope,
            r_small_low=config.thresh_r_small_low,
            r_small_mid=config.thresh_r_small_mid,
            r_large=config.thresh_r_large,
            structure_roughness=config.thresh_structure_roughness,
        )


def classify_points(
    slope_deg: np.ndarray,
    r_small: np.ndarray,
    r_large: np.ndarray,
    thresholds: Optional[ClassificationThresholds] = None,
) -> np.ndarray:
    """
    Classify points using simplified 5-class RAI decision tree.

    Decision tree logic (simplified from Markus et al. 2023):
    ```
    if slope > 80°:
        if r_small < 4° → Structure (5)
        else → Steep/Overhang (4)
    elif r_small < 6°:
        if slope < 42° → Talus (1)
        else → Intact (2)
    elif r_small > 15°:
        → Discontinuous (3)
    elif r_large > 15°:
        → Discontinuous (3)
    else:
        → Intact (2)  # moderate r_small, low r_large
    ```

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    r_small : np.ndarray
        (N,) small-scale roughness in degrees.
    r_large : np.ndarray
        (N,) large-scale roughness in degrees.
    thresholds : ClassificationThresholds, optional
        Classification thresholds (uses defaults if None).

    Returns
    -------
    classes : np.ndarray
        (N,) uint8 array of class codes 0-5.

    Class Codes
    -----------
    0 = Unclassified (invalid data)
    1 = Talus (T) - low angle, stable
    2 = Intact (I) - stable cliff face
    3 = Discontinuous (D) - potential rockfall source
    4 = Steep/Overhang (O) - high risk steep faces
    5 = Structure (St) - seawalls, engineered surfaces
    """
    if thresholds is None:
        thresholds = ClassificationThresholds()

    n_points = len(slope_deg)
    classes = np.zeros(n_points, dtype=np.uint8)

    # Identify invalid points (NaN in any input)
    invalid = np.isnan(slope_deg) | np.isnan(r_small) | np.isnan(r_large)

    # Level 1: Steep faces (slope > 80°)
    steep_mask = slope_deg > thresholds.overhang

    # Structure (St): steep + very smooth (likely seawall/engineered)
    structure_mask = steep_mask & (r_small < thresholds.structure_roughness)
    classes[structure_mask] = 5  # Structure

    # Steep/Overhang (O): steep + rough (natural cliff face or overhang)
    steep_overhang_mask = steep_mask & ~structure_mask
    classes[steep_overhang_mask] = 4  # Steep/Overhang

    # Level 2: Non-steep terrain (slope <= 80°)
    non_steep = ~steep_mask

    # Low roughness (r_small < 6°)
    low_roughness = non_steep & (r_small < thresholds.r_small_low)

    # Talus (T): low slope + low roughness
    talus_mask = low_roughness & (slope_deg < thresholds.talus_slope)
    classes[talus_mask] = 1  # Talus

    # Intact (I): steeper slope + low roughness
    intact_low_rough = low_roughness & (slope_deg >= thresholds.talus_slope)
    classes[intact_low_rough] = 2  # Intact

    # Higher roughness (r_small >= 6°)
    higher_roughness = non_steep & (r_small >= thresholds.r_small_low)

    # Discontinuous (D): high r_small (> 11°) OR moderate r_small with high r_large
    discontinuous_mask = higher_roughness & (
        (r_small > thresholds.r_small_mid) |  # High small-scale roughness
        (r_large > thresholds.r_large)         # Or high large-scale roughness
    )
    classes[discontinuous_mask] = 3  # Discontinuous

    # Intact (I): moderate r_small (6-11°) with low r_large (≤ 12°)
    intact_moderate = higher_roughness & ~discontinuous_mask
    classes[intact_moderate] = 2  # Intact

    # Mark invalid points as Unclassified
    classes[invalid] = 0

    return classes


def smooth_classification(
    classes: np.ndarray,
    xyz: np.ndarray,
    k: int = 25,
    min_agreement: float = 0.0,
) -> np.ndarray:
    """
    Smooth classification using majority voting among k nearest neighbors.

    This reduces salt-and-pepper noise in the classification by assigning
    each point to the most common class among its local neighborhood.

    Parameters
    ----------
    classes : np.ndarray
        (N,) uint8 array of class codes from classify_points().
    xyz : np.ndarray
        (N, 3) point coordinates for spatial indexing.
    k : int
        Number of neighbors to consider for voting (default 25).
    min_agreement : float
        Minimum fraction of neighbors that must agree for reassignment.
        If agreement is below this threshold, keep original class.
        Default 0.0 (always use majority vote).

    Returns
    -------
    smoothed_classes : np.ndarray
        (N,) uint8 array of smoothed class codes.

    Notes
    -----
    - Unclassified points (class 0) are excluded from voting but can be
      reassigned if neighbors have a clear majority.
    - This is a spatial smoothing operation that reduces local noise while
      preserving larger-scale classification patterns.
    """
    from scipy.spatial import cKDTree

    n_points = len(classes)
    smoothed = classes.copy()

    # Build spatial index
    tree = cKDTree(xyz)

    # Query k nearest neighbors for all points at once
    _, indices = tree.query(xyz, k=min(k, n_points))

    # For each point, find majority class among neighbors
    for i in range(n_points):
        neighbor_classes = classes[indices[i]]

        # Exclude unclassified from voting (but include current point)
        valid_classes = neighbor_classes[neighbor_classes > 0]

        if len(valid_classes) == 0:
            continue  # No valid neighbors, keep original

        # Count votes for each class
        unique, counts = np.unique(valid_classes, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_class = unique[majority_idx]
        agreement = counts[majority_idx] / len(valid_classes)

        # Only reassign if agreement threshold is met
        if agreement >= min_agreement:
            smoothed[i] = majority_class

    return smoothed


def get_class_statistics(classes: np.ndarray) -> Dict:
    """
    Calculate classification statistics.

    Parameters
    ----------
    classes : np.ndarray
        (N,) array of class codes (0-5).

    Returns
    -------
    stats : dict
        Dictionary with count and percentage for each class.
    """
    n_total = len(classes)
    stats = {
        "total_points": n_total,
        "classes": {},
    }

    for code in range(6):  # 0-5 for simplified 5-class scheme
        count = int(np.sum(classes == code))
        pct = 100.0 * count / n_total if n_total > 0 else 0.0
        name = RAI_CLASS_NAMES.get(code, f"Unknown ({code})")

        stats["classes"][code] = {
            "name": name,
            "count": count,
            "percentage": pct,
        }

    return stats


def get_class_distribution(classes: np.ndarray) -> Dict[int, int]:
    """
    Get simple count distribution of classes.

    Parameters
    ----------
    classes : np.ndarray
        (N,) array of class codes.

    Returns
    -------
    distribution : dict
        Mapping from class code to count.
    """
    unique, counts = np.unique(classes, return_counts=True)
    return {int(code): int(count) for code, count in zip(unique, counts)}


def compare_classifications(
    classes_a: np.ndarray,
    classes_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
) -> Dict:
    """
    Compare two classification results.

    Parameters
    ----------
    classes_a : np.ndarray
        First classification array.
    classes_b : np.ndarray
        Second classification array.
    name_a : str
        Name for first classification.
    name_b : str
        Name for second classification.

    Returns
    -------
    comparison : dict
        Dictionary with agreement metrics.
    """
    if len(classes_a) != len(classes_b):
        raise ValueError("Classification arrays must have same length")

    n_total = len(classes_a)

    # Overall agreement
    agree = classes_a == classes_b
    n_agree = int(np.sum(agree))
    pct_agree = 100.0 * n_agree / n_total if n_total > 0 else 0.0

    # Confusion matrix (6x6 for simplified 5-class scheme + unclassified)
    confusion = np.zeros((6, 6), dtype=np.int32)
    for i in range(n_total):
        # Clip to valid range in case of legacy data
        ca = min(classes_a[i], 5)
        cb = min(classes_b[i], 5)
        confusion[ca, cb] += 1

    # Per-class agreement
    class_agreement = {}
    for code in range(6):
        mask = (classes_a == code) | (classes_b == code)
        if np.sum(mask) > 0:
            both_same = np.sum((classes_a == code) & (classes_b == code))
            class_agreement[code] = {
                "name": RAI_CLASS_NAMES.get(code, f"Unknown ({code})"),
                "count_a": int(np.sum(classes_a == code)),
                "count_b": int(np.sum(classes_b == code)),
                "agreement": int(both_same),
            }

    return {
        "n_total": n_total,
        "n_agree": n_agree,
        "pct_agree": pct_agree,
        "confusion_matrix": confusion.tolist(),
        "class_agreement": class_agreement,
        "name_a": name_a,
        "name_b": name_b,
    }
