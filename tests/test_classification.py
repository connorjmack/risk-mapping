"""Tests for pc_rai.classification.decision_tree module."""

import numpy as np
import pytest

from pc_rai.classification.decision_tree import classify_points, ClassificationThresholds


@pytest.fixture
def thresholds():
    """Default classification thresholds."""
    return ClassificationThresholds()


def test_talus(thresholds):
    """Low slope + low roughness = Talus (1)."""
    slope = np.array([30.0])  # < 42°
    r_small = np.array([3.0])  # < 6°
    r_large = np.array([5.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 1  # Talus


def test_intact_steep(thresholds):
    """Steep slope + low roughness = Intact (2)."""
    slope = np.array([60.0])  # > 42°
    r_small = np.array([3.0])  # < 6°
    r_large = np.array([5.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 2  # Intact


def test_widely_spaced_discontinuous(thresholds):
    """High r_small = Dw (5)."""
    slope = np.array([60.0])
    r_small = np.array([20.0])  # > 18°
    r_large = np.array([25.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 5  # Dw


def test_closely_spaced_discontinuous(thresholds):
    """Moderate r_small = Dc (4)."""
    slope = np.array([60.0])
    r_small = np.array([15.0])  # 11° < x <= 18°
    r_large = np.array([20.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 4  # Dc


def test_fragmented_discontinuous(thresholds):
    """Intermediate r_small + high r_large = Df (3)."""
    slope = np.array([60.0])
    r_small = np.array([8.0])  # 6° <= x <= 11°
    r_large = np.array([15.0])  # > 12°
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 3  # Df


def test_intact_intermediate(thresholds):
    """Intermediate r_small + low r_large = Intact (2)."""
    slope = np.array([60.0])
    r_small = np.array([8.0])  # 6° <= x <= 11°
    r_large = np.array([8.0])  # <= 12°
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 2  # Intact


def test_shallow_overhang(thresholds):
    """Slope > 90° but <= 150° = Os (6)."""
    slope = np.array([120.0])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 6  # Os


def test_cantilevered_overhang(thresholds):
    """Slope > 150° = Oc (7)."""
    slope = np.array([160.0])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 7  # Oc


def test_overhang_at_boundary(thresholds):
    """Slope exactly at 90° should NOT be overhang."""
    slope = np.array([90.0])
    r_small = np.array([3.0])
    r_large = np.array([5.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] != 6  # Not Os
    assert classes[0] != 7  # Not Oc


def test_cantilever_at_boundary(thresholds):
    """Slope exactly at 150° should be Os, not Oc."""
    slope = np.array([150.0])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 6  # Os (150° is not > 150°)


def test_unclassified_nan_slope(thresholds):
    """NaN slope = Unclassified (0)."""
    slope = np.array([np.nan])
    r_small = np.array([10.0])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 0  # Unclassified


def test_unclassified_nan_r_small(thresholds):
    """NaN r_small = Unclassified (0)."""
    slope = np.array([60.0])
    r_small = np.array([np.nan])
    r_large = np.array([10.0])
    classes = classify_points(slope, r_small, r_large, thresholds)
    assert classes[0] == 0  # Unclassified


def test_unclassified_nan_r_large(thresholds):
    """NaN r_large = Unclassified (0)."""
    slope = np.array([60.0])
    r_small = np.array([10.0])
    r_large = np.array([np.nan])
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


def test_all_classes_possible(thresholds):
    """Test that all classes can be generated."""
    # Create inputs that should produce each class
    test_cases = [
        # (slope, r_small, r_large, expected_class)
        (30.0, 3.0, 5.0, 1),    # Talus
        (60.0, 3.0, 5.0, 2),    # Intact (low rough)
        (60.0, 8.0, 15.0, 3),   # Df
        (60.0, 15.0, 20.0, 4),  # Dc
        (60.0, 20.0, 25.0, 5),  # Dw
        (120.0, 10.0, 10.0, 6), # Os
        (160.0, 10.0, 10.0, 7), # Oc
    ]

    for slope_val, r_small_val, r_large_val, expected in test_cases:
        slope = np.array([slope_val])
        r_small = np.array([r_small_val])
        r_large = np.array([r_large_val])
        classes = classify_points(slope, r_small, r_large, thresholds)
        assert classes[0] == expected, f"Expected class {expected} for slope={slope_val}, r_small={r_small_val}, r_large={r_large_val}, got {classes[0]}"


def test_custom_thresholds():
    """Test with custom threshold values."""
    custom = ClassificationThresholds(
        talus_slope=35.0,  # Original Dunham threshold
    )

    # 35° < slope < 42° with low roughness
    slope = np.array([38.0])
    r_small = np.array([3.0])
    r_large = np.array([5.0])

    # With default thresholds (42°), this would be Talus
    default_classes = classify_points(slope, r_small, r_large)
    assert default_classes[0] == 1  # Talus

    # With custom thresholds (35°), this should be Intact
    custom_classes = classify_points(slope, r_small, r_large, custom)
    assert custom_classes[0] == 2  # Intact


def test_thresholds_from_config():
    """Test creating thresholds from RAIConfig."""
    from pc_rai.config import RAIConfig

    config = RAIConfig(thresh_talus_slope=35.0)
    thresholds = ClassificationThresholds.from_config(config)

    assert thresholds.talus_slope == 35.0
    assert thresholds.overhang == 90.0  # Default


def test_get_class_statistics(thresholds):
    """Test class statistics calculation."""
    from pc_rai.classification.decision_tree import get_class_statistics

    classes = np.array([1, 1, 2, 2, 2, 3, 6, 7])
    stats = get_class_statistics(classes)

    assert stats["total_points"] == 8
    assert stats["classes"][1]["count"] == 2
    assert stats["classes"][2]["count"] == 3
    assert stats["classes"][3]["count"] == 1
    assert stats["classes"][6]["count"] == 1
    assert stats["classes"][7]["count"] == 1
    assert stats["classes"][0]["count"] == 0


def test_get_class_distribution(thresholds):
    """Test class distribution calculation."""
    from pc_rai.classification.decision_tree import get_class_distribution

    classes = np.array([1, 1, 2, 2, 2, 3])
    distribution = get_class_distribution(classes)

    assert distribution[1] == 2
    assert distribution[2] == 3
    assert distribution[3] == 1
    assert 0 not in distribution  # No unclassified


def test_compare_classifications(thresholds):
    """Test classification comparison."""
    from pc_rai.classification.decision_tree import compare_classifications

    classes_a = np.array([1, 2, 3, 4, 5])
    classes_b = np.array([1, 2, 3, 5, 5])  # One disagreement

    comparison = compare_classifications(classes_a, classes_b, "radius", "knn")

    assert comparison["n_total"] == 5
    assert comparison["n_agree"] == 4
    assert comparison["pct_agree"] == 80.0


def test_compare_classifications_mismatched_length():
    """Test that comparison rejects mismatched arrays."""
    from pc_rai.classification.decision_tree import compare_classifications

    classes_a = np.array([1, 2, 3])
    classes_b = np.array([1, 2])

    with pytest.raises(ValueError, match="same length"):
        compare_classifications(classes_a, classes_b)


def test_boundary_r_small_6(thresholds):
    """Test boundary at r_small = 6°."""
    slope = np.array([60.0])
    r_large = np.array([8.0])  # <= 12°

    # r_small < 6 -> Intact (low roughness path)
    r_small_low = np.array([5.9])
    classes_low = classify_points(slope, r_small_low, r_large, thresholds)
    assert classes_low[0] == 2  # Intact

    # r_small >= 6 -> check r_large path
    r_small_high = np.array([6.0])
    classes_high = classify_points(slope, r_small_high, r_large, thresholds)
    assert classes_high[0] == 2  # Intact (r_large <= 12)


def test_boundary_r_small_11(thresholds):
    """Test boundary at r_small = 11°."""
    slope = np.array([60.0])
    r_large = np.array([15.0])

    # r_small <= 11 -> Df (since r_large > 12)
    r_small_low = np.array([11.0])
    classes_low = classify_points(slope, r_small_low, r_large, thresholds)
    assert classes_low[0] == 3  # Df

    # r_small > 11 -> Dc
    r_small_high = np.array([11.1])
    classes_high = classify_points(slope, r_small_high, r_large, thresholds)
    assert classes_high[0] == 4  # Dc


def test_boundary_r_small_18(thresholds):
    """Test boundary at r_small = 18°."""
    slope = np.array([60.0])
    r_large = np.array([15.0])

    # r_small <= 18 -> Dc
    r_small_low = np.array([18.0])
    classes_low = classify_points(slope, r_small_low, r_large, thresholds)
    assert classes_low[0] == 4  # Dc

    # r_small > 18 -> Dw
    r_small_high = np.array([18.1])
    classes_high = classify_points(slope, r_small_high, r_large, thresholds)
    assert classes_high[0] == 5  # Dw
