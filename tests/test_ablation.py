"""Tests for cumulative feature ablation study module."""

import numpy as np
import pandas as pd
import pytest

from pc_rai.ml.ablation import (
    DEFAULT_FEATURE_ORDER,
    FEATURE_GROUP_COLORS,
    AblationResults,
    AblationStep,
    get_feature_group,
    get_group_columns,
    plot_ablation,
    run_ablation,
)
from pc_rai.ml.config import MLConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ablation_df():
    """Create synthetic training DataFrame with known feature structure.

    9 feature groups x 7 stats = 63 features, 200 rows, binary labels.
    slope_mean is made slightly predictive so AUC > 0.5.
    """
    np.random.seed(42)
    n = 200

    groups = [
        "slope", "height", "linearity", "planarity", "curvature",
        "sphericity", "roughness_small", "roughness_large", "roughness_ratio",
    ]
    stats = ["mean", "std", "min", "max", "p10", "p50", "p90"]

    data = {}
    # Metadata columns
    data["survey_date"] = ["20240101"] * n
    data["survey_file"] = ["test.las"] * n
    data["location"] = np.random.choice(["A", "B", "C"], n)
    data["polygon_id"] = np.arange(n, dtype=float)
    data["alongshore_m"] = np.arange(n, dtype=float)
    data["zone"] = np.random.choice(["lower", "middle", "upper"], n)
    data["zone_idx"] = np.random.choice([0, 1, 2], n)
    data["n_points"] = np.random.randint(50, 500, n)
    data["z_min"] = np.random.uniform(0, 5, n)
    data["z_max"] = np.random.uniform(10, 20, n)
    data["z_mean"] = np.random.uniform(5, 15, n)
    data["z_range"] = np.random.uniform(5, 15, n)

    # Feature columns
    for group in groups:
        for stat in stats:
            data[f"{group}_{stat}"] = np.random.uniform(0, 1, n)

    # Make slope_mean predictive of label
    label_proba = 1 / (1 + np.exp(-(data["slope_mean"] - 0.5) * 4))
    data["label"] = (np.random.random(n) < label_proba).astype(int)
    data["event_volume"] = np.where(
        data["label"], np.random.uniform(5, 50, n), np.nan
    )
    data["event_id"] = [f"evt_{i}" if lbl else None for i, lbl in enumerate(data["label"])]
    data["days_before_event"] = np.where(
        data["label"], np.random.uniform(1, 365, n), np.nan
    )

    return pd.DataFrame(data)


@pytest.fixture
def fast_config():
    """MLConfig with small n_estimators for fast tests."""
    return MLConfig(n_estimators=10, cv_n_splits=3)


# ---------------------------------------------------------------------------
# get_feature_group tests
# ---------------------------------------------------------------------------


class TestGetFeatureGroup:
    """Tests for feature group prefix matching."""

    @pytest.mark.parametrize(
        "col, expected",
        [
            ("slope_mean", "slope"),
            ("slope_p90", "slope"),
            ("height_std", "height"),
            ("linearity_max", "linearity"),
            ("planarity_min", "planarity"),
            ("curvature_p50", "curvature"),
            ("sphericity_mean", "sphericity"),
        ],
    )
    def test_basic_groups(self, col, expected):
        assert get_feature_group(col) == expected

    def test_roughness_large(self):
        assert get_feature_group("roughness_large_p90") == "roughness_large"
        assert get_feature_group("roughness_large_mean") == "roughness_large"

    def test_roughness_small(self):
        assert get_feature_group("roughness_small_mean") == "roughness_small"
        assert get_feature_group("roughness_small_std") == "roughness_small"

    def test_roughness_ratio(self):
        assert get_feature_group("roughness_ratio_std") == "roughness_ratio"
        assert get_feature_group("roughness_ratio_max") == "roughness_ratio"

    def test_unknown_column(self):
        assert get_feature_group("unknown_col") == "other"


# ---------------------------------------------------------------------------
# get_group_columns tests
# ---------------------------------------------------------------------------


class TestGetGroupColumns:
    """Tests for filtering columns by feature group."""

    def test_returns_correct_subset(self, ablation_df):
        from pc_rai.ml.training_data import get_feature_columns

        all_cols = get_feature_columns(ablation_df)
        slope_cols = get_group_columns(all_cols, "slope")
        assert len(slope_cols) == 7
        assert all(c.startswith("slope_") for c in slope_cols)

    def test_roughness_large_not_mixed(self, ablation_df):
        from pc_rai.ml.training_data import get_feature_columns

        all_cols = get_feature_columns(ablation_df)
        rl_cols = get_group_columns(all_cols, "roughness_large")
        assert len(rl_cols) == 7
        assert all("roughness_large" in c for c in rl_cols)
        # None should be roughness_small or roughness_ratio
        assert not any("roughness_small" in c or "roughness_ratio" in c for c in rl_cols)

    def test_empty_for_unknown_group(self, ablation_df):
        from pc_rai.ml.training_data import get_feature_columns

        all_cols = get_feature_columns(ablation_df)
        assert get_group_columns(all_cols, "nonexistent") == []


# ---------------------------------------------------------------------------
# DEFAULT_FEATURE_ORDER tests
# ---------------------------------------------------------------------------


class TestDefaultFeatureOrder:
    """Tests for the default feature ordering constant."""

    def test_has_nine_entries(self):
        assert len(DEFAULT_FEATURE_ORDER) == 9

    def test_covers_all_groups(self):
        expected = {
            "slope", "height", "linearity", "planarity", "curvature",
            "sphericity", "roughness_small", "roughness_large", "roughness_ratio",
        }
        assert set(DEFAULT_FEATURE_ORDER) == expected

    def test_no_duplicates(self):
        assert len(DEFAULT_FEATURE_ORDER) == len(set(DEFAULT_FEATURE_ORDER))


# ---------------------------------------------------------------------------
# run_ablation tests
# ---------------------------------------------------------------------------


class TestRunAblation:
    """Tests for the cumulative ablation runner."""

    def test_returns_correct_steps(self, ablation_df, fast_config):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        assert len(results.steps) == 9

    def test_cumulative_features(self, ablation_df, fast_config):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        for i, step in enumerate(results.steps):
            expected_n = (i + 1) * 7
            assert step.n_features == expected_n, (
                f"Step {i} ({step.feature_group}): expected {expected_n} "
                f"features, got {step.n_features}"
            )

    def test_step_has_finite_metrics(self, ablation_df, fast_config):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        for step in results.steps:
            assert np.isfinite(step.auc_roc), f"Step {step.step}: non-finite AUC-ROC"
            assert np.isfinite(step.auc_pr), f"Step {step.step}: non-finite AUC-PR"
            assert 0.0 <= step.auc_roc <= 1.0
            assert 0.0 <= step.auc_pr <= 1.0

    def test_custom_order(self, ablation_df, fast_config):
        order = ["slope", "height", "linearity"]
        results = run_ablation(
            ablation_df, feature_order=order, config=fast_config, verbose=False
        )
        assert len(results.steps) == 3
        assert results.feature_order == order
        assert results.steps[0].feature_group == "slope"
        assert results.steps[1].feature_group == "height"
        assert results.steps[2].feature_group == "linearity"

    def test_invalid_group_raises(self, ablation_df, fast_config):
        with pytest.raises(ValueError, match="nonexistent"):
            run_ablation(
                ablation_df,
                feature_order=["slope", "nonexistent"],
                config=fast_config,
                verbose=False,
            )

    def test_with_groups(self, ablation_df, fast_config):
        """GroupKFold path runs without error."""
        groups = ablation_df["location"].values
        results = run_ablation(
            ablation_df, config=fast_config, groups=groups, verbose=False
        )
        assert len(results.steps) == 9

    def test_feature_order_stored(self, ablation_df, fast_config):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        assert results.feature_order == DEFAULT_FEATURE_ORDER


# ---------------------------------------------------------------------------
# AblationResults.to_dataframe tests
# ---------------------------------------------------------------------------


class TestAblationResultsToDataframe:
    """Tests for the results-to-DataFrame conversion."""

    def test_correct_shape(self, ablation_df, fast_config):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        df = results.to_dataframe()
        assert len(df) == 9
        assert "step" in df.columns
        assert "feature_group" in df.columns
        assert "n_features" in df.columns
        assert "auc_roc" in df.columns
        assert "auc_pr" in df.columns
        assert "auc_roc_std" in df.columns
        assert "auc_pr_std" in df.columns

    def test_fold_columns_present(self, ablation_df, fast_config):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        df = results.to_dataframe()
        n_folds = fast_config.cv_n_splits
        for i in range(n_folds):
            assert f"auc_roc_fold_{i}" in df.columns
            assert f"auc_pr_fold_{i}" in df.columns


# ---------------------------------------------------------------------------
# plot_ablation tests
# ---------------------------------------------------------------------------


class TestPlotAblation:
    """Tests for the ablation figure generation."""

    def test_creates_file(self, ablation_df, fast_config, tmp_path):
        results = run_ablation(ablation_df, config=fast_config, verbose=False)
        out = tmp_path / "ablation_curve.png"
        plot_ablation(results, out)
        assert out.exists()
        assert out.stat().st_size > 0
