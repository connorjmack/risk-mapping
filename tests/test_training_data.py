"""Tests for training data assembly module (pc_rai/ml/training_data.py)."""

import numpy as np
import pandas as pd
import pytest

from pc_rai.ml.training_data import (
    balance_controls,
    create_labels,
    get_feature_columns,
    load_polygon_features,
    load_pre_event_surveys,
    match_survey_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def features_df():
    """Synthetic polygon features DataFrame."""
    np.random.seed(42)
    n = 60
    data = {
        "survey_date": ["20230101"] * 30 + ["20230201"] * 30,
        "survey_file": ["scan_a.laz"] * 30 + ["scan_b.laz"] * 30,
        "location": ["DelMar"] * 30 + ["Torrey"] * 30,
        "polygon_id": list(range(10)) * 6,
        "alongshore_m": ([float(i) for i in range(10)]) * 6,
        "zone": (["lower", "middle", "upper"] * 4)[:10] * 6,
        "zone_idx": ([0, 1, 2] * 4)[:10] * 6,
        "n_points": np.random.randint(50, 500, n),
        "z_min": np.random.uniform(0, 5, n),
        "z_max": np.random.uniform(10, 20, n),
        "z_mean": np.random.uniform(5, 15, n),
        "z_range": np.random.uniform(5, 15, n),
    }
    # Feature columns
    for feat in ["slope", "height", "linearity"]:
        for stat in ["mean", "std", "min", "max", "p10", "p50", "p90"]:
            data[f"{feat}_{stat}"] = np.random.uniform(0, 90, n)
    return pd.DataFrame(data)


@pytest.fixture
def surveys_df():
    """Synthetic pre-event surveys DataFrame."""
    return pd.DataFrame({
        "survey_date": ["2023-01-01", "2023-02-01"],
        "survey_file": ["scan_a.laz", "scan_b.laz"],
        "location": ["DelMar", "Torrey"],
        "event_date": ["2023-03-15", "2023-04-20"],
        "event_id": ["evt_001", "evt_002"],
        "event_volume": [25.0, 50.0],
        "days_before": [73, 78],
        "event_alongshore_start": [2.0, 4.0],
        "event_alongshore_end": [5.0, 7.0],
    })


@pytest.fixture
def features_csv(tmp_path, features_df):
    """Write features_df to a temp CSV and return the path."""
    path = tmp_path / "features.csv"
    features_df.to_csv(path, index=False)
    return path


@pytest.fixture
def surveys_csv(tmp_path, surveys_df):
    """Write surveys_df to a temp CSV and return the path."""
    path = tmp_path / "surveys.csv"
    surveys_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_polygon_features
# ---------------------------------------------------------------------------


class TestLoadPolygonFeatures:
    def test_loads_csv(self, features_csv):
        df = load_polygon_features(features_csv, verbose=False)
        assert len(df) == 60
        assert "survey_date" in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_polygon_features(tmp_path / "nope.csv", verbose=False)


# ---------------------------------------------------------------------------
# load_pre_event_surveys
# ---------------------------------------------------------------------------


class TestLoadPreEventSurveys:
    def test_loads_and_filters_by_volume(self, surveys_csv):
        df = load_pre_event_surveys(surveys_csv, min_volume=5.0, verbose=False)
        assert (df["event_volume"] >= 5.0).all()

    def test_volume_filter_excludes_small(self, surveys_csv):
        df = load_pre_event_surveys(surveys_csv, min_volume=30.0, verbose=False)
        assert (df["event_volume"] >= 30.0).all()
        assert len(df) == 1  # Only the 50m³ event

    def test_height_filter_when_column_present(self, tmp_path):
        df = pd.DataFrame({
            "survey_date": ["2023-01-01", "2023-02-01"],
            "survey_file": ["a.laz", "b.laz"],
            "event_volume": [20.0, 30.0],
            "event_elevation": [4.0, 8.0],
            "location": ["A", "B"],
        })
        path = tmp_path / "surveys.csv"
        df.to_csv(path, index=False)
        result = load_pre_event_surveys(path, min_volume=5.0, min_height=6.0, verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["event_elevation"] == 8.0


# ---------------------------------------------------------------------------
# match_survey_files
# ---------------------------------------------------------------------------


class TestMatchSurveyFiles:
    def test_matching_by_date_and_location(self, features_df, surveys_df):
        feat_matched, surv_matched = match_survey_files(
            features_df, surveys_df, verbose=False
        )
        # Both surveys should match
        assert len(feat_matched) > 0
        assert len(surv_matched) > 0
        assert "survey_key" in feat_matched.columns

    def test_no_match_returns_empty(self, features_df, surveys_df):
        # Change survey dates so nothing matches
        surveys_df = surveys_df.copy()
        surveys_df["survey_date"] = ["2099-01-01", "2099-02-01"]
        feat_matched, surv_matched = match_survey_files(
            features_df, surveys_df, verbose=False
        )
        assert len(feat_matched) == 0
        assert len(surv_matched) == 0

    def test_does_not_modify_originals(self, features_df, surveys_df):
        orig_feat_cols = list(features_df.columns)
        match_survey_files(features_df, surveys_df, verbose=False)
        assert list(features_df.columns) == orig_feat_cols


# ---------------------------------------------------------------------------
# create_labels
# ---------------------------------------------------------------------------


class TestCreateLabels:
    def test_cases_labeled_correctly(self, features_df, surveys_df):
        feat_matched, surv_matched = match_survey_files(
            features_df, surveys_df, verbose=False
        )
        labeled = create_labels(feat_matched, surv_matched, verbose=False)
        assert "label" in labeled.columns
        assert set(labeled["label"].unique()).issubset({0, 1})
        # Should have some cases (events span alongshore 2-5 and 4-7)
        assert (labeled["label"] == 1).sum() > 0

    def test_controls_have_nan_event_info(self, features_df, surveys_df):
        feat_matched, surv_matched = match_survey_files(
            features_df, surveys_df, verbose=False
        )
        labeled = create_labels(feat_matched, surv_matched, verbose=False)
        controls = labeled[labeled["label"] == 0]
        assert controls["event_volume"].isna().all()

    def test_cases_have_event_metadata(self, features_df, surveys_df):
        feat_matched, surv_matched = match_survey_files(
            features_df, surveys_df, verbose=False
        )
        labeled = create_labels(feat_matched, surv_matched, verbose=False)
        cases = labeled[labeled["label"] == 1]
        assert (cases["event_volume"] > 0).all()
        assert cases["event_id"].notna().all()

    def test_no_events_yields_all_controls(self, features_df):
        """No events means everything is labeled as control."""
        empty_surveys = pd.DataFrame(
            columns=["survey_key", "event_alongshore_start",
                      "event_alongshore_end", "event_volume",
                      "event_id", "days_before"]
        )
        features_df = features_df.copy()
        features_df["survey_key"] = "dummy"
        labeled = create_labels(features_df, empty_surveys, verbose=False)
        assert (labeled["label"] == 0).all()


# ---------------------------------------------------------------------------
# balance_controls
# ---------------------------------------------------------------------------


class TestBalanceControls:
    def test_equal_ratio(self):
        df = pd.DataFrame({
            "label": [1] * 20 + [0] * 80,
            "feat": np.random.rand(100),
        })
        balanced = balance_controls(df, control_ratio=1.0, verbose=False)
        n_cases = (balanced["label"] == 1).sum()
        n_controls = (balanced["label"] == 0).sum()
        assert n_cases == 20
        assert n_controls == 20

    def test_2x_ratio(self):
        df = pd.DataFrame({
            "label": [1] * 20 + [0] * 80,
            "feat": np.random.rand(100),
        })
        balanced = balance_controls(df, control_ratio=2.0, verbose=False)
        n_cases = (balanced["label"] == 1).sum()
        n_controls = (balanced["label"] == 0).sum()
        assert n_cases == 20
        assert n_controls == 40

    def test_insufficient_controls_uses_all(self):
        df = pd.DataFrame({
            "label": [1] * 20 + [0] * 5,
            "feat": np.random.rand(25),
        })
        balanced = balance_controls(df, control_ratio=1.0, verbose=False)
        n_controls = (balanced["label"] == 0).sum()
        assert n_controls == 5  # All available

    def test_shuffle_is_deterministic(self):
        df = pd.DataFrame({
            "label": [1] * 10 + [0] * 50,
            "feat": np.random.rand(60),
        })
        b1 = balance_controls(df, random_state=42, verbose=False)
        b2 = balance_controls(df, random_state=42, verbose=False)
        pd.testing.assert_frame_equal(b1, b2)

    def test_output_is_shuffled(self):
        df = pd.DataFrame({
            "label": [1] * 50 + [0] * 50,
            "feat": np.random.rand(100),
        })
        balanced = balance_controls(df, control_ratio=1.0, verbose=False)
        # First 50 should NOT all be cases (shuffled)
        first_half_labels = balanced.iloc[:50]["label"]
        assert first_half_labels.nunique() > 1


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------


class TestGetFeatureColumns:
    def test_excludes_metadata(self, features_df):
        feat_cols = get_feature_columns(features_df)
        for col in feat_cols:
            assert not col.startswith("survey_")
            assert not col.startswith("polygon_")
            assert col != "location"
            assert col != "label"

    def test_includes_numeric_features(self, features_df):
        feat_cols = get_feature_columns(features_df)
        assert "slope_mean" in feat_cols
        assert "height_p90" in feat_cols

    def test_returns_sorted(self, features_df):
        feat_cols = get_feature_columns(features_df)
        assert feat_cols == sorted(feat_cols)

    def test_excludes_non_numeric(self):
        df = pd.DataFrame({
            "survey_date": ["2023"],
            "location": ["DelMar"],
            "slope_mean": [45.0],
            "text_col": ["abc"],
        })
        feat_cols = get_feature_columns(df)
        assert "text_col" not in feat_cols
        assert "slope_mean" in feat_cols
