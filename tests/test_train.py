"""Tests for ML model training module (pc_rai/ml/train.py)."""

import numpy as np
import pandas as pd
import pytest

from pc_rai.ml.config import MLConfig
from pc_rai.ml.train import (
    StabilityModel,
    evaluate_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_data():
    """Synthetic training DataFrame with predictive signal.

    slope_mean is correlated with label so the model can learn something.
    """
    np.random.seed(42)
    n = 200
    labels = np.array([1] * 100 + [0] * 100)

    # Feature columns — cases have higher slope_mean on average
    data = {}
    for feat in ["slope", "height", "linearity"]:
        for stat in ["mean", "std", "min", "max", "p10", "p50", "p90"]:
            col_name = f"{feat}_{stat}"
            if col_name == "slope_mean":
                # Inject signal: cases have higher values
                data[col_name] = np.where(labels == 1,
                                          np.random.uniform(50, 90, n),
                                          np.random.uniform(10, 50, n)).astype(np.float64)
            else:
                data[col_name] = np.random.uniform(0, 90, n).astype(np.float64)

    df = pd.DataFrame(data)
    return df, labels


@pytest.fixture
def small_config():
    """Fast config for testing (few trees, etc.)."""
    return MLConfig(
        n_estimators=10,
        max_depth=3,
        cv_n_splits=2,
        random_state=42,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_returns_stability_model(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        assert isinstance(model, StabilityModel)

    def test_has_feature_names(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        assert model.feature_names == list(X.columns)

    def test_has_feature_importances(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        assert len(model.feature_importances) == len(X.columns)
        # Importances sum to ~1
        total = sum(model.feature_importances.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_cv_metrics_present(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        assert "auc_roc" in model.cv_metrics
        assert "auc_pr" in model.cv_metrics
        assert "fold_metrics" in model.cv_metrics

    def test_auc_above_random(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        # With injected signal, should be better than random
        assert model.cv_metrics["auc_roc"] > 0.5

    def test_train_metadata(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        assert model.train_metadata["n_samples"] == 200
        assert model.train_metadata["n_positive"] == 100
        assert model.train_metadata["n_features"] == len(X.columns)

    def test_with_groups(self, training_data, small_config):
        X, y = training_data
        groups = np.array(["A"] * 100 + ["B"] * 100)
        small_config.cv_n_splits = 2
        model = train_model(X, y, config=small_config, groups=groups, verbose=False)
        assert isinstance(model, StabilityModel)
        assert len(model.cv_metrics["fold_metrics"]) == 2

    def test_with_normalization(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, normalize=True, verbose=False)
        assert model.scaler is not None


# ---------------------------------------------------------------------------
# StabilityModel.predict / predict_proba
# ---------------------------------------------------------------------------


class TestStabilityModelPredict:
    def test_predict_proba_shape(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert proba.min() >= 0
        assert proba.max() <= 1

    def test_predict_binary(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        preds = model.predict(X, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_handles_nan(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        # Inject NaN into features
        X_nan = X.copy()
        X_nan.iloc[0, 0] = np.nan
        # Should not raise
        proba = model.predict_proba(X_nan)
        assert not np.isnan(proba).any()


# ---------------------------------------------------------------------------
# StabilityModel.save / load
# ---------------------------------------------------------------------------


class TestStabilityModelSaveLoad:
    def test_save_and_load_roundtrip(self, training_data, small_config, tmp_path):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)

        loaded = StabilityModel.load(model_path)
        assert loaded.feature_names == model.feature_names
        assert loaded.cv_metrics["auc_roc"] == model.cv_metrics["auc_roc"]

        # Predictions should match
        np.testing.assert_array_equal(
            model.predict_proba(X),
            loaded.predict_proba(X),
        )

    def test_save_creates_json(self, training_data, small_config, tmp_path):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)
        json_path = model_path.with_suffix(".json")
        assert json_path.exists()


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_returns_metrics(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        metrics = evaluate_model(model, X, y, verbose=False)
        assert "auc_roc" in metrics
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics

    def test_accuracy_in_range(self, training_data, small_config):
        X, y = training_data
        model = train_model(X, y, config=small_config, verbose=False)
        metrics = evaluate_model(model, X, y, verbose=False)
        assert 0 <= metrics["accuracy"] <= 1
