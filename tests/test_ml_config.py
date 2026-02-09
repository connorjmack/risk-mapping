"""Tests for ML config module (pc_rai/ml/config.py)."""

from pathlib import Path

import pytest

from pc_rai.ml.config import BeachConfig, MLConfig


class TestMLConfig:
    def test_defaults(self):
        config = MLConfig()
        assert config.min_volume == 5.0
        assert config.n_estimators == 100
        assert config.class_weight == "balanced"
        assert config.cv_n_splits == 5

    def test_custom_values(self):
        config = MLConfig(n_estimators=50, max_depth=10, min_volume=10.0)
        assert config.n_estimators == 50
        assert config.max_depth == 10
        assert config.min_volume == 10.0

    def test_path_conversion(self):
        config = MLConfig(model_output_dir="custom/models")
        assert isinstance(config.model_output_dir, Path)
        assert config.model_output_dir == Path("custom/models")

    def test_default_qc_flags(self):
        config = MLConfig()
        assert "real" in config.qc_flags_include
        assert "unreviewed" in config.qc_flags_include
        assert "construction" in config.qc_flags_exclude
        assert "noise" in config.qc_flags_exclude


class TestBeachConfig:
    def test_path_conversion(self):
        config = BeachConfig(
            name="DelMar",
            events_path="data/events.csv",
            point_cloud_dir="data/clouds",
            transects_path="data/transects.shp",
        )
        assert isinstance(config.events_path, Path)
        assert isinstance(config.point_cloud_dir, Path)
        assert isinstance(config.transects_path, Path)

    def test_optional_alongshore(self):
        config = BeachConfig(
            name="Torrey",
            events_path=Path("a.csv"),
            point_cloud_dir=Path("clouds/"),
            transects_path=Path("t.shp"),
        )
        assert config.alongshore_range is None

    def test_with_alongshore(self):
        config = BeachConfig(
            name="Torrey",
            events_path=Path("a.csv"),
            point_cloud_dir=Path("clouds/"),
            transects_path=Path("t.shp"),
            alongshore_range=(567.0, 581.0),
        )
        assert config.alongshore_range == (567.0, 581.0)
