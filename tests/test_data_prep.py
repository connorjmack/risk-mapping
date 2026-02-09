"""Tests for ML data preparation module (pc_rai/ml/data_prep.py)."""

import numpy as np
import pandas as pd
import pytest

from pc_rai.ml.data_prep import (
    filter_events,
    get_event_summary,
    load_events,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def events_df():
    """Synthetic events DataFrame."""
    return pd.DataFrame({
        "mid_date": pd.to_datetime([
            "2020-01-15", "2020-06-20", "2021-03-10",
            "2021-11-05", "2022-02-14",
        ]),
        "start_date": pd.to_datetime([
            "2020-01-10", "2020-06-15", "2021-03-05",
            "2021-11-01", "2022-02-10",
        ]),
        "end_date": pd.to_datetime([
            "2020-01-20", "2020-06-25", "2021-03-15",
            "2021-11-10", "2022-02-18",
        ]),
        "volume": [3.0, 15.0, 50.0, 2.0, 25.0],
        "qc_flag": ["real", "real", "unreviewed", "noise", "construction"],
        "alongshore_centroid_m": [100.0, 200.0, 300.0, 150.0, 250.0],
    })


@pytest.fixture
def events_csv(tmp_path, events_df):
    """Write events_df to a temp CSV."""
    path = tmp_path / "events.csv"
    events_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_events
# ---------------------------------------------------------------------------


class TestLoadEvents:
    def test_loads_csv(self, events_csv):
        df = load_events(events_csv, verbose=False)
        assert len(df) == 5

    def test_parses_dates(self, events_csv):
        df = load_events(events_csv, verbose=False)
        assert pd.api.types.is_datetime64_any_dtype(df["mid_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["start_date"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_events(tmp_path / "missing.csv")


# ---------------------------------------------------------------------------
# filter_events
# ---------------------------------------------------------------------------


class TestFilterEvents:
    def test_volume_filter(self, events_df):
        result = filter_events(events_df, min_volume=10.0, verbose=False)
        assert (result["volume"] >= 10.0).all()
        assert len(result) == 3

    def test_qc_exclude(self, events_df):
        result = filter_events(
            events_df,
            qc_flags_exclude=["noise", "construction"],
            verbose=False,
        )
        assert "noise" not in result["qc_flag"].values
        assert "construction" not in result["qc_flag"].values

    def test_qc_include(self, events_df):
        result = filter_events(
            events_df,
            qc_flags_include=["real"],
            verbose=False,
        )
        assert (result["qc_flag"] == "real").all()

    def test_date_range(self, events_df):
        result = filter_events(
            events_df,
            date_range=("2020-01-01", "2020-12-31"),
            verbose=False,
        )
        assert (result["mid_date"].dt.year == 2020).all()

    def test_alongshore_range(self, events_df):
        result = filter_events(
            events_df,
            alongshore_range=(100, 200),
            verbose=False,
        )
        assert (result["alongshore_centroid_m"] >= 100).all()
        assert (result["alongshore_centroid_m"] <= 200).all()

    def test_combined_filters(self, events_df):
        result = filter_events(
            events_df,
            min_volume=5.0,
            qc_flags_include=["real"],
            verbose=False,
        )
        assert len(result) == 1  # Only the 15m³ real event
        assert result.iloc[0]["volume"] == 15.0

    def test_no_filters_returns_all(self, events_df):
        result = filter_events(events_df, verbose=False)
        assert len(result) == len(events_df)

    def test_resets_index(self, events_df):
        result = filter_events(events_df, min_volume=10.0, verbose=False)
        assert list(result.index) == list(range(len(result)))

    def test_config_defaults(self, events_df):
        """MLConfig defaults are applied when config is passed."""
        from pc_rai.ml.config import MLConfig
        config = MLConfig(min_volume=10.0, qc_flags_exclude=["noise"])
        result = filter_events(events_df, config=config, verbose=False)
        assert (result["volume"] >= 10.0).all()
        assert "noise" not in result["qc_flag"].values


# ---------------------------------------------------------------------------
# get_event_summary
# ---------------------------------------------------------------------------


class TestGetEventSummary:
    def test_basic_summary(self, events_df):
        summary = get_event_summary(events_df)
        assert summary["n_events"] == 5

    def test_volume_stats(self, events_df):
        summary = get_event_summary(events_df)
        assert "volume_stats" in summary
        assert summary["volume_stats"]["min"] == 2.0
        assert summary["volume_stats"]["max"] == 50.0

    def test_date_range(self, events_df):
        summary = get_event_summary(events_df)
        assert "date_range" in summary
        assert summary["years_covered"] > 1

    def test_qc_distribution(self, events_df):
        summary = get_event_summary(events_df)
        assert "qc_distribution" in summary
        assert summary["qc_distribution"]["real"] == 2

    def test_empty_df(self):
        empty = pd.DataFrame(columns=["volume"])
        summary = get_event_summary(empty)
        assert summary["n_events"] == 0
