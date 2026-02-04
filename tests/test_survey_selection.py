"""Tests for survey selection module."""

import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from pc_rai.ml.survey_selection import (
    _parse_date_from_filename,
    load_survey_catalog,
    find_pre_event_surveys,
    deduplicate_surveys,
    create_pre_event_survey_dataset,
)
from pc_rai.ml.config import MLConfig


class TestParseDateFromFilename:
    """Test date parsing from filenames."""

    def test_standard_format(self):
        """Test standard YYYYMMDD prefix."""
        filename = "20170301_00590_00612_NoWaves_DelMar.las"
        result = _parse_date_from_filename(filename)
        assert result == "2017-03-01"

    def test_different_years(self):
        """Test various years."""
        assert _parse_date_from_filename("20180115_test.las") == "2018-01-15"
        assert _parse_date_from_filename("20251231_test.las") == "2025-12-31"
        assert _parse_date_from_filename("20160101_test.las") == "2016-01-01"

    def test_invalid_date(self):
        """Test invalid date returns None."""
        # Invalid month
        assert _parse_date_from_filename("20171301_test.las") is None
        # Invalid day
        assert _parse_date_from_filename("20170232_test.las") is None

    def test_no_date_prefix(self):
        """Test filename without date prefix."""
        assert _parse_date_from_filename("nodate_test.las") is None
        assert _parse_date_from_filename("test_20170301.las") is None

    def test_short_prefix(self):
        """Test filename with short numeric prefix."""
        assert _parse_date_from_filename("2017030_test.las") is None


class TestLoadSurveyCatalog:
    """Test loading survey catalog."""

    @pytest.fixture
    def sample_catalog(self, tmp_path):
        """Create a sample survey catalog CSV."""
        csv_path = tmp_path / "surveys.csv"
        data = """location,filename,path
DelMar,20170301_test1.las,/path/to/20170301_test1.las
DelMar,20170315_test2.las,/path/to/20170315_test2.las
Solana,20170401_test3.las,/path/to/20170401_test3.las
DelMar,20170501_test4.las,/path/to/20170501_test4.las
"""
        csv_path.write_text(data)
        return csv_path

    def test_load_all(self, sample_catalog):
        """Test loading all surveys."""
        df = load_survey_catalog(sample_catalog, verbose=False)
        assert len(df) == 4
        assert "survey_date" in df.columns
        assert df["survey_date"].notna().all()

    def test_filter_by_location(self, sample_catalog):
        """Test filtering by location."""
        df = load_survey_catalog(sample_catalog, location_filter="DelMar", verbose=False)
        assert len(df) == 3
        assert (df["location"] == "DelMar").all()

    def test_dates_parsed_correctly(self, sample_catalog):
        """Test dates are parsed correctly."""
        df = load_survey_catalog(sample_catalog, verbose=False)
        expected_dates = pd.to_datetime([
            "2017-03-01", "2017-03-15", "2017-04-01", "2017-05-01"
        ])
        # Sort both for comparison
        actual_sorted = sorted(df["survey_date"].tolist())
        expected_sorted = sorted(expected_dates.tolist())
        assert actual_sorted == expected_sorted

    def test_file_not_found(self, tmp_path):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            load_survey_catalog(tmp_path / "nonexistent.csv")


class TestFindPreEventSurveys:
    """Test pre-event survey matching."""

    @pytest.fixture
    def surveys(self):
        """Sample survey dataframe."""
        return pd.DataFrame({
            "survey_date": pd.to_datetime([
                "2017-03-01", "2017-03-15", "2017-04-01", "2017-05-01"
            ]),
            "filename": ["s1.las", "s2.las", "s3.las", "s4.las"],
            "path": ["/path/s1.las", "/path/s2.las", "/path/s3.las", "/path/s4.las"],
        })

    @pytest.fixture
    def events(self):
        """Sample events dataframe."""
        return pd.DataFrame({
            "start_date": pd.to_datetime([
                "2017-03-25",  # Should match s2 (2017-03-15), 10 days before
                "2017-04-10",  # Should match s3 (2017-04-01), 9 days before
                "2017-05-15",  # Should match s4 (2017-05-01), 14 days before
            ]),
            "volume": [10.0, 20.0, 30.0],
            "alongshore_start_m": [100.0, 200.0, 300.0],
            "alongshore_end_m": [110.0, 220.0, 340.0],
        })

    def test_basic_matching(self, surveys, events):
        """Test basic pre-event survey matching."""
        pairs = find_pre_event_surveys(
            surveys, events, min_days_before=7, verbose=False
        )

        assert len(pairs) == 3
        assert (pairs["days_before"] >= 7).all()

    def test_days_before_calculation(self, surveys, events):
        """Test days_before is calculated correctly."""
        pairs = find_pre_event_surveys(
            surveys, events, min_days_before=7, verbose=False
        )

        # First event (2017-03-25) should match survey from 2017-03-15
        first_pair = pairs[pairs["event_date"] == pd.Timestamp("2017-03-25")].iloc[0]
        assert first_pair["days_before"] == 10

    def test_min_days_before_filter(self, surveys, events):
        """Test minimum days filter works."""
        # With min_days_before=14, surveys must be at least 14 days before event
        # Event 2017-03-25: Survey 2017-03-01 is 24 days before ✓
        # Event 2017-04-10: Survey 2017-03-15 is 26 days before ✓
        # Event 2017-05-15: Survey 2017-05-01 is 14 days before ✓
        pairs = find_pre_event_surveys(
            surveys, events, min_days_before=14, verbose=False
        )

        # All events have valid surveys with 14+ days gap
        assert len(pairs) == 3
        assert (pairs["days_before"] >= 14).all()

    def test_min_days_excludes_recent_surveys(self, surveys, events):
        """Test that recent surveys are excluded."""
        # Event 2017-03-25 with survey 2017-03-15 (10 days) - should be excluded with min=11
        pairs = find_pre_event_surveys(
            surveys, events, min_days_before=11, verbose=False
        )

        # First event should use earlier survey (2017-03-01) instead of recent one
        first_pair = pairs[pairs["event_date"] == pd.Timestamp("2017-03-25")].iloc[0]
        assert first_pair["survey_date"] == pd.Timestamp("2017-03-01")
        assert first_pair["days_before"] == 24

    def test_event_with_no_valid_survey(self):
        """Test events before any survey are skipped."""
        surveys = pd.DataFrame({
            "survey_date": pd.to_datetime(["2017-05-01"]),
            "filename": ["s1.las"],
            "path": ["/path/s1.las"],
        })
        events = pd.DataFrame({
            "start_date": pd.to_datetime(["2017-04-01"]),
            "volume": [10.0],
        })

        pairs = find_pre_event_surveys(
            surveys, events, min_days_before=7, verbose=False
        )

        assert len(pairs) == 0

    def test_event_metadata_preserved(self, surveys, events):
        """Test event metadata is preserved in output."""
        pairs = find_pre_event_surveys(
            surveys, events, min_days_before=7, verbose=False
        )

        assert "event_volume" in pairs.columns
        assert "event_alongshore_start" in pairs.columns
        assert "event_alongshore_end" in pairs.columns
        assert pairs["event_volume"].notna().all()


class TestDeduplicateSurveys:
    """Test survey deduplication."""

    def test_deduplication(self):
        """Test duplicate surveys are removed."""
        pairs = pd.DataFrame({
            "survey_date": pd.to_datetime([
                "2017-03-01", "2017-03-01", "2017-04-01"
            ]),
            "survey_path": ["/path/s1.las", "/path/s1.las", "/path/s2.las"],
        })

        unique = deduplicate_surveys(pairs)

        assert len(unique) == 2
        assert "/path/s1.las" in unique
        assert "/path/s2.las" in unique

    def test_sorted_by_date(self):
        """Test output is sorted by date."""
        pairs = pd.DataFrame({
            "survey_date": pd.to_datetime([
                "2017-05-01", "2017-03-01", "2017-04-01"
            ]),
            "survey_path": ["/path/s3.las", "/path/s1.las", "/path/s2.las"],
        })

        unique = deduplicate_surveys(pairs)

        assert unique[0] == "/path/s1.las"
        assert unique[1] == "/path/s2.las"
        assert unique[2] == "/path/s3.las"


class TestCreatePreEventSurveyDataset:
    """Test full pipeline."""

    @pytest.fixture
    def test_data(self, tmp_path):
        """Create test CSV files."""
        # Create surveys CSV
        surveys_csv = tmp_path / "surveys.csv"
        surveys_csv.write_text("""location,filename,path
DelMar,20170301_test1.las,/path/to/20170301_test1.las
DelMar,20170315_test2.las,/path/to/20170315_test2.las
DelMar,20170401_test3.las,/path/to/20170401_test3.las
""")

        # Create events CSV
        events_csv = tmp_path / "events.csv"
        events_csv.write_text("""start_date,mid_date,end_date,volume,qc_flag,alongshore_start_m,alongshore_end_m
2017-03-25,2017-03-26,2017-03-27,10.0,real,100,110
2017-04-10,2017-04-11,2017-04-12,3.0,real,200,220
2017-04-15,2017-04-16,2017-04-17,15.0,construction,300,340
""")

        output_csv = tmp_path / "output.csv"

        return surveys_csv, events_csv, output_csv

    def test_full_pipeline(self, test_data):
        """Test full pipeline execution."""
        surveys_csv, events_csv, output_csv = test_data

        config = MLConfig(
            min_volume=5.0,
            qc_flags_exclude=["construction", "noise"],
        )

        pairs, unique = create_pre_event_survey_dataset(
            surveys_csv=surveys_csv,
            events_csv=events_csv,
            output_path=output_csv,
            location="DelMar",
            config=config,
            min_days_before=7,
            verbose=False,
        )

        # Should have 1 event (10.0 m³ real) that passes filters
        # The 3.0 m³ event is below min_volume, 15.0 m³ is construction
        assert len(pairs) == 1
        assert output_csv.exists()

    def test_output_file_created(self, test_data):
        """Test output CSV is created."""
        surveys_csv, events_csv, output_csv = test_data

        create_pre_event_survey_dataset(
            surveys_csv=surveys_csv,
            events_csv=events_csv,
            output_path=output_csv,
            verbose=False,
        )

        assert output_csv.exists()

        # Verify it's readable
        df = pd.read_csv(output_csv)
        assert "survey_date" in df.columns
        assert "event_date" in df.columns
        assert "days_before" in df.columns


class TestWithRealData:
    """Tests that use actual project data files (skipped if not available)."""

    @pytest.fixture
    def real_surveys_path(self):
        """Path to real survey catalog."""
        path = Path("utiliies/file_lists/all_noveg_files.csv")
        if not path.exists():
            pytest.skip("Real survey catalog not available")
        return path

    @pytest.fixture
    def real_events_path(self):
        """Path to real events file."""
        # Find any events file
        events_dir = Path("utiliies/events")
        if not events_dir.exists():
            pytest.skip("Events directory not available")

        events_files = list(events_dir.glob("*events*.csv"))
        if not events_files:
            pytest.skip("No events files available")

        return events_files[0]

    def test_load_real_surveys(self, real_surveys_path):
        """Test loading real survey catalog."""
        df = load_survey_catalog(real_surveys_path, verbose=False)

        assert len(df) > 0
        assert df["survey_date"].notna().all()
        assert "location" in df.columns

    def test_load_real_surveys_filter_location(self, real_surveys_path):
        """Test filtering real surveys by location."""
        df = load_survey_catalog(
            real_surveys_path, location_filter="DelMar", verbose=False
        )

        assert len(df) > 0
        assert (df["location"] == "DelMar").all()
