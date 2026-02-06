"""
Survey selection for ML training pipeline.

Identifies pre-event surveys from a catalog of available LAS files by matching
survey dates to event dates, ensuring temporal alignment for case-control design.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from .config import MLConfig
from .data_prep import filter_events, load_events

logger = logging.getLogger(__name__)


def load_survey_catalog(
    csv_path: Union[str, Path],
    location_filter: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load survey catalog and parse dates from filenames.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file with columns: location, filename, path
    location_filter : str, optional
        If provided, filter to only this location (e.g., "DelMar")
    verbose : bool
        Print loading information.

    Returns
    -------
    pd.DataFrame
        Survey catalog with parsed survey_date column.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Survey catalog not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse survey date from filename (YYYYMMDD prefix)
    df["survey_date"] = df["filename"].apply(_parse_date_from_filename)

    # Filter out rows where date parsing failed
    n_failed = df["survey_date"].isna().sum()
    if n_failed > 0:
        logger.warning(f"Could not parse date from {n_failed} filenames")
        df = df.dropna(subset=["survey_date"])

    # Convert to datetime
    df["survey_date"] = pd.to_datetime(df["survey_date"])

    # Filter by location if specified
    if location_filter is not None:
        df = df[df["location"] == location_filter].copy()

    # Sort by date
    df = df.sort_values("survey_date").reset_index(drop=True)

    if verbose:
        print(f"Loaded {len(df):,} surveys from {csv_path.name}")
        if location_filter:
            print(f"  Filtered to location: {location_filter}")
        print(
            f"  Date range: {df['survey_date'].min().date()} to {df['survey_date'].max().date()}"
        )

    return df


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYYMMDD date from filename prefix.

    Parameters
    ----------
    filename : str
        Filename starting with YYYYMMDD_...

    Returns
    -------
    str or None
        Date string in YYYY-MM-DD format, or None if parsing fails.
    """
    # Match YYYYMMDD at start of filename
    match = re.match(r"^(\d{8})", filename)
    if match:
        date_str = match.group(1)
        try:
            # Validate it's a real date
            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def find_pre_event_surveys(
    surveys: pd.DataFrame,
    events: pd.DataFrame,
    min_days_before: int = 7,
    max_days_before: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find the most recent survey BEFORE each event.

    For each event, finds the survey that occurred closest to (but before)
    the event's start_date, with a minimum gap of min_days_before.

    Parameters
    ----------
    surveys : pd.DataFrame
        Survey catalog with 'survey_date' column.
    events : pd.DataFrame
        Events dataframe with 'start_date' column.
    min_days_before : int
        Minimum days between survey and event start (default: 7).
    max_days_before : int, optional
        Maximum days between survey and event start. If None, no limit.
    verbose : bool
        Print matching information.

    Returns
    -------
    pd.DataFrame
        Survey-event pairs with columns:
        survey_date, survey_file, survey_path, event_date, event_id,
        days_before, event_volume, event_alongshore_start, event_alongshore_end
    """
    if "start_date" not in events.columns:
        raise ValueError("Events dataframe must have 'start_date' column")

    if "survey_date" not in surveys.columns:
        raise ValueError("Surveys dataframe must have 'survey_date' column")

    # Ensure dates are datetime
    surveys = surveys.copy()
    events = events.copy()
    surveys["survey_date"] = pd.to_datetime(surveys["survey_date"])
    events["start_date"] = pd.to_datetime(events["start_date"])

    # Sort surveys by date for efficient lookup
    surveys_sorted = surveys.sort_values("survey_date")

    pairs = []
    n_skipped_no_survey = 0
    n_skipped_too_recent = 0

    for idx, event in events.iterrows():
        event_date = event["start_date"]
        event_id = idx  # Use index as event ID if no explicit ID column

        # Find surveys before this event (with minimum gap)
        cutoff_date = event_date - timedelta(days=min_days_before)
        valid_surveys = surveys_sorted[surveys_sorted["survey_date"] <= cutoff_date]

        # Apply max_days_before filter if specified
        if max_days_before is not None:
            earliest_date = event_date - timedelta(days=max_days_before)
            valid_surveys = valid_surveys[
                valid_surveys["survey_date"] >= earliest_date
            ]

        if len(valid_surveys) == 0:
            n_skipped_no_survey += 1
            logger.debug(
                f"Event {event_id} ({event_date.date()}): no valid pre-event survey"
            )
            continue

        # Get most recent valid survey
        best_survey = valid_surveys.iloc[-1]
        days_before = (event_date - best_survey["survey_date"]).days

        pairs.append(
            {
                "survey_date": best_survey["survey_date"],
                "survey_file": best_survey["filename"],
                "survey_path": best_survey["path"],
                "location": best_survey.get("location", None),
                "event_date": event_date,
                "event_id": event_id,
                "days_before": days_before,
                "event_volume": event.get("volume", None),
                "event_alongshore_start": event.get("alongshore_start_m", None),
                "event_alongshore_end": event.get("alongshore_end_m", None),
                "event_alongshore_centroid": event.get("alongshore_centroid_m", None),
                "event_elevation": event.get("elevation", None),
                "qc_flag": event.get("qc_flag", None),
            }
        )

    result = pd.DataFrame(pairs)

    if verbose:
        print(f"\nPre-event survey matching:")
        print(f"  Events processed: {len(events):,}")
        print(f"  Matched pairs: {len(pairs):,}")
        if n_skipped_no_survey > 0:
            print(f"  Skipped (no valid survey): {n_skipped_no_survey:,}")
        if len(result) > 0:
            print(f"  Days before (median): {result['days_before'].median():.0f}")
            print(
                f"  Days before (range): {result['days_before'].min():.0f} - {result['days_before'].max():.0f}"
            )

    return result


def deduplicate_surveys(survey_event_pairs: pd.DataFrame) -> List[str]:
    """Get unique list of survey files that need feature extraction.

    Parameters
    ----------
    survey_event_pairs : pd.DataFrame
        Output from find_pre_event_surveys()

    Returns
    -------
    List[str]
        Unique survey file paths, sorted by date.
    """
    if "survey_path" in survey_event_pairs.columns:
        # Deduplicate by path
        unique_df = (
            survey_event_pairs.drop_duplicates(subset=["survey_path"])
            .sort_values("survey_date")
        )
        return unique_df["survey_path"].tolist()
    else:
        # Fallback to filename
        unique_df = (
            survey_event_pairs.drop_duplicates(subset=["survey_file"])
            .sort_values("survey_date")
        )
        return unique_df["survey_file"].tolist()


def create_pre_event_survey_dataset(
    surveys_csv: Union[str, Path],
    events_csv: Union[str, Path],
    output_path: Union[str, Path],
    location: Optional[str] = None,
    config: Optional[MLConfig] = None,
    min_days_before: int = 7,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Full pipeline to identify pre-event surveys and save results.

    Parameters
    ----------
    surveys_csv : str or Path
        Path to survey catalog CSV (all_noveg_files.csv).
    events_csv : str or Path
        Path to events CSV.
    output_path : str or Path
        Where to save the survey-event pairs CSV.
    location : str, optional
        Location to filter surveys (e.g., "DelMar").
    config : MLConfig, optional
        Configuration for event filtering.
    min_days_before : int
        Minimum days between survey and event.
    verbose : bool
        Print progress information.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (survey_event_pairs DataFrame, list of unique survey paths)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and filter events
    if verbose:
        print("=" * 60)
        print("STEP 1: Loading and filtering events")
        print("=" * 60)

    events = load_events(events_csv, verbose=verbose)

    if config is None:
        config = MLConfig()

    events_filtered = filter_events(
        events,
        config=config,
        verbose=verbose,
    )

    # Load survey catalog
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Loading survey catalog")
        print("=" * 60)

    surveys = load_survey_catalog(surveys_csv, location_filter=location, verbose=verbose)

    # Find pre-event surveys
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Matching events to pre-event surveys")
        print("=" * 60)

    pairs = find_pre_event_surveys(
        surveys,
        events_filtered,
        min_days_before=min_days_before,
        verbose=verbose,
    )

    # Get unique surveys
    unique_surveys = deduplicate_surveys(pairs)

    if verbose:
        print(f"\nUnique surveys to process: {len(unique_surveys):,}")

    # Save results
    pairs.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved survey-event pairs to: {output_path}")

    return pairs, unique_surveys
