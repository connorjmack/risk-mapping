"""
Data preparation for ML training pipeline.

Loads event CSV files and applies filtering based on QC flags and volume thresholds.
"""

from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

from .config import MLConfig


def load_events(
    events_path: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame:
    """Load events from CSV file.

    Parameters
    ----------
    events_path : str or Path
        Path to events CSV file.
    verbose : bool
        Print loading information.

    Returns
    -------
    pd.DataFrame
        Events dataframe with parsed dates.
    """
    events_path = Path(events_path)

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    df = pd.read_csv(events_path)

    # Parse date columns
    date_cols = ["mid_date", "start_date", "end_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if verbose:
        print(f"Loaded {len(df):,} events from {events_path.name}")
        if "qc_flag" in df.columns:
            print(f"  QC flag distribution:")
            for flag, count in df["qc_flag"].value_counts().items():
                print(f"    {flag}: {count:,} ({100*count/len(df):.1f}%)")

    return df


def filter_events(
    df: pd.DataFrame,
    config: Optional[MLConfig] = None,
    min_volume: Optional[float] = None,
    qc_flags_include: Optional[List[str]] = None,
    qc_flags_exclude: Optional[List[str]] = None,
    date_range: Optional[tuple] = None,
    alongshore_range: Optional[tuple] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter events based on various criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Events dataframe.
    config : MLConfig, optional
        Configuration object. If provided, uses config values as defaults.
    min_volume : float, optional
        Minimum event volume in m³.
    qc_flags_include : list of str, optional
        QC flags to include (events with these flags are kept).
    qc_flags_exclude : list of str, optional
        QC flags to exclude (events with these flags are removed).
    date_range : tuple of (start, end), optional
        Filter to events within this date range.
    alongshore_range : tuple of (min, max), optional
        Filter to events within this alongshore range.
    verbose : bool
        Print filtering information.

    Returns
    -------
    pd.DataFrame
        Filtered events dataframe.
    """
    # Use config defaults if provided
    if config is not None:
        min_volume = min_volume if min_volume is not None else config.min_volume
        qc_flags_include = (
            qc_flags_include
            if qc_flags_include is not None
            else config.qc_flags_include
        )
        qc_flags_exclude = (
            qc_flags_exclude
            if qc_flags_exclude is not None
            else config.qc_flags_exclude
        )

    initial_count = len(df)
    filtered = df.copy()

    # Filter by QC flags
    if "qc_flag" in filtered.columns:
        if qc_flags_exclude:
            mask = ~filtered["qc_flag"].isin(qc_flags_exclude)
            excluded_count = (~mask).sum()
            filtered = filtered[mask]
            if verbose and excluded_count > 0:
                print(
                    f"  Excluded {excluded_count:,} events with QC flags: {qc_flags_exclude}"
                )

        if qc_flags_include:
            mask = filtered["qc_flag"].isin(qc_flags_include)
            excluded_count = (~mask).sum()
            filtered = filtered[mask]
            if verbose and excluded_count > 0:
                print(
                    f"  Excluded {excluded_count:,} events not in QC flags: {qc_flags_include}"
                )

    # Filter by minimum volume
    if min_volume is not None and "volume" in filtered.columns:
        mask = filtered["volume"] >= min_volume
        excluded_count = (~mask).sum()
        filtered = filtered[mask]
        if verbose and excluded_count > 0:
            print(f"  Excluded {excluded_count:,} events with volume < {min_volume} m³")

    # Filter by date range
    if date_range is not None and "mid_date" in filtered.columns:
        start_date, end_date = date_range
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        mask = (filtered["mid_date"] >= start_date) & (filtered["mid_date"] <= end_date)
        excluded_count = (~mask).sum()
        filtered = filtered[mask]
        if verbose and excluded_count > 0:
            print(f"  Excluded {excluded_count:,} events outside date range")

    # Filter by alongshore range
    if alongshore_range is not None and "alongshore_centroid_m" in filtered.columns:
        min_along, max_along = alongshore_range
        mask = (filtered["alongshore_centroid_m"] >= min_along) & (
            filtered["alongshore_centroid_m"] <= max_along
        )
        excluded_count = (~mask).sum()
        filtered = filtered[mask]
        if verbose and excluded_count > 0:
            print(f"  Excluded {excluded_count:,} events outside alongshore range")

    if verbose:
        print(
            f"  Final: {len(filtered):,} events "
            f"({100*len(filtered)/initial_count:.1f}% of original)"
        )

    return filtered.reset_index(drop=True)


def get_event_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics for events dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Events dataframe.

    Returns
    -------
    dict
        Summary statistics.
    """
    summary = {
        "n_events": len(df),
    }

    if "mid_date" in df.columns:
        summary["date_range"] = (
            df["mid_date"].min().isoformat(),
            df["mid_date"].max().isoformat(),
        )
        summary["years_covered"] = (
            df["mid_date"].max() - df["mid_date"].min()
        ).days / 365.25

    if "volume" in df.columns:
        summary["volume_stats"] = {
            "min": df["volume"].min(),
            "max": df["volume"].max(),
            "mean": df["volume"].mean(),
            "median": df["volume"].median(),
            "total": df["volume"].sum(),
        }

    if "alongshore_centroid_m" in df.columns:
        summary["alongshore_range"] = (
            df["alongshore_centroid_m"].min(),
            df["alongshore_centroid_m"].max(),
        )
        summary["alongshore_extent"] = (
            df["alongshore_centroid_m"].max() - df["alongshore_centroid_m"].min()
        )

    if "qc_flag" in df.columns:
        summary["qc_distribution"] = df["qc_flag"].value_counts().to_dict()

    if "month" in df.columns:
        summary["monthly_distribution"] = df["month"].value_counts().sort_index().to_dict()

    return summary


def print_event_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of the events dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Events dataframe.
    """
    summary = get_event_summary(df)

    print(f"\n{'='*60}")
    print(f"EVENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total events: {summary['n_events']:,}")

    if "date_range" in summary:
        print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        print(f"Years covered: {summary['years_covered']:.1f}")

    if "volume_stats" in summary:
        vs = summary["volume_stats"]
        print(f"\nVolume statistics (m³):")
        print(f"  Min: {vs['min']:.2f}")
        print(f"  Max: {vs['max']:.2f}")
        print(f"  Mean: {vs['mean']:.2f}")
        print(f"  Median: {vs['median']:.2f}")
        print(f"  Total: {vs['total']:.2f}")

    if "alongshore_range" in summary:
        print(f"\nAlongshore range: {summary['alongshore_range'][0]:.1f}m - {summary['alongshore_range'][1]:.1f}m")
        print(f"Alongshore extent: {summary['alongshore_extent']:.1f}m")

    if "qc_distribution" in summary:
        print(f"\nQC flag distribution:")
        for flag, count in summary["qc_distribution"].items():
            pct = 100 * count / summary["n_events"]
            print(f"  {flag}: {count:,} ({pct:.1f}%)")

    print(f"{'='*60}\n")
