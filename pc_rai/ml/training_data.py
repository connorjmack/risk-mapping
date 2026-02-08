"""
Assemble training data by joining polygon features with event labels.

Creates a case-control dataset where:
- Cases: Polygon-zones that had subsequent rockfall events
- Controls: Polygon-zones with no subsequent events

The temporal alignment ensures we're using pre-failure morphology
to predict future events (not describing past failures).
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_polygon_features(
    features_path: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame:
    """Load polygon features CSV.

    Parameters
    ----------
    features_path : str or Path
        Path to polygon features CSV.
    verbose : bool
        Print summary.

    Returns
    -------
    df : pd.DataFrame
        Polygon features with survey metadata.
    """
    df = pd.read_csv(features_path)

    if verbose:
        print(f"Loaded polygon features: {features_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Surveys: {df['survey_file'].nunique()}")
        print(f"  Locations: {df['location'].nunique()}")

    return df


def load_pre_event_surveys(
    surveys_path: Union[str, Path],
    min_volume: float = 5.0,
    min_height: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load pre-event survey matches.

    Parameters
    ----------
    surveys_path : str or Path
        Path to pre-event surveys CSV.
    min_volume : float
        Minimum event volume to include.
    min_height : float, optional
        Minimum event elevation in meters. If provided, only events
        at or above this elevation are included (e.g., 6.0 for
        upper-cliff focus).
    verbose : bool
        Print summary.

    Returns
    -------
    df : pd.DataFrame
        Survey-event pairs.
    """
    df = pd.read_csv(surveys_path)
    n_total = len(df)

    # Filter by volume
    df = df[df['event_volume'] >= min_volume].copy()

    # Filter by elevation if specified
    if min_height is not None and 'event_elevation' in df.columns:
        n_before_height = len(df)
        df = df[df['event_elevation'] >= min_height].copy()
        n_height_filtered = n_before_height - len(df)
    else:
        n_height_filtered = 0

    if verbose:
        print(f"Loaded pre-event surveys: {surveys_path}")
        print(f"  Total survey-event pairs: {n_total:,}")
        print(f"  After volume >= {min_volume}m³: {len(df) + n_height_filtered:,}")
        if min_height is not None:
            print(f"  After elevation >= {min_height}m: {len(df):,}")
        print(f"  Unique surveys: {df['survey_file'].nunique()}")

    return df


def match_survey_files(
    features_df: pd.DataFrame,
    surveys_df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match survey files between features and events.

    Survey filenames may differ slightly (e.g., _noveg vs _subsampled_features).
    This function matches them by date and location.

    Parameters
    ----------
    features_df : pd.DataFrame
        Polygon features.
    surveys_df : pd.DataFrame
        Pre-event surveys.
    verbose : bool
        Print matching summary.

    Returns
    -------
    features_matched : pd.DataFrame
        Features with normalized survey key.
    surveys_matched : pd.DataFrame
        Surveys with normalized survey key.
    """
    # Extract date from survey files (first 8 chars)
    features_df = features_df.copy()
    surveys_df = surveys_df.copy()

    features_df['survey_key'] = features_df['survey_date'].astype(str) + '_' + features_df['location']
    surveys_df['survey_key'] = surveys_df['survey_date'].str.replace('-', '') + '_' + surveys_df['location']

    # Find common survey keys
    feature_keys = set(features_df['survey_key'].unique())
    survey_keys = set(surveys_df['survey_key'].unique())
    common_keys = feature_keys & survey_keys

    if verbose:
        print(f"\nSurvey matching:")
        print(f"  Feature surveys: {len(feature_keys)}")
        print(f"  Event surveys: {len(survey_keys)}")
        print(f"  Matched: {len(common_keys)}")

    # Filter to matched only
    features_matched = features_df[features_df['survey_key'].isin(common_keys)].copy()
    surveys_matched = surveys_df[surveys_df['survey_key'].isin(common_keys)].copy()

    return features_matched, surveys_matched


def create_labels(
    features_df: pd.DataFrame,
    surveys_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create case-control labels for polygon features.

    A polygon-zone is labeled as a case (1) if:
    - An event occurred within the polygon's alongshore range
    - The event elevation overlaps with the zone's elevation range

    Parameters
    ----------
    features_df : pd.DataFrame
        Polygon features with survey_key.
    surveys_df : pd.DataFrame
        Pre-event surveys with survey_key.
    verbose : bool
        Print summary.

    Returns
    -------
    labeled_df : pd.DataFrame
        Features with 'label' column (1=case, 0=control).
    """
    features_df = features_df.copy()

    # Initialize all as controls
    features_df['label'] = 0
    features_df['event_volume'] = np.nan
    features_df['event_id'] = None
    features_df['days_before_event'] = np.nan

    n_cases = 0

    # Group events by survey
    for survey_key, survey_events in surveys_df.groupby('survey_key'):
        # Get features for this survey
        survey_mask = features_df['survey_key'] == survey_key

        if survey_mask.sum() == 0:
            continue

        survey_features = features_df[survey_mask]

        # Check each event
        for _, event in survey_events.iterrows():
            event_start = event['event_alongshore_start']
            event_end = event['event_alongshore_end']
            event_elev = event.get('event_elevation', None)

            # Find polygons that overlap with event alongshore range
            alongshore = survey_features['alongshore_m']
            alongshore_overlap = (alongshore >= event_start - 0.5) & (alongshore <= event_end + 0.5)

            if alongshore_overlap.sum() == 0:
                continue

            # If we have elevation info, also check elevation overlap
            if event_elev is not None and not np.isnan(event_elev):
                # Check if event elevation falls within polygon zone
                z_min = survey_features['z_min']
                z_max = survey_features['z_max']
                # Add some tolerance (events can be at zone boundaries)
                elev_overlap = (event_elev >= z_min - 5) & (event_elev <= z_max + 5)
                overlap_mask = alongshore_overlap & elev_overlap
            else:
                overlap_mask = alongshore_overlap

            # Get indices of overlapping features
            overlap_indices = survey_features[overlap_mask].index

            # Label as cases
            features_df.loc[overlap_indices, 'label'] = 1
            features_df.loc[overlap_indices, 'event_volume'] = event['event_volume']
            features_df.loc[overlap_indices, 'event_id'] = event['event_id']
            features_df.loc[overlap_indices, 'days_before_event'] = event['days_before']

            n_cases += len(overlap_indices)

    # Count unique case polygon-zones
    n_case_rows = (features_df['label'] == 1).sum()
    n_control_rows = (features_df['label'] == 0).sum()

    if verbose:
        print(f"\nLabeling results:")
        print(f"  Cases (label=1): {n_case_rows:,}")
        print(f"  Controls (label=0): {n_control_rows:,}")
        print(f"  Case ratio: {100 * n_case_rows / len(features_df):.1f}%")

    return features_df


def balance_controls(
    df: pd.DataFrame,
    control_ratio: float = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Downsample controls to balance with cases.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled features.
    control_ratio : float
        Ratio of controls to cases (1.0 = equal, 2.0 = 2x controls).
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Print summary.

    Returns
    -------
    balanced_df : pd.DataFrame
        Balanced dataset.
    """
    cases = df[df['label'] == 1]
    controls = df[df['label'] == 0]

    n_cases = len(cases)
    n_controls_target = int(n_cases * control_ratio)

    if n_controls_target >= len(controls):
        # Not enough controls, use all
        balanced = pd.concat([cases, controls], ignore_index=True)
        if verbose:
            print(f"\nBalancing: Using all {len(controls)} controls (target was {n_controls_target})")
    else:
        # Downsample controls
        controls_sampled = controls.sample(n=n_controls_target, random_state=random_state)
        balanced = pd.concat([cases, controls_sampled], ignore_index=True)
        if verbose:
            print(f"\nBalancing: Sampled {n_controls_target} controls from {len(controls)}")

    # Shuffle
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if verbose:
        print(f"  Final dataset: {len(balanced)} rows ({len(cases)} cases, {len(balanced) - len(cases)} controls)")

    return balanced


def assemble_training_data(
    features_path: Union[str, Path],
    surveys_path: Union[str, Path],
    output_path: Union[str, Path],
    min_volume: float = 5.0,
    min_height: Optional[float] = None,
    control_ratio: float = 1.0,
    balance: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full pipeline to assemble training data.

    Parameters
    ----------
    features_path : str or Path
        Path to polygon features CSV.
    surveys_path : str or Path
        Path to pre-event surveys CSV.
    output_path : str or Path
        Output path for training data CSV.
    min_volume : float
        Minimum event volume to include.
    min_height : float, optional
        Minimum event elevation in meters.
    control_ratio : float
        Ratio of controls to cases for balancing.
    balance : bool
        Whether to balance controls with cases.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    df : pd.DataFrame
        Training dataset.
    """
    # Load data
    features_df = load_polygon_features(features_path, verbose=verbose)
    surveys_df = load_pre_event_surveys(surveys_path, min_volume=min_volume,
                                         min_height=min_height, verbose=verbose)

    # Match survey files
    features_matched, surveys_matched = match_survey_files(features_df, surveys_df, verbose=verbose)

    if len(features_matched) == 0:
        raise ValueError("No matching surveys found between features and events")

    # Create labels
    labeled_df = create_labels(features_matched, surveys_matched, verbose=verbose)

    # Balance if requested
    if balance:
        final_df = balance_controls(labeled_df, control_ratio=control_ratio,
                                     random_state=random_state, verbose=verbose)
    else:
        final_df = labeled_df

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nSaved training data: {output_path}")
        print(f"  Total rows: {len(final_df):,}")

    return final_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns for training.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.

    Returns
    -------
    feature_cols : list
        List of feature column names.
    """
    # Exclude metadata and label columns
    exclude_prefixes = ['survey_', 'polygon_', 'alongshore', 'zone', 'event_',
                        'label', 'n_points', 'days_before']
    exclude_exact = ['location', 'z_min', 'z_max', 'z_mean', 'z_range', 'year']

    feature_cols = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue
        # Check if numeric
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)

    return sorted(feature_cols)
