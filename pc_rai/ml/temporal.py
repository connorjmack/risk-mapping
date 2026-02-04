"""
Temporal alignment for ML training.

Implements case-control study design where:
- Cases: pre-failure morphology features from scans taken before events
- Controls: features from transects without subsequent events

This ensures we're training on predictive features (pre-failure state)
rather than descriptive features (post-failure state).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .labels import TransectLabeler
from .features import TransectFeatureExtractor, PointCloudFeatures


@dataclass
class TemporalSample:
    """A single training sample with temporal context.

    Attributes
    ----------
    transect_id : int
        Transect identifier.
    scan_date : pd.Timestamp
        Date of the point cloud scan.
    event_date : pd.Timestamp or None
        Date of subsequent event (None for controls).
    days_to_event : int or None
        Days between scan and event (None for controls).
    features : dict
        Feature values for this sample.
    label : int
        1 for case (pre-failure), 0 for control.
    event_volume : float or None
        Volume of the event (None for controls).
    """
    transect_id: int
    scan_date: pd.Timestamp
    event_date: Optional[pd.Timestamp]
    days_to_event: Optional[int]
    features: Dict[str, float]
    label: int
    event_volume: Optional[float] = None


class TemporalAligner:
    """Aligns events to pre-failure scans for case-control training.

    Parameters
    ----------
    point_cloud_dir : str or Path
        Directory containing point cloud files.
    labeler : TransectLabeler
        Transect labeler with loaded transects.
    half_width : float
        Half-width of transect corridors in meters.
    lookforward_days : int
        Maximum days to look forward for events after each scan.
    min_days_before : int
        Minimum days before event that scan must occur.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        point_cloud_dir: Union[str, Path],
        labeler: TransectLabeler,
        half_width: float = 5.0,
        lookforward_days: int = 365,
        min_days_before: int = 7,
        verbose: bool = True,
    ):
        self.point_cloud_dir = Path(point_cloud_dir)
        self.labeler = labeler
        self.half_width = half_width
        self.lookforward_days = lookforward_days
        self.min_days_before = min_days_before
        self.verbose = verbose

        # Create feature extractor
        self.extractor = TransectFeatureExtractor(
            labeler=labeler,
            half_width=half_width,
            verbose=False,  # Suppress per-file output
        )

        # Cache for loaded features (scan_date -> features_df)
        self._feature_cache: Dict[pd.Timestamp, pd.DataFrame] = {}

        # Scan inventory
        self.scan_dates: List[pd.Timestamp] = []
        self.scan_files: Dict[pd.Timestamp, Path] = {}

    def discover_scans(self, pattern: str = "*_rai.laz") -> int:
        """Discover available point cloud scans.

        Parameters
        ----------
        pattern : str
            Glob pattern for finding scan files.

        Returns
        -------
        int
            Number of scans discovered.
        """
        scan_files = list(self.point_cloud_dir.glob(pattern))

        for f in scan_files:
            # Parse date from filename (assuming YYYYMMDD prefix)
            try:
                date_str = f.stem[:8]
                scan_date = pd.to_datetime(date_str, format="%Y%m%d")
                self.scan_dates.append(scan_date)
                self.scan_files[scan_date] = f
            except Exception:
                if self.verbose:
                    print(f"  Warning: Could not parse date from {f.name}")
                continue

        # Sort by date
        self.scan_dates.sort()

        if self.verbose:
            print(f"Discovered {len(self.scan_dates)} scans")
            if self.scan_dates:
                print(f"  Date range: {self.scan_dates[0].strftime('%Y-%m-%d')} to "
                      f"{self.scan_dates[-1].strftime('%Y-%m-%d')}")

        return len(self.scan_dates)

    def find_pre_event_scan(
        self,
        event_date: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        """Find the most recent scan before an event.

        Parameters
        ----------
        event_date : pd.Timestamp
            Date of the event.

        Returns
        -------
        pd.Timestamp or None
            Date of the most recent pre-event scan, or None if no valid scan.
        """
        # Find scans that are at least min_days_before the event
        max_scan_date = event_date - pd.Timedelta(days=self.min_days_before)

        valid_scans = [d for d in self.scan_dates if d <= max_scan_date]

        if not valid_scans:
            return None

        # Return the most recent valid scan
        return max(valid_scans)

    def get_features_for_scan(
        self,
        scan_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Get or compute features for a scan date.

        Parameters
        ----------
        scan_date : pd.Timestamp
            Date of the scan.

        Returns
        -------
        pd.DataFrame
            Transect-level features for this scan.
        """
        if scan_date in self._feature_cache:
            return self._feature_cache[scan_date]

        scan_file = self.scan_files[scan_date]

        if self.verbose:
            print(f"  Extracting features from {scan_file.name}...")

        features = self.extractor.extract_features(scan_file)
        features["scan_date"] = scan_date

        self._feature_cache[scan_date] = features

        return features

    def create_case_control_dataset(
        self,
        events: pd.DataFrame,
        control_ratio: float = 1.0,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Create a temporally-aligned case-control dataset.

        For each event, finds the most recent pre-event scan and extracts
        features at the event location (case). Then samples control transects
        from the same scan that didn't have events in the lookforward window.

        Parameters
        ----------
        events : pd.DataFrame
            Filtered events dataframe.
        control_ratio : float
            Ratio of controls to cases (1.0 = equal numbers).
        random_state : int
            Random seed for control sampling.

        Returns
        -------
        pd.DataFrame
            Training dataset with features, labels, and metadata.
        """
        if not self.scan_dates:
            raise ValueError("No scans discovered. Call discover_scans() first.")

        np.random.seed(random_state)

        # Ensure events have parsed dates
        if not pd.api.types.is_datetime64_any_dtype(events["mid_date"]):
            events = events.copy()
            events["mid_date"] = pd.to_datetime(events["mid_date"])

        cases = []
        controls_pool = []

        # Track which transects have events by scan date
        scan_event_transects: Dict[pd.Timestamp, set] = {d: set() for d in self.scan_dates}

        if self.verbose:
            print(f"\nProcessing {len(events)} events...")

        n_events_matched = 0
        n_events_no_scan = 0

        for event_idx, event in events.iterrows():
            event_date = event["mid_date"]

            # Find pre-event scan
            scan_date = self.find_pre_event_scan(event_date)

            if scan_date is None:
                n_events_no_scan += 1
                continue

            n_events_matched += 1

            # Get features for this scan
            features_df = self.get_features_for_scan(scan_date)

            # Find transects that overlap with this event
            alongshore_start = event["alongshore_start_m"]
            alongshore_end = event["alongshore_end_m"]
            overlapping_transects = self.labeler.find_overlapping_transects(
                alongshore_start, alongshore_end
            )

            # Mark these transects as having events for this scan
            for t_idx in overlapping_transects:
                transect_id = self.labeler.transects[t_idx].transect_id
                scan_event_transects[scan_date].add(transect_id)

            # Extract case features for overlapping transects
            days_to_event = (event_date - scan_date).days

            for t_idx in overlapping_transects:
                transect_id = self.labeler.transects[t_idx].transect_id

                # Get features for this transect
                transect_features = features_df[features_df["transect_id"] == transect_id]

                if len(transect_features) == 0:
                    continue

                # Create case record
                case_record = transect_features.iloc[0].to_dict()
                case_record["label"] = 1
                case_record["event_date"] = event_date
                case_record["days_to_event"] = days_to_event
                case_record["event_volume"] = event.get("volume", np.nan)
                case_record["event_idx"] = event_idx

                cases.append(case_record)

        if self.verbose:
            print(f"  Matched {n_events_matched} events to pre-event scans")
            print(f"  {n_events_no_scan} events had no valid pre-event scan")
            print(f"  Created {len(cases)} case samples")

        # Sample controls from transects without events
        if self.verbose:
            print(f"\nSampling controls (ratio={control_ratio})...")

        n_controls_target = int(len(cases) * control_ratio)

        for scan_date in self.scan_dates:
            features_df = self.get_features_for_scan(scan_date)
            event_transects = scan_event_transects[scan_date]

            # Find transects without events for this scan
            control_candidates = features_df[
                ~features_df["transect_id"].isin(event_transects) &
                (features_df["point_count"] > 0)  # Only transects with data
            ]

            if len(control_candidates) > 0:
                for _, row in control_candidates.iterrows():
                    control_record = row.to_dict()
                    control_record["label"] = 0
                    control_record["event_date"] = None
                    control_record["days_to_event"] = None
                    control_record["event_volume"] = None
                    control_record["event_idx"] = None

                    controls_pool.append(control_record)

        # Sample controls to match ratio
        if len(controls_pool) > n_controls_target:
            sampled_controls = np.random.choice(
                len(controls_pool), size=n_controls_target, replace=False
            )
            controls = [controls_pool[i] for i in sampled_controls]
        else:
            controls = controls_pool

        if self.verbose:
            print(f"  Sampled {len(controls)} controls from pool of {len(controls_pool)}")

        # Combine cases and controls
        all_records = cases + controls
        dataset = pd.DataFrame(all_records)

        if self.verbose:
            print(f"\nFinal dataset:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Cases (label=1): {(dataset['label'] == 1).sum()}")
            print(f"  Controls (label=0): {(dataset['label'] == 0).sum()}")
            print(f"  Class balance: {dataset['label'].mean()*100:.1f}% positive")

        return dataset


def create_temporal_training_data(
    events: pd.DataFrame,
    point_cloud_dir: Union[str, Path],
    transects_path: Union[str, Path],
    half_width: float = 5.0,
    lookforward_days: int = 365,
    min_days_before: int = 7,
    control_ratio: float = 1.0,
    scan_pattern: str = "*_rai.laz",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, TemporalAligner]:
    """Convenience function to create temporally-aligned training data.

    Parameters
    ----------
    events : pd.DataFrame
        Filtered events dataframe.
    point_cloud_dir : str or Path
        Directory containing point cloud files.
    transects_path : str or Path
        Path to transects shapefile.
    half_width : float
        Half-width of transect corridors in meters.
    lookforward_days : int
        Maximum days to look forward for events.
    min_days_before : int
        Minimum days before event that scan must occur.
    control_ratio : float
        Ratio of controls to cases.
    scan_pattern : str
        Glob pattern for finding scan files.
    verbose : bool
        Print progress information.

    Returns
    -------
    dataset : pd.DataFrame
        Training dataset with features and labels.
    aligner : TemporalAligner
        The aligner object (useful for inspection).
    """
    from .labels import TransectLabeler

    # Load transects
    labeler = TransectLabeler(transects_path, verbose=verbose)

    # Create aligner
    aligner = TemporalAligner(
        point_cloud_dir=point_cloud_dir,
        labeler=labeler,
        half_width=half_width,
        lookforward_days=lookforward_days,
        min_days_before=min_days_before,
        verbose=verbose,
    )

    # Discover scans
    aligner.discover_scans(pattern=scan_pattern)

    # Create dataset
    dataset = aligner.create_case_control_dataset(
        events=events,
        control_ratio=control_ratio,
    )

    return dataset, aligner
