"""
Temporal alignment for ML training using 1m polygon shapefiles.

Implements case-control study design where:
- Cases: pre-failure morphology features from scans taken before events
- Controls: features from polygons without subsequent events

This ensures we're training on predictive features (pre-failure state)
rather than descriptive features (post-failure state).

Polygon IDs correspond directly to alongshore meter positions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import laspy

from .polygons import PolygonLabeler, Polygon


@dataclass
class TemporalSample:
    """A single training sample with temporal context."""
    polygon_id: int
    scan_date: pd.Timestamp
    event_date: Optional[pd.Timestamp]
    days_to_event: Optional[int]
    features: Dict[str, float]
    label: int
    event_volume: Optional[float] = None


def compute_polygon_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute aggregated statistics for a set of values."""
    if len(values) == 0:
        return {"mean": np.nan, "max": np.nan, "std": np.nan, "p90": np.nan}
    return {
        "mean": float(np.nanmean(values)),
        "max": float(np.nanmax(values)),
        "std": float(np.nanstd(values)),
        "p90": float(np.nanpercentile(values, 90)),
    }


def extract_all_polygon_features(
    las_path: Path,
    polygon_labeler: PolygonLabeler,
    polygon_ids: Optional[List[int]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract features for multiple polygons from a single scan.

    This is much more efficient than extracting one polygon at a time
    because the point cloud is loaded only once.

    Parameters
    ----------
    las_path : Path
        Path to the LAZ/LAS file.
    polygon_labeler : PolygonLabeler
        Polygon labeler with loaded 1m polygons.
    polygon_ids : List[int], optional
        Specific polygon IDs to extract. If None, extracts all.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Features for each polygon.
    """
    if verbose:
        print(f"  Loading {las_path.name}...")

    las = laspy.read(las_path)

    # Convert to numpy arrays once
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # Get optional feature arrays
    dim_names = [dim.name for dim in las.point_format.dimensions]
    slope = np.array(las.slope) if "slope" in dim_names else None
    r_small = np.array(las.r_small) if "r_small" in dim_names else None
    r_large = np.array(las.r_large) if "r_large" in dim_names else None

    # Determine which polygons to process
    if polygon_ids is None:
        polygons_to_process = polygon_labeler.polygons
    else:
        polygons_to_process = [
            polygon_labeler.polygon_by_id[pid]
            for pid in polygon_ids
            if pid in polygon_labeler.polygon_by_id
        ]

    if verbose:
        print(f"  Extracting features for {len(polygons_to_process)} polygons...")

    results = []

    for polygon in polygons_to_process:
        # Quick bounding box filter first
        in_bbox = (
            (x >= polygon.x_min) & (x <= polygon.x_max) &
            (y >= polygon.y_min) & (y <= polygon.y_max)
        )

        if not in_bbox.any():
            results.append({
                "polygon_id": polygon.polygon_id,
                "point_count": 0,
            })
            continue

        # Check which bbox points are actually inside polygon
        bbox_indices = np.where(in_bbox)[0]
        mask = np.zeros(len(x), dtype=bool)

        for idx in bbox_indices:
            if polygon.contains_point(x[idx], y[idx]):
                mask[idx] = True

        n_points = mask.sum()

        if n_points == 0:
            results.append({
                "polygon_id": polygon.polygon_id,
                "point_count": 0,
            })
            continue

        # Compute features
        record = {
            "polygon_id": polygon.polygon_id,
            "point_count": int(n_points),
        }

        # Height
        h_stats = compute_polygon_stats(z[mask])
        record.update({f"height_{k}": v for k, v in h_stats.items()})

        # Slope
        if slope is not None:
            s_stats = compute_polygon_stats(slope[mask])
            record.update({f"slope_{k}": v for k, v in s_stats.items()})

        # Roughness small
        if r_small is not None:
            rs_stats = compute_polygon_stats(r_small[mask])
            record.update({f"r_small_{k}": v for k, v in rs_stats.items()})

        # Roughness large
        if r_large is not None:
            rl_stats = compute_polygon_stats(r_large[mask])
            record.update({f"r_large_{k}": v for k, v in rl_stats.items()})

        # Roughness ratio
        if r_small is not None and r_large is not None:
            rs_vals = r_small[mask]
            rl_vals = r_large[mask]
            valid = rl_vals > 0
            if valid.sum() > 0:
                r_ratio = np.zeros_like(rs_vals, dtype=np.float64)
                r_ratio[valid] = rs_vals[valid] / rl_vals[valid]
                r_ratio[~valid] = np.nan
                ratio_stats = compute_polygon_stats(r_ratio[valid])
                record.update({f"r_ratio_{k}": v for k, v in ratio_stats.items()})

        results.append(record)

    return pd.DataFrame(results)


class TemporalAligner:
    """Aligns events to pre-failure scans for case-control training.

    Uses 1m polygon shapefiles for precise spatial matching.
    Polygon IDs correspond directly to alongshore meter positions.
    """

    def __init__(
        self,
        point_cloud_dir: Union[str, Path],
        polygon_labeler: PolygonLabeler,
        lookforward_days: int = 365,
        min_days_before: int = 7,
        verbose: bool = True,
    ):
        self.point_cloud_dir = Path(point_cloud_dir)
        self.polygon_labeler = polygon_labeler
        self.lookforward_days = lookforward_days
        self.min_days_before = min_days_before
        self.verbose = verbose

        # Scan inventory
        self.scan_dates: List[pd.Timestamp] = []
        self.scan_files: Dict[pd.Timestamp, Path] = {}

        # Feature cache: scan_date -> DataFrame of polygon features
        self._scan_features: Dict[pd.Timestamp, pd.DataFrame] = {}

    def discover_scans(self, pattern: str = "*_rai.laz") -> int:
        """Discover available point cloud scans."""
        scan_files = list(self.point_cloud_dir.glob(pattern))

        for f in scan_files:
            try:
                date_str = f.stem[:8]
                scan_date = pd.to_datetime(date_str, format="%Y%m%d")
                self.scan_dates.append(scan_date)
                self.scan_files[scan_date] = f
            except Exception:
                if self.verbose:
                    print(f"  Warning: Could not parse date from {f.name}")
                continue

        self.scan_dates.sort()

        if self.verbose:
            print(f"Discovered {len(self.scan_dates)} scans")
            if self.scan_dates:
                print(f"  Date range: {self.scan_dates[0].strftime('%Y-%m-%d')} to "
                      f"{self.scan_dates[-1].strftime('%Y-%m-%d')}")

        return len(self.scan_dates)

    def find_pre_event_scan(self, event_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Find the most recent scan before an event."""
        max_scan_date = event_date - pd.Timedelta(days=self.min_days_before)
        valid_scans = [d for d in self.scan_dates if d <= max_scan_date]
        return max(valid_scans) if valid_scans else None

    def load_scan_features(self, scan_date: pd.Timestamp) -> pd.DataFrame:
        """Load or compute features for all polygons in a scan."""
        if scan_date in self._scan_features:
            return self._scan_features[scan_date]

        scan_file = self.scan_files[scan_date]
        features_df = extract_all_polygon_features(
            scan_file,
            self.polygon_labeler,
            polygon_ids=None,  # Extract all
            verbose=self.verbose,
        )
        features_df["scan_date"] = scan_date

        self._scan_features[scan_date] = features_df
        return features_df

    def create_case_control_dataset(
        self,
        events: pd.DataFrame,
        control_ratio: float = 1.0,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Create a temporally-aligned case-control dataset."""
        if not self.scan_dates:
            raise ValueError("No scans discovered. Call discover_scans() first.")

        np.random.seed(random_state)

        # Ensure events have parsed dates
        if not pd.api.types.is_datetime64_any_dtype(events["mid_date"]):
            events = events.copy()
            events["mid_date"] = pd.to_datetime(events["mid_date"])

        # Group events by their pre-event scan date
        event_scan_mapping = []
        for event_idx, event in events.iterrows():
            scan_date = self.find_pre_event_scan(event["mid_date"])
            if scan_date is not None:
                event_scan_mapping.append({
                    "event_idx": event_idx,
                    "event": event,
                    "scan_date": scan_date,
                })

        if self.verbose:
            print(f"\nMatched {len(event_scan_mapping)} events to pre-event scans")
            print(f"  ({len(events) - len(event_scan_mapping)} events had no valid pre-event scan)")

        # Group by scan date
        events_by_scan = {}
        for mapping in event_scan_mapping:
            scan_date = mapping["scan_date"]
            if scan_date not in events_by_scan:
                events_by_scan[scan_date] = []
            events_by_scan[scan_date].append(mapping)

        cases = []
        scan_event_polygons: Dict[pd.Timestamp, set] = {d: set() for d in self.scan_dates}

        if self.verbose:
            print(f"\nProcessing {len(events_by_scan)} scans with events...")

        # Process each scan that has events
        from tqdm import tqdm
        scan_iter = tqdm(events_by_scan.items(), desc="Processing scans") if self.verbose else events_by_scan.items()

        for scan_date, scan_events in scan_iter:
            # Load features for this scan (loads point cloud once for all polygons)
            features_df = self.load_scan_features(scan_date)

            # Create feature lookup by polygon_id
            features_lookup = features_df.set_index("polygon_id").to_dict("index")

            # Process each event for this scan
            for mapping in scan_events:
                event = mapping["event"]
                event_idx = mapping["event_idx"]
                event_date = event["mid_date"]
                days_to_event = (event_date - scan_date).days

                # Find polygon IDs that overlap with this event
                polygon_ids = self.polygon_labeler.find_polygons_for_event(
                    event["alongshore_start_m"],
                    event["alongshore_end_m"],
                )

                # Mark these polygons as having events
                for pid in polygon_ids:
                    scan_event_polygons[scan_date].add(pid)

                # Create case records
                for pid in polygon_ids:
                    if pid not in features_lookup:
                        continue

                    features = features_lookup[pid]
                    if features.get("point_count", 0) == 0:
                        continue

                    case_record = dict(features)
                    case_record["polygon_id"] = pid
                    case_record["scan_date"] = scan_date
                    case_record["label"] = 1
                    case_record["event_date"] = event_date
                    case_record["days_to_event"] = days_to_event
                    case_record["event_volume"] = event.get("volume", np.nan)
                    case_record["event_idx"] = event_idx

                    cases.append(case_record)

        if self.verbose:
            print(f"\nCreated {len(cases)} case samples")

        # Sample controls from polygons without events
        if self.verbose:
            print(f"\nSampling controls (ratio={control_ratio})...")

        n_controls_target = int(len(cases) * control_ratio)
        controls = []

        # Sample from scans that have been loaded
        all_polygon_ids = set(self.polygon_labeler.polygon_by_id.keys())

        for scan_date in self._scan_features.keys():
            features_df = self._scan_features[scan_date]
            event_polygons = scan_event_polygons[scan_date]

            # Find polygons without events
            control_df = features_df[
                ~features_df["polygon_id"].isin(event_polygons) &
                (features_df["point_count"] > 0)
            ]

            for _, row in control_df.iterrows():
                control_record = row.to_dict()
                control_record["label"] = 0
                control_record["event_date"] = None
                control_record["days_to_event"] = None
                control_record["event_volume"] = None
                control_record["event_idx"] = None
                controls.append(control_record)

        # Sample controls to match ratio
        if len(controls) > n_controls_target:
            sampled_indices = np.random.choice(
                len(controls), size=n_controls_target, replace=False
            )
            controls = [controls[i] for i in sampled_indices]

        if self.verbose:
            print(f"  Sampled {len(controls)} controls")

        # Combine cases and controls
        all_records = cases + controls
        dataset = pd.DataFrame(all_records)

        if self.verbose:
            print(f"\nFinal dataset:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Cases (label=1): {(dataset['label'] == 1).sum()}")
            print(f"  Controls (label=0): {(dataset['label'] == 0).sum()}")
            if len(dataset) > 0:
                print(f"  Class balance: {dataset['label'].mean()*100:.1f}% positive")

        return dataset


def create_temporal_training_data(
    events: pd.DataFrame,
    point_cloud_dir: Union[str, Path],
    polygon_shapefile: Union[str, Path],
    lookforward_days: int = 365,
    min_days_before: int = 7,
    control_ratio: float = 1.0,
    scan_pattern: str = "*_rai.laz",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, TemporalAligner]:
    """Convenience function to create temporally-aligned training data."""
    polygon_labeler = PolygonLabeler(polygon_shapefile, verbose=verbose)

    aligner = TemporalAligner(
        point_cloud_dir=point_cloud_dir,
        polygon_labeler=polygon_labeler,
        lookforward_days=lookforward_days,
        min_days_before=min_days_before,
        verbose=verbose,
    )

    aligner.discover_scans(pattern=scan_pattern)

    dataset = aligner.create_case_control_dataset(
        events=events,
        control_ratio=control_ratio,
    )

    return dataset, aligner
