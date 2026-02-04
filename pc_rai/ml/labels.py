"""
Label generation for ML training.

Maps rockfall events to transect-level labels based on alongshore position overlap.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shapefile
except ImportError:
    shapefile = None


@dataclass
class Transect:
    """A single transect with its geometry and attributes."""

    transect_id: int
    alongshore_dist: float  # Alongshore position in meters
    tr_id: str  # Human-readable ID (e.g., "MOP 520_001")
    start_point: Tuple[float, float]  # (x, y) UTM coordinates
    end_point: Tuple[float, float]  # (x, y) UTM coordinates

    @property
    def center_point(self) -> Tuple[float, float]:
        """Get the center point of the transect line."""
        return (
            (self.start_point[0] + self.end_point[0]) / 2,
            (self.start_point[1] + self.end_point[1]) / 2,
        )


class TransectLabeler:
    """Maps rockfall events to transect-level labels.

    Events are mapped to transects based on alongshore position overlap.
    Each transect receives a label indicating the number of events that
    overlap with its corridor.

    Parameters
    ----------
    transects_path : str or Path
        Path to transects shapefile.
    transect_spacing : float
        Expected spacing between transects in meters (default 10m).
    verbose : bool
        Print progress information.

    Attributes
    ----------
    transects : list of Transect
        List of all transects loaded from shapefile.
    alongshore_positions : np.ndarray
        Array of alongshore positions for each transect.
    """

    def __init__(
        self,
        transects_path: Union[str, Path],
        transect_spacing: float = 10.0,
        verbose: bool = True,
    ):
        if shapefile is None:
            raise ImportError("pyshp is required. Install with: pip install pyshp")

        self.transects_path = Path(transects_path)
        self.transect_spacing = transect_spacing
        self.verbose = verbose

        self.transects: List[Transect] = []
        self.alongshore_positions: np.ndarray = np.array([])

        self._load_transects()

    def _load_transects(self) -> None:
        """Load transects from shapefile."""
        shp_path = str(self.transects_path)
        if shp_path.endswith(".shp"):
            shp_path = shp_path[:-4]

        sf = shapefile.Reader(shp_path)

        # Get field names
        field_names = [f[0] for f in sf.fields[1:]]

        # Find the relevant field indices
        try:
            idx_transect_i = field_names.index("transect_i")
            idx_dist = field_names.index("dist")
            idx_tr_id = field_names.index("tr_id")
        except ValueError as e:
            raise ValueError(
                f"Required field not found in shapefile. "
                f"Available fields: {field_names}. Error: {e}"
            )

        for shape_rec in sf.iterShapeRecords():
            rec = shape_rec.record
            shape = shape_rec.shape

            # Get line endpoints
            points = shape.points
            if len(points) < 2:
                continue

            start_point = (points[0][0], points[0][1])
            end_point = (points[-1][0], points[-1][1])

            transect = Transect(
                transect_id=rec[idx_transect_i],
                alongshore_dist=rec[idx_dist],
                tr_id=rec[idx_tr_id],
                start_point=start_point,
                end_point=end_point,
            )
            self.transects.append(transect)

        # Sort by alongshore distance
        self.transects.sort(key=lambda t: t.alongshore_dist)

        # Create array of alongshore positions for fast lookup
        self.alongshore_positions = np.array(
            [t.alongshore_dist for t in self.transects]
        )

        if self.verbose:
            print(f"Loaded {len(self.transects):,} transects from {self.transects_path.name}")
            print(
                f"  Alongshore range: {self.alongshore_positions.min():.1f}m - "
                f"{self.alongshore_positions.max():.1f}m"
            )

    def find_overlapping_transects(
        self,
        alongshore_start: float,
        alongshore_end: float,
    ) -> List[int]:
        """Find transect indices that overlap with an alongshore range.

        Parameters
        ----------
        alongshore_start : float
            Start of alongshore range in meters.
        alongshore_end : float
            End of alongshore range in meters.

        Returns
        -------
        list of int
            Indices of overlapping transects.
        """
        # Ensure start < end
        if alongshore_start > alongshore_end:
            alongshore_start, alongshore_end = alongshore_end, alongshore_start

        # Find transects within the range (with half-spacing buffer on each side)
        half_spacing = self.transect_spacing / 2
        mask = (self.alongshore_positions >= alongshore_start - half_spacing) & (
            self.alongshore_positions <= alongshore_end + half_spacing
        )

        return np.where(mask)[0].tolist()

    def map_events_to_transects(
        self,
        events: pd.DataFrame,
        alongshore_start_col: str = "alongshore_start_m",
        alongshore_end_col: str = "alongshore_end_m",
    ) -> pd.DataFrame:
        """Map events to transects based on alongshore overlap.

        Parameters
        ----------
        events : pd.DataFrame
            Events dataframe with alongshore extent columns.
        alongshore_start_col : str
            Column name for alongshore start position.
        alongshore_end_col : str
            Column name for alongshore end position.

        Returns
        -------
        pd.DataFrame
            Transect-level labels with columns:
            - transect_id: Transect index
            - alongshore_dist: Alongshore position
            - tr_id: Human-readable transect ID
            - event_count: Number of overlapping events
            - has_event: Binary flag (1 if any events, 0 otherwise)
            - total_volume: Sum of event volumes
            - max_volume: Maximum event volume
        """
        n_transects = len(self.transects)

        # Initialize label arrays
        event_counts = np.zeros(n_transects, dtype=int)
        total_volumes = np.zeros(n_transects, dtype=float)
        max_volumes = np.zeros(n_transects, dtype=float)
        event_indices = [[] for _ in range(n_transects)]  # Track which events hit each transect

        # Map each event to overlapping transects
        for event_idx, event in events.iterrows():
            alongshore_start = event[alongshore_start_col]
            alongshore_end = event[alongshore_end_col]

            overlapping = self.find_overlapping_transects(alongshore_start, alongshore_end)

            volume = event.get("volume", 1.0)

            for transect_idx in overlapping:
                event_counts[transect_idx] += 1
                total_volumes[transect_idx] += volume
                max_volumes[transect_idx] = max(max_volumes[transect_idx], volume)
                event_indices[transect_idx].append(event_idx)

        # Build output dataframe
        labels_df = pd.DataFrame(
            {
                "transect_id": [t.transect_id for t in self.transects],
                "alongshore_dist": self.alongshore_positions,
                "tr_id": [t.tr_id for t in self.transects],
                "event_count": event_counts,
                "has_event": (event_counts > 0).astype(int),
                "total_volume": total_volumes,
                "max_volume": max_volumes,
            }
        )

        if self.verbose:
            n_with_events = (event_counts > 0).sum()
            print(f"Mapped {len(events):,} events to {n_transects:,} transects")
            print(
                f"  Transects with events: {n_with_events:,} "
                f"({100*n_with_events/n_transects:.1f}%)"
            )
            print(f"  Total event-transect overlaps: {event_counts.sum():,}")

        return labels_df

    def get_transect_by_alongshore(
        self,
        alongshore_position: float,
    ) -> Optional[Transect]:
        """Get the transect closest to an alongshore position.

        Parameters
        ----------
        alongshore_position : float
            Alongshore position in meters.

        Returns
        -------
        Transect or None
            The closest transect, or None if no transects loaded.
        """
        if len(self.transects) == 0:
            return None

        idx = np.argmin(np.abs(self.alongshore_positions - alongshore_position))
        return self.transects[idx]

    def get_transect_centers(self) -> np.ndarray:
        """Get UTM coordinates of all transect centers.

        Returns
        -------
        np.ndarray
            (N, 2) array of (x, y) coordinates.
        """
        return np.array([t.center_point for t in self.transects])


def create_temporal_labels(
    events: pd.DataFrame,
    labeler: TransectLabeler,
    scan_dates: List[pd.Timestamp],
    lookforward_days: int = 365,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create labels for each transect-scan combination.

    For each scan date, labels indicate whether an event occurred
    in the subsequent lookforward_days period.

    Parameters
    ----------
    events : pd.DataFrame
        Events dataframe with 'mid_date' column.
    labeler : TransectLabeler
        Transect labeler with loaded transects.
    scan_dates : list of pd.Timestamp
        Dates of point cloud scans.
    lookforward_days : int
        Number of days after scan to look for events.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Labels with columns:
        - transect_id
        - scan_date
        - event_count
        - has_event
        - days_to_next_event (if any)
    """
    records = []

    for scan_date in scan_dates:
        # Filter events to lookforward window
        window_start = scan_date
        window_end = scan_date + pd.Timedelta(days=lookforward_days)

        window_events = events[
            (events["mid_date"] >= window_start) & (events["mid_date"] < window_end)
        ]

        # Map events to transects
        labels = labeler.map_events_to_transects(window_events)
        labels["scan_date"] = scan_date

        records.append(labels)

    result = pd.concat(records, ignore_index=True)

    if verbose:
        n_positive = (result["has_event"] == 1).sum()
        n_total = len(result)
        print(
            f"Created {n_total:,} transect-scan labels "
            f"({n_positive:,} positive, {100*n_positive/n_total:.1f}%)"
        )

    return result
