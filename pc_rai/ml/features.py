"""
Feature extraction and aggregation for ML training.

Extracts point cloud features and aggregates them to transect-level statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

try:
    import laspy
except ImportError:
    laspy = None

from .labels import TransectLabeler, Transect


@dataclass
class PointCloudFeatures:
    """Container for point cloud features.

    Attributes
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    slope : np.ndarray
        (N,) slope angles in degrees.
    roughness_small : np.ndarray
        (N,) small-scale roughness values.
    roughness_large : np.ndarray
        (N,) large-scale roughness values.
    source_file : str
        Path to source file.
    scan_date : pd.Timestamp, optional
        Date of scan if parseable from filename.
    """

    xyz: np.ndarray
    slope: np.ndarray
    roughness_small: np.ndarray
    roughness_large: np.ndarray
    source_file: str
    scan_date: Optional[pd.Timestamp] = None

    @property
    def n_points(self) -> int:
        return len(self.xyz)

    @property
    def roughness_ratio(self) -> np.ndarray:
        """Compute roughness ratio (small / large)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ratio = np.divide(
                self.roughness_small,
                self.roughness_large,
                out=np.zeros_like(self.roughness_small),
                where=self.roughness_large > 0,
            )
        return ratio


class TransectFeatureExtractor:
    """Extracts and aggregates features from point clouds to transects.

    Parameters
    ----------
    labeler : TransectLabeler
        Transect labeler with loaded transects.
    half_width : float
        Half-width of transect corridors in meters.
    feature_names : list of str
        Names of features to extract from point cloud.
    aggregation_stats : list of str
        Statistics to compute for aggregation.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        labeler: TransectLabeler,
        half_width: float = 5.0,
        feature_names: Optional[List[str]] = None,
        aggregation_stats: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.labeler = labeler
        self.half_width = half_width
        self.verbose = verbose

        # Default features
        self.feature_names = feature_names or [
            "slope_deg",
            "roughness_small_knn",
            "roughness_large_knn",
        ]

        # Default aggregation statistics
        self.aggregation_stats = aggregation_stats or ["mean", "max", "std", "p90"]

        # Precompute transect geometries for fast point assignment
        self._precompute_transect_geometries()

    def _precompute_transect_geometries(self) -> None:
        """Precompute transect line vectors for point-to-transect assignment."""
        self.transect_starts = np.array(
            [t.start_point for t in self.labeler.transects]
        )
        self.transect_ends = np.array([t.end_point for t in self.labeler.transects])

        # Line vectors and lengths
        self.line_vecs = self.transect_ends - self.transect_starts
        self.line_lengths = np.linalg.norm(self.line_vecs, axis=1)

        # Unit vectors along each transect
        self.line_units = self.line_vecs / self.line_lengths[:, np.newaxis]

    def load_point_cloud(
        self,
        las_path: Union[str, Path],
    ) -> PointCloudFeatures:
        """Load a processed point cloud with RAI features.

        Parameters
        ----------
        las_path : str or Path
            Path to LAS/LAZ file with RAI features.

        Returns
        -------
        PointCloudFeatures
            Extracted features from point cloud.
        """
        if laspy is None:
            raise ImportError("laspy is required. Install with: pip install laspy")

        las_path = Path(las_path)
        las = laspy.read(str(las_path))

        # Extract coordinates
        xyz = np.column_stack([las.x, las.y, las.z])

        # Extract features
        try:
            slope = np.asarray(las["slope_deg"], dtype=np.float32)
        except Exception:
            raise ValueError(f"slope_deg not found in {las_path.name}")

        try:
            roughness_small = np.asarray(las["roughness_small_knn"], dtype=np.float32)
        except Exception:
            roughness_small = np.zeros(len(xyz), dtype=np.float32)
            if self.verbose:
                print(f"  Warning: roughness_small_knn not found, using zeros")

        try:
            roughness_large = np.asarray(las["roughness_large_knn"], dtype=np.float32)
        except Exception:
            roughness_large = np.zeros(len(xyz), dtype=np.float32)
            if self.verbose:
                print(f"  Warning: roughness_large_knn not found, using zeros")

        # Try to parse scan date from filename (assuming YYYYMMDD prefix)
        scan_date = None
        try:
            date_str = las_path.stem[:8]
            scan_date = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            pass

        if self.verbose:
            print(f"Loaded {len(xyz):,} points from {las_path.name}")
            if scan_date:
                print(f"  Scan date: {scan_date.strftime('%Y-%m-%d')}")

        return PointCloudFeatures(
            xyz=xyz,
            slope=slope,
            roughness_small=roughness_small,
            roughness_large=roughness_large,
            source_file=str(las_path),
            scan_date=scan_date,
        )

    def assign_points_to_transects(
        self,
        xyz: np.ndarray,
    ) -> np.ndarray:
        """Assign each point to the nearest transect within corridor.

        Parameters
        ----------
        xyz : np.ndarray
            (N, 3) point coordinates.

        Returns
        -------
        np.ndarray
            (N,) array of transect indices, -1 if point not in any corridor.
        """
        n_points = len(xyz)
        n_transects = len(self.labeler.transects)

        # Initialize assignments to -1 (no transect)
        assignments = np.full(n_points, -1, dtype=np.int32)

        # For each transect, find points within corridor
        xy = xyz[:, :2]

        for t_idx in range(n_transects):
            start = self.transect_starts[t_idx]
            line_unit = self.line_units[t_idx]
            line_length = self.line_lengths[t_idx]

            # Vector from transect start to each point
            point_vecs = xy - start

            # Project points onto transect line
            t = np.dot(point_vecs, line_unit)

            # Perpendicular distance from line
            proj = np.outer(t, line_unit)
            perp_vecs = point_vecs - proj
            perp_dist = np.linalg.norm(perp_vecs, axis=1)

            # Points within corridor: within half_width of line,
            # and between start and end (0 <= t <= line_length)
            in_corridor = (perp_dist <= self.half_width) & (t >= 0) & (t <= line_length)

            # Assign points to this transect (may overwrite previous assignment
            # if point is in multiple corridors - keeps last one)
            assignments[in_corridor] = t_idx

        return assignments

    def aggregate_features(
        self,
        features: PointCloudFeatures,
        assignments: np.ndarray,
    ) -> pd.DataFrame:
        """Aggregate point-level features to transect-level statistics.

        Parameters
        ----------
        features : PointCloudFeatures
            Point cloud features.
        assignments : np.ndarray
            Transect assignments for each point.

        Returns
        -------
        pd.DataFrame
            Transect-level aggregated features.
        """
        n_transects = len(self.labeler.transects)

        # Prepare feature arrays
        feature_arrays = {
            "slope": features.slope,
            "r_small": features.roughness_small,
            "r_large": features.roughness_large,
            "r_ratio": features.roughness_ratio,
            "height": features.xyz[:, 2],  # Z coordinate
        }

        # Initialize output dictionary
        output = {
            "transect_id": [t.transect_id for t in self.labeler.transects],
            "alongshore_dist": self.labeler.alongshore_positions.copy(),
            "point_count": np.zeros(n_transects, dtype=int),
        }

        # Compute aggregated statistics for each feature
        for feat_name, feat_array in feature_arrays.items():
            for stat in self.aggregation_stats:
                col_name = f"{feat_name}_{stat}"
                output[col_name] = np.full(n_transects, np.nan)

        # Height range is a special case
        output["height_range"] = np.full(n_transects, np.nan)

        # Aggregate for each transect
        for t_idx in range(n_transects):
            mask = assignments == t_idx
            count = mask.sum()
            output["point_count"][t_idx] = count

            if count == 0:
                continue

            for feat_name, feat_array in feature_arrays.items():
                values = feat_array[mask]
                valid_values = values[~np.isnan(values)]

                if len(valid_values) == 0:
                    continue

                for stat in self.aggregation_stats:
                    col_name = f"{feat_name}_{stat}"

                    if stat == "mean":
                        output[col_name][t_idx] = np.mean(valid_values)
                    elif stat == "max":
                        output[col_name][t_idx] = np.max(valid_values)
                    elif stat == "min":
                        output[col_name][t_idx] = np.min(valid_values)
                    elif stat == "std":
                        output[col_name][t_idx] = np.std(valid_values)
                    elif stat == "p90":
                        output[col_name][t_idx] = np.percentile(valid_values, 90)
                    elif stat == "p10":
                        output[col_name][t_idx] = np.percentile(valid_values, 10)
                    elif stat == "median":
                        output[col_name][t_idx] = np.median(valid_values)

            # Height range
            heights = features.xyz[mask, 2]
            valid_heights = heights[~np.isnan(heights)]
            if len(valid_heights) > 0:
                output["height_range"][t_idx] = np.max(valid_heights) - np.min(valid_heights)

        df = pd.DataFrame(output)

        if self.verbose:
            n_with_points = (df["point_count"] > 0).sum()
            print(
                f"Aggregated features to {n_transects:,} transects "
                f"({n_with_points:,} with points)"
            )

        return df

    def extract_features(
        self,
        las_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Extract and aggregate features from a point cloud file.

        This is the main entry point for feature extraction.

        Parameters
        ----------
        las_path : str or Path
            Path to LAS/LAZ file with RAI features.

        Returns
        -------
        pd.DataFrame
            Transect-level aggregated features.
        """
        # Load point cloud
        features = self.load_point_cloud(las_path)

        # Assign points to transects
        if self.verbose:
            print("Assigning points to transects...")
        assignments = self.assign_points_to_transects(features.xyz)

        n_assigned = (assignments >= 0).sum()
        if self.verbose:
            print(
                f"  {n_assigned:,} points ({100*n_assigned/features.n_points:.1f}%) "
                f"assigned to transects"
            )

        # Aggregate features
        df = self.aggregate_features(features, assignments)

        # Add scan date if available
        if features.scan_date is not None:
            df["scan_date"] = features.scan_date

        df["source_file"] = features.source_file

        return df


def extract_features_from_multiple_scans(
    las_paths: List[Union[str, Path]],
    labeler: TransectLabeler,
    half_width: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract features from multiple point cloud scans.

    Parameters
    ----------
    las_paths : list of str or Path
        Paths to LAS/LAZ files.
    labeler : TransectLabeler
        Transect labeler with loaded transects.
    half_width : float
        Half-width of transect corridors.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        Combined features from all scans.
    """
    extractor = TransectFeatureExtractor(
        labeler=labeler,
        half_width=half_width,
        verbose=verbose,
    )

    all_features = []

    for i, las_path in enumerate(las_paths):
        if verbose:
            print(f"\nProcessing {i+1}/{len(las_paths)}: {Path(las_path).name}")

        try:
            features = extractor.extract_features(las_path)
            all_features.append(features)
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue

    if not all_features:
        raise ValueError("No features extracted from any files")

    result = pd.concat(all_features, ignore_index=True)

    if verbose:
        print(f"\nCombined {len(result):,} transect-scan feature records")

    return result
