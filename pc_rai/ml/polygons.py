"""
Polygon-based spatial matching for ML training.

Uses 1m polygon shapefiles for precise spatial matching between
events and point cloud features. Polygon IDs correspond directly
to alongshore meter positions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import shapefile
from matplotlib.path import Path as MplPath


@dataclass
class Polygon:
    """A single 1m alongshore polygon.

    Attributes
    ----------
    polygon_id : int
        Polygon ID (equals alongshore meter position).
    vertices : np.ndarray
        Polygon vertices as (N, 2) array of (x, y) coordinates.
    x_min : float
        Minimum X coordinate.
    x_max : float
        Maximum X coordinate.
    y_min : float
        Minimum Y coordinate.
    y_max : float
        Maximum Y coordinate.
    """
    polygon_id: int
    vertices: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    _path: MplPath = field(default=None, repr=False)

    def __post_init__(self):
        """Create matplotlib Path for efficient point-in-polygon tests."""
        if self._path is None:
            self._path = MplPath(self.vertices)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this polygon."""
        return self._path.contains_point((x, y))

    def points_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return mask of points inside this polygon (vectorized).

        Parameters
        ----------
        x : np.ndarray
            X coordinates of points.
        y : np.ndarray
            Y coordinates of points.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates point is inside polygon.
        """
        # Quick bounding box check first
        in_bbox = (
            (x >= self.x_min) & (x <= self.x_max) &
            (y >= self.y_min) & (y <= self.y_max)
        )

        # Full polygon check only for points in bounding box (vectorized)
        mask = np.zeros(len(x), dtype=bool)
        bbox_indices = np.where(in_bbox)[0]

        if len(bbox_indices) > 0:
            bbox_points = np.column_stack([x[bbox_indices], y[bbox_indices]])
            inside = self._path.contains_points(bbox_points)
            mask[bbox_indices] = inside

        return mask


class PolygonLabeler:
    """Loads and manages 1m polygon shapefiles for spatial matching.

    Polygon IDs correspond directly to alongshore meter positions.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to the polygon shapefile (without extension).
    verbose : bool
        Print loading information.
    """

    def __init__(
        self,
        shapefile_path: Union[str, Path],
        verbose: bool = True,
    ):
        self.shapefile_path = Path(shapefile_path)
        self.verbose = verbose

        # Storage
        self.polygons: List[Polygon] = []
        self.polygon_by_id: Dict[int, Polygon] = {}

        # Load polygons
        self._load_polygons()

    def _load_polygons(self):
        """Load polygons from shapefile."""
        # Handle path - remove extension if present
        path_str = str(self.shapefile_path)
        for ext in ['.shp', '.dbf', '.shx']:
            if path_str.endswith(ext):
                path_str = path_str[:-4]
                break

        sf = shapefile.Reader(path_str)

        for shape_rec in sf.iterShapeRecords():
            rec = shape_rec.record
            shape = shape_rec.shape

            polygon_id = rec[0]  # First field is Id
            vertices = np.array(shape.points)

            polygon = Polygon(
                polygon_id=polygon_id,
                vertices=vertices,
                x_min=vertices[:, 0].min(),
                x_max=vertices[:, 0].max(),
                y_min=vertices[:, 1].min(),
                y_max=vertices[:, 1].max(),
            )

            self.polygons.append(polygon)
            self.polygon_by_id[polygon_id] = polygon

        # Sort by ID
        self.polygons.sort(key=lambda p: p.polygon_id)

        if self.verbose:
            print(f"Loaded {len(self.polygons)} polygons")
            print(f"  ID range: {self.polygons[0].polygon_id} - {self.polygons[-1].polygon_id}")

    def find_polygons_for_event(
        self,
        alongshore_start: float,
        alongshore_end: float,
    ) -> List[int]:
        """Find polygon IDs that overlap with an event's alongshore extent.

        Since polygon IDs equal alongshore meter positions, this is a
        simple range lookup.

        Parameters
        ----------
        alongshore_start : float
            Start of event alongshore extent in meters.
        alongshore_end : float
            End of event alongshore extent in meters.

        Returns
        -------
        List[int]
            List of polygon IDs that overlap with the event.
        """
        # Round to nearest meter (polygon resolution)
        start_id = int(np.floor(alongshore_start))
        end_id = int(np.ceil(alongshore_end))

        # Find all polygon IDs in range that exist
        polygon_ids = [
            pid for pid in range(start_id, end_id + 1)
            if pid in self.polygon_by_id
        ]

        return polygon_ids

    def get_polygon(self, polygon_id: int) -> Optional[Polygon]:
        """Get a polygon by ID.

        Parameters
        ----------
        polygon_id : int
            The polygon ID (alongshore meter position).

        Returns
        -------
        Polygon or None
            The polygon, or None if not found.
        """
        return self.polygon_by_id.get(polygon_id)

    def assign_points_to_polygons(
        self,
        x: np.ndarray,
        y: np.ndarray,
        polygon_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Assign points to polygons.

        Parameters
        ----------
        x : np.ndarray
            X coordinates of points.
        y : np.ndarray
            Y coordinates of points.
        polygon_ids : List[int], optional
            Specific polygon IDs to check. If None, checks all polygons.

        Returns
        -------
        np.ndarray
            Array of polygon IDs for each point (-1 if not in any polygon).
        """
        assignments = np.full(len(x), -1, dtype=np.int32)

        polygons_to_check = (
            [self.polygon_by_id[pid] for pid in polygon_ids if pid in self.polygon_by_id]
            if polygon_ids is not None
            else self.polygons
        )

        for polygon in polygons_to_check:
            mask = polygon.points_inside(x, y)
            assignments[mask] = polygon.polygon_id

        return assignments

    @property
    def min_id(self) -> int:
        """Minimum polygon ID."""
        return self.polygons[0].polygon_id if self.polygons else 0

    @property
    def max_id(self) -> int:
        """Maximum polygon ID."""
        return self.polygons[-1].polygon_id if self.polygons else 0

    @property
    def id_range(self) -> Tuple[int, int]:
        """Range of polygon IDs as (min, max)."""
        return (self.min_id, self.max_id)
