"""
Aggregate point-level features to 1m polygon bins with elevation zones.

Uses polygon shapefiles for spatial matching to ensure polygon IDs
align with event data. Each polygon is split into relative elevation
zones (lower/middle/upper thirds) to capture different cliff behavior.

Output: One row per polygon-zone with aggregated feature statistics.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import shapefile
    HAS_SHAPEFILE = True
except ImportError:
    HAS_SHAPEFILE = False

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

logger = logging.getLogger(__name__)

# Feature columns to aggregate
FEATURE_COLUMNS = [
    "slope",
    "roughness_small",
    "roughness_large",
    "roughness_ratio",
    "height",
    "planarity",
    "linearity",
    "sphericity",
    "curvature",
]

# Aggregation statistics to compute
AGG_STATS = ["mean", "std", "min", "max", "p10", "p50", "p90"]

# Location to shapefile directory mapping
SHAPEFILE_DIRS = {
    "Blacks": "BlacksPolygons520to567at1m",
    "DelMar": "DelMarPolygons595to620at1m",
    "Encinitas": "EncinitasPolygons708to764at1m",
    "SanElijo": "SanElijoPolygons683to708at1m",
    "Solana": "SolanaPolygons637to666at1m",
    "Cardiff": "SolanaPolygons637to666at1m",  # Cardiff uses Solana polygons
    "Torrey": "TorreyPolygons567to581at1m",
}


def load_polygons(
    shapefile_dir: Union[str, Path],
    location: str,
    verbose: bool = True,
) -> Tuple[List[dict], Dict[int, int]]:
    """Load polygon geometries from shapefile.

    Parameters
    ----------
    shapefile_dir : str or Path
        Base directory containing polygon subdirectories.
    location : str
        Location name (e.g., 'DelMar', 'Torrey').
    verbose : bool
        Print progress.

    Returns
    -------
    polygons : list of dict
        List of polygon dicts with 'mop_id', 'points', 'bbox'.
    mop_to_idx : dict
        Mapping from integer MOP ID to polygon list index.
    """
    if not HAS_SHAPEFILE:
        raise ImportError("pyshp is required for shapefile loading: pip install pyshp")

    shapefile_dir = Path(shapefile_dir)

    # Get shapefile subdirectory for this location
    if location not in SHAPEFILE_DIRS:
        raise ValueError(f"Unknown location '{location}'. Known: {list(SHAPEFILE_DIRS.keys())}")

    subdir = SHAPEFILE_DIRS[location]
    shp_subdir = shapefile_dir / subdir

    if not shp_subdir.exists():
        raise FileNotFoundError(f"Shapefile directory not found: {shp_subdir}")

    # Find .shp file
    shp_files = list(shp_subdir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp files found in {shp_subdir}")

    shp_path = shp_files[0]

    if verbose:
        print(f"  Loading polygons from: {shp_path.name}")

    # Read shapefile
    sf = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in sf.fields[1:]]

    # Check if this is DelMar (needs special handling - derive MOP from Y coordinate)
    is_delmar = location == "DelMar" and "MOP_ID" not in fields

    if is_delmar:
        # DelMar shapefile uses arbitrary 'Id' field, not MOP coordinates
        # Derive MOP from Y coordinate using linear mapping
        # MOP range 595-620, Y range from shapefile extent
        mop_min, mop_max = 595, 620

        # First pass: get Y extent
        y_values = []
        for shape in sf.shapes():
            points = np.array(shape.points)
            centroid_y = points[:, 1].mean()
            y_values.append(centroid_y)

        y_min, y_max = min(y_values), max(y_values)
        y_extent = y_max - y_min
        mop_extent = mop_max - mop_min
        scale = mop_extent / y_extent  # MOP per meter Y

        if verbose:
            print(f"  DelMar: deriving MOP from Y coordinate (scale={scale:.6f} MOP/m)")

        # Second pass: create polygons with derived MOP
        polygons = []
        mop_to_idx = {}

        for i, shape in enumerate(sf.shapes()):
            points = np.array(shape.points)
            centroid_y = points[:, 1].mean()

            # Derive MOP from Y coordinate - keep full precision
            mop_float = mop_min + (centroid_y - y_min) * scale

            # Compute bounding box
            bbox = (points[:, 0].min(), points[:, 1].min(),
                    points[:, 0].max(), points[:, 1].max())

            polygons.append({
                "mop_id": mop_float,  # Use float MOP, not rounded
                "points": points,
                "bbox": bbox,
                "idx": i,
            })

            # Map MOP to index using rounded key for lookup
            mop_key = round(mop_float, 3)
            if mop_key not in mop_to_idx:
                mop_to_idx[mop_key] = i

    else:
        # Standard handling: use MOP_ID field
        if "MOP_ID" in fields:
            mop_idx = fields.index("MOP_ID")
        else:
            raise ValueError(f"Cannot find MOP_ID field. Fields: {fields}")

        polygons = []
        mop_to_idx = {}

        for i, (shape, rec) in enumerate(zip(sf.shapes(), sf.records())):
            # Get MOP ID from field - keep full precision
            mop_value = rec[mop_idx]
            if isinstance(mop_value, str):
                mop_float = float(mop_value)
            else:
                mop_float = float(mop_value)

            # Get polygon points
            points = np.array(shape.points)

            # Compute bounding box
            bbox = (points[:, 0].min(), points[:, 1].min(),
                    points[:, 0].max(), points[:, 1].max())

            polygons.append({
                "mop_id": mop_float,  # Use float MOP, not rounded
                "points": points,
                "bbox": bbox,
                "idx": i,
            })

            # Map MOP to index using rounded key for lookup
            mop_key = round(mop_float, 3)
            if mop_key not in mop_to_idx:
                mop_to_idx[mop_key] = i

    if verbose:
        mop_ids = [p["mop_id"] for p in polygons]
        print(f"  Loaded {len(polygons)} polygons, MOP range: {min(mop_ids)}-{max(mop_ids)}")

    return polygons, mop_to_idx


def points_in_polygon_vectorized(
    points_xy: np.ndarray,
    poly_points: np.ndarray,
) -> np.ndarray:
    """Vectorized point-in-polygon test using ray casting.

    Parameters
    ----------
    points_xy : np.ndarray
        (N, 2) array of point coordinates to test.
    poly_points : np.ndarray
        (M, 2) array of polygon vertices.

    Returns
    -------
    inside : np.ndarray
        (N,) boolean array, True if point is inside polygon.
    """
    n_points = len(points_xy)
    n_verts = len(poly_points)

    if n_points == 0:
        return np.array([], dtype=bool)

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    inside = np.zeros(n_points, dtype=bool)

    # Get polygon vertices as arrays
    px = poly_points[:, 0]
    py = poly_points[:, 1]

    # Ray casting algorithm - vectorized over all points
    j = n_verts - 1
    for i in range(n_verts):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]

        # Vectorized condition check
        cond1 = (yi > y) != (yj > y)

        # Avoid division by zero
        dy = yj - yi
        if abs(dy) > 1e-10:
            cond2 = x < (xj - xi) * (y - yi) / dy + xi
            mask = cond1 & cond2
            inside[mask] = ~inside[mask]

        j = i

    return inside


def assign_points_to_polygons(
    xyz: np.ndarray,
    polygons: List[dict],
    verbose: bool = True,
) -> np.ndarray:
    """Assign each point to a polygon based on XY coordinates.

    Uses vectorized point-in-polygon testing for speed.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    polygons : list of dict
        Polygon definitions with 'mop_id', 'points', 'bbox'.
    verbose : bool
        Print progress.

    Returns
    -------
    mop_ids : np.ndarray
        (N,) MOP IDs for each point (-1 if not in any polygon).
        MOP IDs are floats with full precision.
    """
    n_points = len(xyz)
    mop_ids = np.full(n_points, -1.0, dtype=np.float64)  # Float for precision
    xy = xyz[:, :2]

    if verbose:
        print(f"  Assigning {n_points:,} points to {len(polygons)} polygons...")

    n_assigned = 0

    for i, poly in enumerate(polygons):
        bbox = poly["bbox"]

        # Vectorized bounding box filter
        in_bbox = (
            (xy[:, 0] >= bbox[0]) &
            (xy[:, 0] <= bbox[2]) &
            (xy[:, 1] >= bbox[1]) &
            (xy[:, 1] <= bbox[3])
        )

        # Only test unassigned points in bounding box
        candidates = in_bbox & (mop_ids < 0)

        if not candidates.any():
            continue

        # Get candidate points
        candidate_indices = np.where(candidates)[0]
        candidate_points = xy[candidate_indices]

        # Vectorized point-in-polygon test
        inside = points_in_polygon_vectorized(candidate_points, poly["points"])

        # Assign MOP IDs (float)
        mop_ids[candidate_indices[inside]] = poly["mop_id"]
        n_assigned += inside.sum()

        # Progress every 500 polygons
        if verbose and (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(polygons)} polygons...")

    if verbose:
        pct = 100 * n_assigned / n_points if n_points > 0 else 0
        print(f"  Assigned {n_assigned:,}/{n_points:,} points ({pct:.1f}%)")

    return mop_ids


def aggregate_features(values: np.ndarray) -> dict:
    """Compute aggregation statistics for a feature array.

    Parameters
    ----------
    values : np.ndarray
        1D array of feature values (may contain NaN).

    Returns
    -------
    stats : dict
        Dictionary with keys: mean, std, min, max, p10, p50, p90
    """
    valid = values[~np.isnan(values)]
    n_valid = len(valid)

    if n_valid == 0:
        return {stat: np.nan for stat in AGG_STATS}

    return {
        "mean": np.mean(valid),
        "std": np.std(valid) if n_valid > 1 else 0.0,
        "min": np.min(valid),
        "max": np.max(valid),
        "p10": np.percentile(valid, 10),
        "p50": np.percentile(valid, 50),
        "p90": np.percentile(valid, 90),
    }


def get_elevation_zone(z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Assign points to elevation zones (lower/middle/upper thirds).

    Parameters
    ----------
    z : np.ndarray
        (N,) elevation values.
    z_min : float
        Minimum elevation in the polygon.
    z_max : float
        Maximum elevation in the polygon.

    Returns
    -------
    zones : np.ndarray
        (N,) zone labels: 0=lower, 1=middle, 2=upper
    """
    z_range = z_max - z_min
    if z_range < 0.1:  # Very flat - all middle
        return np.ones(len(z), dtype=np.int32)

    # Compute tercile boundaries
    z_lower = z_min + z_range / 3
    z_upper = z_min + 2 * z_range / 3

    zones = np.ones(len(z), dtype=np.int32)  # Default to middle
    zones[z < z_lower] = 0  # Lower
    zones[z >= z_upper] = 2  # Upper

    return zones


def aggregate_polygon_zone(
    xyz: np.ndarray,
    features: dict,
    mask: np.ndarray,
) -> Optional[dict]:
    """Aggregate features for points within a polygon-zone.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) all point coordinates.
    features : dict
        Dictionary of feature arrays, each (N,).
    mask : np.ndarray
        (N,) boolean mask for points in this polygon-zone.

    Returns
    -------
    row : dict or None
        Aggregated statistics for all features plus metadata.
    """
    n_points = mask.sum()

    if n_points == 0:
        return None

    z_values = xyz[mask, 2]

    row = {
        "n_points": n_points,
        "z_min": z_values.min(),
        "z_max": z_values.max(),
        "z_mean": z_values.mean(),
        "z_range": z_values.max() - z_values.min(),
    }

    # Aggregate each feature
    for feat_name in FEATURE_COLUMNS:
        if feat_name not in features:
            continue

        feat_values = features[feat_name][mask]
        stats = aggregate_features(feat_values)

        for stat_name, stat_value in stats.items():
            row[f"{feat_name}_{stat_name}"] = stat_value

    return row


def aggregate_survey(
    las_path: Union[str, Path],
    shapefile_dir: Union[str, Path],
    location: str,
    min_points_per_zone: int = 5,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Aggregate point features to polygon-zone level for a single survey.

    Parameters
    ----------
    las_path : str or Path
        Path to LAZ file with extracted features.
    shapefile_dir : str or Path
        Base directory containing polygon shapefiles.
    location : str
        Location name for shapefile selection.
    min_points_per_zone : int
        Minimum points for valid zone aggregation.
    verbose : bool
        Print progress.

    Returns
    -------
    df : pd.DataFrame or None
        DataFrame with one row per polygon-zone, or None if failed.
    """
    if not HAS_LASPY:
        raise ImportError("laspy is required: pip install laspy")

    las_path = Path(las_path)
    shapefile_dir = Path(shapefile_dir)

    if verbose:
        print(f"Aggregating: {las_path.name}")

    # Load polygons
    try:
        polygons, mop_to_idx = load_polygons(shapefile_dir, location, verbose=verbose)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load polygons for {location}: {e}")
        return None

    # Load LAZ file
    try:
        las = laspy.read(las_path)
    except Exception as e:
        logger.error(f"Failed to read {las_path}: {e}")
        return None

    # Get coordinates
    xyz = np.column_stack([las.x, las.y, las.z])
    n_points = len(xyz)

    if verbose:
        print(f"  Points: {n_points:,}")

    # Load features
    features = {}
    available_features = []
    for feat_name in FEATURE_COLUMNS:
        if feat_name in las.point_format.dimension_names:
            features[feat_name] = np.array(las[feat_name])
            available_features.append(feat_name)

    if verbose:
        print(f"  Features: {', '.join(available_features)}")

    if len(features) == 0:
        logger.error(f"No features found in {las_path}")
        return None

    # Assign points to polygons
    mop_ids = assign_points_to_polygons(xyz, polygons, verbose=verbose)

    # Get unique MOP IDs (excluding -1)
    # MOP IDs are now floats, so filter properly
    valid_mask = mop_ids >= 0
    unique_mops = np.unique(mop_ids[valid_mask])

    if len(unique_mops) == 0:
        logger.warning(f"No points fell within polygons for {las_path}")
        return None

    if verbose:
        print(f"  Unique polygons with points: {len(unique_mops)}")

    # Zone names
    zone_names = ["lower", "middle", "upper"]

    # Aggregate each polygon-zone
    rows = []

    for mop_id in unique_mops:
        mop_mask = mop_ids == mop_id
        n_in_polygon = mop_mask.sum()

        if n_in_polygon < min_points_per_zone:
            continue

        # Get elevation range for this polygon
        z_in_poly = xyz[mop_mask, 2]
        z_min_poly = z_in_poly.min()
        z_max_poly = z_in_poly.max()

        # Assign elevation zones
        zones = get_elevation_zone(z_in_poly, z_min_poly, z_max_poly)

        # Aggregate each zone
        for zone_idx, zone_name in enumerate(zone_names):
            # Create mask for this polygon-zone
            zone_mask_local = zones == zone_idx

            if zone_mask_local.sum() < min_points_per_zone:
                continue

            # Convert to global mask
            global_indices = np.where(mop_mask)[0]
            zone_indices = global_indices[zone_mask_local]
            global_mask = np.zeros(n_points, dtype=bool)
            global_mask[zone_indices] = True

            # Aggregate
            row = aggregate_polygon_zone(xyz, features, global_mask)

            if row is None:
                continue

            # Add metadata - use full precision MOP
            row["polygon_id"] = round(mop_id, 3)
            row["alongshore_m"] = round(mop_id, 3)
            row["zone"] = zone_name
            row["zone_idx"] = zone_idx

            rows.append(row)

    if len(rows) == 0:
        logger.warning(f"No valid polygon-zones in {las_path}")
        return None

    df = pd.DataFrame(rows)

    # Reorder columns - metadata first
    meta_cols = ["polygon_id", "alongshore_m", "zone", "zone_idx",
                 "n_points", "z_min", "z_max", "z_mean", "z_range"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + sorted(feat_cols)]

    if verbose:
        print(f"  Output: {len(df)} polygon-zones")

    return df


def aggregate_survey_batch(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    shapefile_dir: Union[str, Path],
    min_points_per_zone: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate features for all surveys in a directory.

    Each survey is processed against ALL shapefiles that overlap with
    its MOP range, not just one location.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing LAZ files with features.
    output_path : str or Path
        Output CSV path.
    shapefile_dir : str or Path
        Base directory containing polygon shapefiles.
    min_points_per_zone : int
        Minimum points for valid zone aggregation.
    verbose : bool
        Print progress.

    Returns
    -------
    df : pd.DataFrame
        Combined DataFrame for all surveys.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    shapefile_dir = Path(shapefile_dir)

    # Find all LAZ files
    laz_files = sorted(input_dir.glob("*.laz"))

    if len(laz_files) == 0:
        raise ValueError(f"No LAZ files found in {input_dir}")

    if verbose:
        print(f"Found {len(laz_files)} LAZ files in {input_dir}")
        print("=" * 60)

    all_dfs = []

    for laz_path in laz_files:
        # Extract survey date from filename (YYYYMMDD at start)
        filename = laz_path.stem
        survey_date = filename[:8]

        # Get MOP range from filename to find overlapping shapefiles
        mop_range = extract_mop_range(filename)

        if mop_range:
            # Get all locations that overlap with this file's MOP range
            locations = get_overlapping_locations(mop_range[0], mop_range[1])
            if verbose:
                print(f"\n{laz_path.name}")
                print(f"  MOP range: {mop_range[0]}-{mop_range[1]}")
                print(f"  Overlapping locations: {locations}")
        else:
            # Fall back to extracting from filename
            locations = extract_all_locations(filename)
            if not locations:
                if verbose:
                    print(f"\nSkipping {laz_path.name}: Unknown location")
                continue

        # Process against each overlapping location's shapefile
        for location in locations:
            if verbose:
                print(f"  Processing for {location}...")

            df = aggregate_survey(
                laz_path,
                shapefile_dir,
                location,
                min_points_per_zone=min_points_per_zone,
                verbose=False,  # Reduce verbosity for multi-location
            )

            if df is not None and len(df) > 0:
                df["survey_date"] = survey_date
                df["survey_file"] = laz_path.name
                df["location"] = location
                all_dfs.append(df)
                if verbose:
                    print(f"    -> {len(df)} polygon-zones")
            else:
                if verbose:
                    print(f"    -> No points in {location} polygons")

    if len(all_dfs) == 0:
        raise ValueError("No surveys were successfully aggregated")

    # Combine all DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns - survey metadata first
    survey_cols = ["survey_date", "survey_file", "location"]
    other_cols = [c for c in combined.columns if c not in survey_cols]
    combined = combined[survey_cols + other_cols]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    if verbose:
        print()
        print("=" * 60)
        print(f"Saved: {output_path}")
        print(f"  Total rows: {len(combined):,}")
        print(f"  Surveys: {combined['survey_file'].nunique()}")
        print(f"  Locations: {combined['location'].nunique()}")
        by_loc = combined.groupby('location').size()
        print(f"  By location:")
        for loc, count in by_loc.items():
            print(f"    {loc}: {count:,}")

    return combined


def extract_location(filename: str) -> str:
    """Extract primary location name from filename.

    Parameters
    ----------
    filename : str
        Filename like '20181121_00567_00622_NoWaves_Torrey_DelMar_...'

    Returns
    -------
    location : str
        Primary location name (e.g., 'DelMar', 'Torrey').
    """
    locations = extract_all_locations(filename)
    return locations[0] if locations else "Unknown"


def extract_all_locations(filename: str) -> List[str]:
    """Extract ALL location names from filename.

    Parameters
    ----------
    filename : str
        Filename like '20181121_00567_00622_NoWaves_Torrey_DelMar_...'

    Returns
    -------
    locations : list of str
        All location names found in filename.
    """
    filename_lower = filename.lower()

    # Known locations
    location_keywords = [
        ("sanelijo", "SanElijo"),
        ("encinitas", "Encinitas"),
        ("moonlight", "Encinitas"),
        ("ponto", "Encinitas"),
        ("cardiff", "Cardiff"),
        ("solana", "Solana"),
        ("delmar", "DelMar"),
        ("torrey", "Torrey"),
        ("blacks", "Blacks"),
    ]

    found = []
    for key, name in location_keywords:
        if key in filename_lower and name not in found:
            found.append(name)

    return found


def extract_mop_range(filename: str) -> Optional[Tuple[int, int]]:
    """Extract MOP range from filename.

    Filenames typically have format: YYYYMMDD_MOPSTART_MOPEND_...
    e.g., '20181121_00567_00622_NoWaves_...' -> (567, 622)

    Parameters
    ----------
    filename : str
        Filename to parse.

    Returns
    -------
    mop_range : tuple of (int, int) or None
        (mop_start, mop_end) or None if not parseable.
    """
    parts = filename.split('_')
    if len(parts) < 3:
        return None

    try:
        mop_start = int(parts[1])
        mop_end = int(parts[2])
        return (mop_start, mop_end)
    except ValueError:
        return None


# MOP ranges for each shapefile (approximate)
SHAPEFILE_MOP_RANGES = {
    "Blacks": (520, 567),
    "Torrey": (567, 581),
    "DelMar": (595, 620),
    "Solana": (637, 666),
    "Cardiff": (637, 666),  # Same as Solana
    "SanElijo": (683, 708),
    "Encinitas": (708, 764),
}


def get_overlapping_locations(mop_start: int, mop_end: int) -> List[str]:
    """Get locations whose shapefiles overlap with given MOP range.

    Parameters
    ----------
    mop_start : int
        Start of MOP range.
    mop_end : int
        End of MOP range.

    Returns
    -------
    locations : list of str
        Location names with overlapping shapefiles.
    """
    overlapping = []
    for location, (shp_start, shp_end) in SHAPEFILE_MOP_RANGES.items():
        # Check for overlap
        if mop_start <= shp_end and mop_end >= shp_start:
            if location not in overlapping:
                overlapping.append(location)
    return overlapping
