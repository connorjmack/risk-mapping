#!/usr/bin/env python3
"""Generate a county-wide coastal risk map from multiple LAZ surveys.

Combines processed *_rai.laz files onto transect corridors defined by a
shapefile.  Each transect is colored by the energy from the most recent
survey that covers it.  Transects with no coverage are drawn as gray
outlines.

Usage
-----
    python scripts/risk_map_regional.py --laz-dir output/subsamp1/rai/

Outputs to output/figures/main/county_risk.png by default.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from pyproj import Transformer

from pc_rai.visualization.risk_map import (
    compute_transect_energy,
    get_transect_corners,
    parse_transects,
)

# Energy colormap matching existing visualizations (white → red)
ENERGY_COLORS = [
    (0.95, 0.95, 0.95),  # Near-white (0 kJ)
    (1.0, 1.0, 0.7),     # Light yellow
    (1.0, 0.8, 0.4),     # Yellow-orange
    (1.0, 0.5, 0.2),     # Orange
    (0.8, 0.2, 0.1),     # Red-orange
    (0.6, 0.0, 0.0),     # Dark red (max)
]

_DATE_RE = re.compile(r"^(\d{8})")


def parse_date_from_filename(path: Path) -> Optional[datetime]:
    """Extract YYYYMMDD survey date from a LAZ filename.

    Parameters
    ----------
    path : Path
        Path to a LAZ file whose stem starts with YYYYMMDD.

    Returns
    -------
    datetime or None
        Parsed date, or None if the pattern doesn't match.
    """
    match = _DATE_RE.match(path.stem)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def find_overlapping_transects(
    laz_bounds: Tuple[float, float, float, float],
    transects: List[Tuple[np.ndarray, np.ndarray]],
    half_width: float,
) -> List[int]:
    """Find transects whose corridors intersect the LAZ bounding box.

    Parameters
    ----------
    laz_bounds : tuple
        (xmin, ymin, xmax, ymax) of the LAZ point cloud.
    transects : list of (start, end) tuples
        All transect endpoints.
    half_width : float
        Half-width of transect corridors in meters.

    Returns
    -------
    list of int
        Indices of overlapping transects.
    """
    lx_min, ly_min, lx_max, ly_max = laz_bounds
    indices = []

    for i, (start, end) in enumerate(transects):
        # Transect corridor bounding box
        tx_min = min(start[0], end[0]) - half_width
        tx_max = max(start[0], end[0]) + half_width
        ty_min = min(start[1], end[1]) - half_width
        ty_max = max(start[1], end[1]) + half_width

        # Axis-aligned bbox overlap test
        if tx_max >= lx_min and tx_min <= lx_max and ty_max >= ly_min and ty_min <= ly_max:
            indices.append(i)

    return indices


def process_laz_surveys(
    laz_dir: Path,
    transects: List[Tuple[np.ndarray, np.ndarray]],
    half_width: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Process all LAZ files and aggregate transect energy data.

    Surveys are processed chronologically (oldest first) so that newer
    data overwrites older data for overlapping transects.

    Parameters
    ----------
    laz_dir : Path
        Directory containing ``*_rai.laz`` files.
    transects : list of (start, end)
        Transect endpoints from shapefile.
    half_width : float
        Corridor half-width in meters.

    Returns
    -------
    dict
        Keys: ``energy`` (float64), ``counts`` (int), ``dates`` (object),
        ``has_data`` (bool).  All arrays have length ``len(transects)``.
    """
    n = len(transects)
    energy = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=int)
    dates = np.empty(n, dtype=object)  # datetime or None
    has_data = np.zeros(n, dtype=bool)

    # Discover and sort LAZ files by date
    laz_files = sorted(laz_dir.glob("*_rai.laz"))
    if not laz_files:
        laz_files = sorted(laz_dir.glob("*_rai.LAZ"))
    if not laz_files:
        print(f"  No *_rai.laz files found in {laz_dir}")
        return {"energy": energy, "counts": counts, "dates": dates, "has_data": has_data}

    dated = []
    for f in laz_files:
        d = parse_date_from_filename(f)
        if d is None:
            print(f"  Warning: could not parse date from {f.name}, skipping")
            continue
        dated.append((d, f))

    # Sort oldest first so newest overwrites
    dated.sort(key=lambda x: x[0])

    print(f"  Found {len(dated)} dated LAZ files spanning "
          f"{dated[0][0].strftime('%Y-%m-%d')} to {dated[-1][0].strftime('%Y-%m-%d')}")

    for survey_date, laz_path in dated:
        print(f"  Processing {laz_path.name} ({survey_date.strftime('%Y-%m-%d')}) ...")
        las = laspy.read(str(laz_path))
        xyz = np.column_stack([las.x, las.y, las.z])

        try:
            e = np.asarray(las["energy_kj_knn"], dtype=np.float64)
        except Exception:
            print(f"    Warning: no energy_kj_knn dimension, skipping")
            continue

        bounds = (xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max(), xyz[:, 1].max())
        overlap_idx = find_overlapping_transects(bounds, transects, half_width)

        if not overlap_idx:
            print(f"    No transect overlap, skipping")
            continue

        # Build subset of transects for compute_transect_energy
        sub_transects = [transects[j] for j in overlap_idx]
        t_energy, t_counts = compute_transect_energy(xyz, e, sub_transects, half_width)

        # Overwrite stored data for transects that have points
        for k, j in enumerate(overlap_idx):
            if t_counts[k] > 0:
                energy[j] = t_energy[k]
                counts[j] = t_counts[k]
                dates[j] = survey_date
                has_data[j] = True

        n_updated = (t_counts > 0).sum()
        print(f"    Updated {n_updated}/{len(overlap_idx)} transects")

    return {"energy": energy, "counts": counts, "dates": dates, "has_data": has_data}


def render_regional_risk_map(
    transects: List[Tuple[np.ndarray, np.ndarray]],
    transect_data: Dict[str, np.ndarray],
    output_path: Path,
    half_width: float = 5.0,
    figsize: Tuple[float, float] = (12, 20),
    dpi: int = 300,
    vmax: Optional[float] = None,
    crs: str = "EPSG:32611",
    title: str = "Coastal Risk Map",
    buffer_m: float = 200.0,
) -> plt.Figure:
    """Render county-wide risk map with satellite basemap.

    Plots in Web Mercator (EPSG:3857) for high-resolution satellite tiles,
    matching the style of ``render_transect_risk_map_3d``.

    Parameters
    ----------
    transects : list of (start, end)
        All transect endpoints.
    transect_data : dict
        Output from :func:`process_laz_surveys`.
    output_path : Path
        Where to save the figure.
    half_width : float
        Corridor half-width in meters.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output DPI.
    vmax : float, optional
        Max energy for colorbar.  Auto-detected if None.
    crs : str
        Coordinate reference system string.
    title : str
        Figure title.
    buffer_m : float
        Buffer around transects for satellite imagery in meters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    energy = transect_data["energy"]
    has_data = transect_data["has_data"]
    dates = transect_data["dates"]

    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", ENERGY_COLORS)

    valid_energy = energy[has_data]
    if vmax is None:
        vmax = valid_energy.max() if len(valid_energy) > 0 and valid_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # Compute bounds from ALL transects (UTM)
    all_starts = np.array([t[0] for t in transects])
    all_ends = np.array([t[1] for t in transects])
    all_pts = np.vstack([all_starts, all_ends])
    xmin = all_pts[:, 0].min() - half_width - buffer_m
    xmax = all_pts[:, 0].max() + half_width + buffer_m
    ymin = all_pts[:, 1].min() - half_width - buffer_m
    ymax = all_pts[:, 1].max() + half_width + buffer_m

    # Transform to Web Mercator for plotting
    transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    wm_xmin, wm_ymin = transformer.transform(xmin, ymin)
    wm_xmax, wm_ymax = transformer.transform(xmax, ymax)

    print(f"  Map extent (UTM): X=[{xmin:.0f}, {xmax:.0f}], Y=[{ymin:.0f}, {ymax:.0f}]")
    print(f"  Map extent (WebMerc): X=[{wm_xmin:.0f}, {wm_xmax:.0f}], Y=[{wm_ymin:.0f}, {wm_ymax:.0f}]")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(wm_xmin, wm_xmax)
    ax.set_ylim(wm_ymin, wm_ymax)

    # Satellite basemap (fetch before drawing transects so tiles are behind)
    try:
        import contextily as ctx

        extent_width_m = xmax - xmin
        if extent_width_m < 2000:
            zoom_level = 17
        elif extent_width_m < 10000:
            zoom_level = 15
        else:
            zoom_level = "auto"

        print(f"  Fetching satellite imagery (zoom={zoom_level}) ...")
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            zoom=zoom_level,
            attribution_size=6,
        )
        print("  Basemap loaded")
    except ImportError:
        print("  Warning: contextily not installed, skipping basemap")
        ax.set_facecolor("lightgray")
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")
        ax.set_facecolor("#d4e4bc")

    # Draw all transects (transform corners to Web Mercator)
    for i, transect in enumerate(transects):
        corners = get_transect_corners(transect, half_width)
        if corners is None:
            continue

        # Transform corners to Web Mercator
        corners_wm = []
        for c in corners:
            wm_x, wm_y = transformer.transform(c[0], c[1])
            corners_wm.append((wm_x, wm_y))

        if has_data[i]:
            color = energy_cmap(norm(energy[i]))
            poly = Polygon(
                corners_wm,
                facecolor=color,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.7,
                zorder=5,
            )
        else:
            poly = Polygon(
                corners_wm,
                facecolor="none",
                edgecolor="gray",
                linewidth=0.2,
                alpha=0.3,
                zorder=4,
            )
        ax.add_patch(poly)

    # Colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Total Energy per transect (kJ)", fontsize=10)

    # Statistics box
    n_total = len(transects)
    n_covered = int(has_data.sum())
    total_energy = valid_energy.sum()
    max_e = valid_energy.max() if len(valid_energy) > 0 else 0.0
    mean_e = valid_energy.mean() if len(valid_energy) > 0 else 0.0

    # Date range
    valid_dates = [d for d in dates if d is not None]
    if valid_dates:
        date_min = min(valid_dates).strftime("%Y-%m-%d")
        date_max = max(valid_dates).strftime("%Y-%m-%d")
        date_str = f"Surveys: {date_min} to {date_max}"
    else:
        date_str = "No survey dates"

    stats_text = (
        f"Transects: {n_covered}/{n_total} ({100 * n_covered / n_total:.0f}%)\n"
        f"Total: {total_energy:.2f} kJ\n"
        f"Max: {max_e:.3f} kJ\n"
        f"Mean: {mean_e:.3f} kJ\n"
        f"{date_str}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        zorder=10,
    )

    # Clean map styling — no axis ticks, matching render_transect_risk_map_3d
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate county-wide coastal risk map from processed LAZ surveys."
    )
    parser.add_argument(
        "--laz-dir",
        type=Path,
        required=True,
        help="Directory containing *_rai.laz files.",
    )
    parser.add_argument(
        "--transects",
        type=Path,
        default=Path("utiliies/transects_10m/transect_lines.shp"),
        help="Path to transect shapefile (default: utiliies/transects_10m/transect_lines.shp).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/main/county_risk.png"),
        help="Output image path (default: output/figures/main/county_risk.png).",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=5.0,
        help="Transect corridor half-width in meters (default: 5.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Max energy for color scale.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=200.0,
        help="Buffer around transects for satellite imagery in meters (default: 200).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="San Diego County Coastal Risk Map",
        help="Figure title.",
    )
    args = parser.parse_args()

    if not args.laz_dir.is_dir():
        raise FileNotFoundError(f"LAZ directory not found: {args.laz_dir}")
    if not args.transects.exists():
        raise FileNotFoundError(f"Transect shapefile not found: {args.transects}")

    print("Parsing transects ...")
    transects = parse_transects(args.transects)
    print(f"  {len(transects)} transects loaded")

    print("Processing LAZ surveys ...")
    transect_data = process_laz_surveys(args.laz_dir, transects, args.half_width)

    print("Rendering regional risk map ...")
    fig = render_regional_risk_map(
        transects,
        transect_data,
        output_path=args.output,
        half_width=args.half_width,
        dpi=args.dpi,
        vmax=args.vmax,
        title=args.title,
        buffer_m=args.buffer,
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
