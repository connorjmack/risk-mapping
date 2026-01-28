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
import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle
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


def _create_hillshade_from_transects(
    transects: List[Tuple[np.ndarray, np.ndarray]],
    has_data: np.ndarray,
    energy: np.ndarray,
    transformer,
    bounds_wm: Tuple[float, float, float, float],
    resolution: int = 500,
) -> Optional[np.ndarray]:
    """Create a pseudo-hillshade from transect energy for visual depth.

    This creates a subtle relief effect based on energy variation
    along the coastline.

    Parameters
    ----------
    transects : list
        Transect endpoints.
    has_data : np.ndarray
        Boolean mask for transects with data.
    energy : np.ndarray
        Energy values per transect.
    transformer : Transformer
        Coordinate transformer (UTM to Web Mercator).
    bounds_wm : tuple
        (xmin, ymin, xmax, ymax) in Web Mercator.
    resolution : int
        Grid resolution for hillshade.

    Returns
    -------
    np.ndarray or None
        RGBA hillshade array or None if insufficient data.
    """
    from matplotlib.colors import LightSource

    wm_xmin, wm_ymin, wm_xmax, wm_ymax = bounds_wm

    # Create grid
    x_grid = np.linspace(wm_xmin, wm_xmax, resolution)
    y_grid = np.linspace(wm_ymin, wm_ymax, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Initialize elevation surface (will represent energy-based "elevation")
    Z = np.zeros_like(X)

    # Accumulate energy at each grid cell based on nearby transects
    for i, transect in enumerate(transects):
        if not has_data[i]:
            continue

        start, end = transect
        # Transform transect center to Web Mercator
        center = (start + end) / 2
        cx, cy = transformer.transform(center[0], center[1])

        # Add energy contribution as a gaussian bump
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        sigma = (wm_xmax - wm_xmin) / 50  # Spread based on map width
        Z += energy[i] * np.exp(-dist_sq / (2 * sigma ** 2))

    if Z.max() < 1e-10:
        return None

    # Normalize
    Z = Z / Z.max()

    # Create hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(Z, vert_exag=2.0)

    # Convert to RGBA with transparency
    rgba = np.zeros((resolution, resolution, 4))
    rgba[:, :, 0] = hillshade
    rgba[:, :, 1] = hillshade
    rgba[:, :, 2] = hillshade
    rgba[:, :, 3] = 0.3  # Subtle overlay

    return rgba


def _add_scale_bar_regional(
    ax: plt.Axes,
    transformer,
    crs: str,
    position: str = "lower-right",
) -> None:
    """Add a scale bar to the regional map.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    transformer : Transformer
        Coordinate transformer.
    crs : str
        Source CRS (e.g., EPSG:32611).
    position : str
        Position: "lower-right", "lower-left", "upper-right", "upper-left".
    """
    # Get axis limits in Web Mercator
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Estimate map width in meters using inverse transform at center
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2

    inv_transformer = Transformer.from_crs("EPSG:3857", crs, always_xy=True)
    utm_x1, utm_y1 = inv_transformer.transform(xlim[0], y_center)
    utm_x2, utm_y2 = inv_transformer.transform(xlim[1], y_center)
    map_width_m = abs(utm_x2 - utm_x1)

    # Choose appropriate scale bar length
    if map_width_m > 50000:
        bar_length_m = 10000
        label = "10 km"
    elif map_width_m > 20000:
        bar_length_m = 5000
        label = "5 km"
    elif map_width_m > 10000:
        bar_length_m = 2000
        label = "2 km"
    elif map_width_m > 5000:
        bar_length_m = 1000
        label = "1 km"
    elif map_width_m > 2000:
        bar_length_m = 500
        label = "500 m"
    elif map_width_m > 500:
        bar_length_m = 100
        label = "100 m"
    else:
        bar_length_m = 50
        label = "50 m"

    # Convert bar length to Web Mercator
    bar_length_wm = (xlim[1] - xlim[0]) * (bar_length_m / map_width_m)

    # Position
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    margin = 0.05

    if "right" in position:
        bar_x = xlim[1] - x_range * margin - bar_length_wm
    else:
        bar_x = xlim[0] + x_range * margin

    if "lower" in position:
        bar_y = ylim[0] + y_range * margin
    else:
        bar_y = ylim[1] - y_range * margin

    # Draw scale bar
    bar_height = y_range * 0.008
    rect = plt.Rectangle(
        (bar_x, bar_y),
        bar_length_wm,
        bar_height,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=15,
    )
    ax.add_patch(rect)

    # Add alternating black/white segments for better readability
    segment_width = bar_length_wm / 4
    for j in range(4):
        color = "black" if j % 2 == 0 else "white"
        segment = plt.Rectangle(
            (bar_x + j * segment_width, bar_y),
            segment_width,
            bar_height,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=16,
        )
        ax.add_patch(segment)

    # Label
    ax.text(
        bar_x + bar_length_wm / 2,
        bar_y + bar_height * 2.5,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="white",
        path_effects=[
            plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
        ],
        zorder=17,
    )


def render_regional_risk_map(
    transects: List[Tuple[np.ndarray, np.ndarray]],
    transect_data: Dict[str, np.ndarray],
    output_path: Path,
    half_width: float = 5.0,
    figsize: Tuple[float, float] = (12, 16),
    dpi: int = 300,
    vmax: Optional[float] = None,
    crs: str = "EPSG:32611",
    title: str = "Coastal Risk Map",
    subtitle: Optional[str] = None,
    buffer_m: float = 200.0,
    show_hillshade: bool = True,
    show_scale_bar: bool = True,
    show_empty_transects: bool = True,
) -> plt.Figure:
    """Render publication-quality county-wide risk map with hillshading.

    Creates a professional map with satellite basemap, hillshade overlay,
    and transect corridors colored by energy. Zooms to show only the
    area with data coverage.

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
        Max energy for colorbar. Auto-detected if None.
    crs : str
        Coordinate reference system string.
    title : str
        Figure title.
    subtitle : str, optional
        Subtitle (e.g., survey date range).
    buffer_m : float
        Buffer around data extent for imagery in meters.
    show_hillshade : bool
        Whether to add hillshade overlay for depth.
    show_scale_bar : bool
        Whether to add a scale bar.
    show_empty_transects : bool
        Whether to show transects without data as gray outlines.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patheffects as patheffects

    energy = transect_data["energy"]
    has_data = transect_data["has_data"]
    dates = transect_data["dates"]

    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", ENERGY_COLORS)

    valid_energy = energy[has_data]
    if vmax is None:
        vmax = valid_energy.max() if len(valid_energy) > 0 and valid_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # Compute bounds from ONLY transects with data (zoom to data extent)
    valid_transects = [t for t, has in zip(transects, has_data) if has]
    if len(valid_transects) > 0:
        valid_starts = np.array([t[0] for t in valid_transects])
        valid_ends = np.array([t[1] for t in valid_transects])
        valid_pts = np.vstack([valid_starts, valid_ends])
        xmin = valid_pts[:, 0].min() - half_width - buffer_m
        xmax = valid_pts[:, 0].max() + half_width + buffer_m
        ymin = valid_pts[:, 1].min() - half_width - buffer_m
        ymax = valid_pts[:, 1].max() + half_width + buffer_m
    else:
        # Fallback to all transects if no data
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

    print(f"  Data extent (UTM): X=[{xmin:.0f}, {xmax:.0f}], Y=[{ymin:.0f}, {ymax:.0f}]")
    print(f"  Data extent (WebMerc): X=[{wm_xmin:.0f}, {wm_xmax:.0f}], Y=[{wm_ymin:.0f}, {wm_ymax:.0f}]")

    # Create figure with clean white background
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.set_xlim(wm_xmin, wm_xmax)
    ax.set_ylim(wm_ymin, wm_ymax)

    # Calculate extent for zoom level selection
    extent_width_m = xmax - xmin

    # Satellite basemap
    basemap_loaded = False
    try:
        import contextily as ctx

        if extent_width_m < 2000:
            zoom_level = 17
        elif extent_width_m < 5000:
            zoom_level = 16
        elif extent_width_m < 10000:
            zoom_level = 15
        elif extent_width_m < 30000:
            zoom_level = 14
        else:
            zoom_level = 13

        print(f"  Fetching satellite imagery (zoom={zoom_level}) ...")
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            zoom=zoom_level,
            attribution_size=5,
        )
        basemap_loaded = True
        print("  Basemap loaded")
    except ImportError:
        print("  Warning: contextily not installed, skipping basemap")
        ax.set_facecolor("#e8e8e8")
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")
        ax.set_facecolor("#d4e4bc")

    # Add hillshade overlay for depth/texture
    if show_hillshade and basemap_loaded:
        try:
            hillshade = _create_hillshade_from_transects(
                transects, has_data, energy, transformer,
                (wm_xmin, wm_ymin, wm_xmax, wm_ymax),
                resolution=400,
            )
            if hillshade is not None:
                ax.imshow(
                    hillshade,
                    extent=[wm_xmin, wm_xmax, wm_ymin, wm_ymax],
                    origin="lower",
                    aspect="auto",
                    zorder=2,
                    alpha=0.25,
                )
                print("  Hillshade overlay added")
        except Exception as e:
            print(f"  Warning: Could not create hillshade: {e}")

    # Draw transects (with data on top, empty ones behind)
    for i, transect in enumerate(transects):
        if not has_data[i] and not show_empty_transects:
            continue

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
                linewidth=0.4,
                alpha=0.85,
                zorder=5,
            )
        else:
            poly = Polygon(
                corners_wm,
                facecolor="none",
                edgecolor="#666666",
                linewidth=0.15,
                alpha=0.25,
                zorder=3,
            )
        ax.add_patch(poly)

    # Professional colorbar
    cbar_ax = fig.add_axes([0.88, 0.25, 0.025, 0.5])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Total Energy (kJ)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Statistics box (top-left, clean styling)
    n_total = len(transects)
    n_covered = int(has_data.sum())
    total_energy = valid_energy.sum()
    max_e = valid_energy.max() if len(valid_energy) > 0 else 0.0
    mean_e = valid_energy.mean() if len(valid_energy) > 0 else 0.0

    stats_text = (
        f"Coverage: {n_covered:,} / {n_total:,} transects ({100 * n_covered / n_total:.1f}%)\n"
        f"Total Energy: {total_energy:.2f} kJ\n"
        f"Max: {max_e:.4f} kJ  •  Mean: {mean_e:.4f} kJ"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.92,
            edgecolor="#cccccc",
            linewidth=0.8,
        ),
        zorder=15,
    )

    # Date range box (below stats)
    valid_dates = [d for d in dates if d is not None]
    if valid_dates:
        date_min = min(valid_dates).strftime("%b %d, %Y")
        date_max = max(valid_dates).strftime("%b %d, %Y")
        if date_min == date_max:
            date_str = f"Survey: {date_min}"
        else:
            date_str = f"Surveys: {date_min} – {date_max}"

        ax.text(
            0.02, 0.88, date_str,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontstyle="italic",
            color="#444444",
            zorder=15,
        )

    # Add scale bar
    if show_scale_bar:
        try:
            _add_scale_bar_regional(ax, transformer, crs, position="lower-right")
        except Exception as e:
            print(f"  Warning: Could not add scale bar: {e}")

    # Clean map styling
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title with optional subtitle
    if title:
        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=15,
            loc="left",
        )

    if subtitle:
        # Add subtitle on right side of title area
        ax.text(
            0.99, 1.02, subtitle,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            fontstyle="italic",
            color="#555555",
        )

    plt.tight_layout(rect=[0, 0, 0.86, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
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
