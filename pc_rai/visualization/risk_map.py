"""
Alongshore energy risk map visualization.

Creates a 10m-binned risk map showing normalized energy (kJ) per alongshore slice,
overlaid on a satellite basemap.

Supports two binning modes:
1. Auto-detect: Determines alongshore axis (X or Y) and bins along it
2. Shapefile transects: Uses user-provided transect lines from shapefile

Also provides 3D visualization with DEM terrain and satellite imagery texture.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import Normalize, LinearSegmentedColormap, LightSource
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple, Union, List
from pathlib import Path
import tempfile

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


def parse_transects(transect_path: Union[str, Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse transect lines from a shapefile.

    Each transect should be a LineString (2+ points defining the transect).
    Coordinates are expected to be in the same CRS as the point cloud (UTM).

    Parameters
    ----------
    transect_path : str or Path
        Path to shapefile (.shp) containing transect lines.

    Returns
    -------
    transects : list of (start, end) tuples
        Each tuple contains two np.ndarray of shape (2,) for start/end XY coords.
        Transects are returned in the order they appear in the shapefile.
    """
    transect_path = Path(transect_path)

    # Use pyshp (shapefile) for reading - pure Python, no GDAL dependency
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "pyshp is required for reading shapefiles. "
            "Install with: pip install pyshp"
        )

    # Read shapefile (can pass path without .shp extension)
    shp_path = str(transect_path)
    if shp_path.endswith('.shp'):
        shp_path = shp_path[:-4]

    sf = shapefile.Reader(shp_path)

    transects = []

    for shape in sf.shapes():
        # Get points from the shape
        points = shape.points

        if len(points) >= 2:
            # Use first and last point as transect endpoints
            start = np.array(points[0][:2])  # Take only X, Y (ignore Z if present)
            end = np.array(points[-1][:2])
            transects.append((start, end))

    return transects


# Alias for backwards compatibility
parse_kml_transects = parse_transects


def compute_transect_energy(
    xyz: np.ndarray,
    energy: np.ndarray,
    transects: List[Tuple[np.ndarray, np.ndarray]],
    half_width: float = 5.0,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute total energy for each transect corridor.

    For each transect line, finds all points within half_width meters
    of the line and sums their energy.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    energy : np.ndarray
        (N,) energy values in kJ.
    transects : list of (start, end) tuples
        Transect line endpoints from parse_kml_transects().
    half_width : float
        Half-width of transect corridor in meters (default 5m = 10m total width).

    Returns
    -------
    transect_energy : np.ndarray
        (M,) total energy per transect in kJ.
    transect_counts : np.ndarray
        (M,) number of points in each transect corridor.
    """
    n_transects = len(transects)
    transect_energy = np.zeros(n_transects)
    transect_counts = np.zeros(n_transects, dtype=int)

    xy = xyz[:, :2]  # Just X, Y coordinates

    for i, (start, end) in enumerate(transects):
        # Vector along the transect
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            continue  # Skip degenerate transects

        line_unit = line_vec / line_len

        # Vector from start to each point
        point_vecs = xy - start

        # Project points onto line
        # t = dot(point_vec, line_unit) gives position along line
        t = np.dot(point_vecs, line_unit)

        # Perpendicular distance from line
        # perp = point_vec - t * line_unit
        proj = np.outer(t, line_unit)
        perp_vecs = point_vecs - proj
        perp_dist = np.linalg.norm(perp_vecs, axis=1)

        # Points within corridor: within half_width of line,
        # and between start and end (0 <= t <= line_len)
        in_corridor = (perp_dist <= half_width) & (t >= 0) & (t <= line_len)

        # Sum energy for points in this transect's corridor
        transect_counts[i] = in_corridor.sum()
        transect_energy[i] = energy[in_corridor].sum()

    return transect_energy, transect_counts


def get_transect_center(transect: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Get the center point of a transect line."""
    start, end = transect
    return (start + end) / 2


def get_transect_corners(
    transect: Tuple[np.ndarray, np.ndarray],
    half_width: float = 5.0,
) -> np.ndarray:
    """
    Get the 4 corners of a transect corridor rectangle.

    Returns corners in order suitable for matplotlib Polygon.
    """
    start, end = transect
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return None

    # Perpendicular unit vector
    perp = np.array([-line_vec[1], line_vec[0]]) / line_len

    # Four corners
    corners = np.array([
        start - perp * half_width,
        start + perp * half_width,
        end + perp * half_width,
        end - perp * half_width,
    ])

    return corners


def _get_alongshore_axis(xyz: np.ndarray) -> Tuple[int, float, float]:
    """
    Determine which axis is alongshore (the longer horizontal extent).

    Returns
    -------
    axis : int
        0 for X, 1 for Y
    min_val : float
        Minimum value along that axis
    max_val : float
        Maximum value along that axis
    """
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()

    if x_range >= y_range:
        return 0, xyz[:, 0].min(), xyz[:, 0].max()
    else:
        return 1, xyz[:, 1].min(), xyz[:, 1].max()


def _get_cross_shore_bounds(xyz: np.ndarray, alongshore_axis: int) -> Tuple[float, float]:
    """Get the cross-shore (perpendicular) axis bounds."""
    cross_axis = 1 if alongshore_axis == 0 else 0
    return xyz[:, cross_axis].min(), xyz[:, cross_axis].max()


def compute_binned_energy(
    xyz: np.ndarray,
    energy: np.ndarray,
    bin_size: float = 10.0,
    alongshore_axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute total energy per alongshore bin.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    energy : np.ndarray
        (N,) energy values in kJ.
    bin_size : float
        Bin size in meters (default 10m).
    alongshore_axis : int, optional
        Which axis is alongshore (0=X, 1=Y). Auto-detected if None.

    Returns
    -------
    bin_edges : np.ndarray
        (M+1,) bin edge positions.
    bin_energy : np.ndarray
        (M,) total energy per bin in kJ.
    alongshore_axis : int
        The axis used for binning.
    """
    if alongshore_axis is None:
        alongshore_axis, min_val, max_val = _get_alongshore_axis(xyz)
    else:
        min_val = xyz[:, alongshore_axis].min()
        max_val = xyz[:, alongshore_axis].max()

    # Create bins
    n_bins = int(np.ceil((max_val - min_val) / bin_size))
    bin_edges = np.linspace(min_val, min_val + n_bins * bin_size, n_bins + 1)

    # Assign points to bins
    positions = xyz[:, alongshore_axis]
    bin_indices = np.digitize(positions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Sum energy per bin
    bin_energy = np.zeros(n_bins)
    np.add.at(bin_energy, bin_indices, energy)

    return bin_edges, bin_energy, alongshore_axis


def render_risk_map(
    xyz: np.ndarray,
    energy: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    bin_size: float = 10.0,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300,
    title: Optional[str] = None,
    crs: str = "EPSG:32611",  # UTM Zone 11N (Southern California)
    add_basemap: bool = True,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Create an alongshore energy risk map with satellite basemap.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates in UTM.
    energy : np.ndarray
        (N,) energy values in kJ.
    output_path : str or Path, optional
        If provided, save figure to this path.
    bin_size : float
        Alongshore bin size in meters (default 10m).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Figure title.
    crs : str
        Coordinate reference system for basemap (default UTM 11N).
    add_basemap : bool
        Whether to add satellite basemap (requires contextily).
    vmax : float, optional
        Maximum energy for color scale. If None, uses max bin energy.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Compute binned energy
    bin_edges, bin_energy, alongshore_axis = compute_binned_energy(
        xyz, energy, bin_size
    )

    # Get cross-shore bounds for rectangle height
    cross_min, cross_max = _get_cross_shore_bounds(xyz, alongshore_axis)
    cross_width = cross_max - cross_min

    # Add some padding for visualization
    padding = cross_width * 0.2

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Energy colormap (white -> yellow -> orange -> red)
    colors_list = [
        (0.95, 0.95, 0.95),  # Near-white (0 kJ)
        (1.0, 1.0, 0.7),     # Light yellow
        (1.0, 0.8, 0.4),     # Yellow-orange
        (1.0, 0.5, 0.2),     # Orange
        (0.8, 0.2, 0.1),     # Red-orange
        (0.6, 0.0, 0.0),     # Dark red (max)
    ]
    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", colors_list)

    # Determine color normalization
    if vmax is None:
        vmax = bin_energy.max() if bin_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # Draw rectangles for each bin
    n_bins = len(bin_energy)
    for i in range(n_bins):
        if alongshore_axis == 0:  # X is alongshore
            x = bin_edges[i]
            y = cross_min - padding
            width = bin_size
            height = cross_width + 2 * padding
        else:  # Y is alongshore
            x = cross_min - padding
            y = bin_edges[i]
            width = cross_width + 2 * padding
            height = bin_size

        color = energy_cmap(norm(bin_energy[i]))
        rect = Rectangle(
            (x, y), width, height,
            facecolor=color,
            edgecolor='black',
            linewidth=0.3,
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Set axis limits
    if alongshore_axis == 0:
        ax.set_xlim(bin_edges[0] - bin_size, bin_edges[-1] + bin_size)
        ax.set_ylim(cross_min - padding * 2, cross_max + padding * 2)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    else:
        ax.set_xlim(cross_min - padding * 2, cross_max + padding * 2)
        ax.set_ylim(bin_edges[0] - bin_size, bin_edges[-1] + bin_size)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    # Add satellite basemap if available
    if add_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=crs,
                source=ctx.providers.Esri.WorldImagery,
                alpha=0.6,
                attribution_size=6,
            )
        except ImportError:
            print("  Warning: contextily not installed, skipping basemap")
        except Exception as e:
            print(f"  Warning: Could not add basemap: {e}")

    # Add colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"Total Energy per {bin_size:.0f}m bin (kJ)", fontsize=10)

    # Add statistics annotation
    total_energy = bin_energy.sum()
    max_energy = bin_energy.max()
    mean_energy = bin_energy.mean()
    stats_text = (
        f"Total: {total_energy:.2f} kJ\n"
        f"Max bin: {max_energy:.3f} kJ\n"
        f"Mean bin: {mean_energy:.3f} kJ"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f"Alongshore Energy Risk Map ({bin_size:.0f}m bins)", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig


def render_risk_map_profile(
    xyz: np.ndarray,
    energy: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    bin_size: float = 10.0,
    figsize: Tuple[float, float] = (14, 4),
    dpi: int = 300,
    title: Optional[str] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Create a simple 1D bar chart of alongshore energy.

    This is a simpler alternative to the map view, showing energy
    as a bar chart along the cliff profile.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    energy : np.ndarray
        (N,) energy values in kJ.
    output_path : str or Path, optional
        If provided, save figure to this path.
    bin_size : float
        Alongshore bin size in meters (default 10m).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Figure title.
    vmax : float, optional
        Maximum energy for y-axis. If None, auto-scale.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Compute binned energy
    bin_edges, bin_energy, alongshore_axis = compute_binned_energy(
        xyz, energy, bin_size
    )

    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Color bars by energy value
    colors_list = [
        (0.95, 0.95, 0.95),
        (1.0, 1.0, 0.7),
        (1.0, 0.8, 0.4),
        (1.0, 0.5, 0.2),
        (0.8, 0.2, 0.1),
        (0.6, 0.0, 0.0),
    ]
    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", colors_list)

    if vmax is None:
        vmax = bin_energy.max() if bin_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    colors = [energy_cmap(norm(e)) for e in bin_energy]

    # Draw bars
    ax.bar(
        bin_centers, bin_energy,
        width=bin_size * 0.9,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
    )

    # Labels
    ax.set_xlabel("Distance alongshore (m)", fontsize=10)
    ax.set_ylabel(f"Energy per {bin_size:.0f}m (kJ)", fontsize=10)

    # Add colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Energy (kJ)", fontsize=9)

    # Statistics
    total_energy = bin_energy.sum()
    ax.text(
        0.98, 0.95,
        f"Total: {total_energy:.2f} kJ",
        transform=ax.transAxes,
        fontsize=9,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig


def render_transect_risk_map(
    xyz: np.ndarray,
    energy: np.ndarray,
    transects: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    half_width: float = 5.0,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300,
    title: Optional[str] = None,
    crs: str = "EPSG:32611",
    add_basemap: bool = True,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Create an energy risk map using KML-defined transects.

    Each transect corridor (±half_width from line) is colored by total energy.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates in UTM.
    energy : np.ndarray
        (N,) energy values in kJ.
    transects : list of (start, end) tuples
        Transect line endpoints from parse_kml_transects().
    output_path : str or Path, optional
        If provided, save figure to this path.
    half_width : float
        Half-width of transect corridor in meters (default 5m).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Figure title.
    crs : str
        Coordinate reference system for basemap (default UTM 11N).
    add_basemap : bool
        Whether to add satellite basemap (requires contextily).
    vmax : float, optional
        Maximum energy for color scale. If None, uses max transect energy.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Compute energy per transect
    transect_energy, transect_counts = compute_transect_energy(xyz, energy, transects, half_width)

    # Filter to only transects with valid data (points in corridor)
    valid_mask = transect_counts > 0
    valid_transects = [t for t, v in zip(transects, valid_mask) if v]
    valid_energy = transect_energy[valid_mask]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Energy colormap (white -> yellow -> orange -> red)
    colors_list = [
        (0.95, 0.95, 0.95),  # Near-white (0 kJ)
        (1.0, 1.0, 0.7),     # Light yellow
        (1.0, 0.8, 0.4),     # Yellow-orange
        (1.0, 0.5, 0.2),     # Orange
        (0.8, 0.2, 0.1),     # Red-orange
        (0.6, 0.0, 0.0),     # Dark red (max)
    ]
    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", colors_list)

    # Determine color normalization
    if vmax is None:
        vmax = valid_energy.max() if len(valid_energy) > 0 and valid_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # Track bounds for axis limits
    all_x, all_y = [], []

    # Draw transect corridors as polygons (only where data exists)
    for i, transect in enumerate(valid_transects):
        corners = get_transect_corners(transect, half_width)
        if corners is None:
            continue

        all_x.extend(corners[:, 0])
        all_y.extend(corners[:, 1])

        color = energy_cmap(norm(valid_energy[i]))
        poly = Polygon(
            corners,
            facecolor=color,
            edgecolor='black',
            linewidth=0.3,
            alpha=0.7,
        )
        ax.add_patch(poly)

    # Set axis limits with padding
    if all_x and all_y:
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        padding = max(x_range, y_range) * 0.1

        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Add satellite basemap if available
    if add_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=crs,
                source=ctx.providers.Esri.WorldImagery,
                alpha=0.6,
                attribution_size=6,
            )
        except ImportError:
            print("  Warning: contextily not installed, skipping basemap")
        except Exception as e:
            print(f"  Warning: Could not add basemap: {e}")

    # Add colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"Total Energy per transect (kJ)", fontsize=10)

    # Add statistics annotation
    total_energy = valid_energy.sum()
    max_energy = valid_energy.max() if len(valid_energy) > 0 else 0.0
    mean_energy = valid_energy.mean() if len(valid_energy) > 0 else 0.0
    stats_text = (
        f"Transects: {len(valid_transects)} of {len(transects)}\n"
        f"Total: {total_energy:.2f} kJ\n"
        f"Max: {max_energy:.3f} kJ\n"
        f"Mean: {mean_energy:.3f} kJ"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f"Transect Energy Risk Map ({len(valid_transects)} of {len(transects)} transects)", fontsize=12)

    ax.set_aspect('equal')
    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig


def render_transect_risk_profile(
    xyz: np.ndarray,
    energy: np.ndarray,
    transects: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    half_width: float = 5.0,
    figsize: Tuple[float, float] = (14, 4),
    dpi: int = 300,
    title: Optional[str] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Create a 1D bar chart of energy per transect.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    energy : np.ndarray
        (N,) energy values in kJ.
    transects : list of (start, end) tuples
        Transect line endpoints from parse_kml_transects().
    output_path : str or Path, optional
        If provided, save figure to this path.
    half_width : float
        Half-width of transect corridor in meters (default 5m).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Figure title.
    vmax : float, optional
        Maximum energy for y-axis. If None, auto-scale.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Compute energy per transect
    transect_energy, transect_counts = compute_transect_energy(xyz, energy, transects, half_width)

    # Filter to only transects with valid data (points in corridor)
    valid_mask = transect_counts > 0
    valid_energy = transect_energy[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Color bars by energy value
    colors_list = [
        (0.95, 0.95, 0.95),
        (1.0, 1.0, 0.7),
        (1.0, 0.8, 0.4),
        (1.0, 0.5, 0.2),
        (0.8, 0.2, 0.1),
        (0.6, 0.0, 0.0),
    ]
    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", colors_list)

    if vmax is None:
        vmax = valid_energy.max() if len(valid_energy) > 0 and valid_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    colors = [energy_cmap(norm(e)) for e in valid_energy]

    # X positions use original transect indices (1-based for display)
    x_positions = valid_indices + 1

    # Draw bars
    ax.bar(
        x_positions, valid_energy,
        width=0.9,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
    )

    # Labels
    ax.set_xlabel("Transect Number", fontsize=10)
    ax.set_ylabel(f"Energy per transect (kJ)", fontsize=10)

    # Set x-axis ticks at reasonable intervals
    if len(x_positions) > 20:
        tick_step = max(1, len(x_positions) // 10)
        ax.set_xticks(x_positions[::tick_step])
    else:
        ax.set_xticks(x_positions)

    # Add colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Energy (kJ)", fontsize=9)

    # Statistics
    total_energy = valid_energy.sum()
    ax.text(
        0.98, 0.95,
        f"Total: {total_energy:.2f} kJ\nTransects: {len(valid_energy)} of {len(transects)}",
        transform=ax.transAxes,
        fontsize=9,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig


def _download_dem_for_bounds(
    bounds: Tuple[float, float, float, float],
    crs: str = "EPSG:32611",
    buffer_m: float = 100.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Download DEM data from OpenTopography for the given bounds.

    Parameters
    ----------
    bounds : tuple
        (xmin, ymin, xmax, ymax) in the given CRS.
    crs : str
        Coordinate reference system of bounds (default UTM 11N).
    buffer_m : float
        Buffer in meters to add around bounds.

    Returns
    -------
    X, Y, Z : np.ndarray or None
        Meshgrid arrays for the DEM, or None if download fails.
    """
    if not HAS_PYPROJ or not HAS_RASTERIO:
        print("  Warning: pyproj and rasterio required for DEM download")
        return None

    import requests

    # Convert bounds to WGS84 for OpenTopography API
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    xmin, ymin, xmax, ymax = bounds
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m

    # Transform corners to get lon/lat bounds
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    # OpenTopography Global DEM API (1m resolution USGS 3DEP)
    api_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "USGS1m",  # 1m USGS 3DEP
        "south": lat_min,
        "north": lat_max,
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
    }

    print(f"  Downloading DEM for bounds: {lon_min:.5f}W to {lon_max:.5f}E, {lat_min:.5f}S to {lat_max:.5f}N")

    try:
        response = requests.get(api_url, params=params, timeout=60)
        if response.status_code != 200:
            # Try fallback to SRTM
            print(f"  USGS1m not available, trying SRTMGL1...")
            params["demtype"] = "SRTMGL1"
            response = requests.get(api_url, params=params, timeout=60)

        if response.status_code != 200:
            print(f"  Warning: DEM download failed with status {response.status_code}")
            return None

        # Save to temp file and read with rasterio
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            dem_data = src.read(1)
            dem_transform = src.transform
            dem_crs = src.crs

            # Create coordinate arrays
            rows, cols = dem_data.shape
            xs = np.array([dem_transform[2] + dem_transform[0] * c for c in range(cols)])
            ys = np.array([dem_transform[5] + dem_transform[4] * r for r in range(rows)])

            X, Y = np.meshgrid(xs, ys)

            # Reproject to target CRS if needed
            if str(dem_crs) != crs:
                transformer_back = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                X_flat, Y_flat = transformer_back.transform(X.flatten(), Y.flatten())
                X = X_flat.reshape(X.shape)
                Y = Y_flat.reshape(Y.shape)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

        return X, Y, dem_data

    except Exception as e:
        print(f"  Warning: DEM download failed: {e}")
        return None


def _fetch_satellite_for_bounds(
    bounds: Tuple[float, float, float, float],
    crs: str = "EPSG:32611",
    buffer_m: float = 100.0,
    resolution: int = 1,
) -> Optional[Tuple[np.ndarray, Tuple]]:
    """
    Fetch satellite imagery for the given bounds.

    Parameters
    ----------
    bounds : tuple
        (xmin, ymin, xmax, ymax) in the given CRS.
    crs : str
        Coordinate reference system of bounds.
    buffer_m : float
        Buffer in meters to add around bounds.
    resolution : int
        Target resolution in meters.

    Returns
    -------
    img, extent : tuple or None
        RGB image array and (xmin, xmax, ymin, ymax) extent.
    """
    if not HAS_CONTEXTILY or not HAS_PYPROJ:
        return None

    try:
        xmin, ymin, xmax, ymax = bounds
        xmin -= buffer_m
        ymin -= buffer_m
        xmax += buffer_m
        ymax += buffer_m

        # Contextily needs Web Mercator bounds
        transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
        wm_xmin, wm_ymin = transformer.transform(xmin, ymin)
        wm_xmax, wm_ymax = transformer.transform(xmax, ymax)

        # Fetch tiles
        img, ext = ctx.bounds2img(
            wm_xmin, wm_ymin, wm_xmax, wm_ymax,
            source=ctx.providers.Esri.WorldImagery,
            ll=False,
        )

        # Reproject image to target CRS
        transformer_back = Transformer.from_crs("EPSG:3857", crs, always_xy=True)

        # Create new extent in target CRS
        new_xmin, new_ymin = transformer_back.transform(ext[0], ext[2])
        new_xmax, new_ymax = transformer_back.transform(ext[1], ext[3])

        return img, (new_xmin, new_xmax, new_ymin, new_ymax)

    except Exception as e:
        print(f"  Warning: Satellite fetch failed: {e}")
        return None


def render_transect_risk_map_3d(
    xyz: np.ndarray,
    energy: np.ndarray,
    transects: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    half_width: float = 5.0,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300,
    title: Optional[str] = None,
    crs: str = "EPSG:32611",
    vertical_exaggeration: float = 2.0,
    azimuth: float = -130,
    elevation: float = 45,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Create a 3D energy risk map with DEM terrain and satellite imagery.

    Only shows the area where transects contain data, zoomed to that extent.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates in UTM.
    energy : np.ndarray
        (N,) energy values in kJ.
    transects : list of (start, end) tuples
        Transect line endpoints from parse_transects().
    output_path : str or Path, optional
        If provided, save figure to this path.
    half_width : float
        Half-width of transect corridor in meters (default 5m).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Figure title.
    crs : str
        Coordinate reference system (default UTM 11N).
    vertical_exaggeration : float
        Factor to exaggerate elevation (default 2.0).
    azimuth : float
        Viewing azimuth angle in degrees.
    elevation : float
        Viewing elevation angle in degrees.
    vmax : float, optional
        Maximum energy for color scale. If None, uses max transect energy.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Compute energy per transect
    transect_energy, transect_counts = compute_transect_energy(xyz, energy, transects, half_width)

    # Filter to only transects with valid data
    valid_mask = transect_counts > 0
    valid_transects = [t for t, v in zip(transects, valid_mask) if v]
    valid_energy = transect_energy[valid_mask]

    if len(valid_transects) == 0:
        print("  Warning: No transects contain data points")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No data in transects", ha='center', va='center')
        return fig

    # Get bounds of valid transects only
    all_x, all_y = [], []
    for start, end in valid_transects:
        all_x.extend([start[0], end[0]])
        all_y.extend([start[1], end[1]])

    xmin, xmax = min(all_x) - half_width, max(all_x) + half_width
    ymin, ymax = min(all_y) - half_width, max(all_y) + half_width
    bounds = (xmin, ymin, xmax, ymax)

    print(f"  Valid transects: {len(valid_transects)} of {len(transects)}")
    print(f"  Data bounds: X=[{xmin:.1f}, {xmax:.1f}], Y=[{ymin:.1f}, {ymax:.1f}]")

    # Create figure with 3D axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Energy colormap
    colors_list = [
        (0.95, 0.95, 0.95, 0.7),  # Near-white with alpha
        (1.0, 1.0, 0.7, 0.75),
        (1.0, 0.8, 0.4, 0.8),
        (1.0, 0.5, 0.2, 0.85),
        (0.8, 0.2, 0.1, 0.9),
        (0.6, 0.0, 0.0, 0.95),
    ]
    energy_cmap = LinearSegmentedColormap.from_list("energy_risk", colors_list)

    if vmax is None:
        vmax = valid_energy.max() if len(valid_energy) > 0 and valid_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # Try to download DEM
    dem_result = _download_dem_for_bounds(bounds, crs, buffer_m=50.0)

    if dem_result is not None:
        X, Y, Z = dem_result
        Z = Z * vertical_exaggeration

        # Try to get satellite texture
        sat_result = _fetch_satellite_for_bounds(bounds, crs, buffer_m=50.0)

        if sat_result is not None:
            img, extent = sat_result
            # Create facecolors from satellite image
            # Resample satellite to match DEM grid
            from scipy.ndimage import map_coordinates

            # Normalize coordinates to image space
            img_rows, img_cols = img.shape[:2]
            x_norm = (X - extent[0]) / (extent[1] - extent[0]) * (img_cols - 1)
            y_norm = (1 - (Y - extent[2]) / (extent[3] - extent[2])) * (img_rows - 1)

            # Sample RGB channels
            facecolors = np.zeros((*X.shape, 4))
            for c in range(3):
                facecolors[:, :, c] = map_coordinates(
                    img[:, :, c].astype(float),
                    [y_norm.flatten(), x_norm.flatten()],
                    order=1,
                ).reshape(X.shape) / 255.0
            facecolors[:, :, 3] = 0.8  # Alpha

            # Apply hillshade lighting
            ls = LightSource(azdeg=315, altdeg=45)
            shaded = ls.shade_rgb(facecolors[:, :, :3], Z / vertical_exaggeration)
            facecolors[:, :, :3] = shaded

            ax.plot_surface(
                X, Y, Z,
                facecolors=facecolors,
                rstride=1, cstride=1,
                antialiased=True,
                shade=False,
            )
        else:
            # Just plot DEM with terrain colormap
            ax.plot_surface(
                X, Y, Z,
                cmap='terrain',
                rstride=1, cstride=1,
                antialiased=True,
                alpha=0.7,
            )

        z_base = Z.min()
        z_offset = (Z.max() - Z.min()) * 0.02  # Small offset above terrain

    else:
        # No DEM available - create flat base
        print("  No DEM available, using flat base")
        z_base = 0
        z_offset = 0.5

        # Create flat surface with satellite if available
        sat_result = _fetch_satellite_for_bounds(bounds, crs, buffer_m=50.0)
        if sat_result is not None:
            img, extent = sat_result
            # Create meshgrid for flat surface
            X = np.linspace(extent[0], extent[1], img.shape[1])
            Y = np.linspace(extent[2], extent[3], img.shape[0])
            X, Y = np.meshgrid(X, Y)
            Z = np.zeros_like(X)

            # Normalize image for facecolors
            facecolors = img[::-1, :, :3] / 255.0  # Flip Y axis

            ax.plot_surface(
                X, Y, Z,
                facecolors=facecolors,
                rstride=1, cstride=1,
                antialiased=True,
                shade=False,
            )

    # Draw transect corridors as 3D polygons elevated above terrain
    for i, transect in enumerate(valid_transects):
        corners = get_transect_corners(transect, half_width)
        if corners is None:
            continue

        # Get z elevation for this transect (sample from DEM or use base)
        if dem_result is not None:
            # Find approximate Z values at transect corners
            from scipy.interpolate import RegularGridInterpolator

            try:
                # Create interpolator from DEM
                dem_X, dem_Y, dem_Z = dem_result
                dem_Z_scaled = dem_Z * vertical_exaggeration

                # Get unique sorted coordinates
                x_unique = np.unique(dem_X[0, :])
                y_unique = np.unique(dem_Y[:, 0])[::-1]  # Flip for correct order
                z_grid = dem_Z_scaled[::-1, :]  # Flip corresponding Z

                interp = RegularGridInterpolator(
                    (y_unique, x_unique), z_grid,
                    bounds_error=False, fill_value=z_base
                )

                # Sample Z at corner positions
                corner_z = interp(corners[:, ::-1])  # Swap x,y order for interpolator
                z_height = corner_z.max() + z_offset
            except Exception:
                z_height = z_base + z_offset
        else:
            z_height = z_base + z_offset

        # Create 3D polygon vertices
        verts_3d = [(c[0], c[1], z_height) for c in corners]

        color = energy_cmap(norm(valid_energy[i]))
        poly = Poly3DCollection(
            [verts_3d],
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_collection3d(poly)

    # Set axis limits
    padding = max(xmax - xmin, ymax - ymin) * 0.05
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)

    if dem_result is not None:
        ax.set_zlim(dem_result[2].min() * vertical_exaggeration * 0.9,
                    dem_result[2].max() * vertical_exaggeration * 1.1)
    else:
        ax.set_zlim(-1, 10)

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Labels
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.set_zlabel('Elevation (m)', fontsize=10)

    # Add colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Energy per transect (kJ)', fontsize=10)

    # Statistics text
    total_energy = valid_energy.sum()
    max_energy = valid_energy.max() if len(valid_energy) > 0 else 0.0
    stats_text = (
        f"Transects: {len(valid_transects)}/{len(transects)}\n"
        f"Total: {total_energy:.2f} kJ\n"
        f"Max: {max_energy:.3f} kJ"
    )
    ax.text2D(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    else:
        ax.set_title(
            f"3D Transect Energy Risk Map\n(VE={vertical_exaggeration}x)",
            fontsize=12
        )

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig