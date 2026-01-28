"""
Alongshore energy risk map visualization.

Creates a 10m-binned risk map showing normalized energy (kJ) per alongshore slice,
overlaid on a satellite basemap.

Supports two binning modes:
1. Auto-detect: Determines alongshore axis (X or Y) and bins along it
2. Shapefile transects: Uses user-provided transect lines from shapefile
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from typing import Optional, Tuple, Union, List
from pathlib import Path

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False


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
    transects : list
        Same transects list (for convenience).
    """
    n_transects = len(transects)
    transect_energy = np.zeros(n_transects)

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
        transect_energy[i] = energy[in_corridor].sum()

    return transect_energy, transects


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
    transect_energy, _ = compute_transect_energy(xyz, energy, transects, half_width)

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
        vmax = transect_energy.max() if transect_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # Track bounds for axis limits
    all_x, all_y = [], []

    # Draw transect corridors as polygons
    for i, transect in enumerate(transects):
        corners = get_transect_corners(transect, half_width)
        if corners is None:
            continue

        all_x.extend(corners[:, 0])
        all_y.extend(corners[:, 1])

        color = energy_cmap(norm(transect_energy[i]))
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
    total_energy = transect_energy.sum()
    max_energy = transect_energy.max()
    mean_energy = transect_energy.mean()
    stats_text = (
        f"Transects: {len(transects)}\n"
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
        ax.set_title(f"Transect Energy Risk Map ({len(transects)} transects)", fontsize=12)

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
    transect_energy, _ = compute_transect_energy(xyz, energy, transects, half_width)

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
        vmax = transect_energy.max() if transect_energy.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    colors = [energy_cmap(norm(e)) for e in transect_energy]

    # X positions are just transect indices (1-based for display)
    x_positions = np.arange(1, len(transects) + 1)

    # Draw bars
    ax.bar(
        x_positions, transect_energy,
        width=0.9,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
    )

    # Labels
    ax.set_xlabel("Transect Number", fontsize=10)
    ax.set_ylabel(f"Energy per transect (kJ)", fontsize=10)

    # Set x-axis ticks at reasonable intervals
    if len(transects) > 20:
        tick_step = max(1, len(transects) // 10)
        ax.set_xticks(x_positions[::tick_step])
    else:
        ax.set_xticks(x_positions)

    # Add colorbar
    sm = ScalarMappable(cmap=energy_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Energy (kJ)", fontsize=9)

    # Statistics
    total_energy = transect_energy.sum()
    ax.text(
        0.98, 0.95,
        f"Total: {total_energy:.2f} kJ\nTransects: {len(transects)}",
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