"""
3D point cloud rendering for RAI visualization.

Uses matplotlib for rendering classified point clouds to static images.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from typing import Dict, Optional, Tuple

from pc_rai.config import RAI_CLASS_COLORS, RAI_CLASS_NAMES


def create_rai_colormap() -> ListedColormap:
    """
    Create matplotlib colormap for RAI classes.

    Returns
    -------
    ListedColormap
        Colormap with 8 colors for RAI classes 0-7.
    """
    colors = [RAI_CLASS_COLORS[i] for i in range(8)]
    return ListedColormap(colors, name="rai_classes")


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def get_class_colors(classes: np.ndarray) -> np.ndarray:
    """
    Get RGB colors for each point based on class.

    Parameters
    ----------
    classes : np.ndarray
        (N,) array of class codes 0-7.

    Returns
    -------
    np.ndarray
        (N, 3) array of RGB colors (0-1 range).
    """
    colors = np.zeros((len(classes), 3), dtype=np.float32)
    for class_code, hex_color in RAI_CLASS_COLORS.items():
        mask = classes == class_code
        if np.any(mask):
            rgb = hex_to_rgb(hex_color)
            colors[mask] = rgb
    return colors


def get_view_params(view: str, bounds: Dict[str, Tuple[float, float]]) -> Dict:
    """
    Get camera parameters for different view angles.

    Parameters
    ----------
    view : str
        View name: "front", "oblique", "top", "side".
    bounds : dict
        Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples.

    Returns
    -------
    dict
        Dictionary with 'elev' and 'azim' for matplotlib 3D view.
    """
    view_params = {
        "front": {"elev": 0, "azim": -90},
        "oblique": {"elev": 30, "azim": -60},
        "top": {"elev": 90, "azim": -90},
        "side": {"elev": 0, "azim": 0},
    }

    if view not in view_params:
        raise ValueError(f"Unknown view '{view}'. Options: {list(view_params.keys())}")

    return view_params[view]


def _subsample_points(
    xyz: np.ndarray,
    colors: np.ndarray,
    max_points: int = 100000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample points for visualization if too many.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    colors : np.ndarray
        (N, 3) or (N,) color array.
    max_points : int
        Maximum number of points to render.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Subsampled (xyz, colors).
    """
    n_points = len(xyz)
    if n_points <= max_points:
        return xyz, colors

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_points, max_points, replace=False)
    return xyz[indices], colors[indices] if colors.ndim > 1 else colors[indices]


def render_classification(
    xyz: np.ndarray,
    classes: np.ndarray,
    view: str = "front",
    title: str = "RAI Classification",
    figsize: Tuple[int, int] = (12, 10),
    point_size: float = 1.0,
    dpi: int = 300,
    show_legend: bool = True,
    max_points: int = 100000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render classified point cloud to matplotlib figure.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    classes : np.ndarray
        (N,) class codes 0-7.
    view : str
        View angle: "front", "oblique", "top", "side".
    title : str
        Figure title.
    figsize : tuple
        Figure size in inches.
    point_size : float
        Point size multiplier.
    dpi : int
        Output resolution.
    show_legend : bool
        Whether to show class legend.
    max_points : int
        Maximum points to render (subsamples if exceeded).
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Get colors for each point
    colors = get_class_colors(classes)

    # Subsample if needed
    xyz_plot, colors_plot = _subsample_points(xyz, colors, max_points)

    # Calculate bounds
    bounds = {
        "x": (xyz[:, 0].min(), xyz[:, 0].max()),
        "y": (xyz[:, 1].min(), xyz[:, 1].max()),
        "z": (xyz[:, 2].min(), xyz[:, 2].max()),
    }

    # Get view parameters
    view_params = get_view_params(view, bounds)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Render points
    ax.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=colors_plot,
        s=point_size,
        marker=".",
        alpha=0.8,
    )

    # Set view
    ax.view_init(elev=view_params["elev"], azim=view_params["azim"])

    # Set labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # Equal aspect ratio
    _set_axes_equal(ax)

    # Add legend
    if show_legend:
        # Find which classes are present
        unique_classes = np.unique(classes)
        legend_elements = [
            Patch(
                facecolor=RAI_CLASS_COLORS[c],
                edgecolor="black",
                label=f"{RAI_CLASS_NAMES[c]} ({(classes == c).sum():,})",
            )
            for c in sorted(unique_classes)
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def render_continuous(
    xyz: np.ndarray,
    values: np.ndarray,
    view: str = "front",
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
    figsize: Tuple[int, int] = (12, 10),
    point_size: float = 1.0,
    dpi: int = 300,
    max_points: int = 100000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render point cloud with continuous colormap (for slope, roughness).

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    values : np.ndarray
        (N,) values to map to colors.
    view : str
        View angle: "front", "oblique", "top", "side".
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap name.
    vmin : float, optional
        Minimum value for colormap scaling.
    vmax : float, optional
        Maximum value for colormap scaling.
    colorbar_label : str
        Label for the colorbar.
    figsize : tuple
        Figure size in inches.
    point_size : float
        Point size multiplier.
    dpi : int
        Output resolution.
    max_points : int
        Maximum points to render.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    # Determine color scaling
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    # Subsample if needed
    xyz_plot, values_plot = _subsample_points(xyz, values, max_points)

    # Calculate bounds
    bounds = {
        "x": (xyz[:, 0].min(), xyz[:, 0].max()),
        "y": (xyz[:, 1].min(), xyz[:, 1].max()),
        "z": (xyz[:, 2].min(), xyz[:, 2].max()),
    }

    # Get view parameters
    view_params = get_view_params(view, bounds)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Render points
    scatter = ax.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=values_plot,
        s=point_size,
        marker=".",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
    )

    # Set view
    ax.view_init(elev=view_params["elev"], azim=view_params["azim"])

    # Set labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # Equal aspect ratio
    _set_axes_equal(ax)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def render_slope(
    xyz: np.ndarray,
    slope_deg: np.ndarray,
    view: str = "front",
    title: str = "Slope Angle",
    dpi: int = 300,
    max_points: int = 100000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render slope angle visualization.

    Convenience wrapper for render_continuous with slope-specific settings.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    view : str
        View angle.
    title : str
        Figure title.
    dpi : int
        Output resolution.
    max_points : int
        Maximum points to render.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    return render_continuous(
        xyz,
        slope_deg,
        view=view,
        title=title,
        cmap="RdYlBu_r",  # Red for steep, blue for flat
        vmin=0,
        vmax=180,
        colorbar_label="Slope (°)",
        dpi=dpi,
        max_points=max_points,
        output_path=output_path,
    )


def render_roughness(
    xyz: np.ndarray,
    roughness: np.ndarray,
    view: str = "front",
    title: str = "Roughness",
    dpi: int = 300,
    max_points: int = 100000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render roughness visualization.

    Convenience wrapper for render_continuous with roughness-specific settings.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    roughness : np.ndarray
        (N,) roughness values in degrees.
    view : str
        View angle.
    title : str
        Figure title.
    dpi : int
        Output resolution.
    max_points : int
        Maximum points to render.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Auto-scale vmax to 95th percentile of valid values
    valid_roughness = roughness[~np.isnan(roughness)]
    if len(valid_roughness) > 0:
        vmax = np.percentile(valid_roughness, 95)
    else:
        vmax = 30

    return render_continuous(
        xyz,
        roughness,
        view=view,
        title=title,
        cmap="plasma",
        vmin=0,
        vmax=vmax,
        colorbar_label="Roughness (°)",
        dpi=dpi,
        max_points=max_points,
        output_path=output_path,
    )


def _set_axes_equal(ax: Axes3D) -> None:
    """
    Set 3D axes to equal scale.

    Makes the scaling equal on all axes so the point cloud is not distorted.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes object.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])
