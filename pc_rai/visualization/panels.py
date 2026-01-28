"""
Dunham-style panel visualization for RAI results.

Creates publication-quality figures matching the style of Dunham et al. (2017),
with 2D projected views of point clouds showing:
- Panel A: Shaded relief / intensity
- Panel B: RAI morphological classification
- Panel C: Energy source mapping
- Panel D: Large-scale roughness (optional)
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PathCollection
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from pc_rai.config import RAI_CLASS_COLORS, RAI_CLASS_NAMES, RAI_CLASS_ABBREV


# Dunham-style colors (matching the paper's figure style)
# Simplified 5-class scheme
DUNHAM_CLASS_COLORS = {
    0: "#9E9E9E",  # Unclassified - Gray
    1: "#C8A2C8",  # Talus (T) - Light Purple
    2: "#4CAF50",  # Intact (I) - Green
    3: "#2196F3",  # Discontinuous (D) - Blue
    4: "#FF9800",  # Steep/Overhang (O) - Orange (high risk)
    5: "#795548",  # Structure (St) - Brown (seawalls, engineered)
}


def _project_to_profile(
    xyz: np.ndarray,
    profile_azimuth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D profile view (distance along cliff, elevation).

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    profile_azimuth : float, optional
        Azimuth angle in degrees for profile direction. If None, auto-detect
        from principal axis of point cloud.

    Returns
    -------
    dist : np.ndarray
        (N,) distance along profile direction.
    elev : np.ndarray
        (N,) elevation (Z coordinate).
    """
    if profile_azimuth is None:
        # Auto-detect profile direction from data extent
        x_range = xyz[:, 0].max() - xyz[:, 0].min()
        y_range = xyz[:, 1].max() - xyz[:, 1].min()

        if x_range >= y_range:
            # Cliff runs along X axis
            dist = xyz[:, 0] - xyz[:, 0].min()
        else:
            # Cliff runs along Y axis
            dist = xyz[:, 1] - xyz[:, 1].min()
    else:
        # Project onto specified azimuth
        azimuth_rad = np.radians(profile_azimuth)
        dist = xyz[:, 0] * np.cos(azimuth_rad) + xyz[:, 1] * np.sin(azimuth_rad)
        dist = dist - dist.min()

    elev = xyz[:, 2]
    return dist, elev


def _subsample_for_viz(
    arrays: List[np.ndarray],
    max_points: int = 500000,
    seed: int = 42,
) -> List[np.ndarray]:
    """Subsample multiple arrays consistently."""
    n_points = len(arrays[0])
    if n_points <= max_points:
        return arrays

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_points, max_points, replace=False)
    return [arr[indices] for arr in arrays]


def _calculate_aspect_ratio(dist: np.ndarray, elev: np.ndarray, max_aspect: float = 10.0) -> float:
    """
    Calculate appropriate aspect ratio for cliff visualization.

    For very long, thin cliffs, we exaggerate the vertical scale to make
    the morphology visible while keeping it reasonable.

    Parameters
    ----------
    dist : np.ndarray
        Distance along profile.
    elev : np.ndarray
        Elevation values.
    max_aspect : float
        Maximum aspect ratio (height exaggeration factor).

    Returns
    -------
    float
        Aspect ratio to use (vertical_scale / horizontal_scale).
    """
    dist_range = dist.max() - dist.min()
    elev_range = elev.max() - elev.min()

    if elev_range < 1e-6:
        return 1.0

    natural_aspect = dist_range / elev_range

    # If the cliff is more than max_aspect times wider than tall,
    # exaggerate vertical by max_aspect
    if natural_aspect > max_aspect:
        return max_aspect
    elif natural_aspect > 1:
        # Moderate exaggeration for somewhat elongated cliffs
        return min(natural_aspect, max_aspect)
    else:
        return 1.0


def render_intensity_panel(
    ax: plt.Axes,
    dist: np.ndarray,
    elev: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    point_size: float = 0.1,
    title: str = "A",
    aspect_ratio: Optional[float] = None,
) -> None:
    """
    Render RGB / intensity / grayscale panel.

    Uses RGB if available, falls back to intensity, then grayscale.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on.
    dist : np.ndarray
        (N,) distance along profile.
    elev : np.ndarray
        (N,) elevation.
    intensity : np.ndarray, optional
        (N,) intensity values. If None and no RGB, uses grayscale.
    rgb : np.ndarray, optional
        (N, 3) RGB values in [0, 1]. Takes priority over intensity.
    point_size : float
        Point size for scatter plot.
    title : str
        Panel label (e.g., "A").
    aspect_ratio : float, optional
        Aspect ratio for the plot. If None, auto-calculate with vertical exaggeration.
    """
    if rgb is not None:
        colors = rgb
    elif intensity is not None:
        # Normalize intensity to 0-1
        i_min, i_max = np.nanpercentile(intensity, [2, 98])
        intensity_norm = np.clip((intensity - i_min) / (i_max - i_min + 1e-10), 0, 1)
        colors = plt.cm.gray(intensity_norm)
    else:
        # Use simple gray shading
        colors = "#666666"

    ax.scatter(dist, elev, c=colors, s=point_size, marker=",", linewidths=0, rasterized=True)

    # Set aspect ratio with vertical exaggeration for elongated cliffs
    if aspect_ratio is None:
        aspect_ratio = _calculate_aspect_ratio(dist, elev)
    ax.set_aspect(aspect_ratio)

    ax.set_ylabel("Elevation (m)")
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_classification_panel(
    ax: plt.Axes,
    dist: np.ndarray,
    elev: np.ndarray,
    classes: np.ndarray,
    point_size: float = 0.1,
    title: str = "B",
    show_legend: bool = True,
    aspect_ratio: Optional[float] = None,
) -> None:
    """
    Render RAI classification panel with Dunham-style colors.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on.
    dist : np.ndarray
        (N,) distance along profile.
    elev : np.ndarray
        (N,) elevation.
    classes : np.ndarray
        (N,) RAI class codes (0-7).
    point_size : float
        Point size for scatter plot.
    title : str
        Panel label (e.g., "B").
    show_legend : bool
        Whether to show the class legend.
    aspect_ratio : float, optional
        Aspect ratio for the plot. If None, auto-calculate with vertical exaggeration.
    """
    # Create color array
    colors = np.zeros((len(classes), 4))
    for class_code, hex_color in DUNHAM_CLASS_COLORS.items():
        mask = classes == class_code
        if np.any(mask):
            # Convert hex to RGBA
            hex_clean = hex_color.lstrip("#")
            r, g, b = tuple(int(hex_clean[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            colors[mask] = [r, g, b, 1.0]

    ax.scatter(dist, elev, c=colors, s=point_size, marker=",", linewidths=0, rasterized=True)

    # Set aspect ratio with vertical exaggeration for elongated cliffs
    if aspect_ratio is None:
        aspect_ratio = _calculate_aspect_ratio(dist, elev)
    ax.set_aspect(aspect_ratio)
    ax.set_ylabel("Elevation (m)")
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    if show_legend:
        # Simplified 5-class scheme labels
        legend_labels = {
            1: "Talus",
            2: "Intact",
            3: "Discontinuous",
            4: "Steep/Ovhg.",
            5: "Structure",
        }
        unique_classes = np.unique(classes)
        legend_elements = []
        for c in [1, 2, 3, 4, 5]:  # Simplified 5-class scheme  # Ordered by class code
            if c in unique_classes:
                legend_elements.append(
                    Patch(
                        facecolor=DUNHAM_CLASS_COLORS[c],
                        edgecolor="black",
                        linewidth=0.5,
                        label=legend_labels[c],
                    )
                )

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            ncol=1,
            fontsize=8,
            framealpha=0.9,
            edgecolor="black",
        )


def render_energy_panel(
    ax: plt.Axes,
    dist: np.ndarray,
    elev: np.ndarray,
    energy: np.ndarray,
    point_size: float = 0.1,
    title: str = "C",
    show_colorbar: bool = True,
    vmax: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
) -> Optional[plt.cm.ScalarMappable]:
    """
    Render energy source mapping panel with Dunham-style colormap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on.
    dist : np.ndarray
        (N,) distance along profile.
    elev : np.ndarray
        (N,) elevation.
    energy : np.ndarray
        (N,) energy values in kJ.
    point_size : float
        Point size for scatter plot.
    title : str
        Panel label (e.g., "C").
    show_colorbar : bool
        Whether to show colorbar.
    vmax : float, optional
        Maximum value for color scaling. If None, uses 95th percentile.
    aspect_ratio : float, optional
        Aspect ratio for the plot. If None, auto-calculate with vertical exaggeration.

    Returns
    -------
    ScalarMappable or None
        The mappable for colorbar creation, if show_colorbar is True.
    """
    # Create energy colormap (cream to orange to red, matching Dunham Fig 6C)
    colors_list = [
        (1.0, 1.0, 0.9),    # Cream (0 kJ)
        (1.0, 0.9, 0.6),    # Light yellow
        (1.0, 0.7, 0.3),    # Orange
        (0.9, 0.3, 0.1),    # Dark orange-red
        (0.7, 0.0, 0.0),    # Dark red (max)
    ]
    energy_cmap = LinearSegmentedColormap.from_list("energy", colors_list)

    # Determine vmax
    valid_energy = energy[energy > 0]
    if vmax is None:
        if len(valid_energy) > 0:
            vmax = np.percentile(valid_energy, 95)
        else:
            vmax = 0.3

    # Ensure vmax is at least a small positive value
    vmax = max(vmax, 0.01)

    scatter = ax.scatter(
        dist, elev,
        c=energy,
        s=point_size,
        marker=",",
        linewidths=0,
        cmap=energy_cmap,
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )

    # Set aspect ratio with vertical exaggeration for elongated cliffs
    if aspect_ratio is None:
        aspect_ratio = _calculate_aspect_ratio(dist, elev)
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel("Distance along cliff (m)")
    ax.set_ylabel("Elevation (m)")
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return scatter


def render_roughness_panel(
    ax: plt.Axes,
    dist: np.ndarray,
    elev: np.ndarray,
    roughness: np.ndarray,
    point_size: float = 0.1,
    title: str = "D",
    label: str = "Large-scale roughness (°)",
    vmax: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
) -> PathCollection:
    """
    Render roughness panel with plasma colormap.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on.
    dist : np.ndarray
        (N,) distance along profile.
    elev : np.ndarray
        (N,) elevation.
    roughness : np.ndarray
        (N,) roughness values in degrees.
    point_size : float
        Point size for scatter plot.
    title : str
        Panel label (e.g., "D").
    label : str
        Colorbar label.
    vmax : float, optional
        Maximum value for color scaling. If None, uses 95th percentile.
    aspect_ratio : float, optional
        Aspect ratio for the plot.

    Returns
    -------
    PathCollection
        The scatter plot mappable for colorbar creation.
    """
    if vmax is None:
        valid = roughness[~np.isnan(roughness)]
        vmax = np.percentile(valid, 95) if len(valid) > 0 else 30.0

    scatter = ax.scatter(
        dist, elev,
        c=roughness,
        s=point_size,
        marker=",",
        linewidths=0,
        cmap="plasma",
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )

    if aspect_ratio is None:
        aspect_ratio = _calculate_aspect_ratio(dist, elev)
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel("Distance along cliff (m)")
    ax.set_ylabel("Elevation (m)")
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return scatter


def render_dunham_figure(
    xyz: np.ndarray,
    classes: np.ndarray,
    energy: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    roughness_large: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (14, 13),
    dpi: int = 300,
    max_points: int = 500000,
    point_size: float = 0.3,
    profile_azimuth: Optional[float] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create a Dunham et al. (2017) style 4-panel figure.

    Panel A: RGB (preferred), intensity, or grayscale
    Panel B: RAI morphological classification
    Panel C: Energy source mapping
    Panel D: Large-scale roughness (if provided)

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    classes : np.ndarray
        (N,) RAI class codes (0-5).
    energy : np.ndarray
        (N,) energy values in kJ.
    rgb : np.ndarray, optional
        (N, 3) RGB values in [0, 1] for Panel A. Takes priority over intensity.
    intensity : np.ndarray, optional
        (N,) intensity values for Panel A. Used if rgb is None.
    roughness_large : np.ndarray, optional
        (N,) large-scale roughness in degrees for Panel D. If None, 3-panel figure.
    output_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    max_points : int
        Maximum points to render (subsamples if exceeded).
    point_size : float
        Point size for scatter plots.
    profile_azimuth : float, optional
        Azimuth for profile projection. If None, auto-detect.
    title : str, optional
        Overall figure title.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Project to 2D profile view
    dist, elev = _project_to_profile(xyz, profile_azimuth)

    # Subsample consistently across all arrays
    n_points = len(dist)
    if n_points > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_points, max_points, replace=False)
        dist = dist[indices]
        elev = elev[indices]
        classes = classes[indices]
        energy = energy[indices]
        if rgb is not None:
            rgb = rgb[indices]
        if intensity is not None:
            intensity = intensity[indices]
        if roughness_large is not None:
            roughness_large = roughness_large[indices]

    # Determine number of panels
    n_panels = 4 if roughness_large is not None else 3

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, dpi=dpi, sharex=True)

    # Calculate aspect ratio once for all panels (consistent vertical exaggeration)
    aspect_ratio = _calculate_aspect_ratio(dist, elev)

    # Render panels with consistent aspect ratio
    render_intensity_panel(axes[0], dist, elev, intensity, rgb=rgb, point_size=point_size, title="A", aspect_ratio=aspect_ratio)
    render_classification_panel(axes[1], dist, elev, classes, point_size, title="B", aspect_ratio=aspect_ratio)
    energy_scatter = render_energy_panel(axes[2], dist, elev, energy, point_size, title="C", aspect_ratio=aspect_ratio)

    if roughness_large is not None:
        roughness_scatter = render_roughness_panel(
            axes[3], dist, elev, roughness_large, point_size, title="D",
            label="Small-scale roughness (°)", aspect_ratio=aspect_ratio,
        )
        # Add scale bar to bottom panel (D)
        _add_scale_bar(axes[3], dist)
    else:
        roughness_scatter = None
        # Add scale bar to bottom panel (C)
        _add_scale_bar(axes[2], dist)

    # Add energy colorbar
    if energy_scatter is not None:
        cbar_ax = fig.add_axes([0.92, 0.30 if n_panels == 4 else 0.11, 0.02, 0.15])
        cbar = fig.colorbar(energy_scatter, cax=cbar_ax)
        cbar.set_label("Energy (kJ)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Add roughness colorbar
    if roughness_scatter is not None:
        cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.15])
        cbar = fig.colorbar(roughness_scatter, cax=cbar_ax)
        cbar.set_label("Roughness (°)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Add class legend colorbar-style on right side
    _add_class_legend(fig, classes)

    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 0.90, 0.96])

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


def _add_scale_bar(ax: plt.Axes, dist: np.ndarray, target_length: float = None) -> None:
    """Add a scale bar to the axes."""
    dist_range = dist.max() - dist.min()

    # Choose appropriate scale bar length
    if target_length is None:
        if dist_range > 100:
            bar_length = 20
        elif dist_range > 50:
            bar_length = 10
        else:
            bar_length = 5

    # Position in lower right
    x_pos = dist.max() - bar_length - dist_range * 0.02
    y_min = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_pos = y_min + y_range * 0.05

    # Draw scale bar
    ax.plot([x_pos, x_pos + bar_length], [y_pos, y_pos], "k-", linewidth=2)
    ax.text(
        x_pos + bar_length / 2, y_pos - y_range * 0.03,
        f"{bar_length} m",
        ha="center", va="top", fontsize=8
    )


def _add_class_legend(fig: plt.Figure, classes: np.ndarray) -> None:
    """Add a vertical class legend matching Dunham style."""
    unique_classes = np.unique(classes)

    # Create legend axes on right side
    legend_ax = fig.add_axes([0.92, 0.45, 0.06, 0.35])
    legend_ax.axis("off")

    # Classes in order (bottom to top in legend) - simplified 5-class scheme
    class_order = [1, 2, 3, 4, 5]
    y_positions = np.linspace(0, 1, len(class_order) + 1)[:-1]

    for i, class_code in enumerate(class_order):
        if class_code in unique_classes:
            alpha = 1.0
        else:
            alpha = 0.3

        # Color box
        rect = Rectangle(
            (0, y_positions[i]), 0.3, 0.12,
            facecolor=DUNHAM_CLASS_COLORS[class_code],
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha,
        )
        legend_ax.add_patch(rect)

        # Label
        legend_ax.text(
            0.4, y_positions[i] + 0.06,
            RAI_CLASS_ABBREV[class_code],
            va="center", ha="left",
            fontsize=9, fontstyle="italic",
            alpha=alpha,
        )

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)


def render_single_panel(
    xyz: np.ndarray,
    values: np.ndarray,
    panel_type: str = "classification",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 4),
    dpi: int = 300,
    max_points: int = 500000,
    point_size: float = 0.5,
    profile_azimuth: Optional[float] = None,
    title: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Render a single panel visualization.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    values : np.ndarray
        (N,) values to visualize (classes, energy, intensity, etc.).
    panel_type : str
        Type of panel: "classification", "energy", "intensity", "slope", "roughness".
    output_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    max_points : int
        Maximum points to render.
    point_size : float
        Point size for scatter plot.
    profile_azimuth : float, optional
        Azimuth for profile projection.
    title : str, optional
        Figure title.
    **kwargs
        Additional arguments passed to specific render functions.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Project to 2D
    dist, elev = _project_to_profile(xyz, profile_azimuth)

    # Subsample
    dist, elev, values = _subsample_for_viz([dist, elev, values], max_points)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Calculate aspect ratio for vertical exaggeration
    aspect_ratio = _calculate_aspect_ratio(dist, elev)

    if panel_type == "classification":
        render_classification_panel(ax, dist, elev, values.astype(np.uint8), point_size, title="", aspect_ratio=aspect_ratio, **kwargs)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")

    elif panel_type == "energy":
        scatter = render_energy_panel(ax, dist, elev, values, point_size, title="", aspect_ratio=aspect_ratio, **kwargs)
        if scatter:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Energy (kJ)")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")

    elif panel_type == "intensity":
        render_intensity_panel(ax, dist, elev, values, point_size, title="", aspect_ratio=aspect_ratio)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")

    elif panel_type in ["slope", "roughness"]:
        # Use a diverging colormap for slope/roughness
        cmap = "RdYlBu_r" if panel_type == "slope" else "plasma"
        vmin = kwargs.get("vmin", 0)
        vmax = kwargs.get("vmax", np.nanpercentile(values, 95))

        scatter = ax.scatter(
            dist, elev, c=values, s=point_size, marker=",",
            linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
        )
        ax.set_aspect(aspect_ratio)
        ax.set_xlabel("Distance along cliff (m)")
        ax.set_ylabel("Elevation (m)")

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        label = "Slope (°)" if panel_type == "slope" else "Roughness (°)"
        cbar.set_label(label)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig
