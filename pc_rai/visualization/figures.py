"""
Multi-panel figure generation for RAI visualization.

Creates comparison figures, summary panels, and distribution histograms.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, Optional, Tuple

from pc_rai.config import RAI_CLASS_COLORS, RAI_CLASS_NAMES
from pc_rai.visualization.render_3d import (
    get_class_colors,
    get_view_params,
    _subsample_points,
    _set_axes_equal,
)


def create_comparison_figure(
    xyz: np.ndarray,
    classes_radius: np.ndarray,
    classes_knn: np.ndarray,
    view: str = "front",
    title: str = "Classification Comparison: Radius vs k-NN",
    figsize: Tuple[int, int] = (16, 8),
    point_size: float = 1.0,
    dpi: int = 300,
    max_points: int = 50000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create side-by-side comparison of radius vs k-NN classification.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    classes_radius : np.ndarray
        (N,) class codes from radius method.
    classes_knn : np.ndarray
        (N,) class codes from k-NN method.
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
    max_points : int
        Maximum points to render per subplot.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Get colors for each method
    colors_radius = get_class_colors(classes_radius)
    colors_knn = get_class_colors(classes_knn)

    # Subsample consistently
    xyz_plot, colors_radius_plot = _subsample_points(xyz, colors_radius, max_points)
    _, colors_knn_plot = _subsample_points(xyz, colors_knn, max_points)

    # Calculate bounds
    bounds = {
        "x": (xyz[:, 0].min(), xyz[:, 0].max()),
        "y": (xyz[:, 1].min(), xyz[:, 1].max()),
        "z": (xyz[:, 2].min(), xyz[:, 2].max()),
    }
    view_params = get_view_params(view, bounds)

    # Create figure with two subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Radius method subplot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=colors_radius_plot,
        s=point_size,
        marker=".",
        alpha=0.8,
    )
    ax1.view_init(elev=view_params["elev"], azim=view_params["azim"])
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Radius Method")
    _set_axes_equal(ax1)

    # k-NN method subplot
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=colors_knn_plot,
        s=point_size,
        marker=".",
        alpha=0.8,
    )
    ax2.view_init(elev=view_params["elev"], azim=view_params["azim"])
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title("k-NN Method")
    _set_axes_equal(ax2)

    # Overall title
    fig.suptitle(title, fontsize=14, y=0.98)

    # Add shared legend
    all_classes = np.unique(np.concatenate([classes_radius, classes_knn]))
    legend_elements = [
        Patch(
            facecolor=RAI_CLASS_COLORS[c],
            edgecolor="black",
            label=RAI_CLASS_NAMES[c],
        )
        for c in sorted(all_classes)
    ]
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def create_summary_figure(
    xyz: np.ndarray,
    slope: np.ndarray,
    roughness_small: np.ndarray,
    roughness_large: np.ndarray,
    classes: np.ndarray,
    view: str = "front",
    title: str = "RAI Analysis Summary",
    figsize: Tuple[int, int] = (16, 14),
    point_size: float = 0.5,
    dpi: int = 300,
    max_points: int = 30000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create 4-panel summary figure: slope, r_small, r_large, classification.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) point coordinates.
    slope : np.ndarray
        (N,) slope angles in degrees.
    roughness_small : np.ndarray
        (N,) small-scale roughness in degrees.
    roughness_large : np.ndarray
        (N,) large-scale roughness in degrees.
    classes : np.ndarray
        (N,) class codes 0-7.
    view : str
        View angle for all panels.
    title : str
        Figure title.
    figsize : tuple
        Figure size in inches.
    point_size : float
        Point size multiplier.
    dpi : int
        Output resolution.
    max_points : int
        Maximum points to render per subplot.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Subsample all data consistently
    indices = np.arange(len(xyz))
    if len(xyz) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(xyz), max_points, replace=False)

    xyz_plot = xyz[indices]
    slope_plot = slope[indices]
    r_small_plot = roughness_small[indices]
    r_large_plot = roughness_large[indices]
    classes_plot = classes[indices]

    # Calculate view parameters
    bounds = {
        "x": (xyz[:, 0].min(), xyz[:, 0].max()),
        "y": (xyz[:, 1].min(), xyz[:, 1].max()),
        "z": (xyz[:, 2].min(), xyz[:, 2].max()),
    }
    view_params = get_view_params(view, bounds)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Panel 1: Slope
    ax1 = fig.add_subplot(221, projection="3d")
    sc1 = ax1.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=slope_plot,
        s=point_size,
        marker=".",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=180,
        alpha=0.8,
    )
    ax1.view_init(elev=view_params["elev"], azim=view_params["azim"])
    ax1.set_title("Slope Angle")
    _set_axes_equal(ax1)
    cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.5, pad=0.1)
    cbar1.set_label("Slope (°)")

    # Panel 2: Small-scale roughness
    ax2 = fig.add_subplot(222, projection="3d")
    valid_r_small = r_small_plot[~np.isnan(r_small_plot)]
    vmax_small = np.percentile(valid_r_small, 95) if len(valid_r_small) > 0 else 30
    sc2 = ax2.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=r_small_plot,
        s=point_size,
        marker=".",
        cmap="plasma",
        vmin=0,
        vmax=vmax_small,
        alpha=0.8,
    )
    ax2.view_init(elev=view_params["elev"], azim=view_params["azim"])
    ax2.set_title("Small-Scale Roughness")
    _set_axes_equal(ax2)
    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.1)
    cbar2.set_label("R_small (°)")

    # Panel 3: Large-scale roughness
    ax3 = fig.add_subplot(223, projection="3d")
    valid_r_large = r_large_plot[~np.isnan(r_large_plot)]
    vmax_large = np.percentile(valid_r_large, 95) if len(valid_r_large) > 0 else 30
    sc3 = ax3.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=r_large_plot,
        s=point_size,
        marker=".",
        cmap="plasma",
        vmin=0,
        vmax=vmax_large,
        alpha=0.8,
    )
    ax3.view_init(elev=view_params["elev"], azim=view_params["azim"])
    ax3.set_title("Large-Scale Roughness")
    _set_axes_equal(ax3)
    cbar3 = fig.colorbar(sc3, ax=ax3, shrink=0.5, pad=0.1)
    cbar3.set_label("R_large (°)")

    # Panel 4: Classification
    ax4 = fig.add_subplot(224, projection="3d")
    colors = get_class_colors(classes_plot)
    ax4.scatter(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        xyz_plot[:, 2],
        c=colors,
        s=point_size,
        marker=".",
        alpha=0.8,
    )
    ax4.view_init(elev=view_params["elev"], azim=view_params["azim"])
    ax4.set_title("RAI Classification")
    _set_axes_equal(ax4)

    # Add legend for classification panel
    unique_classes = np.unique(classes_plot)
    legend_elements = [
        Patch(
            facecolor=RAI_CLASS_COLORS[c],
            edgecolor="black",
            label=RAI_CLASS_NAMES[c],
        )
        for c in sorted(unique_classes)
    ]
    ax4.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=7,
    )

    # Overall title
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def create_histogram_figure(
    classes: np.ndarray,
    title: str = "RAI Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    show_percentages: bool = True,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create bar chart of class distribution.

    Parameters
    ----------
    classes : np.ndarray
        (N,) class codes 0-7.
    title : str
        Figure title.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    show_percentages : bool
        Whether to show percentage labels on bars.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Count each class
    n_total = len(classes)
    counts = []
    colors = []
    labels = []

    for class_code in range(6):  # 6 classes in simplified scheme
        count = (classes == class_code).sum()
        counts.append(count)
        colors.append(RAI_CLASS_COLORS[class_code])
        labels.append(f"{RAI_CLASS_NAMES[class_code]}\n({count:,})")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create bars
    x = np.arange(6)  # 6 classes in simplified scheme
    bars = ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add percentage labels
    if show_percentages:
        for bar, count in zip(bars, counts):
            pct = 100 * count / n_total if n_total > 0 else 0
            height = bar.get_height()
            ax.annotate(
                f"{pct:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(
        [RAI_CLASS_NAMES[i] for i in range(6)],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Point Count")
    ax.set_title(title)

    # Add total count annotation
    ax.text(
        0.99,
        0.99,
        f"Total: {n_total:,} points",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def create_method_agreement_figure(
    classes_radius: np.ndarray,
    classes_knn: np.ndarray,
    title: str = "Classification Agreement: Radius vs k-NN",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create confusion matrix heatmap showing agreement between methods.

    Parameters
    ----------
    classes_radius : np.ndarray
        (N,) class codes from radius method.
    classes_knn : np.ndarray
        (N,) class codes from k-NN method.
    title : str
        Figure title.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Build confusion matrix
    confusion = np.zeros((8, 8), dtype=np.int64)
    for i in range(len(classes_radius)):
        confusion[classes_radius[i], classes_knn[i]] += 1

    # Calculate agreement
    agreement = np.trace(confusion)
    total = len(classes_radius)
    agreement_pct = 100 * agreement / total if total > 0 else 0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot heatmap
    im = ax.imshow(confusion, cmap="Blues")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Point Count")

    # Set ticks (6 classes in simplified scheme)
    class_abbrevs = ["U", "T", "I", "D", "O", "St"]
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(class_abbrevs)
    ax.set_yticklabels(class_abbrevs)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            count = confusion[i, j]
            if count > 0:
                # Use white text on dark cells
                text_color = "white" if count > confusion.max() / 2 else "black"
                ax.text(j, i, f"{count:,}", ha="center", va="center", color=text_color, fontsize=8)

    # Labels
    ax.set_xlabel("k-NN Classification")
    ax.set_ylabel("Radius Classification")
    ax.set_title(f"{title}\n(Agreement: {agreement_pct:.1f}%)")

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
