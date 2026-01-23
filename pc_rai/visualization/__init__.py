"""Visualization module for rendering point clouds."""

from pc_rai.visualization.render_3d import (
    create_rai_colormap,
    get_class_colors,
    get_view_params,
    render_classification,
    render_continuous,
    render_roughness,
    render_slope,
)
from pc_rai.visualization.figures import (
    create_comparison_figure,
    create_histogram_figure,
    create_method_agreement_figure,
    create_summary_figure,
)

__all__ = [
    # render_3d
    "create_rai_colormap",
    "get_class_colors",
    "get_view_params",
    "render_classification",
    "render_continuous",
    "render_roughness",
    "render_slope",
    # figures
    "create_comparison_figure",
    "create_histogram_figure",
    "create_method_agreement_figure",
    "create_summary_figure",
]
