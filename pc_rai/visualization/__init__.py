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
from pc_rai.visualization.panels import (
    render_dunham_figure,
    render_single_panel,
    render_intensity_panel,
    render_classification_panel,
    render_energy_panel,
)
from pc_rai.visualization.risk_map import (
    render_risk_map,
    render_risk_map_profile,
    compute_binned_energy,
    # Transect-based risk maps
    parse_transects,
    parse_kml_transects,  # Alias for backwards compatibility
    compute_transect_energy,
    render_transect_risk_map,
    render_transect_risk_profile,
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
    # panels (Dunham-style)
    "render_dunham_figure",
    "render_single_panel",
    "render_intensity_panel",
    "render_classification_panel",
    "render_energy_panel",
    # risk map (axis-aligned bins)
    "render_risk_map",
    "render_risk_map_profile",
    "compute_binned_energy",
    # risk map (shapefile transects)
    "parse_transects",
    "parse_kml_transects",  # Alias for backwards compatibility
    "compute_transect_energy",
    "render_transect_risk_map",
    "render_transect_risk_profile",
]
