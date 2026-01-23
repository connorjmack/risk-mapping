"""Reporting module for statistics and report generation."""

from pc_rai.reporting.statistics import (
    calculate_all_statistics,
    calculate_classification_stats,
    calculate_feature_stats,
    calculate_method_agreement,
)
from pc_rai.reporting.report_writer import (
    generate_config_summary,
    write_json_report,
    write_markdown_report,
)

__all__ = [
    # statistics
    "calculate_all_statistics",
    "calculate_classification_stats",
    "calculate_feature_stats",
    "calculate_method_agreement",
    # report_writer
    "generate_config_summary",
    "write_json_report",
    "write_markdown_report",
]
