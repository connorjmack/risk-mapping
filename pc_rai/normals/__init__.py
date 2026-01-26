"""Normal computation module using CloudCompare."""

from pc_rai.normals.cloudcompare import (
    CloudCompareError,
    CloudCompareNotFoundError,
    compute_normals_cloudcompare,
    compute_normals_for_cloud,
    extract_normals_from_las,
    find_cloudcompare,
    get_cloudcompare_version,
    is_cloudcompare_available,
    is_cloudcompare_flatpak_installed,
    is_flatpak_available,
    is_xvfb_available,
)

__all__ = [
    "CloudCompareError",
    "CloudCompareNotFoundError",
    "find_cloudcompare",
    "is_cloudcompare_available",
    "is_cloudcompare_flatpak_installed",
    "is_flatpak_available",
    "is_xvfb_available",
    "compute_normals_cloudcompare",
    "compute_normals_for_cloud",
    "extract_normals_from_las",
    "get_cloudcompare_version",
]
