"""
CloudCompare CLI wrapper for normal computation.

Provides functions to compute surface normals using CloudCompare's
command-line interface.

Supports:
- Standard CloudCompare installations (direct executable)
- Flatpak installations (flatpak run org.cloudcompare.CloudCompare)
- Headless operation via xvfb-run (Linux)
"""

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import laspy
import numpy as np

logger = logging.getLogger(__name__)

# Flatpak app ID for CloudCompare
FLATPAK_APP_ID = "org.cloudcompare.CloudCompare"


# Common CloudCompare executable names and paths
CLOUDCOMPARE_NAMES = [
    "CloudCompare",
    "cloudcompare",
    "CloudCompare.exe",
    "ccViewer",
]

CLOUDCOMPARE_PATHS = [
    # Linux
    "/usr/bin/CloudCompare",
    "/usr/local/bin/CloudCompare",
    "/usr/bin/cloudcompare",
    "/usr/local/bin/cloudcompare",
    # macOS
    "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare",
    "/Applications/CloudCompare/CloudCompare.app/Contents/MacOS/CloudCompare",
    # Windows
    "C:\\Program Files\\CloudCompare\\CloudCompare.exe",
    "C:\\Program Files (x86)\\CloudCompare\\CloudCompare.exe",
]


class CloudCompareError(Exception):
    """Exception raised when CloudCompare operations fail."""

    pass


class CloudCompareNotFoundError(CloudCompareError):
    """Exception raised when CloudCompare executable cannot be found."""

    pass


def is_flatpak_available() -> bool:
    """Check if flatpak command is available."""
    return shutil.which("flatpak") is not None


def is_cloudcompare_flatpak_installed() -> bool:
    """Check if CloudCompare is installed via Flatpak."""
    if not is_flatpak_available():
        return False

    try:
        result = subprocess.run(
            ["flatpak", "info", FLATPAK_APP_ID],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def is_xvfb_available() -> bool:
    """Check if xvfb-run is available for headless operation."""
    return shutil.which("xvfb-run") is not None


def find_cloudcompare(check_flatpak: bool = True) -> Optional[str]:
    """
    Find CloudCompare executable on system.

    Searches common executable names in PATH, common installation paths,
    and Flatpak installations.

    Parameters
    ----------
    check_flatpak : bool
        If True, also check for Flatpak installation.

    Returns
    -------
    str or None
        Path to CloudCompare executable, "flatpak" if installed via Flatpak,
        or None if not found.
    """
    # Check PATH first
    for name in CLOUDCOMPARE_NAMES:
        path = shutil.which(name)
        if path:
            return path

    # Check common installation paths
    for path_str in CLOUDCOMPARE_PATHS:
        path = Path(path_str)
        if path.exists() and path.is_file():
            return str(path)

    # Check Flatpak installation
    if check_flatpak and is_cloudcompare_flatpak_installed():
        return "flatpak"

    return None


def is_cloudcompare_available() -> bool:
    """
    Check if CloudCompare is available on the system.

    Checks for both direct installations and Flatpak installations.

    Returns
    -------
    bool
        True if CloudCompare is found and executable.
    """
    return find_cloudcompare(check_flatpak=True) is not None


def compute_normals_cloudcompare(
    input_path: Path,
    output_path: Path,
    radius: float = 1.0,
    mst_neighbors: int = 12,
    cloudcompare_path: Optional[str] = None,
    timeout: int = 300,
    use_xvfb: Optional[bool] = None,
) -> bool:
    """
    Compute normals using CloudCompare CLI.

    Parameters
    ----------
    input_path : Path
        Input LAS/LAZ file.
    output_path : Path
        Output path for file with normals.
    radius : float
        Local radius for normal estimation (in point cloud units).
    mst_neighbors : int
        Number of neighbors for MST orientation.
    cloudcompare_path : str, optional
        Path to CloudCompare executable, or "flatpak" for Flatpak installation.
        If None, auto-detect.
    timeout : int
        Maximum time in seconds to wait for CloudCompare.
    use_xvfb : bool, optional
        If True, wrap command with xvfb-run for headless operation.
        If None, auto-detect (True on Linux, False otherwise).

    Returns
    -------
    success : bool
        True if normals were computed successfully.

    Raises
    ------
    CloudCompareNotFoundError
        If CloudCompare executable cannot be found.
    CloudCompareError
        If normal computation fails.
    FileNotFoundError
        If input file does not exist.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Find CloudCompare
    use_flatpak = False
    if cloudcompare_path:
        if cloudcompare_path == "flatpak":
            use_flatpak = True
            if not is_cloudcompare_flatpak_installed():
                raise CloudCompareNotFoundError(
                    f"CloudCompare Flatpak not installed. Install with: "
                    f"flatpak install {FLATPAK_APP_ID}"
                )
        else:
            cc_path = Path(cloudcompare_path)
            if not cc_path.exists():
                raise CloudCompareNotFoundError(
                    f"Specified CloudCompare path not found: {cloudcompare_path}"
                )
    else:
        cc_path_or_flatpak = find_cloudcompare()
        if cc_path_or_flatpak is None:
            raise CloudCompareNotFoundError(
                "CloudCompare not found. Install CloudCompare or specify path."
            )
        if cc_path_or_flatpak == "flatpak":
            use_flatpak = True
        else:
            cc_path = Path(cc_path_or_flatpak)

    # Auto-detect xvfb usage on Linux
    if use_xvfb is None:
        use_xvfb = sys.platform.startswith("linux") and is_xvfb_available()

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    # CloudCompare CLI reference: https://www.cloudcompare.org/doc/wiki/index.php/Command_line_mode
    cmd = []

    # Prepend xvfb-run for headless operation
    if use_xvfb:
        cmd.extend(["xvfb-run", "-a"])

    # Add CloudCompare invocation (flatpak or direct)
    if use_flatpak:
        cmd.extend(["flatpak", "run", FLATPAK_APP_ID])
    else:
        cmd.append(str(cc_path))

    # Add CloudCompare arguments
    cmd.extend([
        "-SILENT",  # No GUI
        "-O", str(input_path),  # Open file
        "-OCTREE_NORMALS", str(radius),  # Compute normals with octree
        "-ORIENT_NORMS_MST", str(mst_neighbors),  # Orient using MST
        "-C_EXPORT_FMT", "LAS",  # Export format
        "-SAVE_CLOUDS", "FILE", str(output_path),  # Save to specific path
    ])

    logger.info(f"Running CloudCompare: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.error(f"CloudCompare stderr: {result.stderr}")
            raise CloudCompareError(
                f"CloudCompare failed with return code {result.returncode}: {result.stderr}"
            )

        # CloudCompare may save with modified filename - check for output
        if not output_path.exists():
            # Try common CloudCompare output naming patterns
            possible_outputs = _find_cloudcompare_output(input_path, output_path)
            if possible_outputs:
                # Move the found file to expected location
                possible_outputs[0].rename(output_path)
            else:
                raise CloudCompareError(
                    f"CloudCompare completed but output file not found: {output_path}"
                )

        logger.info(f"Normals computed successfully: {output_path}")
        return True

    except subprocess.TimeoutExpired:
        raise CloudCompareError(
            f"CloudCompare timed out after {timeout} seconds"
        )
    except subprocess.SubprocessError as e:
        raise CloudCompareError(f"Failed to run CloudCompare: {e}")


def _find_cloudcompare_output(
    input_path: Path, expected_output: Path
) -> list:
    """
    Find CloudCompare output file with potentially modified name.

    CloudCompare often appends suffixes like "_WITH_NORMALS" to output files.
    """
    output_dir = expected_output.parent
    input_stem = input_path.stem

    # Patterns CloudCompare might use
    patterns = [
        f"{input_stem}_WITH_NORMALS.las",
        f"{input_stem}_WITH_NORMALS.laz",
        f"{input_stem}_OCTREE_NORMALS.las",
        f"{input_stem}_OCTREE_NORMALS.laz",
        f"{input_stem}.las",
        f"{input_stem}.laz",
    ]

    found = []
    for pattern in patterns:
        candidate = output_dir / pattern
        if candidate.exists() and candidate != expected_output:
            found.append(candidate)

    return found


def extract_normals_from_las(las_path: Path) -> Optional[np.ndarray]:
    """
    Extract normal vectors from LAS file.

    CloudCompare saves normals as NormalX, NormalY, NormalZ extra dimensions.

    Parameters
    ----------
    las_path : Path
        Path to LAS file with normals.

    Returns
    -------
    normals : np.ndarray or None
        (N, 3) array of normal vectors, or None if not found.
    """
    las_path = Path(las_path)

    if not las_path.exists():
        return None

    las = laspy.read(las_path)

    # Get extra dimension names
    extra_dim_names = [dim.name for dim in las.point_format.extra_dimensions]

    # Check for CloudCompare normal naming convention
    normal_names = [
        ("NormalX", "NormalY", "NormalZ"),
        ("normalx", "normaly", "normalz"),
        ("Nx", "Ny", "Nz"),
        ("nx", "ny", "nz"),
    ]

    for nx_name, ny_name, nz_name in normal_names:
        if all(name in extra_dim_names for name in (nx_name, ny_name, nz_name)):
            nx = np.array(las[nx_name], dtype=np.float32)
            ny = np.array(las[ny_name], dtype=np.float32)
            nz = np.array(las[nz_name], dtype=np.float32)
            return np.column_stack([nx, ny, nz])

    return None


def compute_normals_for_cloud(
    input_path: Path,
    radius: float = 1.0,
    mst_neighbors: int = 12,
    cloudcompare_path: Optional[str] = None,
    cleanup: bool = True,
) -> Tuple[np.ndarray, Path]:
    """
    Compute normals for a point cloud file and return the normals array.

    This is a convenience function that handles temporary files.

    Parameters
    ----------
    input_path : Path
        Input LAS/LAZ file without normals.
    radius : float
        Local radius for normal estimation.
    mst_neighbors : int
        Number of neighbors for MST orientation.
    cloudcompare_path : str, optional
        Path to CloudCompare executable.
    cleanup : bool
        If True, delete temporary files after extraction.

    Returns
    -------
    normals : np.ndarray
        (N, 3) array of computed normal vectors.
    output_path : Path
        Path to the output file with normals.

    Raises
    ------
    CloudCompareError
        If normal computation fails.
    """
    input_path = Path(input_path)

    # Create output path in temp directory or same directory as input
    if cleanup:
        temp_dir = tempfile.mkdtemp(prefix="pcrai_")
        output_path = Path(temp_dir) / f"{input_path.stem}_normals.las"
    else:
        output_path = input_path.parent / f"{input_path.stem}_normals.las"

    # Compute normals
    compute_normals_cloudcompare(
        input_path,
        output_path,
        radius=radius,
        mst_neighbors=mst_neighbors,
        cloudcompare_path=cloudcompare_path,
    )

    # Extract normals
    normals = extract_normals_from_las(output_path)

    if normals is None:
        raise CloudCompareError(
            f"Failed to extract normals from CloudCompare output: {output_path}"
        )

    return normals, output_path


def get_cloudcompare_version(cloudcompare_path: Optional[str] = None) -> Optional[str]:
    """
    Get CloudCompare version string.

    Parameters
    ----------
    cloudcompare_path : str, optional
        Path to CloudCompare executable.

    Returns
    -------
    version : str or None
        Version string, or None if cannot be determined.
    """
    if cloudcompare_path:
        cc_path = Path(cloudcompare_path)
    else:
        cc_path = find_cloudcompare()

    if cc_path is None:
        return None

    try:
        # Try to get version (CloudCompare doesn't have a standard --version flag)
        # This may not work on all platforms
        result = subprocess.run(
            [str(cc_path), "-SILENT", "-VERSION"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Parse version from output if available
        if result.stdout:
            return result.stdout.strip()
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass

    return None
