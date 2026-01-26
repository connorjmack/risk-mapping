#!/usr/bin/env python3
"""
Compute normals for LAS/LAZ files listed in a CSV.

Processes files with graceful error handling and restart capability.
Progress is tracked in a JSON file, allowing interrupted runs to resume.

Usage:
    python -m pc_rai.normals.normals_from_list input_files.csv --output-dir ./output
    python -m pc_rai.normals.normals_from_list input_files.csv --in-place

CSV Format:
    input_path,output_path (optional)
    /path/to/file1.laz,/path/to/output1.laz
    /path/to/file2.laz

If output_path is not provided, files are saved to --output-dir with '_normals' suffix.
Use --in-place to overwrite original files (writes to temp first for safety).
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from pc_rai.normals.cloudcompare import (
    CloudCompareError,
    CloudCompareNotFoundError,
    compute_normals_cloudcompare,
    find_cloudcompare,
    is_cloudcompare_flatpak_installed,
    is_xvfb_available,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class FileStatus:
    """Status of a single file processing."""

    input_path: str
    output_path: str
    status: str = "pending"  # pending, success, failed, skipped
    error: Optional[str] = None
    processed_at: Optional[str] = None


@dataclass
class ProcessingProgress:
    """Overall processing progress for restart capability."""

    csv_path: str
    output_dir: Optional[str]
    started_at: str
    updated_at: str
    files: dict = field(default_factory=dict)  # input_path -> FileStatus

    def save(self, progress_path: Path) -> None:
        """Save progress to JSON file."""
        self.updated_at = datetime.now().isoformat()
        data = {
            "csv_path": self.csv_path,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "files": {k: asdict(v) for k, v in self.files.items()},
        }
        with open(progress_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, progress_path: Path) -> "ProcessingProgress":
        """Load progress from JSON file."""
        with open(progress_path) as f:
            data = json.load(f)

        progress = cls(
            csv_path=data["csv_path"],
            output_dir=data.get("output_dir"),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
        )
        for input_path, file_data in data.get("files", {}).items():
            progress.files[input_path] = FileStatus(**file_data)

        return progress


def read_csv_file_list(
    csv_path: Path,
    path_column: str = "path",
    output_column: Optional[str] = None,
) -> list[tuple[Path, Optional[Path]]]:
    """
    Read file paths from CSV.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with columns: input_path, output_path (optional)
    path_column : str
        Column containing input file paths. Can be column name (e.g., "path")
        or zero-based index (e.g., "2"). Default is "path".
    output_column : str, optional
        Column containing output file paths. Can be column name or index.
        If None, output paths are not read from CSV.

    Returns
    -------
    list[tuple[Path, Optional[Path]]]
        List of (input_path, output_path) tuples. output_path may be None.
    """
    files = []

    with open(csv_path, newline="") as f:
        # Detect if there's a header
        sample = f.read(1024)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)

        reader = csv.reader(f)

        # Get header if present
        header = None
        if has_header:
            header = next(reader)
            header = [h.strip().lower() for h in header]

        # Resolve column indices
        def resolve_column(col_spec: str, header: Optional[list]) -> int:
            """Convert column name or index string to integer index."""
            # Try as integer index first
            try:
                return int(col_spec)
            except ValueError:
                pass

            # Try as column name
            if header is not None:
                col_lower = col_spec.strip().lower()
                if col_lower in header:
                    return header.index(col_lower)

            raise ValueError(
                f"Column '{col_spec}' not found. "
                f"Available columns: {header if header else 'no header detected'}"
            )

        path_idx = resolve_column(path_column, header)
        output_idx = resolve_column(output_column, header) if output_column else None

        for row_num, row in enumerate(reader, start=2 if has_header else 1):
            if not row:
                continue  # Skip empty rows

            # Get input path
            if path_idx >= len(row) or not row[path_idx].strip():
                continue  # Skip rows without path

            input_path = Path(row[path_idx].strip())

            # Get output path if specified
            output_path = None
            if output_idx is not None and output_idx < len(row) and row[output_idx].strip():
                output_path = Path(row[output_idx].strip())

            files.append((input_path, output_path))

    return files


def get_output_path(
    input_path: Path,
    output_path: Optional[Path],
    output_dir: Optional[Path],
    suffix: str = "_normals",
    in_place: bool = False,
) -> Path:
    """Determine output path for a file."""
    if in_place:
        return input_path

    if output_path is not None:
        return output_path

    if output_dir is not None:
        return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"

    # Default: same directory as input with suffix
    return input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"


def process_files(
    files: list[tuple[Path, Optional[Path]]],
    output_dir: Optional[Path],
    progress: ProcessingProgress,
    progress_path: Path,
    radius: float = 0.1,
    mst_neighbors: int = 10,
    suffix: str = "_normals",
    skip_existing: bool = True,
    in_place: bool = False,
    cloudcompare_path: Optional[str] = None,
    use_xvfb: Optional[bool] = None,
) -> tuple[int, int, int]:
    """
    Process all files with graceful error handling.

    Parameters
    ----------
    files : list[tuple[Path, Optional[Path]]]
        List of (input_path, output_path) tuples.
    output_dir : Path, optional
        Default output directory.
    progress : ProcessingProgress
        Progress tracker for restart capability.
    progress_path : Path
        Path to save progress file.
    radius : float
        Local radius for normal estimation.
    mst_neighbors : int
        Number of neighbors for MST orientation.
    suffix : str
        Suffix for output filenames.
    skip_existing : bool
        If True, skip files that already have output.
    in_place : bool
        If True, overwrite original files with normals added.
    cloudcompare_path : str, optional
        Path to CloudCompare executable, or "flatpak" for Flatpak installation.
    use_xvfb : bool, optional
        If True, use xvfb-run for headless operation. Auto-detects on Linux if None.

    Returns
    -------
    tuple[int, int, int]
        (success_count, failed_count, skipped_count)
    """
    success_count = 0
    failed_count = 0
    skipped_count = 0

    total = len(files)

    for i, (input_path, output_path) in enumerate(files, start=1):
        input_str = str(input_path)
        final_output = get_output_path(input_path, output_path, output_dir, suffix, in_place)

        logger.info(f"[{i}/{total}] Processing: {input_path.name}")

        # Check if already processed successfully in previous run
        if input_str in progress.files:
            prev_status = progress.files[input_str]
            if prev_status.status == "success":
                logger.info(f"  Skipping (already processed in previous run)")
                skipped_count += 1
                continue

        # Check if output already exists (skip for in-place mode)
        if skip_existing and not in_place and final_output.exists():
            logger.info(f"  Skipping (output exists): {final_output}")
            progress.files[input_str] = FileStatus(
                input_path=input_str,
                output_path=str(final_output),
                status="skipped",
                processed_at=datetime.now().isoformat(),
            )
            progress.save(progress_path)
            skipped_count += 1
            continue

        # Validate input exists
        if not input_path.exists():
            logger.error(f"  Input file not found: {input_path}")
            progress.files[input_str] = FileStatus(
                input_path=input_str,
                output_path=str(final_output),
                status="failed",
                error="Input file not found",
                processed_at=datetime.now().isoformat(),
            )
            progress.save(progress_path)
            failed_count += 1
            continue

        # Process the file
        try:
            if in_place:
                # Write to temp file first, then replace original on success
                temp_fd, temp_path = tempfile.mkstemp(suffix=input_path.suffix)
                os.close(temp_fd)  # Close fd, CloudCompare will write to path
                temp_path = Path(temp_path)
                try:
                    compute_normals_cloudcompare(
                        input_path=input_path,
                        output_path=temp_path,
                        radius=radius,
                        mst_neighbors=mst_neighbors,
                        cloudcompare_path=cloudcompare_path,
                        use_xvfb=use_xvfb,
                    )
                    # Success - replace original with temp file
                    shutil.move(str(temp_path), str(input_path))
                    logger.info(f"  Success (in-place): {input_path}")
                finally:
                    # Clean up temp file if it still exists (failed case)
                    if temp_path.exists():
                        temp_path.unlink()
            else:
                compute_normals_cloudcompare(
                    input_path=input_path,
                    output_path=final_output,
                    radius=radius,
                    mst_neighbors=mst_neighbors,
                    cloudcompare_path=cloudcompare_path,
                    use_xvfb=use_xvfb,
                )
                logger.info(f"  Success: {final_output}")

            progress.files[input_str] = FileStatus(
                input_path=input_str,
                output_path=str(final_output),
                status="success",
                processed_at=datetime.now().isoformat(),
            )
            success_count += 1

        except CloudCompareError as e:
            logger.error(f"  Failed: {e}")
            progress.files[input_str] = FileStatus(
                input_path=input_str,
                output_path=str(final_output),
                status="failed",
                error=str(e),
                processed_at=datetime.now().isoformat(),
            )
            failed_count += 1

        except Exception as e:
            logger.error(f"  Unexpected error: {e}")
            progress.files[input_str] = FileStatus(
                input_path=input_str,
                output_path=str(final_output),
                status="failed",
                error=f"Unexpected: {e}",
                processed_at=datetime.now().isoformat(),
            )
            failed_count += 1

        # Save progress after each file
        progress.save(progress_path)

    return success_count, failed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Compute normals for LAS/LAZ files listed in a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Format:
  input_path,output_path
  /path/to/file1.laz,/path/to/output1.laz
  /path/to/file2.laz

The output_path column is optional. If omitted, files are saved to
--output-dir (or alongside input files) with the specified suffix.

Examples:
  %(prog)s files.csv --output-dir ./normals_output
  %(prog)s files.csv --path-column path --output-dir ./output  # Use 'path' column
  %(prog)s files.csv --path-column 2 --output-dir ./output  # Use 3rd column (0-indexed)
  %(prog)s files.csv --in-place  # Overwrite originals with normals
  %(prog)s files.csv --n-clouds 5  # Test with first 5 files only
  %(prog)s files.csv --radius 0.2 --mst-neighbors 12
  %(prog)s files.csv --resume  # Resume interrupted run
        """,
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file containing input file paths",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="path",
        help=(
            "Column containing input file paths. Can be column name (e.g., 'path') "
            "or zero-based index (e.g., '2'). Default: 'path'"
        ),
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default=None,
        help=(
            "Column containing output file paths (optional). Can be column name or index."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for processed files",
    )
    parser.add_argument(
        "-r", "--radius",
        type=float,
        default=0.1,
        help="Local radius for normal estimation (default: 0.1)",
    )
    parser.add_argument(
        "-n", "--mst-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for MST orientation (default: 10)",
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str,
        default="_normals",
        help="Suffix for output filenames (default: '_normals')",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Path to progress file (default: <csv_name>_progress.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress file",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess files even if output already exists",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files with normals added (writes to temp first for safety)",
    )
    parser.add_argument(
        "--n-clouds",
        type=int,
        default=None,
        help="Only process the first N clouds (useful for testing)",
    )
    parser.add_argument(
        "--cloudcompare-path",
        type=str,
        default=None,
        help=(
            "Path to CloudCompare executable, or 'flatpak' for Flatpak installation. "
            "Auto-detected if not specified (including Flatpak)."
        ),
    )
    parser.add_argument(
        "--xvfb",
        action="store_true",
        default=None,
        help="Use xvfb-run for headless operation (auto-enabled on Linux if available)",
    )
    parser.add_argument(
        "--no-xvfb",
        action="store_true",
        help="Disable xvfb-run even on Linux",
    )

    args = parser.parse_args()

    # Validate CSV exists
    if not args.csv_path.exists():
        logger.error(f"CSV file not found: {args.csv_path}")
        sys.exit(1)

    # Check CloudCompare availability
    cc_path = args.cloudcompare_path or find_cloudcompare()
    if cc_path is None:
        logger.error(
            "CloudCompare not found. Install CloudCompare, install via Flatpak, "
            "or specify --cloudcompare-path"
        )
        sys.exit(1)

    # Determine xvfb usage
    if args.no_xvfb:
        use_xvfb = False
    elif args.xvfb:
        use_xvfb = True
    else:
        # Auto-detect on Linux
        use_xvfb = None  # Let compute_normals_cloudcompare auto-detect

    # Log configuration
    if cc_path == "flatpak":
        logger.info("Using CloudCompare: Flatpak (org.cloudcompare.CloudCompare)")
        if is_cloudcompare_flatpak_installed():
            logger.info("  Flatpak installation verified")
    else:
        logger.info(f"Using CloudCompare: {cc_path}")

    if use_xvfb is True or (use_xvfb is None and is_xvfb_available()):
        logger.info("Using xvfb-run for headless operation")
    elif use_xvfb is False:
        logger.info("xvfb-run disabled")

    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Progress file path
    progress_path = args.progress_file or args.csv_path.with_suffix(".progress.json")

    # Load or create progress
    if args.resume and progress_path.exists():
        logger.info(f"Resuming from progress file: {progress_path}")
        progress = ProcessingProgress.load(progress_path)
    else:
        progress = ProcessingProgress(
            csv_path=str(args.csv_path),
            output_dir=str(args.output_dir) if args.output_dir else None,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

    # Read file list
    try:
        files = read_csv_file_list(
            args.csv_path,
            path_column=args.path_column,
            output_column=args.output_column,
        )
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    if not files:
        logger.error("No files found in CSV")
        sys.exit(1)

    total_in_csv = len(files)

    # Limit to first N clouds if requested
    if args.n_clouds is not None and args.n_clouds < total_in_csv:
        files = files[: args.n_clouds]
        logger.info(f"Processing first {len(files)} of {total_in_csv} clouds (--n-clouds)")
    else:
        logger.info(f"Found {len(files)} files to process")
    logger.info("-" * 60)

    # Warn if in-place mode
    if args.in_place:
        logger.warning("In-place mode: original files will be overwritten with normals")

    # Process files
    success, failed, skipped = process_files(
        files=files,
        output_dir=args.output_dir,
        progress=progress,
        progress_path=progress_path,
        radius=args.radius,
        mst_neighbors=args.mst_neighbors,
        suffix=args.suffix,
        skip_existing=not args.no_skip_existing,
        in_place=args.in_place,
        cloudcompare_path=cc_path,
        use_xvfb=use_xvfb,
    )

    # Summary
    logger.info("=" * 60)
    logger.info(f"Processing complete:")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Total:   {len(files)}")
    logger.info(f"Progress saved to: {progress_path}")

    if failed > 0:
        logger.info(f"\nFailed files:")
        for input_path, status in progress.files.items():
            if status.status == "failed":
                logger.info(f"  {input_path}: {status.error}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
