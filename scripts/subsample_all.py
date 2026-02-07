#!/usr/bin/env python3
"""
Subsample all noveg LAS files to LAZ for the full training pipeline.

Reads file paths from utiliies/file_lists/all_noveg_files.csv and writes
voxel-subsampled LAZ files to data/laz_files/. These LAZ files are then
enriched in-place with normals and features in subsequent pipeline steps.

Usage:
    python scripts/subsample_all.py
    python scripts/subsample_all.py --voxel-size 0.5 --dry-run
    python scripts/subsample_all.py --location DelMar  # single location
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pc_rai.ml.feature_extraction import subsample_survey

logger = logging.getLogger(__name__)

FILE_LIST = Path(__file__).parent.parent / "utiliies" / "file_lists" / "all_noveg_files.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "laz_files"


def load_file_list(csv_path: Path, location: str = None) -> list[dict]:
    """Load file list CSV, optionally filtering by location."""
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if location and row["location"] != location:
                continue
            entries.append(row)
    return entries


def expected_output_name(filename: str) -> str:
    """Derive the expected output .laz filename from input .las filename."""
    stem = Path(filename).stem
    if "_noveg" in stem:
        stem = stem.replace("_noveg", "_subsampled")
    else:
        stem = stem + "_subsampled"
    return f"{stem}.laz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subsample all noveg files to LAZ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=FILE_LIST,
        help=f"CSV file list (default: {FILE_LIST})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
        help="Voxel size in meters (default: 0.5)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Process only this location (e.g., DelMar, Torrey)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess files even if output exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Load file list
    if not args.file_list.exists():
        print(f"Error: File list not found: {args.file_list}", file=sys.stderr)
        return 1

    entries = load_file_list(args.file_list, location=args.location)
    if not entries:
        print("Error: No files matched", file=sys.stderr)
        return 1

    # Count by location
    locations = {}
    for e in entries:
        loc = e["location"]
        locations[loc] = locations.get(loc, 0) + 1

    print(f"File list: {args.file_list}")
    print(f"Output dir: {args.output_dir}")
    print(f"Voxel size: {args.voxel_size}m")
    print(f"Total files: {len(entries)}")
    for loc, count in sorted(locations.items()):
        print(f"  {loc}: {count}")

    # Check which already exist
    skip_existing = not args.no_skip_existing
    to_process = []
    n_skipped = 0

    for entry in entries:
        out_name = expected_output_name(entry["filename"])
        out_path = args.output_dir / out_name
        if skip_existing and out_path.exists():
            n_skipped += 1
        else:
            to_process.append(entry)

    if n_skipped > 0:
        print(f"Skipping {n_skipped} already processed files")
    print(f"To process: {len(to_process)}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for entry in to_process[:10]:
            print(f"  {entry['location']}: {entry['filename']}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return 0

    if len(to_process) == 0:
        print("Nothing to do.")
        return 0

    # Process
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_failed = 0
    failed_files = []
    t_start = time.time()

    for i, entry in enumerate(to_process):
        input_path = Path(entry["path"])
        loc = entry["location"]

        print(f"\n[{i+1}/{len(to_process)}] {loc}: {entry['filename']}")

        if not input_path.exists():
            print(f"  SKIP: file not found at {input_path}")
            n_failed += 1
            failed_files.append((entry["filename"], "not found"))
            continue

        try:
            result = subsample_survey(
                input_path,
                args.output_dir,
                voxel_size=args.voxel_size,
                verbose=args.verbose,
            )
            if result is not None:
                n_processed += 1
            else:
                n_failed += 1
                failed_files.append((entry["filename"], "subsample returned None"))
        except Exception as e:
            print(f"  ERROR: {e}")
            n_failed += 1
            failed_files.append((entry["filename"], str(e)))

        # Progress summary every 50 files
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(to_process) - i - 1) / rate
            print(f"\n  --- Progress: {i+1}/{len(to_process)} "
                  f"({n_processed} ok, {n_failed} failed) "
                  f"~{remaining/60:.0f} min remaining ---")

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped (existing): {n_skipped}")
    print(f"  Failed: {n_failed}")

    if failed_files:
        print(f"\nFailed files:")
        for fname, reason in failed_files:
            print(f"  {fname}: {reason}")

    return 1 if n_processed == 0 and len(to_process) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
