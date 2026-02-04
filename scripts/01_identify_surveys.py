#!/usr/bin/env python3
"""
Step 1: Identify Pre-Event Surveys

Matches rockfall events to the most recent survey taken BEFORE each event,
enabling case-control study design for ML training.

Usage:
    python scripts/01_identify_surveys.py \
        --events utiliies/events/DelMar_events_qc_*.csv \
        --surveys utiliies/file_lists/all_noveg_files.csv \
        --output data/pre_event_surveys.csv \
        --location DelMar

Output:
    CSV file with columns:
    - survey_date: Date of the pre-event survey
    - survey_file: Filename of the survey
    - survey_path: Full path to the survey file
    - event_date: Start date of the event
    - event_id: Index of the event in the events file
    - days_before: Days between survey and event
    - event_volume: Volume of the event (m³)
    - event_alongshore_start: Alongshore start position (m)
    - event_alongshore_end: Alongshore end position (m)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pc_rai.ml.config import MLConfig
from pc_rai.ml.survey_selection import (
    create_pre_event_survey_dataset,
    deduplicate_surveys,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify pre-event surveys for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Path to events CSV file",
    )

    parser.add_argument(
        "--surveys",
        type=Path,
        required=True,
        help="Path to survey catalog CSV (all_noveg_files.csv)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for survey-event pairs CSV",
    )

    parser.add_argument(
        "--location",
        "-l",
        type=str,
        default=None,
        help="Location to filter surveys (e.g., 'DelMar')",
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=5.0,
        help="Minimum event volume in m³ (default: 5.0)",
    )

    parser.add_argument(
        "--min-days-before",
        type=int,
        default=7,
        help="Minimum days between survey and event (default: 7)",
    )

    parser.add_argument(
        "--qc-include",
        type=str,
        nargs="+",
        default=["real", "unreviewed"],
        help="QC flags to include (default: real unreviewed)",
    )

    parser.add_argument(
        "--qc-exclude",
        type=str,
        nargs="+",
        default=["construction", "noise"],
        help="QC flags to exclude (default: construction noise)",
    )

    parser.add_argument(
        "--list-surveys",
        action="store_true",
        help="Also output a simple list of unique survey paths",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
        verbose = False
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        verbose = True
    else:
        logging.basicConfig(level=logging.INFO)
        verbose = True

    # Validate inputs
    if not args.events.exists():
        print(f"Error: Events file not found: {args.events}", file=sys.stderr)
        return 1

    if not args.surveys.exists():
        print(f"Error: Surveys file not found: {args.surveys}", file=sys.stderr)
        return 1

    # Configure filtering
    config = MLConfig(
        min_volume=args.min_volume,
        qc_flags_include=args.qc_include,
        qc_flags_exclude=args.qc_exclude,
    )

    try:
        # Run the pipeline
        pairs, unique_surveys = create_pre_event_survey_dataset(
            surveys_csv=args.surveys,
            events_csv=args.events,
            output_path=args.output,
            location=args.location,
            config=config,
            min_days_before=args.min_days_before,
            verbose=verbose,
        )

        # Optionally output survey list
        if args.list_surveys:
            list_path = args.output.with_suffix(".survey_list.txt")
            with open(list_path, "w") as f:
                for path in unique_surveys:
                    f.write(path + "\n")
            if verbose:
                print(f"Saved survey list to: {list_path}")

        # Summary
        if verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Survey-event pairs: {len(pairs):,}")
            print(f"Unique surveys: {len(unique_surveys):,}")
            print(f"Output file: {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
