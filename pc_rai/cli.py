"""Command-line interface for PC-RAI."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from pc_rai import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="pc-rai",
        description="Point Cloud Rockfall Activity Index Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  pc-rai process input.las -o output/

  # Process with custom config
  pc-rai process input.las -o output/ --config config.yaml

  # Batch process a directory
  pc-rai process ./data/ -o output/ --batch

  # Skip normal computation (use existing normals)
  pc-rai process input.las -o output/ --skip-normals

  # Use only radius method
  pc-rai process input.las -o output/ --methods radius

  # Generate visualizations from processed file
  pc-rai visualize output/input_rai.las -o figures/
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process point cloud(s) with RAI classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    process_parser.add_argument(
        "input",
        type=Path,
        help="Input LAS/LAZ file or directory",
    )
    process_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    process_parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Configuration YAML file",
    )
    process_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all LAS/LAZ files in directory",
    )
    process_parser.add_argument(
        "--skip-normals",
        action="store_true",
        help="Skip normal computation (use existing normals)",
    )
    process_parser.add_argument(
        "--methods",
        choices=["radius", "knn", "both"],
        default="knn",
        help="Roughness computation method(s) (default: knn)",
    )
    process_parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation",
    )
    process_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation",
    )
    process_parser.add_argument(
        "--pca",
        action="store_true",
        help="Run PCA-based unsupervised classification",
    )
    process_parser.add_argument(
        "--pca-clusters",
        type=int,
        default=None,
        metavar="N",
        help="Number of PCA clusters (skips auto-detection, much faster)",
    )
    process_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Visualize command (for already-processed files)
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate visualizations from processed LAS file",
    )
    viz_parser.add_argument(
        "input",
        type=Path,
        help="Processed LAS file with RAI attributes",
    )
    viz_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    viz_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image resolution (default: 300)",
    )
    viz_parser.add_argument(
        "--views",
        nargs="+",
        choices=["front", "oblique", "top", "side"],
        default=["front", "oblique"],
        help="View angles to render (default: front oblique)",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 1

    try:
        if parsed.command == "process":
            return run_process(parsed)
        elif parsed.command == "visualize":
            return run_visualize(parsed)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def run_process(args) -> int:
    """Run processing command."""
    from pc_rai.classifier import RAIClassifier
    from pc_rai.config import RAIConfig, load_config

    # Load configuration
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = load_config(args.config)
        if args.verbose:
            print(f"Loaded config from {args.config}")
    else:
        config = RAIConfig()

    # Apply CLI overrides
    if args.methods == "radius":
        config.methods = ["radius"]
    elif args.methods == "knn":
        config.methods = ["knn"]
    else:
        config.methods = ["radius", "knn"]

    if args.skip_normals:
        config.compute_normals = False

    # Create classifier
    classifier = RAIClassifier(config)

    # Determine input files
    input_path = args.input
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}", file=sys.stderr)
        return 1

    if args.batch or input_path.is_dir():
        if not input_path.is_dir():
            print(f"Error: --batch requires a directory, got: {input_path}", file=sys.stderr)
            return 1
        input_files = list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))
        input_files += list(input_path.glob("*.LAS")) + list(input_path.glob("*.LAZ"))
        input_files = sorted(set(input_files))
        if not input_files:
            print(f"Error: No LAS/LAZ files found in {input_path}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Found {len(input_files)} files to process")
    else:
        if not input_path.is_file():
            print(f"Error: Input is not a file: {input_path}", file=sys.stderr)
            return 1
        input_files = [input_path]

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process files
    success_count = 0
    error_count = 0

    for i, filepath in enumerate(input_files):
        if len(input_files) > 1:
            print(f"\n[{i + 1}/{len(input_files)}] {filepath.name}")

        try:
            result = classifier.process_file(
                filepath,
                args.output,
                compute_normals=not args.skip_normals,
                generate_visualizations=not args.no_visualize,
                generate_report=not args.no_report,
                run_pca=args.pca or args.pca_clusters is not None,
                pca_clusters=args.pca_clusters,
                verbose=args.verbose,
            )

            # Print summary
            if args.verbose:
                _print_result_summary(result)

            success_count += 1

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            error_count += 1

    # Final summary
    print(f"\nCompleted: {success_count} succeeded, {error_count} failed")

    return 0 if error_count == 0 else 1


def run_visualize(args) -> int:
    """Run visualization command."""
    from datetime import date
    import laspy
    import numpy as np

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pc_rai.visualization import (
        render_classification,
        render_slope,
        render_roughness,
        create_histogram_figure,
    )

    # Load processed file
    input_path = args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading {input_path}...")
    las = laspy.read(input_path)

    xyz = np.column_stack([las.x, las.y, las.z])
    print(f"  Loaded {len(xyz):,} points")

    # Get extra dimensions
    dim_names = [dim.name for dim in las.point_format.extra_dimensions]

    # Create output directory with date subfolder
    output_dir = args.output / "figures" / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = input_path.stem

    print("Generating visualizations...")

    # Classification visualizations
    for method in ["radius", "knn"]:
        class_name = f"rai_class_{method}"
        if class_name in dim_names:
            classes = las[class_name].astype(np.uint8)
            for view in args.views:
                output_file = output_dir / f"{basename}_classification_{method}_{view}.png"
                fig = render_classification(
                    xyz,
                    classes,
                    view=view,
                    title=f"RAI Classification ({method.title()}) - {view.title()}",
                    dpi=args.dpi,
                    output_path=str(output_file),
                )
                plt.close(fig)
                print(f"  Created {output_file.name}")

            # Histogram
            output_file = output_dir / f"{basename}_histogram_{method}.png"
            fig = create_histogram_figure(
                classes,
                title=f"RAI Class Distribution ({method.title()})",
                dpi=args.dpi,
                output_path=str(output_file),
            )
            plt.close(fig)
            print(f"  Created {output_file.name}")

    # Slope visualization
    if "slope_deg" in dim_names:
        slope = las["slope_deg"]
        output_file = output_dir / f"{basename}_slope.png"
        fig = render_slope(
            xyz,
            slope,
            view="front",
            dpi=args.dpi,
            output_path=str(output_file),
        )
        plt.close(fig)
        print(f"  Created {output_file.name}")

    # Roughness visualizations
    for roughness_name in ["roughness_small_radius", "roughness_large_radius",
                           "roughness_small_knn", "roughness_large_knn"]:
        if roughness_name in dim_names:
            roughness = las[roughness_name]
            output_file = output_dir / f"{basename}_{roughness_name}.png"
            fig = render_roughness(
                xyz,
                roughness,
                view="front",
                title=roughness_name.replace("_", " ").title(),
                dpi=args.dpi,
                output_path=str(output_file),
            )
            plt.close(fig)
            print(f"  Created {output_file.name}")

    print("Done!")
    return 0


def _print_result_summary(result) -> None:
    """Print a summary of processing results."""
    from pc_rai.config import RAI_CLASS_NAMES

    print(f"\n  Summary:")
    print(f"    Points: {result.n_points:,}")
    print(f"    Total time: {result.timing.get('total', 0):.2f}s")

    # Classification summary
    if result.rai_class_radius is not None:
        print(f"\n    Classification (Radius):")
        _print_class_distribution(result.rai_class_radius)

    if result.rai_class_knn is not None:
        print(f"\n    Classification (k-NN):")
        _print_class_distribution(result.rai_class_knn)

    # Agreement
    if result.statistics.get("method_agreement"):
        agreement = result.statistics["method_agreement"]
        print(f"\n    Method Agreement: {agreement['agreement_pct']:.1f}%")
        print(f"    Cohen's Kappa: {agreement['cohens_kappa']:.3f}")

    # PCA classification
    if result.pca_result is not None:
        pca = result.pca_result
        print(f"\n    PCA Classification:")
        print(f"      Clusters found: {pca.n_clusters}")
        print(f"      Silhouette score: {pca.silhouette_avg:.3f}")
        print(f"      Variance explained: {sum(pca.explained_variance_ratio)*100:.1f}%")
        print(f"\n      Cluster distribution:")
        _print_pca_clusters(pca)


def _print_pca_clusters(pca_result) -> None:
    """Print PCA cluster distribution."""
    from pc_rai.classification.pca_classifier import get_cluster_interpretation

    interpretations = get_cluster_interpretation(pca_result)

    for cluster_id in range(pca_result.n_clusters):
        stats = pca_result.cluster_stats[cluster_id]
        n_points = stats["n_points"]
        pct = stats["percentage"]
        interp = interpretations[cluster_id]
        print(f"        {cluster_id}: {n_points:8,} ({pct:5.1f}%) - {interp}")


def _print_class_distribution(classes) -> None:
    """Print class distribution."""
    import numpy as np
    from pc_rai.config import RAI_CLASS_NAMES, RAI_CLASS_ABBREV

    n_total = len(classes)
    for code in range(8):
        count = (classes == code).sum()
        if count > 0:
            pct = 100 * count / n_total
            name = RAI_CLASS_NAMES[code]
            abbrev = RAI_CLASS_ABBREV[code]
            print(f"      {abbrev:3s} {name:30s} {count:8,} ({pct:5.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
