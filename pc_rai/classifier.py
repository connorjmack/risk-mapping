"""
RAI Classifier - Main processing pipeline for point cloud classification.

Provides a unified interface for the complete RAI processing workflow:
normals → slope → roughness → classification → statistics.
"""

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pc_rai.config import RAIConfig
from pc_rai.io.las_reader import PointCloud, load_point_cloud
from pc_rai.io.las_writer import save_point_cloud
from pc_rai.normals.cloudcompare import compute_normals_cloudcompare, extract_normals_from_las
from pc_rai.features.slope import calculate_slope
from pc_rai.features.roughness import calculate_all_roughness
from pc_rai.classification.decision_tree import classify_points, ClassificationThresholds
from pc_rai.classification.pca_classifier import (
    PCAClassificationResult,
    classify_pca,
    compare_with_rai,
    get_cluster_interpretation,
)
from pc_rai.utils.spatial import SpatialIndex
from pc_rai.reporting.statistics import calculate_all_statistics
from pc_rai.reporting.report_writer import (
    write_markdown_report,
    write_json_report,
    generate_config_summary,
)
from pc_rai.visualization import (
    render_classification,
    render_slope,
    render_roughness,
    create_comparison_figure,
    create_summary_figure,
    create_histogram_figure,
)


@dataclass
class RAIResult:
    """Container for RAI processing results.

    Attributes
    ----------
    source_file : str
        Name of input file.
    n_points : int
        Number of points in the cloud.
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    roughness_small_radius : np.ndarray, optional
        (N,) small-scale roughness from radius method.
    roughness_large_radius : np.ndarray, optional
        (N,) large-scale roughness from radius method.
    roughness_small_knn : np.ndarray, optional
        (N,) small-scale roughness from k-NN method.
    roughness_large_knn : np.ndarray, optional
        (N,) large-scale roughness from k-NN method.
    neighbor_count_small : np.ndarray
        (N,) neighbor counts at small scale.
    neighbor_count_large : np.ndarray
        (N,) neighbor counts at large scale.
    rai_class_radius : np.ndarray, optional
        (N,) RAI class codes from radius method.
    rai_class_knn : np.ndarray, optional
        (N,) RAI class codes from k-NN method.
    statistics : dict
        Computed statistics.
    timing : dict
        Processing timing information.
    """

    source_file: str
    n_points: int

    # Features
    slope_deg: np.ndarray
    roughness_small_radius: Optional[np.ndarray] = None
    roughness_large_radius: Optional[np.ndarray] = None
    roughness_small_knn: Optional[np.ndarray] = None
    roughness_large_knn: Optional[np.ndarray] = None
    neighbor_count_small: Optional[np.ndarray] = None
    neighbor_count_large: Optional[np.ndarray] = None

    # Classifications
    rai_class_radius: Optional[np.ndarray] = None
    rai_class_knn: Optional[np.ndarray] = None
    pca_result: Optional[PCAClassificationResult] = None

    # Statistics and timing
    statistics: Dict = field(default_factory=dict)
    timing: Dict = field(default_factory=dict)


class RAIClassifier:
    """Main class for RAI point cloud classification.

    Parameters
    ----------
    config : RAIConfig, optional
        Configuration parameters. Uses defaults if not provided.

    Examples
    --------
    >>> from pc_rai import RAIClassifier, load_point_cloud
    >>> classifier = RAIClassifier()
    >>> cloud = load_point_cloud("input.las")
    >>> result = classifier.process(cloud)
    >>> print(f"Classified {result.n_points} points")
    """

    def __init__(self, config: Optional[RAIConfig] = None):
        self.config = config or RAIConfig()

    def process(
        self,
        cloud: PointCloud,
        compute_normals: bool = True,
        run_pca: bool = False,
        pca_clusters: Optional[int] = None,
        verbose: bool = False,
    ) -> RAIResult:
        """
        Run full RAI processing pipeline.

        Parameters
        ----------
        cloud : PointCloud
            Point cloud to process.
        compute_normals : bool
            Whether to compute normals (if not present).
        run_pca : bool
            Whether to run PCA-based unsupervised classification.
        pca_clusters : int, optional
            Number of PCA clusters. If None, auto-detect (slower).
        verbose : bool
            Print progress information.

        Returns
        -------
        RAIResult
            Processing results including features, classifications, and statistics.
        """
        timing = {}
        total_start = time.time()

        source_file = str(cloud.source_file) if cloud.source_file else "unknown"
        n_points = cloud.n_points

        if verbose:
            print(f"Processing {n_points:,} points from {source_file}")

        # Step 1: Ensure normals exist
        if not cloud.has_normals:
            if compute_normals and self.config.compute_normals:
                raise ValueError(
                    "Point cloud has no normals and compute_normals requires "
                    "saving to a file first. Use process_file() instead."
                )
            else:
                raise ValueError(
                    "Point cloud has no normals. Set compute_normals=True or "
                    "provide a cloud with pre-computed normals."
                )

        # Step 2: Calculate slope
        t0 = time.time()
        if verbose:
            print("  Calculating slope...")
        slope_deg = calculate_slope(cloud.normals, self.config.up_vector)
        timing["slope"] = time.time() - t0

        # Step 3: Build spatial index
        t0 = time.time()
        if verbose:
            print("  Building spatial index...")
        spatial_index = SpatialIndex(cloud.xyz)
        timing["spatial_index"] = time.time() - t0

        # Step 4: Calculate roughness
        t0 = time.time()
        if verbose:
            print("  Calculating roughness...")
        roughness = calculate_all_roughness(
            slope_deg,
            spatial_index,
            radius_small=self.config.radius_small,
            radius_large=self.config.radius_large,
            k_small=self.config.k_small,
            k_large=self.config.k_large,
            min_neighbors=self.config.min_neighbors,
            methods=self.config.methods,
        )
        timing["roughness"] = time.time() - t0

        # Step 5: Classify points
        t0 = time.time()
        if verbose:
            print("  Classifying points...")

        thresholds = ClassificationThresholds.from_config(self.config)

        rai_class_radius = None
        rai_class_knn = None

        if "radius" in self.config.methods:
            rai_class_radius = classify_points(
                slope_deg,
                roughness["roughness_small_radius"],
                roughness["roughness_large_radius"],
                thresholds,
            )

        if "knn" in self.config.methods:
            rai_class_knn = classify_points(
                slope_deg,
                roughness["roughness_small_knn"],
                roughness["roughness_large_knn"],
                thresholds,
            )

        timing["classification"] = time.time() - t0

        # Step 6 (optional): PCA-based classification
        pca_result = None
        if run_pca:
            t0 = time.time()
            if verbose:
                print("  Running PCA classification...")

            # Use knn roughness if available, otherwise radius
            r_small = roughness.get("roughness_small_knn")
            if r_small is None:
                r_small = roughness.get("roughness_small_radius")
            r_large = roughness.get("roughness_large_knn")
            if r_large is None:
                r_large = roughness.get("roughness_large_radius")

            if r_small is not None and r_large is not None:
                try:
                    pca_result = classify_pca(
                        slope_deg=slope_deg,
                        roughness_small=r_small,
                        roughness_large=r_large,
                        n_clusters=pca_clusters,
                    )
                    if verbose:
                        print(f"    Found {pca_result.n_clusters} clusters "
                              f"(silhouette: {pca_result.silhouette_avg:.3f})")
                except ValueError as e:
                    if verbose:
                        print(f"    PCA classification failed: {e}")

            timing["pca_classification"] = time.time() - t0

        # Step 7: Calculate statistics
        t0 = time.time()
        if verbose:
            print("  Computing statistics...")
        statistics = calculate_all_statistics(
            slope_deg=slope_deg,
            roughness_small_radius=roughness.get("roughness_small_radius"),
            roughness_large_radius=roughness.get("roughness_large_radius"),
            roughness_small_knn=roughness.get("roughness_small_knn"),
            roughness_large_knn=roughness.get("roughness_large_knn"),
            classes_radius=rai_class_radius,
            classes_knn=rai_class_knn,
        )
        timing["statistics"] = time.time() - t0

        timing["total"] = time.time() - total_start

        if verbose:
            print(f"  Done in {timing['total']:.2f}s")

        return RAIResult(
            source_file=source_file,
            n_points=n_points,
            slope_deg=slope_deg,
            roughness_small_radius=roughness.get("roughness_small_radius"),
            roughness_large_radius=roughness.get("roughness_large_radius"),
            roughness_small_knn=roughness.get("roughness_small_knn"),
            roughness_large_knn=roughness.get("roughness_large_knn"),
            neighbor_count_small=roughness.get("neighbor_count_small"),
            neighbor_count_large=roughness.get("neighbor_count_large"),
            rai_class_radius=rai_class_radius,
            rai_class_knn=rai_class_knn,
            pca_result=pca_result,
            statistics=statistics,
            timing=timing,
        )

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        compute_normals: bool = True,
        generate_visualizations: bool = True,
        generate_report: bool = True,
        run_pca: bool = False,
        pca_clusters: Optional[int] = None,
        verbose: bool = False,
    ) -> RAIResult:
        """
        Process a single file end-to-end.

        Loads input, processes, saves output with attributes,
        and optionally generates visualizations and report.

        Parameters
        ----------
        input_path : Path
            Path to input LAS/LAZ file.
        output_dir : Path
            Directory for output files.
        compute_normals : bool
            Compute normals if not present.
        generate_visualizations : bool
            Generate visualization images.
        generate_report : bool
            Generate Markdown and JSON reports.
        run_pca : bool
            Run PCA-based unsupervised classification.
        pca_clusters : int, optional
            Number of PCA clusters. If None, auto-detect (slower).
        verbose : bool
            Print progress information.

        Returns
        -------
        RAIResult
            Processing results.
        """
        import tempfile

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timing = {}
        total_start = time.time()

        basename = input_path.stem

        if verbose:
            print(f"Loading {input_path}...")

        # Load point cloud
        cloud = load_point_cloud(input_path)

        if verbose:
            print(f"  Loaded {cloud.n_points:,} points")

        # Compute normals if needed
        if not cloud.has_normals and compute_normals and self.config.compute_normals:
            if verbose:
                print("  Computing normals via CloudCompare...")

            t0 = time.time()

            # Save to temp file with normals
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_output = Path(tmp_dir) / f"{basename}_normals.las"

                success = compute_normals_cloudcompare(
                    input_path,
                    tmp_output,
                    radius=self.config.normal_radius,
                    mst_neighbors=self.config.mst_neighbors,
                    cloudcompare_path=self.config.cloudcompare_path,
                )

                if not success:
                    raise RuntimeError("Failed to compute normals with CloudCompare")

                # Extract normals from output
                normals = extract_normals_from_las(tmp_output)
                if normals is None:
                    raise RuntimeError("Failed to extract normals from CloudCompare output")

                cloud.normals = normals

            timing["normals"] = time.time() - t0

        # Process
        result = self.process(
            cloud,
            compute_normals=False,
            run_pca=run_pca,
            pca_clusters=pca_clusters,
            verbose=verbose,
        )

        # Merge timing
        result.timing = {**timing, **result.timing}
        result.timing["total"] = time.time() - total_start

        # Save classified point cloud to output/rai/
        if verbose:
            print("  Saving classified point cloud...")

        rai_dir = output_dir / "rai"
        rai_dir.mkdir(parents=True, exist_ok=True)
        output_las = rai_dir / f"{basename}_rai.laz" if self.config.compress_output else rai_dir / f"{basename}_rai.las"

        attributes = {"slope_deg": result.slope_deg}

        if result.roughness_small_radius is not None:
            attributes["roughness_small_radius"] = result.roughness_small_radius
        if result.roughness_large_radius is not None:
            attributes["roughness_large_radius"] = result.roughness_large_radius
        if result.roughness_small_knn is not None:
            attributes["roughness_small_knn"] = result.roughness_small_knn
        if result.roughness_large_knn is not None:
            attributes["roughness_large_knn"] = result.roughness_large_knn
        if result.rai_class_radius is not None:
            attributes["rai_class_radius"] = result.rai_class_radius
        if result.rai_class_knn is not None:
            attributes["rai_class_knn"] = result.rai_class_knn
        if result.pca_result is not None:
            # Save PCA cluster labels (convert -1 invalid to 255 for uint8)
            pca_labels = result.pca_result.labels.copy()
            pca_labels[pca_labels < 0] = 255
            attributes["pca_cluster"] = pca_labels.astype(np.uint8)
        save_point_cloud(cloud, attributes, output_las, compress=self.config.compress_output)

        # Generate visualizations to output/figures/<date>/
        if generate_visualizations:
            if verbose:
                print("  Generating visualizations...")
            figures_dir = output_dir / "figures" / date.today().isoformat()
            figures_dir.mkdir(parents=True, exist_ok=True)
            self._generate_visualizations(cloud.xyz, result, figures_dir, basename)

        # Generate reports to output/reports/<date>/
        if generate_report:
            if verbose:
                print("  Generating reports...")
            reports_dir = output_dir / "reports" / date.today().isoformat()
            reports_dir.mkdir(parents=True, exist_ok=True)
            self._generate_reports(cloud, result, reports_dir, basename)

        if verbose:
            print(f"  Complete! Total time: {result.timing['total']:.2f}s")

        return result

    def process_batch(
        self,
        input_paths: List[Path],
        output_dir: Path,
        **kwargs,
    ) -> List[RAIResult]:
        """
        Process multiple files.

        Parameters
        ----------
        input_paths : list of Path
            List of input file paths.
        output_dir : Path
            Directory for output files.
        **kwargs
            Additional arguments passed to process_file.

        Returns
        -------
        list of RAIResult
            Results for each file.
        """
        results = []
        for i, path in enumerate(input_paths):
            print(f"[{i + 1}/{len(input_paths)}] Processing {path.name}...")
            try:
                result = self.process_file(path, output_dir, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"  Error processing {path}: {e}")
                results.append(None)
        return results

    def _generate_visualizations(
        self,
        xyz: np.ndarray,
        result: RAIResult,
        output_dir: Path,
        basename: str,
    ) -> None:
        """Generate visualization images."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        dpi = self.config.visualization_dpi
        views = self.config.visualization_views

        # Classification visualizations
        for view in views:
            if result.rai_class_radius is not None:
                fig = render_classification(
                    xyz,
                    result.rai_class_radius,
                    view=view,
                    title=f"RAI Classification (Radius) - {view.title()}",
                    dpi=dpi,
                    output_path=str(output_dir / f"{basename}_classification_radius_{view}.png"),
                )
                plt.close(fig)

            if result.rai_class_knn is not None:
                fig = render_classification(
                    xyz,
                    result.rai_class_knn,
                    view=view,
                    title=f"RAI Classification (k-NN) - {view.title()}",
                    dpi=dpi,
                    output_path=str(output_dir / f"{basename}_classification_knn_{view}.png"),
                )
                plt.close(fig)

        # Slope visualization
        fig = render_slope(
            xyz,
            result.slope_deg,
            view="front",
            title="Slope Angle",
            dpi=dpi,
            output_path=str(output_dir / f"{basename}_slope.png"),
        )
        plt.close(fig)

        # Roughness visualizations
        if result.roughness_small_radius is not None:
            fig = render_roughness(
                xyz,
                result.roughness_small_radius,
                view="front",
                title="Small-Scale Roughness (Radius)",
                dpi=dpi,
                output_path=str(output_dir / f"{basename}_roughness_small_radius.png"),
            )
            plt.close(fig)

        if result.roughness_large_radius is not None:
            fig = render_roughness(
                xyz,
                result.roughness_large_radius,
                view="front",
                title="Large-Scale Roughness (Radius)",
                dpi=dpi,
                output_path=str(output_dir / f"{basename}_roughness_large_radius.png"),
            )
            plt.close(fig)

        # Comparison figure (if both methods)
        if result.rai_class_radius is not None and result.rai_class_knn is not None:
            fig = create_comparison_figure(
                xyz,
                result.rai_class_radius,
                result.rai_class_knn,
                view="front",
                dpi=dpi,
                output_path=str(output_dir / f"{basename}_comparison.png"),
            )
            plt.close(fig)

        # Summary figure
        r_small = result.roughness_small_radius if result.roughness_small_radius is not None else result.roughness_small_knn
        r_large = result.roughness_large_radius if result.roughness_large_radius is not None else result.roughness_large_knn
        classes = result.rai_class_radius if result.rai_class_radius is not None else result.rai_class_knn

        if r_small is not None and r_large is not None and classes is not None:
            fig = create_summary_figure(
                xyz,
                result.slope_deg,
                r_small,
                r_large,
                classes,
                view="front",
                dpi=dpi,
                output_path=str(output_dir / f"{basename}_summary.png"),
            )
            plt.close(fig)

        # Histogram
        if classes is not None:
            fig = create_histogram_figure(
                classes,
                title="RAI Class Distribution",
                dpi=dpi,
                output_path=str(output_dir / f"{basename}_histogram.png"),
            )
            plt.close(fig)

    def _generate_reports(
        self,
        cloud: PointCloud,
        result: RAIResult,
        output_dir: Path,
        basename: str,
    ) -> None:
        """Generate Markdown and JSON reports."""
        config_summary = generate_config_summary(self.config)
        extent = cloud.bounds

        # Markdown report
        write_markdown_report(
            result.statistics,
            output_dir / f"{basename}_report.md",
            result.source_file,
            config_summary,
            extent=extent,
            timing=result.timing,
        )

        # JSON report
        write_json_report(
            result.statistics,
            output_dir / f"{basename}_report.json",
            result.source_file,
            config_summary,
            extent=extent,
            timing=result.timing,
        )
