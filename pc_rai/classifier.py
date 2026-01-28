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
from pc_rai.classification.energy import (
    RAIEnergyParams,
    calculate_point_energy,
    get_energy_statistics,
)
from pc_rai.classification.pca_classifier import (
    PCAClassificationResult,
    classify_pca,
    compare_with_rai,
    get_cluster_interpretation,
)
from pc_rai.utils.spatial import SpatialIndex, voxel_subsample, smooth_values_radius
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
    render_dunham_figure,
    render_single_panel,
    render_risk_map,
    render_risk_map_profile,
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
    energy_radius : np.ndarray, optional
        (N,) per-point energy (kJ) from radius classification.
    energy_knn : np.ndarray, optional
        (N,) per-point energy (kJ) from k-NN classification.
    energy_statistics : dict, optional
        Energy statistics by class and overall.
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

    # Energy (Dunham et al. 2017)
    energy_radius: Optional[np.ndarray] = None
    energy_knn: Optional[np.ndarray] = None
    energy_statistics: Optional[Dict] = None

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
        smooth_slope: Optional[float] = None,
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
        smooth_slope : float, optional
            Radius for slope smoothing (meters). None = no smoothing.
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

        # Step 3b (optional): Smooth slope values
        if smooth_slope is not None and smooth_slope > 0:
            t0 = time.time()
            if verbose:
                print(f"  Smoothing slope (radius={smooth_slope}m)...")
            slope_deg = smooth_values_radius(slope_deg, spatial_index, smooth_slope)
            timing["smooth_slope"] = time.time() - t0

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

        # Step 5b: Calculate RAI energy (Dunham et al. 2017)
        t0 = time.time()
        if verbose:
            print("  Calculating RAI energy...")

        energy_radius = None
        energy_knn = None
        energy_statistics = {}

        # Get Z coordinates for fall height calculation
        z_coords = cloud.xyz[:, 2]

        if rai_class_radius is not None:
            energy_radius = calculate_point_energy(
                rai_class_radius,
                z_coords,
                base_elevation=None,  # Auto-detect from min Z
            )
            energy_statistics["radius"] = get_energy_statistics(
                energy_radius, rai_class_radius
            )

        if rai_class_knn is not None:
            energy_knn = calculate_point_energy(
                rai_class_knn,
                z_coords,
                base_elevation=None,  # Auto-detect from min Z
            )
            energy_statistics["knn"] = get_energy_statistics(
                energy_knn, rai_class_knn
            )

        timing["energy"] = time.time() - t0

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
            energy_radius=energy_radius,
            energy_knn=energy_knn,
            energy_statistics=energy_statistics if energy_statistics else None,
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
        subsample: Optional[float] = None,
        smooth_slope: Optional[float] = None,
        transects_kml: Optional[Path] = None,
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
        subsample : float, optional
            Voxel grid spacing for subsampling (meters). None = no subsampling.
        smooth_slope : float, optional
            Radius for slope smoothing (meters). None = no smoothing.
        transects_kml : Path, optional
            Path to KML file with transect lines for risk map binning.
            If provided, uses transects instead of axis-aligned bins.
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

        # Subsample if requested
        if subsample is not None and subsample > 0:
            t0 = time.time()
            original_count = cloud.n_points
            xyz_sub, normals_sub, _ = voxel_subsample(
                cloud.xyz, subsample, cloud.normals
            )
            cloud.xyz = xyz_sub
            cloud.normals = normals_sub
            # Clear _las_data so save_point_cloud creates fresh file with subsampled points
            cloud._las_data = None
            timing["subsample"] = time.time() - t0
            if verbose:
                reduction = 100 * (1 - cloud.n_points / original_count)
                print(f"  Subsampled to {cloud.n_points:,} points ({reduction:.1f}% reduction)")

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
            smooth_slope=smooth_slope,
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
        # Energy fields (Dunham et al. 2017)
        if result.energy_radius is not None:
            attributes["energy_kj_radius"] = result.energy_radius
        if result.energy_knn is not None:
            attributes["energy_kj_knn"] = result.energy_knn
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
            self._generate_visualizations(cloud.xyz, result, figures_dir, basename, transects_kml)

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
        transects_kml: Optional[Path] = None,
    ) -> None:
        """Generate visualization images.

        Currently generates only the Dunham-style 3-panel figure
        (intensity, classification, energy) as the primary output.

        Figures are organized into subfolders:
        - panels/: Dunham-style 3-panel figures
        - hist/: Histogram distribution figures
        - heatmap/: Risk map heatmaps

        Parameters
        ----------
        xyz : np.ndarray
            (N, 3) point coordinates.
        result : RAIResult
            Processing results.
        output_dir : Path
            Output directory for figures.
        basename : str
            Base name for output files.
        transects_kml : Path, optional
            Path to KML file with transect lines. If provided, uses transect-based
            risk maps instead of axis-aligned bins.
        """
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        dpi = self.config.visualization_dpi

        # Create subfolders for organized output
        panels_dir = output_dir / "panels"
        hist_dir = output_dir / "hist"
        heatmap_dir = output_dir / "heatmap"

        panels_dir.mkdir(exist_ok=True)
        hist_dir.mkdir(exist_ok=True)
        heatmap_dir.mkdir(exist_ok=True)

        # Get the primary classification and energy results
        classes = result.rai_class_knn if result.rai_class_knn is not None else result.rai_class_radius
        energy = result.energy_knn if result.energy_knn is not None else result.energy_radius

        # Dunham-style 3-panel figure (intensity, classification, energy)
        if classes is not None and energy is not None:
            fig = render_dunham_figure(
                xyz,
                classes,
                energy,
                intensity=None,  # TODO: pass intensity if available from LAS
                output_path=str(panels_dir / f"{basename}_dunham_panels.png"),
                dpi=dpi,
                title=basename,
            )
            plt.close(fig)

            # Risk map generation - only if transects provided
            if transects_kml is not None:
                # Parse shapefile transects and generate 3D transect risk map
                try:
                    from pc_rai.visualization.risk_map import (
                        parse_transects,
                        render_transect_risk_map_3d,
                    )

                    transects = parse_transects(transects_kml)
                    print(f"  Loaded {len(transects)} transects from {transects_kml.name}")

                    # 3D transect risk map with satellite imagery
                    fig = render_transect_risk_map_3d(
                        xyz,
                        energy,
                        transects,
                        output_path=str(heatmap_dir / f"{basename}_risk_map_3d.png"),
                        half_width=5.0,
                        dpi=dpi,
                        title=f"{basename} - 3D Transect Risk Map",
                    )
                    plt.close(fig)
                    print(f"  Generated 3D transect map: heatmap/{basename}_risk_map_3d.png")

                except Exception as e:
                    print(f"  Warning: Could not generate 3D transect map: {e}")

    def _generate_axis_aligned_risk_maps(
        self,
        xyz: np.ndarray,
        energy: np.ndarray,
        heatmap_dir: Path,
        basename: str,
        dpi: int,
    ) -> None:
        """Generate axis-aligned (E-W or N-S) risk maps."""
        import matplotlib.pyplot as plt

        # 10m alongshore energy risk map (profile view - simpler, always works)
        fig = render_risk_map_profile(
            xyz,
            energy,
            output_path=str(heatmap_dir / f"{basename}_risk_map_10m.png"),
            bin_size=10.0,
            dpi=dpi,
            title=f"{basename} - Alongshore Energy Risk (10m bins)",
        )
        plt.close(fig)

        # Generate spatial map view with basemap (requires contextily)
        try:
            fig = render_risk_map(
                xyz,
                energy,
                output_path=str(heatmap_dir / f"{basename}_risk_map_spatial.png"),
                bin_size=10.0,
                dpi=dpi,
                title=f"{basename} - Spatial Energy Risk Map",
                add_basemap=True,
            )
            plt.close(fig)
            print(f"  Generated spatial risk map: heatmap/{basename}_risk_map_spatial.png")
        except Exception as e:
            print(f"  Warning: Could not generate spatial risk map: {e}")

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
