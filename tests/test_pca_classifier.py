"""Tests for PCA-based classification."""

import numpy as np
import pytest


class TestClassifyPCA:
    """Tests for classify_pca function."""

    def test_basic_clustering(self):
        """Test basic PCA classification with synthetic data."""
        from pc_rai.classification.pca_classifier import classify_pca

        # Create synthetic data with 3 distinct clusters
        n = 300
        rng = np.random.default_rng(42)

        # Cluster 1: low slope, low roughness (talus-like)
        slope1 = rng.normal(30, 5, n // 3)
        r_small1 = rng.normal(3, 1, n // 3)
        r_large1 = rng.normal(5, 1, n // 3)

        # Cluster 2: steep slope, medium roughness (cliff-like)
        slope2 = rng.normal(80, 10, n // 3)
        r_small2 = rng.normal(10, 2, n // 3)
        r_large2 = rng.normal(8, 2, n // 3)

        # Cluster 3: overhang, high roughness
        slope3 = rng.normal(120, 15, n // 3)
        r_small3 = rng.normal(20, 3, n // 3)
        r_large3 = rng.normal(15, 3, n // 3)

        slope = np.concatenate([slope1, slope2, slope3])
        r_small = np.concatenate([r_small1, r_small2, r_small3])
        r_large = np.concatenate([r_large1, r_large2, r_large3])

        result = classify_pca(slope, r_small, r_large)

        assert result.n_clusters >= 2  # Should find at least 2 clusters
        assert result.labels.shape == (n,)
        assert result.silhouette_avg > 0  # Clusters should be reasonably separated
        assert len(result.cluster_stats) == result.n_clusters

    def test_specified_clusters(self):
        """Test PCA classification with specified number of clusters."""
        from pc_rai.classification.pca_classifier import classify_pca

        n = 200
        rng = np.random.default_rng(42)
        slope = rng.uniform(0, 180, n)
        r_small = rng.uniform(0, 30, n)
        r_large = rng.uniform(0, 20, n)

        result = classify_pca(slope, r_small, r_large, n_clusters=5)

        assert result.n_clusters == 5
        assert len(np.unique(result.labels[result.labels >= 0])) == 5

    def test_handles_nan_values(self):
        """Test PCA classification handles NaN values correctly."""
        from pc_rai.classification.pca_classifier import classify_pca

        n = 200
        rng = np.random.default_rng(42)
        slope = rng.uniform(0, 180, n)
        r_small = rng.uniform(0, 30, n)
        r_large = rng.uniform(0, 20, n)

        # Add some NaN values
        slope[10:20] = np.nan
        r_small[50:60] = np.nan

        result = classify_pca(slope, r_small, r_large)

        # Invalid points should have label -1
        assert (result.labels[10:20] == -1).all()
        assert (result.labels[50:60] == -1).all()

        # Valid points should have labels >= 0
        valid_mask = ~np.isnan(slope) & ~np.isnan(r_small) & ~np.isnan(r_large)
        assert (result.labels[valid_mask] >= 0).all()

    def test_pca_components(self):
        """Test PCA components are computed correctly."""
        from pc_rai.classification.pca_classifier import classify_pca

        n = 200
        rng = np.random.default_rng(42)
        slope = rng.uniform(0, 180, n)
        r_small = rng.uniform(0, 30, n)
        r_large = rng.uniform(0, 20, n)

        result = classify_pca(slope, r_small, r_large)

        # Should have 3 components (same as features)
        assert result.pca_components.shape == (n, 3)
        assert len(result.explained_variance_ratio) == 3
        assert sum(result.explained_variance_ratio) <= 1.0

    def test_cluster_centers(self):
        """Test cluster centers are in original feature space."""
        from pc_rai.classification.pca_classifier import classify_pca

        n = 200
        rng = np.random.default_rng(42)
        slope = rng.uniform(0, 180, n)
        r_small = rng.uniform(0, 30, n)
        r_large = rng.uniform(0, 20, n)

        result = classify_pca(slope, r_small, r_large, n_clusters=4)

        # Cluster centers should be in original feature space
        assert result.cluster_centers.shape == (4, 3)

        # Centers should be within reasonable ranges
        assert (result.cluster_centers[:, 0] >= 0).all()  # slope
        assert (result.cluster_centers[:, 0] <= 180).all()


class TestCompareWithRAI:
    """Tests for compare_with_rai function."""

    def test_comparison_basic(self):
        """Test basic comparison between PCA and RAI labels."""
        from pc_rai.classification.pca_classifier import compare_with_rai

        pca_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        rai_labels = np.array([1, 1, 1, 2, 2, 2, 6, 6, 6])

        result = compare_with_rai(pca_labels, rai_labels)

        assert result["n_pca_clusters"] == 3
        assert result["n_rai_classes"] == 8
        assert result["confusion_matrix"].shape == (3, 8)
        assert result["overall_purity"] == 1.0  # Perfect cluster purity

    def test_comparison_handles_invalid(self):
        """Test comparison handles invalid PCA labels (-1)."""
        from pc_rai.classification.pca_classifier import compare_with_rai

        pca_labels = np.array([0, 0, -1, 1, 1, -1, 2, 2, -1])
        rai_labels = np.array([1, 1, 1, 2, 2, 2, 6, 6, 6])

        result = compare_with_rai(pca_labels, rai_labels)

        # Should only count valid points
        assert result["n_valid_points"] == 6


class TestGetClusterInterpretation:
    """Tests for get_cluster_interpretation function."""

    def test_interpretations(self):
        """Test cluster interpretations are generated."""
        from pc_rai.classification.pca_classifier import (
            classify_pca,
            get_cluster_interpretation,
        )

        n = 200
        rng = np.random.default_rng(42)
        slope = rng.uniform(0, 180, n)
        r_small = rng.uniform(0, 30, n)
        r_large = rng.uniform(0, 20, n)

        result = classify_pca(slope, r_small, r_large, n_clusters=3)
        interpretations = get_cluster_interpretation(result)

        assert len(interpretations) == 3
        for cluster_id, interp in interpretations.items():
            assert isinstance(interp, str)
            assert len(interp) > 0


class TestPCAClassifierIntegration:
    """Integration tests for PCA classifier."""

    def test_with_classifier(self, synthetic_cloud_with_normals):
        """Test PCA classification through main classifier."""
        from pc_rai.classifier import RAIClassifier
        from pc_rai.config import RAIConfig

        config = RAIConfig(methods=["knn"])
        classifier = RAIClassifier(config)

        result = classifier.process(
            synthetic_cloud_with_normals,
            compute_normals=False,
            run_pca=True,
        )

        assert result.pca_result is not None
        assert result.pca_result.n_clusters >= 2
        assert result.pca_result.labels.shape == (synthetic_cloud_with_normals.n_points,)


@pytest.fixture
def synthetic_cloud_with_normals():
    """Create a synthetic point cloud with normals."""
    from pc_rai.io.las_reader import PointCloud

    n = 500
    rng = np.random.default_rng(42)

    # Random points in a box
    xyz = rng.uniform(0, 10, (n, 3)).astype(np.float64)

    # Normals mostly pointing up with variation
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    normals += rng.normal(0, 0.3, (n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return PointCloud(xyz=xyz, normals=normals)
