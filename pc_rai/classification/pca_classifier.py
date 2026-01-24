"""
PCA-based unsupervised classification for point clouds.

Uses Principal Component Analysis for dimensionality reduction followed by
K-means clustering to find natural groupings in the feature space.

Performance notes:
- Uses MiniBatchKMeans for O(N) clustering instead of O(N²)
- Subsamples for silhouette score calculation (O(N²) -> O(sample²))
- Subsamples for optimal cluster detection
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# Sample size for silhouette score (only used for final reporting, not cluster selection)
SILHOUETTE_SAMPLE_SIZE = 10_000


@dataclass
class PCAClassificationResult:
    """Results from PCA-based classification.

    Attributes
    ----------
    labels : np.ndarray
        (N,) cluster labels for each point.
    n_clusters : int
        Number of clusters found.
    pca_components : np.ndarray
        (N, n_components) PCA-transformed features.
    explained_variance_ratio : np.ndarray
        Variance explained by each principal component.
    cluster_centers : np.ndarray
        (n_clusters, n_features) cluster centers in original feature space.
    cluster_centers_pca : np.ndarray
        (n_clusters, n_components) cluster centers in PCA space.
    silhouette_avg : float
        Average silhouette score for the clustering.
    feature_names : List[str]
        Names of features used.
    pca_loadings : np.ndarray
        (n_components, n_features) PCA loadings matrix.
    cluster_stats : Dict
        Statistics for each cluster.
    """

    labels: np.ndarray
    n_clusters: int
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray
    cluster_centers: np.ndarray
    cluster_centers_pca: np.ndarray
    silhouette_avg: float
    feature_names: List[str]
    pca_loadings: np.ndarray
    cluster_stats: Dict


def classify_pca(
    slope_deg: np.ndarray,
    roughness_small: np.ndarray,
    roughness_large: np.ndarray,
    n_clusters: Optional[int] = None,
    min_clusters: int = 3,
    max_clusters: int = 12,
    n_components: int = 3,
    random_state: int = 42,
) -> PCAClassificationResult:
    """
    Classify points using PCA and K-means clustering.

    Parameters
    ----------
    slope_deg : np.ndarray
        (N,) slope angles in degrees.
    roughness_small : np.ndarray
        (N,) small-scale roughness values.
    roughness_large : np.ndarray
        (N,) large-scale roughness values.
    n_clusters : int, optional
        Number of clusters. If None, auto-detect using Calinski-Harabasz index.
    min_clusters : int
        Minimum clusters to try when auto-detecting (default: 3).
    max_clusters : int
        Maximum clusters to try when auto-detecting (default: 12).
    n_components : int
        Number of PCA components to use (default: 3, i.e., all features).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    PCAClassificationResult
        Classification results including labels, PCA components, and statistics.
    """
    n_points = len(slope_deg)
    feature_names = ["slope_deg", "roughness_small", "roughness_large"]

    # Build feature matrix, handling NaN values
    features = np.column_stack([slope_deg, roughness_small, roughness_large])

    # Identify valid points (no NaN in any feature)
    valid_mask = ~np.any(np.isnan(features), axis=1)
    n_valid = valid_mask.sum()

    if n_valid < max_clusters:
        raise ValueError(
            f"Too few valid points ({n_valid}) for clustering. "
            f"Need at least {max_clusters} points."
        )

    features_valid = features[valid_mask]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)

    # Apply PCA
    n_components = min(n_components, features_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(features_scaled)

    # Determine optimal number of clusters using Calinski-Harabasz (O(N))
    if n_clusters is None:
        n_clusters = _find_optimal_clusters(
            pca_features,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_state=random_state,
        )

    # Run MiniBatchKMeans clustering (O(N) instead of O(N²))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=3,  # Reduced from 10 - sufficient for most cases
        batch_size=min(1024, n_valid),
    )
    labels_valid = kmeans.fit_predict(pca_features)

    # Calculate silhouette score on small subsample (for reporting only)
    if n_clusters > 1:
        silhouette_avg = _subsample_silhouette_score(
            pca_features, labels_valid, random_state
        )
    else:
        silhouette_avg = 0.0

    # Map labels back to full array (invalid points get label -1)
    labels = np.full(n_points, -1, dtype=np.int32)
    labels[valid_mask] = labels_valid

    # Map PCA components back to full array
    pca_components_full = np.full((n_points, n_components), np.nan, dtype=np.float32)
    pca_components_full[valid_mask] = pca_features

    # Get cluster centers in original feature space
    cluster_centers_scaled = kmeans.cluster_centers_
    # Transform back through PCA and scaler
    cluster_centers_pca = cluster_centers_scaled
    cluster_centers_original = scaler.inverse_transform(
        pca.inverse_transform(cluster_centers_scaled)
    )

    # Calculate cluster statistics
    cluster_stats = _compute_cluster_stats(
        features_valid, labels_valid, n_clusters, feature_names
    )

    return PCAClassificationResult(
        labels=labels,
        n_clusters=n_clusters,
        pca_components=pca_components_full,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cluster_centers=cluster_centers_original,
        cluster_centers_pca=cluster_centers_pca,
        silhouette_avg=silhouette_avg,
        feature_names=feature_names,
        pca_loadings=pca.components_,
        cluster_stats=cluster_stats,
    )


def _subsample_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> float:
    """
    Calculate silhouette score on a small subsample for reporting.

    This is only used to provide a quality metric to users, not for
    cluster selection (which uses Calinski-Harabasz).

    Parameters
    ----------
    features : np.ndarray
        (N, n_features) feature matrix.
    labels : np.ndarray
        (N,) cluster labels.
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    float
        Silhouette score computed on subsample.
    """
    n_points = len(features)
    if n_points <= SILHOUETTE_SAMPLE_SIZE:
        return silhouette_score(features, labels)

    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n_points, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
    return silhouette_score(features[sample_idx], labels[sample_idx])


def _find_optimal_clusters(
    features: np.ndarray,
    min_clusters: int = 3,
    max_clusters: int = 12,
    random_state: int = 42,
) -> int:
    """
    Find optimal number of clusters using Calinski-Harabasz index.

    Calinski-Harabasz is O(N) vs silhouette's O(N²), making it
    suitable for large point clouds.

    Parameters
    ----------
    features : np.ndarray
        (N, n_features) feature matrix.
    min_clusters : int
        Minimum number of clusters to try.
    max_clusters : int
        Maximum number of clusters to try.
    random_state : int
        Random seed.

    Returns
    -------
    int
        Optimal number of clusters.
    """
    n_points = len(features)

    # Limit max_clusters based on dataset size
    max_clusters = min(max_clusters, n_points // 10)
    max_clusters = max(max_clusters, min_clusters + 1)

    best_score = -1
    best_k = min_clusters

    for k in range(min_clusters, max_clusters + 1):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=3,
            batch_size=min(1024, n_points),
        )
        labels = kmeans.fit_predict(features)

        # Calinski-Harabasz: O(N) - uses centroids, not pairwise distances
        score = calinski_harabasz_score(features, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _compute_cluster_stats(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    feature_names: List[str],
) -> Dict:
    """
    Compute statistics for each cluster.

    Parameters
    ----------
    features : np.ndarray
        (N, n_features) feature matrix.
    labels : np.ndarray
        (N,) cluster labels.
    n_clusters : int
        Number of clusters.
    feature_names : List[str]
        Names of features.

    Returns
    -------
    Dict
        Statistics for each cluster.
    """
    stats = {}

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_features = features[mask]
        n_points = mask.sum()

        cluster_stats = {
            "n_points": int(n_points),
            "percentage": 100.0 * n_points / len(labels),
        }

        for i, name in enumerate(feature_names):
            values = cluster_features[:, i]
            cluster_stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

        stats[cluster_id] = cluster_stats

    return stats


def compare_with_rai(
    pca_labels: np.ndarray,
    rai_labels: np.ndarray,
) -> Dict:
    """
    Compare PCA-based clusters with RAI classification.

    Parameters
    ----------
    pca_labels : np.ndarray
        (N,) PCA cluster labels (-1 for invalid).
    rai_labels : np.ndarray
        (N,) RAI class labels (0-7).

    Returns
    -------
    Dict
        Comparison statistics including confusion matrix and agreement metrics.
    """
    # Only compare valid points
    valid_mask = pca_labels >= 0
    pca_valid = pca_labels[valid_mask]
    rai_valid = rai_labels[valid_mask]

    n_pca_clusters = pca_valid.max() + 1
    n_rai_classes = 8  # RAI has 8 classes (0-7)

    # Build confusion matrix: rows = PCA clusters, cols = RAI classes
    confusion = np.zeros((n_pca_clusters, n_rai_classes), dtype=np.int32)
    for pca_label, rai_label in zip(pca_valid, rai_valid):
        confusion[pca_label, rai_label] += 1

    # For each PCA cluster, find dominant RAI class
    cluster_to_rai = {}
    for cluster_id in range(n_pca_clusters):
        row = confusion[cluster_id]
        dominant_rai = int(np.argmax(row))
        dominant_count = int(row[dominant_rai])
        total_count = int(row.sum())
        purity = dominant_count / total_count if total_count > 0 else 0

        cluster_to_rai[cluster_id] = {
            "dominant_rai_class": dominant_rai,
            "dominant_count": dominant_count,
            "total_count": total_count,
            "purity": purity,
        }

    # Calculate overall purity (weighted average)
    total_dominant = sum(c["dominant_count"] for c in cluster_to_rai.values())
    overall_purity = total_dominant / len(pca_valid) if len(pca_valid) > 0 else 0

    return {
        "confusion_matrix": confusion,
        "cluster_to_rai_mapping": cluster_to_rai,
        "overall_purity": overall_purity,
        "n_pca_clusters": n_pca_clusters,
        "n_rai_classes": n_rai_classes,
        "n_valid_points": int(valid_mask.sum()),
    }


def get_cluster_interpretation(
    result: PCAClassificationResult,
) -> Dict[int, str]:
    """
    Generate human-readable interpretations of each cluster.

    Based on the cluster center values, provide a description of what
    each cluster represents in terms of surface characteristics.

    Parameters
    ----------
    result : PCAClassificationResult
        Classification result.

    Returns
    -------
    Dict[int, str]
        Interpretation string for each cluster.
    """
    interpretations = {}

    for cluster_id in range(result.n_clusters):
        stats = result.cluster_stats[cluster_id]
        slope_mean = stats["slope_deg"]["mean"]
        r_small_mean = stats["roughness_small"]["mean"]
        r_large_mean = stats["roughness_large"]["mean"]

        # Build interpretation based on feature values
        parts = []

        # Slope interpretation
        if slope_mean > 150:
            parts.append("cantilevered overhang")
        elif slope_mean > 90:
            parts.append("overhang")
        elif slope_mean > 70:
            parts.append("very steep")
        elif slope_mean > 45:
            parts.append("steep")
        elif slope_mean > 20:
            parts.append("moderate slope")
        else:
            parts.append("gentle slope")

        # Roughness interpretation
        if r_small_mean > 18:
            parts.append("very rough (small-scale)")
        elif r_small_mean > 11:
            parts.append("rough (small-scale)")
        elif r_small_mean > 6:
            parts.append("moderate roughness")
        else:
            parts.append("smooth")

        if r_large_mean > 12:
            parts.append("fragmented (large-scale)")

        interpretations[cluster_id] = ", ".join(parts)

    return interpretations
