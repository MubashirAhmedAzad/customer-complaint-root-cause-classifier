"""
Complaint Clustering Module

Performs unsupervised clustering on complaint embeddings using:
  1. UMAP for dimensionality reduction (preserving local manifold structure)
  2. HDBSCAN for density-based clustering (handles arbitrary shapes, identifies noise)
  3. Silhouette scoring and cluster quality diagnostics
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional

from config.settings import RANDOM_SEED
from src.utils.logger import get_logger
from src.utils.helpers import timer_decorator

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# UMAP Dimensionality Reduction
# ---------------------------------------------------------------------------

@timer_decorator
def reduce_dimensions_umap(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Reduce dense embeddings into lower-dimensional space for HDBSCAN density clustering.

    Parameters
    ----------
    embeddings : np.ndarray
        Dense embedding vectors (N, 384).
    n_components : int
        Target dimension (5D is recommended for HDBSCAN to avoid curse of dimensionality).
    n_neighbors : int
        Local neighborhood size.
    min_dist : float
        Minimum distance between points in low-dim space (0.0 packs dense clusters).
    metric : str
        Distance metric ('cosine' is optimal for sentence embeddings).

    Returns
    -------
    np.ndarray
        Reduced array (N, n_components).
    """
    import umap

    logger.info(
        "Reducing embeddings from %dD to %dD via UMAP (n_neighbors=%d, metric='%s')...",
        embeddings.shape[1],
        n_components,
        n_neighbors,
        metric,
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1,
    )

    reduced = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction complete. Output shape: %s", reduced.shape)
    return reduced


@timer_decorator
def compute_2d_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Compute 2D coordinates for interactive scatter plot visualization."""
    import umap

    logger.info("Computing 2D UMAP projection for visualization...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1,
    )
    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d


# ---------------------------------------------------------------------------
# HDBSCAN Density Clustering
# ---------------------------------------------------------------------------

@timer_decorator
def cluster_with_hdbscan(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int = 10,
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster low-dimensional vectors using density-based HDBSCAN.

    Parameters
    ----------
    reduced_embeddings : np.ndarray
        Vectors from UMAP (e.g. 5D).
    min_cluster_size : int
        Smallest group size to be considered an autonomous cluster.
    min_samples : int
        Conservative clustering parameter; higher values lead to more points labeled as noise.
    metric : str
        Distance metric ('euclidean' on UMAP output).

    Returns
    -------
    np.ndarray
        Cluster assignments (-1 indicates unclustered noise).
    """
    import hdbscan

    logger.info(
        "Clustering with HDBSCAN (min_cluster_size=%d, min_samples=%d)...",
        min_cluster_size,
        min_samples,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True,
    )

    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    n_clusters = len(set(cluster_labels) - {-1})
    n_noise = (cluster_labels == -1).sum()
    noise_pct = (n_noise / len(cluster_labels)) * 100

    logger.info(
        "HDBSCAN discovered %d distinct clusters (%d noise outliers, %.1f%%)",
        n_clusters,
        n_noise,
        noise_pct,
    )
    return cluster_labels


# ---------------------------------------------------------------------------
# Clustering Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_clustering_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, Any]:
    """Calculate diagnostic metrics for clustering quality."""
    from sklearn.metrics import silhouette_score

    n_samples = len(cluster_labels)
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int((cluster_labels == -1).sum())
    noise_ratio = float(n_noise / n_samples)

    metrics: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "noise_ratio": noise_ratio,
        "silhouette_score": None,
    }

    # Silhouette score on non-noise points if at least 2 clusters exist
    non_noise_mask = cluster_labels != -1
    if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
        try:
            # Sample for speed if dataset is large
            eval_size = min(5000, non_noise_mask.sum())
            indices = np.random.choice(
                np.where(non_noise_mask)[0], size=eval_size, replace=False
            )
            score = float(
                silhouette_score(
                    embeddings[indices], cluster_labels[indices], metric="cosine"
                )
            )
            metrics["silhouette_score"] = score
            logger.info("Non-noise Silhouette score (cosine, N=%d): %.4f", eval_size, score)
        except Exception as e:
            logger.warning("Could not calculate silhouette score: %s", e)

    return metrics


# ---------------------------------------------------------------------------
# Full End-to-End Pipeline
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    min_cluster_size: int = 30,
    min_samples: int = 10,
    n_umap_components: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """End-to-end clustering pipeline: UMAP 5D -> HDBSCAN -> UMAP 2D projection."""
    df_clustered = df.copy()

    # 1. 5D UMAP for clustering
    reduced_5d = reduce_dimensions_umap(
        embeddings, n_components=n_umap_components, min_dist=0.0
    )

    # 2. HDBSCAN clustering
    cluster_labels = cluster_with_hdbscan(
        reduced_5d, min_cluster_size=min_cluster_size, min_samples=min_samples
    )

    # 3. 2D UMAP for plotting
    coords_2d = compute_2d_projection(embeddings, min_dist=0.1)

    # 4. Attach columns to DataFrame
    df_clustered["cluster_id"] = cluster_labels
    df_clustered["umap_x"] = coords_2d[:, 0]
    df_clustered["umap_y"] = coords_2d[:, 1]

    # 5. Metrics
    metrics = compute_clustering_metrics(embeddings, cluster_labels)

    return df_clustered, metrics

