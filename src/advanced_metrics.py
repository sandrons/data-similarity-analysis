"""
Advanced similarity metrics using machine learning techniques.

Includes embedding-based (PCA/SVD/kernel), LDA topic modeling,
and clustering-based (KMeans, DBSCAN) approaches for comparing
numerical datasets.
"""
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from typing import Dict, Optional


class AdvancedSimilarityAnalyzer:
    """
    Advanced similarity metrics using embedding, topic modeling, and clustering.

    Designed for 2D numerical datasets of shape (n_samples, n_features).
    All methods accept 1D inputs and promote them to 2D automatically.
    """

    # ------------------------------------------------------------------
    # Embedding-based similarity
    # ------------------------------------------------------------------

    @staticmethod
    def pca_embedding_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        n_components: int = 2,
    ) -> float:
        """
        Project both datasets into a shared PCA embedding space and compute
        cosine similarity between their mean embeddings.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            n_components: Number of principal components to retain

        Returns:
            Cosine similarity in PCA space, range [-1, 1]
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        combined = np.vstack([data1, data2])
        k = min(n_components, combined.shape[1], combined.shape[0])

        pca = PCA(n_components=k)
        pca.fit(combined)

        emb1 = pca.transform(data1).mean(axis=0)
        emb2 = pca.transform(data2).mean(axis=0)

        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def svd_embedding_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        n_components: int = 2,
    ) -> float:
        """
        Use TruncatedSVD to embed datasets into a latent space and compare
        their mean embeddings via cosine similarity.

        Args:
            data1: First dataset
            data2: Second dataset
            n_components: Number of SVD components

        Returns:
            Cosine similarity in SVD space, range [-1, 1]
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        combined = np.vstack([data1, data2])
        k = min(n_components, combined.shape[1], combined.shape[0] - 1)

        svd = TruncatedSVD(n_components=k, random_state=42)
        svd.fit(combined)

        emb1 = svd.transform(data1).mean(axis=0)
        emb2 = svd.transform(data2).mean(axis=0)

        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def kernel_mmd_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        gamma: Optional[float] = None,
    ) -> float:
        """
        Compute RBF-kernel Maximum Mean Discrepancy (MMD) between two datasets
        and convert it to a similarity score via exp(-MMD²).

        A score close to 1 means the distributions are very similar; close to 0
        means they are very different.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            gamma: RBF kernel bandwidth; defaults to 1/n_features

        Returns:
            Kernel MMD similarity in [0, 1]
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        if data1.shape[1] != data2.shape[1]:
            raise ValueError("data1 and data2 must have the same number of features")

        if gamma is None:
            gamma = 1.0 / data1.shape[1]

        k11 = rbf_kernel(data1, data1, gamma=gamma).mean()
        k22 = rbf_kernel(data2, data2, gamma=gamma).mean()
        k12 = rbf_kernel(data1, data2, gamma=gamma).mean()

        mmd_sq = k11 + k22 - 2 * k12
        return float(np.exp(-mmd_sq))

    # ------------------------------------------------------------------
    # LDA topic-based similarity
    # ------------------------------------------------------------------

    @staticmethod
    def lda_topic_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        n_topics: int = 5,
        random_state: int = 42,
    ) -> float:
        """
        Fit a Latent Dirichlet Allocation model on the combined data (treating
        features as pseudo-word frequencies), infer topic distributions for
        each dataset, and return 1 - Jensen-Shannon divergence between them.

        Negative values are shifted to zero before fitting so that any numeric
        data can be used.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            n_topics: Number of latent topics
            random_state: Reproducibility seed

        Returns:
            Topic similarity in [0, 1] (1 = identical topic mixture)
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        # LDA requires non-negative values
        shift = min(data1.min(), data2.min())
        if shift < 0:
            data1 = data1 - shift
            data2 = data2 - shift

        combined = np.vstack([data1, data2])
        k = min(n_topics, combined.shape[0], combined.shape[1])

        lda = LatentDirichletAllocation(
            n_components=k, random_state=random_state, max_iter=20
        )
        lda.fit(combined)

        topic_dist1 = lda.transform(data1).mean(axis=0)
        topic_dist2 = lda.transform(data2).mean(axis=0)

        # Jensen-Shannon divergence (symmetric, bounded in [0, log 2])
        m = 0.5 * (topic_dist1 + topic_dist2)
        js_div = 0.5 * entropy(topic_dist1, m) + 0.5 * entropy(topic_dist2, m)
        return float(1.0 - js_div / np.log(2))

    # ------------------------------------------------------------------
    # Clustering-based similarity
    # ------------------------------------------------------------------

    @staticmethod
    def kmeans_cluster_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Fit KMeans on the combined data then compare cluster-assignment
        distributions of the two datasets.

        Returned metrics:
        - ``cluster_cosine_similarity``: cosine similarity of the two cluster
          frequency vectors (how similarly distributed across clusters)
        - ``cluster_overlap``: sum of per-cluster minimum proportions
          (intersection-over-union proxy)
        - ``cluster_nmi``: Normalized Mutual Information between dataset
          membership and cluster assignment

        Args:
            data1: First dataset
            data2: Second dataset
            n_clusters: Number of KMeans clusters
            random_state: Reproducibility seed

        Returns:
            Dict with three float metrics
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        combined = np.vstack([data1, data2])
        k = min(n_clusters, combined.shape[0])

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(combined)

        n1 = len(data1)
        labels1, labels2 = labels[:n1], labels[n1:]

        def cluster_dist(lbl):
            counts = np.bincount(lbl, minlength=k).astype(float)
            return counts / counts.sum()

        dist1, dist2 = cluster_dist(labels1), cluster_dist(labels2)

        norm1, norm2 = np.linalg.norm(dist1), np.linalg.norm(dist2)
        cos_sim = (
            float(np.dot(dist1, dist2) / (norm1 * norm2))
            if norm1 > 0 and norm2 > 0
            else 0.0
        )

        overlap = float(np.minimum(dist1, dist2).sum())

        group_labels = np.concatenate(
            [np.zeros(n1, dtype=int), np.ones(len(data2), dtype=int)]
        )
        nmi = float(normalized_mutual_info_score(group_labels, labels))

        return {
            "cluster_cosine_similarity": cos_sim,
            "cluster_overlap": overlap,
            "cluster_nmi": nmi,
        }

    @staticmethod
    def dbscan_structure_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 2,
    ) -> Dict[str, float]:
        """
        Run DBSCAN independently on each (standardized) dataset and compare
        the resulting cluster structures.

        Returned metrics:
        - ``cluster_count_similarity``: 1 - |nc1-nc2| / max(nc1, nc2)
        - ``noise_ratio_similarity``: 1 - |nr1-nr2|
        - ``n_clusters_data1 / 2``: raw cluster counts
        - ``noise_ratio_data1 / 2``: fraction of noise points

        Args:
            data1: First dataset
            data2: Second dataset
            eps: DBSCAN neighborhood radius (applied after standardization)
            min_samples: Minimum points to form a core point

        Returns:
            Dict with structural similarity scores and raw DBSCAN statistics
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        s1 = StandardScaler().fit_transform(data1)
        s2 = StandardScaler().fit_transform(data2)

        labels1 = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(s1)
        labels2 = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(s2)

        def stats(labels):
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = float((labels == -1).sum() / len(labels))
            return n_clusters, noise_ratio

        nc1, nr1 = stats(labels1)
        nc2, nr2 = stats(labels2)

        max_clusters = max(nc1, nc2, 1)
        return {
            "cluster_count_similarity": float(1.0 - abs(nc1 - nc2) / max_clusters),
            "noise_ratio_similarity": float(1.0 - abs(nr1 - nr2)),
            "n_clusters_data1": nc1,
            "n_clusters_data2": nc2,
            "noise_ratio_data1": nr1,
            "noise_ratio_data2": nr2,
        }

    @staticmethod
    def adjusted_rand_similarity(
        data1: np.ndarray,
        data2: np.ndarray,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> float:
        """
        Cluster each dataset independently with KMeans and compute the
        Adjusted Rand Index (ARI) between the two label assignments.

        Both datasets must contain the same number of samples so that labels
        are paired row-by-row.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (n_samples x n_features)
            n_clusters: Number of KMeans clusters
            random_state: Reproducibility seed

        Returns:
            ARI in [-1, 1] (1 = perfect agreement, 0 = random)
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        if len(data1) != len(data2):
            raise ValueError(
                "data1 and data2 must have the same number of samples for ARI"
            )

        k = min(n_clusters, len(data1))
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)

        labels1 = km.fit_predict(data1)
        labels2 = km.fit_predict(data2)

        return float(adjusted_rand_score(labels1, labels2))

    # ------------------------------------------------------------------
    # Aggregation helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare_all(
        data1: np.ndarray,
        data2: np.ndarray,
        n_components: int = 2,
        n_topics: int = 5,
        n_clusters: int = 3,
    ) -> Dict[str, object]:
        """
        Run all advanced similarity metrics and return results as a flat dict.

        Args:
            data1: First dataset  (2D numerical array)
            data2: Second dataset (2D numerical array)
            n_components: PCA / SVD embedding dimensions
            n_topics: Number of LDA topics
            n_clusters: Number of KMeans clusters

        Returns:
            Dict mapping metric names to values (None on failure, with
            a companion ``<name>_error`` key describing the exception)
        """
        analyzer = AdvancedSimilarityAnalyzer()
        results: Dict[str, object] = {}

        scalar_metrics = [
            (
                "pca_embedding_similarity",
                lambda: analyzer.pca_embedding_similarity(data1, data2, n_components),
            ),
            (
                "svd_embedding_similarity",
                lambda: analyzer.svd_embedding_similarity(data1, data2, n_components),
            ),
            (
                "kernel_mmd_similarity",
                lambda: analyzer.kernel_mmd_similarity(data1, data2),
            ),
            (
                "lda_topic_similarity",
                lambda: analyzer.lda_topic_similarity(data1, data2, n_topics),
            ),
        ]

        for name, fn in scalar_metrics:
            try:
                results[name] = fn()
            except Exception as exc:
                results[name] = None
                results[f"{name}_error"] = str(exc)

        dict_metrics = [
            (
                "kmeans",
                lambda: analyzer.kmeans_cluster_similarity(data1, data2, n_clusters),
            ),
            (
                "dbscan",
                lambda: analyzer.dbscan_structure_similarity(data1, data2),
            ),
        ]

        for name, fn in dict_metrics:
            try:
                results.update(fn())
            except Exception as exc:
                results[f"{name}_error"] = str(exc)

        return results
