"""
Advanced similarity metrics using machine learning techniques.

Includes embedding-based (PCA/SVD/kernel), LDA topic modeling,
and clustering-based (KMeans, DBSCAN) approaches for comparing
numerical datasets.

Caching
-------
Instantiate ``AdvancedSimilarityAnalyzer`` and call ``fit(reference_data)``
once to pre-fit PCA, SVD, LDA, and KMeans on a reference dataset.
Subsequent calls to any metric method will reuse the cached models instead
of re-fitting on the combined data, which is significantly faster when
comparing many query datasets against the same reference.
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

    When comparing many datasets against a single reference, call
    ``fit(reference_data)`` first to cache the expensive model fits.
    """

    def __init__(self):
        self._cache: Dict = {}

    # ------------------------------------------------------------------
    # Reference fitting (caching)
    # ------------------------------------------------------------------

    def fit(
        self,
        reference_data: np.ndarray,
        n_components: int = 2,
        n_topics: int = 5,
        n_clusters: int = 3,
    ) -> "AdvancedSimilarityAnalyzer":
        """
        Fit and cache PCA, SVD, LDA, and KMeans on *reference_data*.

        After calling this method, all metric methods will reuse the cached
        models rather than re-fitting on the combined (reference + query) data.
        This is the main speed-up when comparing many queries against one
        reference.

        Args:
            reference_data: The reference dataset (n_samples x n_features)
            n_components: PCA / SVD embedding dimensions to retain
            n_topics: Number of LDA latent topics
            n_clusters: Number of KMeans clusters

        Returns:
            self  (allows method chaining)
        """
        data = np.atleast_2d(np.asarray(reference_data, dtype=float))
        n, d = data.shape

        # PCA
        k_pca = min(n_components, d, n)
        pca = PCA(n_components=k_pca)
        pca.fit(data)
        self._cache["pca"] = pca

        # SVD
        k_svd = min(n_components, d, n - 1)
        svd = TruncatedSVD(n_components=k_svd, random_state=42)
        svd.fit(data)
        self._cache["svd"] = svd

        # LDA (needs non-negative values)
        shift = float(data.min())
        lda_data = data - shift if shift < 0 else data
        k_lda = min(n_topics, n, d)
        lda = LatentDirichletAllocation(
            n_components=k_lda, random_state=42, max_iter=20
        )
        lda.fit(lda_data)
        self._cache["lda"] = lda
        self._cache["lda_shift"] = shift if shift < 0 else 0.0

        # KMeans
        k_km = min(n_clusters, n)
        kmeans = KMeans(n_clusters=k_km, random_state=42, n_init=10)
        kmeans.fit(data)
        self._cache["kmeans"] = kmeans
        self._cache["n_clusters"] = k_km

        return self

    # ------------------------------------------------------------------
    # Embedding-based similarity
    # ------------------------------------------------------------------

    def pca_embedding_similarity(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_components: int = 2,
    ) -> float:
        """
        Project both datasets into a PCA embedding space and compute
        cosine similarity between their mean embeddings.

        Uses a cached PCA model (fitted on reference data) when available;
        otherwise fits PCA on the combined data.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            n_components: Number of principal components (ignored if cached)

        Returns:
            Cosine similarity in PCA space, range [-1, 1]
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        if "pca" in self._cache:
            pca = self._cache["pca"]
        else:
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

    def svd_embedding_similarity(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_components: int = 2,
    ) -> float:
        """
        Use TruncatedSVD to embed datasets into a latent space and compare
        their mean embeddings via cosine similarity.

        Uses a cached SVD model (fitted on reference data) when available;
        otherwise fits SVD on the combined data.

        Args:
            data1: First dataset
            data2: Second dataset
            n_components: Number of SVD components (ignored if cached)

        Returns:
            Cosine similarity in SVD space, range [-1, 1]
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        if "svd" in self._cache:
            svd = self._cache["svd"]
        else:
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

    def lda_topic_similarity(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_topics: int = 5,
        random_state: int = 42,
    ) -> float:
        """
        Fit a Latent Dirichlet Allocation model on the combined data (treating
        features as pseudo-word frequencies), infer topic distributions for
        each dataset, and return 1 - Jensen-Shannon divergence between them.

        Uses a cached LDA model (fitted on reference data) when available;
        otherwise fits LDA on the combined data.

        Negative values are shifted to zero before fitting so that any numeric
        data can be used.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            n_topics: Number of latent topics (ignored if cached)
            random_state: Reproducibility seed (ignored if cached)

        Returns:
            Topic similarity in [0, 1] (1 = identical topic mixture)
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        if "lda" in self._cache:
            lda = self._cache["lda"]
            # Shift enough to make both arrays non-negative (may differ from the
            # cached shift if query data is more negative than the reference)
            shift = min(data1.min(), data2.min())
            d1 = data1 - shift if shift < 0 else data1
            d2 = data2 - shift if shift < 0 else data2
        else:
            shift = min(data1.min(), data2.min())
            if shift < 0:
                data1 = data1 - shift
                data2 = data2 - shift
            d1, d2 = data1, data2

            combined = np.vstack([d1, d2])
            k = min(n_topics, combined.shape[0], combined.shape[1])
            lda = LatentDirichletAllocation(
                n_components=k, random_state=random_state, max_iter=20
            )
            lda.fit(combined)

        topic_dist1 = lda.transform(d1).mean(axis=0)
        topic_dist2 = lda.transform(d2).mean(axis=0)

        m = 0.5 * (topic_dist1 + topic_dist2)
        js_div = 0.5 * entropy(topic_dist1, m) + 0.5 * entropy(topic_dist2, m)
        return float(1.0 - js_div / np.log(2))

    # ------------------------------------------------------------------
    # Clustering-based similarity
    # ------------------------------------------------------------------

    def kmeans_cluster_similarity(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Assign points from both datasets to clusters and compare their
        cluster-assignment distributions.

        Uses a cached KMeans model (fitted on reference data) when available;
        otherwise fits KMeans on the combined data.

        Returned metrics:
        - ``cluster_cosine_similarity``: cosine similarity of the two cluster
          frequency vectors
        - ``cluster_overlap``: sum of per-cluster minimum proportions
        - ``cluster_nmi``: Normalized Mutual Information between dataset
          membership and cluster assignment

        Args:
            data1: First dataset
            data2: Second dataset
            n_clusters: Number of KMeans clusters (ignored if cached)
            random_state: Reproducibility seed (ignored if cached)

        Returns:
            Dict with three float metrics
        """
        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        if "kmeans" in self._cache:
            kmeans = self._cache["kmeans"]
            k = self._cache["n_clusters"]
            labels1 = kmeans.predict(data1)
            labels2 = kmeans.predict(data2)
        else:
            combined = np.vstack([data1, data2])
            k = min(n_clusters, combined.shape[0])
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            all_labels = kmeans.fit_predict(combined)
            n1 = len(data1)
            labels1, labels2 = all_labels[:n1], all_labels[n1:]

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
            [np.zeros(len(data1), dtype=int), np.ones(len(data2), dtype=int)]
        )
        nmi = float(normalized_mutual_info_score(group_labels, np.concatenate([labels1, labels2])))

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

    def adjusted_rand_similarity(
        self,
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
        km1 = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km2 = KMeans(n_clusters=k, random_state=random_state, n_init=10)

        labels1 = km1.fit_predict(data1)
        labels2 = km2.fit_predict(data2)

        return float(adjusted_rand_score(labels1, labels2))

    # ------------------------------------------------------------------
    # Aggregation helper
    # ------------------------------------------------------------------

    def compare_all(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_components: int = 2,
        n_topics: int = 5,
        n_clusters: int = 3,
    ) -> Dict[str, object]:
        """
        Run all advanced similarity metrics and return results as a flat dict.

        Cached models (from a prior ``fit()`` call) are reused automatically,
        making repeated calls against the same reference dataset fast.

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
        results: Dict[str, object] = {}

        scalar_metrics = [
            (
                "pca_embedding_similarity",
                lambda: self.pca_embedding_similarity(data1, data2, n_components),
            ),
            (
                "svd_embedding_similarity",
                lambda: self.svd_embedding_similarity(data1, data2, n_components),
            ),
            (
                "kernel_mmd_similarity",
                lambda: self.kernel_mmd_similarity(data1, data2),
            ),
            (
                "lda_topic_similarity",
                lambda: self.lda_topic_similarity(data1, data2, n_topics),
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
                lambda: self.kmeans_cluster_similarity(data1, data2, n_clusters),
            ),
            (
                "dbscan",
                lambda: self.dbscan_structure_similarity(data1, data2),
            ),
        ]

        for name, fn in dict_metrics:
            try:
                results.update(fn())
            except Exception as exc:
                results[f"{name}_error"] = str(exc)

        return results
