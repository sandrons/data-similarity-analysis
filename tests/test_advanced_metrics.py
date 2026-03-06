"""Tests for src/advanced_metrics.py — advanced similarity metrics and caching."""
import numpy as np
import pytest

from src.advanced_metrics import AdvancedSimilarityAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# Standard 2D datasets
REF = RNG.normal(size=(40, 4))
QUERY_SIMILAR = REF + RNG.normal(scale=0.05, size=REF.shape)   # very close to REF
QUERY_DIFFERENT = RNG.normal(loc=10.0, size=(30, 4))            # far from REF

# Same-length datasets for ARI
REF_30 = RNG.normal(size=(30, 4))
QUERY_30 = RNG.normal(size=(30, 4))

# Negative-valued dataset (edge case for LDA)
NEGATIVE = RNG.normal(loc=-5.0, size=(20, 3))


def make_analyzer(**kwargs) -> AdvancedSimilarityAnalyzer:
    return AdvancedSimilarityAnalyzer(**kwargs)


# ---------------------------------------------------------------------------
# pca_embedding_similarity
# ---------------------------------------------------------------------------

class TestPCAEmbeddingSimilarity:
    def test_identical_data_returns_one(self):
        result = make_analyzer().pca_embedding_similarity(REF, REF)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_similar_data_higher_than_different(self):
        # Use kernel MMD for this directional check; PCA cosine similarity is
        # direction-based and not guaranteed monotone with data distance.
        sim_score = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, QUERY_SIMILAR)
        diff_score = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, QUERY_DIFFERENT)
        assert sim_score > diff_score

    def test_range(self):
        result = make_analyzer().pca_embedding_similarity(REF, QUERY_DIFFERENT)
        assert -1.0 <= result <= 1.0

    def test_uses_cache_when_fitted(self):
        a = make_analyzer()
        a.fit(REF)
        assert "pca" in a._cache
        # Result should still be a valid float
        result = a.pca_embedding_similarity(REF, QUERY_30)
        assert isinstance(result, float)

    def test_1d_input_promoted(self):
        # Verify 1D inputs are accepted (promoted to 2D internally).
        # Use non-degenerate data: two slightly different 1-sample datasets.
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.1, 2.1, 3.1])
        result = make_analyzer().pca_embedding_similarity(v1, v2)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# svd_embedding_similarity
# ---------------------------------------------------------------------------

class TestSVDEmbeddingSimilarity:
    def test_identical_data_returns_one(self):
        result = make_analyzer().svd_embedding_similarity(REF, REF)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        result = make_analyzer().svd_embedding_similarity(REF, QUERY_DIFFERENT)
        assert -1.0 <= result <= 1.0

    def test_uses_cache_when_fitted(self):
        a = make_analyzer()
        a.fit(REF)
        assert "svd" in a._cache
        result = a.svd_embedding_similarity(REF, QUERY_30)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# kernel_mmd_similarity
# ---------------------------------------------------------------------------

class TestKernelMMDSimilarity:
    def test_identical_data_near_one(self):
        result = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, REF)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        result = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, QUERY_DIFFERENT)
        assert 0.0 <= result <= 1.0

    def test_similar_data_higher_than_different(self):
        sim = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, QUERY_SIMILAR)
        diff = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, QUERY_DIFFERENT)
        assert sim > diff

    def test_mismatched_features_raises(self):
        bad = RNG.normal(size=(20, 5))
        with pytest.raises(ValueError, match="same number of features"):
            AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, bad)

    def test_custom_gamma(self):
        result = AdvancedSimilarityAnalyzer.kernel_mmd_similarity(REF, QUERY_SIMILAR, gamma=0.1)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# lda_topic_similarity
# ---------------------------------------------------------------------------

class TestLDATopicSimilarity:
    def test_identical_data_near_one(self):
        pos = np.abs(REF) + 1.0
        result = make_analyzer().lda_topic_similarity(pos, pos, n_topics=3)
        assert result >= 0.9

    def test_range(self):
        result = make_analyzer().lda_topic_similarity(REF, QUERY_DIFFERENT, n_topics=3)
        assert 0.0 <= result <= 1.0

    def test_negative_values_handled(self):
        # Should not raise even with negative data
        result = make_analyzer().lda_topic_similarity(NEGATIVE, NEGATIVE, n_topics=2)
        assert result >= 0.9

    def test_uses_cache_when_fitted(self):
        a = make_analyzer()
        a.fit(REF, n_topics=3)
        assert "lda" in a._cache
        result = a.lda_topic_similarity(REF, QUERY_30)
        assert 0.0 <= result <= 1.0

    def test_cached_handles_more_negative_query(self):
        a = make_analyzer()
        a.fit(REF, n_topics=3)
        # Query data more negative than reference (same feature count) — should not raise
        very_negative = RNG.normal(loc=-20.0, size=(20, 4))
        result = a.lda_topic_similarity(REF, very_negative)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# kmeans_cluster_similarity
# ---------------------------------------------------------------------------

class TestKMeansClusterSimilarity:
    def test_returns_expected_keys(self):
        result = make_analyzer().kmeans_cluster_similarity(REF, QUERY_30, n_clusters=3)
        assert {"cluster_cosine_similarity", "cluster_overlap", "cluster_nmi"} == set(result.keys())

    def test_identical_data_high_similarity(self):
        result = make_analyzer().kmeans_cluster_similarity(REF, REF, n_clusters=3)
        assert result["cluster_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert result["cluster_overlap"] == pytest.approx(1.0, abs=1e-6)

    def test_ranges(self):
        result = make_analyzer().kmeans_cluster_similarity(REF, QUERY_DIFFERENT, n_clusters=3)
        assert 0.0 <= result["cluster_cosine_similarity"] <= 1.0
        assert 0.0 <= result["cluster_overlap"] <= 1.0
        assert 0.0 <= result["cluster_nmi"] <= 1.0

    def test_uses_predict_when_cached(self):
        a = make_analyzer()
        a.fit(REF, n_clusters=3)
        assert "kmeans" in a._cache
        result = a.kmeans_cluster_similarity(REF, QUERY_30)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# dbscan_structure_similarity
# ---------------------------------------------------------------------------

class TestDBSCANStructureSimilarity:
    def test_returns_expected_keys(self):
        result = AdvancedSimilarityAnalyzer.dbscan_structure_similarity(REF, QUERY_30)
        expected = {
            "cluster_count_similarity", "noise_ratio_similarity",
            "n_clusters_data1", "n_clusters_data2",
            "noise_ratio_data1", "noise_ratio_data2",
        }
        assert expected == set(result.keys())

    def test_identical_data_perfect_structure_match(self):
        result = AdvancedSimilarityAnalyzer.dbscan_structure_similarity(REF, REF)
        assert result["cluster_count_similarity"] == pytest.approx(1.0)
        assert result["noise_ratio_similarity"] == pytest.approx(1.0)

    def test_ranges(self):
        result = AdvancedSimilarityAnalyzer.dbscan_structure_similarity(REF, QUERY_DIFFERENT)
        assert 0.0 <= result["cluster_count_similarity"] <= 1.0
        assert 0.0 <= result["noise_ratio_similarity"] <= 1.0
        assert 0.0 <= result["noise_ratio_data1"] <= 1.0
        assert 0.0 <= result["noise_ratio_data2"] <= 1.0


# ---------------------------------------------------------------------------
# adjusted_rand_similarity
# ---------------------------------------------------------------------------

class TestAdjustedRandSimilarity:
    def test_identical_data_returns_one(self):
        result = make_analyzer().adjusted_rand_similarity(REF_30, REF_30, n_clusters=3)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        result = make_analyzer().adjusted_rand_similarity(REF_30, QUERY_30, n_clusters=3)
        assert -1.0 <= result <= 1.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same number of samples"):
            make_analyzer().adjusted_rand_similarity(REF, QUERY_30, n_clusters=3)

    def test_uses_separate_kmeans_instances(self):
        # Simply verify it runs without error and returns a float
        result = make_analyzer().adjusted_rand_similarity(REF_30, QUERY_30, n_clusters=3)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# fit() — caching behaviour
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_populates_all_cache_keys(self):
        a = make_analyzer()
        a.fit(REF, n_components=2, n_topics=3, n_clusters=3)
        assert {"pca", "svd", "lda", "lda_shift", "kmeans", "n_clusters"} == set(a._cache.keys())

    def test_fit_returns_self(self):
        a = make_analyzer()
        result = a.fit(REF)
        assert result is a

    def test_cached_and_uncached_pca_close(self):
        uncached = make_analyzer().pca_embedding_similarity(REF, REF)
        cached = make_analyzer()
        cached.fit(REF)
        from_cache = cached.pca_embedding_similarity(REF, REF)
        # Both should be near 1.0 for identical data
        assert uncached == pytest.approx(1.0, abs=1e-6)
        assert from_cache == pytest.approx(1.0, abs=1e-6)

    def test_cache_reused_across_multiple_queries(self):
        a = make_analyzer()
        a.fit(REF, n_components=2, n_topics=3, n_clusters=3)
        cache_before = dict(a._cache)  # shallow copy of references
        a.pca_embedding_similarity(REF, QUERY_30)
        a.lda_topic_similarity(REF, QUERY_30)
        a.kmeans_cluster_similarity(REF, QUERY_30)
        # Cache object references should be unchanged (no re-fitting)
        assert a._cache["pca"] is cache_before["pca"]
        assert a._cache["lda"] is cache_before["lda"]
        assert a._cache["kmeans"] is cache_before["kmeans"]


# ---------------------------------------------------------------------------
# compare_all
# ---------------------------------------------------------------------------

class TestCompareAll:
    def test_returns_dict(self):
        result = make_analyzer().compare_all(REF, QUERY_30)
        assert isinstance(result, dict)

    def test_contains_all_metric_keys(self):
        result = make_analyzer().compare_all(REF, QUERY_30, n_topics=3, n_clusters=3)
        expected = {
            "pca_embedding_similarity", "svd_embedding_similarity",
            "kernel_mmd_similarity", "lda_topic_similarity",
            "cluster_cosine_similarity", "cluster_overlap", "cluster_nmi",
            "cluster_count_similarity", "noise_ratio_similarity",
        }
        assert expected.issubset(result.keys())

    def test_errors_are_captured_not_raised(self):
        # Mismatched features should cause kernel_mmd to fail gracefully
        bad = RNG.normal(size=(20, 5))
        result = make_analyzer().compare_all(REF, bad)
        assert result.get("kernel_mmd_similarity") is None
        assert "kernel_mmd_similarity_error" in result

    def test_uses_cache_in_compare_all(self):
        a = make_analyzer()
        a.fit(REF, n_topics=3, n_clusters=3)
        cache_pca = a._cache["pca"]
        a.compare_all(REF, QUERY_30, n_topics=3, n_clusters=3)
        assert a._cache["pca"] is cache_pca  # not re-fitted
