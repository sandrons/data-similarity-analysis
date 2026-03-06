"""Tests for the top-level src.compare() convenience function."""
import numpy as np
import pytest

from src import compare, SimilarityAnalyzer, AdvancedSimilarityAnalyzer, DataLoader, Visualizer


RNG = np.random.default_rng(99)
DATA_A = RNG.normal(size=(30, 3))
DATA_B = RNG.normal(size=(30, 3))
DATA_DIFF_SIZE = RNG.normal(size=(20, 3))


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:
    def test_returns_dict(self):
        result = compare(DATA_A, DATA_B)
        assert isinstance(result, dict)

    def test_contains_basic_metric_keys(self):
        result = compare(DATA_A, DATA_B)
        assert "euclidean_distance" in result
        assert "cosine_similarity" in result

    def test_contains_advanced_metric_keys(self):
        result = compare(DATA_A, DATA_B, n_topics=3, n_clusters=3)
        assert "pca_embedding_similarity" in result
        assert "kernel_mmd_similarity" in result
        assert "lda_topic_similarity" in result

    def test_different_size_basic_metrics_skipped_gracefully(self):
        result = compare(DATA_A, DATA_DIFF_SIZE, n_topics=3, n_clusters=3)
        # Basic metrics should be skipped (not raise), advanced should still work
        assert result.get("basic_metrics") is None
        assert "basic_metrics_error" in result
        assert "pca_embedding_similarity" in result

    def test_identical_data_cosine_near_one(self):
        result = compare(DATA_A, DATA_A)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-9)

    def test_identical_data_euclidean_zero(self):
        result = compare(DATA_A, DATA_A)
        assert result["euclidean_distance"] == pytest.approx(0.0, abs=1e-9)

    def test_1d_input_accepted(self):
        v = RNG.normal(size=15)
        result = compare(v, v)
        assert isinstance(result, dict)

    def test_n_clusters_parameter_forwarded(self):
        # Just verify it runs without error with a custom value
        result = compare(DATA_A, DATA_B, n_clusters=2, n_topics=2)
        assert "cluster_cosine_similarity" in result


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_similarity_analyzer_importable(self):
        assert SimilarityAnalyzer is not None

    def test_advanced_analyzer_importable(self):
        assert AdvancedSimilarityAnalyzer is not None

    def test_data_loader_importable(self):
        assert DataLoader is not None

    def test_visualizer_importable(self):
        assert Visualizer is not None

    def test_compare_is_callable(self):
        assert callable(compare)
