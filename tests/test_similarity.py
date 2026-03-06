"""Tests for src/similarity.py — basic similarity metrics."""
import numpy as np
import pytest

from src.similarity import SimilarityAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
A = RNG.normal(size=20)
B = RNG.normal(size=20)
IDENTICAL = A.copy()


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_identical_arrays_returns_zero(self):
        assert SimilarityAnalyzer.euclidean_distance(A, IDENTICAL) == pytest.approx(0.0)

    def test_non_negative(self):
        assert SimilarityAnalyzer.euclidean_distance(A, B) >= 0

    def test_symmetry(self):
        assert SimilarityAnalyzer.euclidean_distance(A, B) == pytest.approx(
            SimilarityAnalyzer.euclidean_distance(B, A)
        )

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert SimilarityAnalyzer.euclidean_distance(a, b) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# manhattan_distance
# ---------------------------------------------------------------------------

class TestManhattanDistance:
    def test_identical_arrays_returns_zero(self):
        assert SimilarityAnalyzer.manhattan_distance(A, IDENTICAL) == pytest.approx(0.0)

    def test_non_negative(self):
        assert SimilarityAnalyzer.manhattan_distance(A, B) >= 0

    def test_symmetry(self):
        assert SimilarityAnalyzer.manhattan_distance(A, B) == pytest.approx(
            SimilarityAnalyzer.manhattan_distance(B, A)
        )

    def test_known_value(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert SimilarityAnalyzer.manhattan_distance(a, b) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert SimilarityAnalyzer.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors_return_minus_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert SimilarityAnalyzer.cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-9)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert SimilarityAnalyzer.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_range(self):
        result = SimilarityAnalyzer.cosine_similarity(A, B)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# pearson_correlation
# ---------------------------------------------------------------------------

class TestPearsonCorrelation:
    def test_identical_arrays_return_one(self):
        corr, _ = SimilarityAnalyzer.pearson_correlation(A, IDENTICAL)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_opposite_arrays_return_minus_one(self):
        corr, _ = SimilarityAnalyzer.pearson_correlation(A, -A)
        assert corr == pytest.approx(-1.0, abs=1e-9)

    def test_returns_tuple_of_floats(self):
        result = SimilarityAnalyzer.pearson_correlation(A, B)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_pvalue_range(self):
        _, pval = SimilarityAnalyzer.pearson_correlation(A, B)
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation:
    def test_identical_arrays_return_one(self):
        corr, _ = SimilarityAnalyzer.spearman_correlation(A, IDENTICAL)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_range(self):
        corr, _ = SimilarityAnalyzer.spearman_correlation(A, B)
        assert -1.0 <= corr <= 1.0

    def test_pvalue_range(self):
        _, pval = SimilarityAnalyzer.spearman_correlation(A, B)
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# chebyshev_distance
# ---------------------------------------------------------------------------

class TestChebyshevDistance:
    def test_identical_arrays_return_zero(self):
        assert SimilarityAnalyzer.chebyshev_distance(A, IDENTICAL) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 3.0])
        assert SimilarityAnalyzer.chebyshev_distance(a, b) == pytest.approx(3.0)

    def test_symmetry(self):
        assert SimilarityAnalyzer.chebyshev_distance(A, B) == pytest.approx(
            SimilarityAnalyzer.chebyshev_distance(B, A)
        )


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_binary_returns_one(self):
        v = np.array([1.0, 0.0, 1.0, 1.0])
        assert SimilarityAnalyzer.jaccard_similarity(v, v) == pytest.approx(1.0)

    def test_disjoint_returns_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert SimilarityAnalyzer.jaccard_similarity(a, b, threshold=0.5) == pytest.approx(0.0)

    def test_range(self):
        result = SimilarityAnalyzer.jaccard_similarity(
            np.abs(A), np.abs(B), threshold=np.median(np.abs(A))
        )
        assert 0.0 <= result <= 1.0

    def test_all_below_threshold_returns_zero(self):
        a = np.array([0.1, 0.2])
        b = np.array([0.3, 0.4])
        assert SimilarityAnalyzer.jaccard_similarity(a, b, threshold=0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# minkowski_distance
# ---------------------------------------------------------------------------

class TestMinkowskiDistance:
    def test_p1_matches_manhattan(self):
        assert SimilarityAnalyzer.minkowski_distance(A, B, p=1) == pytest.approx(
            SimilarityAnalyzer.manhattan_distance(A, B)
        )

    def test_p2_matches_euclidean(self):
        assert SimilarityAnalyzer.minkowski_distance(A, B, p=2) == pytest.approx(
            SimilarityAnalyzer.euclidean_distance(A, B)
        )

    def test_identical_arrays_return_zero(self):
        assert SimilarityAnalyzer.minkowski_distance(A, IDENTICAL, p=3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compare_all
# ---------------------------------------------------------------------------

class TestCompareAll:
    def test_returns_dict(self):
        result = SimilarityAnalyzer.compare_all(A, B)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        result = SimilarityAnalyzer.compare_all(A, B)
        expected = {
            "euclidean_distance", "manhattan_distance", "cosine_similarity",
            "chebyshev_distance", "jaccard_similarity", "hellinger_distance",
            "pearson_correlation", "pearson_pvalue",
            "spearman_correlation", "spearman_pvalue",
        }
        assert expected.issubset(result.keys())

    def test_identical_input_sanity(self):
        pos = np.abs(A) + 1e-6  # hellinger needs positive values
        result = SimilarityAnalyzer.compare_all(pos, pos)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-9)
        assert result["euclidean_distance"] == pytest.approx(0.0, abs=1e-9)
        assert result["pearson_correlation"] == pytest.approx(1.0, abs=1e-9)
