"""
data-similarity-analysis
========================
Basic and advanced similarity metrics for numerical datasets.

Quick start
-----------
::

    from src import compare, SimilarityAnalyzer, AdvancedSimilarityAnalyzer

    # One-liner: runs all basic + advanced metrics
    results = compare(data1, data2)

    # Or use the classes directly for more control
    basic   = SimilarityAnalyzer.compare_all(data1, data2)
    analyzer = AdvancedSimilarityAnalyzer()
    analyzer.fit(reference)          # cache expensive fits once
    advanced = analyzer.compare_all(reference, query)
"""

from .similarity import SimilarityAnalyzer
from .advanced_metrics import AdvancedSimilarityAnalyzer
from .data_loader import DataLoader
from .visualizer import Visualizer

import numpy as np
from typing import Dict


def compare(
    data1: np.ndarray,
    data2: np.ndarray,
    n_components: int = 2,
    n_topics: int = 5,
    n_clusters: int = 3,
) -> Dict[str, object]:
    """
    Compare two datasets using all available basic and advanced metrics.

    This is a convenience wrapper that combines
    :meth:`SimilarityAnalyzer.compare_all` and
    :meth:`AdvancedSimilarityAnalyzer.compare_all` into a single call.

    Args:
        data1: First dataset  (1D or 2D numerical array)
        data2: Second dataset (1D or 2D numerical array)
        n_components: PCA / SVD embedding dimensions
        n_topics: Number of LDA latent topics
        n_clusters: Number of KMeans clusters

    Returns:
        Flat dict with all metric names as keys.  Failed metrics have a
        ``None`` value and a companion ``<name>_error`` key.
    """
    data1 = np.atleast_2d(np.asarray(data1, dtype=float))
    data2 = np.atleast_2d(np.asarray(data2, dtype=float))

    results: Dict[str, object] = {}

    # Basic metrics (require equal total element count when flattened)
    f1, f2 = data1.flatten(), data2.flatten()
    if f1.shape == f2.shape:
        try:
            results.update(SimilarityAnalyzer.compare_all(f1, f2))
        except Exception as exc:
            results["basic_metrics"] = None
            results["basic_metrics_error"] = str(exc)
    else:
        results["basic_metrics"] = None
        results["basic_metrics_error"] = (
            f"Basic metrics skipped: arrays have different flattened sizes "
            f"({f1.size} vs {f2.size})."
        )

    # Advanced metrics
    results.update(
        AdvancedSimilarityAnalyzer().compare_all(
            data1, data2,
            n_components=n_components,
            n_topics=n_topics,
            n_clusters=n_clusters,
        )
    )

    return results


__version__ = "0.2.0"
__all__ = [
    "SimilarityAnalyzer",
    "AdvancedSimilarityAnalyzer",
    "DataLoader",
    "Visualizer",
    "compare",
]
