"""
Visualization utilities for similarity analysis results.

All methods return a ``matplotlib.figure.Figure`` so callers can display,
save, or embed it freely.  No ``plt.show()`` is called internally.

Example
-------
::

    from src.visualizer import Visualizer

    fig = Visualizer.pca_scatter(ref, query)
    fig.savefig("pca.png", dpi=150)

    fig = Visualizer.feature_distributions(ref, query)
    fig.savefig("distributions.png", dpi=150)

    fig = Visualizer.metrics_heatmap({
        "query_A": analyzer.compare_all(ref, query_a),
        "query_B": analyzer.compare_all(ref, query_b),
    })
    fig.savefig("heatmap.png", dpi=150)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Visualizer:
    """Static plotting helpers for data similarity analysis."""

    # ------------------------------------------------------------------
    # PCA scatter
    # ------------------------------------------------------------------

    @staticmethod
    def pca_scatter(
        data1: np.ndarray,
        data2: np.ndarray,
        labels: Tuple[str, str] = ("Dataset 1", "Dataset 2"),
        n_components: int = 2,
        alpha: float = 0.6,
        figsize: Tuple[float, float] = (7, 5),
    ):
        """
        Project both datasets into a shared 2-component PCA space and plot
        a scatter of each dataset in a different colour.

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            labels: Legend labels for data1 and data2
            n_components: Number of PCA components (2 for 2-D scatter,
                          3 for a 3-D scatter; anything else defaults to 2-D)
            alpha: Point transparency
            figsize: Figure size in inches

        Returns:
            ``matplotlib.figure.Figure``
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        combined = np.vstack([data1, data2])
        k = min(n_components, combined.shape[1], combined.shape[0])
        k = max(k, 1)

        pca = PCA(n_components=k)
        pca.fit(combined)
        proj1 = pca.transform(data1)
        proj2 = pca.transform(data2)

        var_ratio = pca.explained_variance_ratio_

        fig = plt.figure(figsize=figsize)

        if k >= 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(proj1[:, 0], proj1[:, 1], proj1[:, 2],
                       label=labels[0], alpha=alpha, s=30)
            ax.scatter(proj2[:, 0], proj2[:, 1], proj2[:, 2],
                       label=labels[1], alpha=alpha, s=30)
            ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
            ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
            ax.set_zlabel(f"PC3 ({var_ratio[2]:.1%})")
        else:
            ax = fig.add_subplot(111)
            ax.scatter(proj1[:, 0], proj1[:, -1],
                       label=labels[0], alpha=alpha, s=30)
            ax.scatter(proj2[:, 0], proj2[:, -1],
                       label=labels[1], alpha=alpha, s=30)
            xlabel = f"PC1 ({var_ratio[0]:.1%})"
            ylabel = f"PC2 ({var_ratio[1]:.1%})" if k >= 2 else "PC1"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        ax.set_title("PCA Projection")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Per-feature distribution overlay
    # ------------------------------------------------------------------

    @staticmethod
    def feature_distributions(
        data1: np.ndarray,
        data2: np.ndarray,
        labels: Tuple[str, str] = ("Dataset 1", "Dataset 2"),
        feature_names: Optional[Sequence[str]] = None,
        max_features: int = 8,
        bins: int = 30,
        figsize_per_col: Tuple[float, float] = (3.5, 2.8),
    ):
        """
        Plot side-by-side histogram overlays for each feature (column).

        Args:
            data1: First dataset  (n_samples x n_features)
            data2: Second dataset (m_samples x n_features)
            labels: Legend labels for data1 and data2
            feature_names: Column names; defaults to ``["f0", "f1", ...]``
            max_features: Cap the number of subplots to avoid huge figures
            bins: Number of histogram bins
            figsize_per_col: Width × height per subplot cell

        Returns:
            ``matplotlib.figure.Figure``
        """
        import matplotlib.pyplot as plt

        data1 = np.atleast_2d(np.asarray(data1, dtype=float))
        data2 = np.atleast_2d(np.asarray(data2, dtype=float))

        n_features = min(data1.shape[1], data2.shape[1], max_features)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]
        else:
            feature_names = list(feature_names)[:n_features]

        ncols = min(n_features, 4)
        nrows = (n_features + ncols - 1) // ncols

        fw = figsize_per_col[0] * ncols
        fh = figsize_per_col[1] * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)

        for i in range(n_features):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            combined_col = np.concatenate([data1[:, i], data2[:, i]])
            lo, hi = combined_col.min(), combined_col.max()
            bin_edges = np.linspace(lo, hi, bins + 1)
            ax.hist(data1[:, i], bins=bin_edges, alpha=0.5, label=labels[0], density=True)
            ax.hist(data2[:, i], bins=bin_edges, alpha=0.5, label=labels[1], density=True)
            ax.set_title(feature_names[i], fontsize=9)
            ax.tick_params(labelsize=7)

        # Legend on the first subplot only
        axes[0][0].legend(fontsize=7)

        # Hide unused subplots
        for i in range(n_features, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle("Feature Distributions", y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Metrics heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def metrics_heatmap(
        results: Dict[str, Dict[str, object]],
        title: str = "Similarity Metrics",
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = "RdYlGn",
    ):
        """
        Heatmap of similarity metric values across multiple query comparisons.

        Rows = query labels, columns = metric names.  Only numeric, non-error
        values are shown; metrics that failed for all queries are dropped.

        Args:
            results: ``{query_label: {metric_name: value, ...}, ...}``
                     as returned by ``AdvancedSimilarityAnalyzer.compare_all``
                     or ``SimilarityAnalyzer.compare_all``.
            title: Figure title
            figsize: Override auto-computed figure size
            cmap: Matplotlib colormap name

        Returns:
            ``matplotlib.figure.Figure``
        """
        import matplotlib.pyplot as plt

        # Collect all numeric metric names (exclude _error keys and None values)
        metric_names: List[str] = []
        for row_results in results.values():
            for k, v in row_results.items():
                if not k.endswith("_error") and isinstance(v, (int, float)) and k not in metric_names:
                    metric_names.append(k)

        query_labels = list(results.keys())

        # Build matrix (NaN for missing / failed metrics)
        matrix = np.full((len(query_labels), len(metric_names)), np.nan)
        for r, label in enumerate(query_labels):
            for c, metric in enumerate(metric_names):
                v = results[label].get(metric)
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    matrix[r, c] = float(v)

        # Drop columns that are entirely NaN
        valid_cols = ~np.all(np.isnan(matrix), axis=0)
        matrix = matrix[:, valid_cols]
        metric_names = [m for m, ok in zip(metric_names, valid_cols) if ok]

        if figsize is None:
            fw = max(6.0, len(metric_names) * 0.9)
            fh = max(3.0, len(query_labels) * 0.6 + 1.5)
            figsize = (fw, fh)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(query_labels)))
        ax.set_yticklabels(query_labels, fontsize=8)

        # Annotate cells
        for r in range(len(query_labels)):
            for c in range(len(metric_names)):
                v = matrix[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

        ax.set_title(title)
        fig.tight_layout()
        return fig
