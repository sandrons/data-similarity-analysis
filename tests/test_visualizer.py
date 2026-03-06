"""Tests for src/visualizer.py."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any pyplot import

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.visualizer import Visualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(11)
DATA1 = RNG.normal(size=(40, 4))
DATA2 = RNG.normal(loc=2.0, size=(30, 4))
DATA_1D = RNG.normal(size=20)


def close_all():
    plt.close("all")


# ---------------------------------------------------------------------------
# pca_scatter
# ---------------------------------------------------------------------------

class TestPCAScatter:
    def teardown_method(self):
        close_all()

    def test_returns_figure(self):
        fig = Visualizer.pca_scatter(DATA1, DATA2)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_one_axes(self):
        fig = Visualizer.pca_scatter(DATA1, DATA2)
        assert len(fig.axes) == 1

    def test_legend_contains_labels(self):
        labels = ("Ref", "Query")
        fig = Visualizer.pca_scatter(DATA1, DATA2, labels=labels)
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Ref" in legend_texts
        assert "Query" in legend_texts

    def test_title_is_set(self):
        fig = Visualizer.pca_scatter(DATA1, DATA2)
        assert fig.axes[0].get_title() == "PCA Projection"

    def test_1d_input_accepted(self):
        fig = Visualizer.pca_scatter(DATA_1D, DATA_1D)
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self):
        fig = Visualizer.pca_scatter(DATA1, DATA2, figsize=(10, 6))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(10.0)
        assert h == pytest.approx(6.0)

    def test_3d_scatter(self):
        data_3f = RNG.normal(size=(30, 5))
        fig = Visualizer.pca_scatter(data_3f, data_3f, n_components=3)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# feature_distributions
# ---------------------------------------------------------------------------

class TestFeatureDistributions:
    def teardown_method(self):
        close_all()

    def test_returns_figure(self):
        fig = Visualizer.feature_distributions(DATA1, DATA2)
        assert isinstance(fig, plt.Figure)

    def test_subplot_count_matches_features(self):
        data = RNG.normal(size=(20, 3))
        fig = Visualizer.feature_distributions(data, data)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 3

    def test_max_features_caps_subplots(self):
        data = RNG.normal(size=(20, 10))
        fig = Visualizer.feature_distributions(data, data, max_features=4)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 4

    def test_custom_feature_names(self):
        names = ["alpha", "beta", "gamma", "delta"]
        fig = Visualizer.feature_distributions(DATA1, DATA2, feature_names=names)
        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        assert titles[:4] == names

    def test_different_row_counts_accepted(self):
        d1 = RNG.normal(size=(50, 3))
        d2 = RNG.normal(size=(20, 3))
        fig = Visualizer.feature_distributions(d1, d2)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# metrics_heatmap
# ---------------------------------------------------------------------------

class TestMetricsHeatmap:
    def teardown_method(self):
        close_all()

    def _make_results(self):
        return {
            "query_A": {"metric1": 0.8, "metric2": 0.5, "metric3": None, "metric3_error": "fail"},
            "query_B": {"metric1": 0.3, "metric2": 0.9, "metric3": 0.6},
        }

    def test_returns_figure(self):
        fig = Visualizer.metrics_heatmap(self._make_results())
        assert isinstance(fig, plt.Figure)

    def test_figure_has_axes(self):
        fig = Visualizer.metrics_heatmap(self._make_results())
        assert len(fig.axes) >= 1

    def test_ytick_labels_are_query_names(self):
        results = self._make_results()
        fig = Visualizer.metrics_heatmap(results)
        ax = fig.axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert set(ytick_labels) == set(results.keys())

    def test_error_keys_excluded_from_columns(self):
        results = self._make_results()
        fig = Visualizer.metrics_heatmap(results)
        ax = fig.axes[0]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert not any(k.endswith("_error") for k in xtick_labels)

    def test_none_values_excluded(self):
        results = self._make_results()
        fig = Visualizer.metrics_heatmap(results)
        ax = fig.axes[0]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        # metric3 is None for query_A but present for query_B, so it appears
        assert "metric3" in xtick_labels

    def test_all_none_metric_dropped(self):
        results = {
            "q1": {"good": 0.5, "bad": None},
            "q2": {"good": 0.7, "bad": None},
        }
        fig = Visualizer.metrics_heatmap(results)
        ax = fig.axes[0]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "bad" not in xtick_labels
        assert "good" in xtick_labels

    def test_custom_title(self):
        fig = Visualizer.metrics_heatmap(self._make_results(), title="My Heatmap")
        assert fig.axes[0].get_title() == "My Heatmap"

    def test_single_query(self):
        results = {"only_query": {"m1": 0.4, "m2": 0.9}}
        fig = Visualizer.metrics_heatmap(results)
        assert isinstance(fig, plt.Figure)
