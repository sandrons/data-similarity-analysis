"""
Command-line interface for data-similarity-analysis.

Usage
-----
Compare two files head-to-head::

    similarity compare data1.csv data2.csv

Compare many query files against one reference (fits models once, reuses cache)::

    similarity compare-ref reference.csv query1.csv query2.csv query3.csv

Plot visualisations::

    similarity plot pca   ref.csv query.csv --output pca.png
    similarity plot dist  ref.csv query.csv --output dist.png
    similarity plot heatmap ref.csv q1.csv q2.csv --output heatmap.png

Options (available on compare / compare-ref subcommands)::

    --metrics   basic | advanced | all   (default: all)
    --format    table | json             (default: table)
    --n-components N                     PCA/SVD dimensions  (default: 2)
    --n-topics N                         LDA topics          (default: 5)
    --n-clusters N                       KMeans clusters     (default: 3)

Supported file types: .csv, .npy, .npz, .parquet
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load(path: str) -> np.ndarray:
    from src.data_loader import DataLoader

    p = Path(path)
    if p.suffix == ".csv":
        return DataLoader.from_csv(p)
    if p.suffix in (".npy", ".npz"):
        return DataLoader.from_numpy(p)
    if p.suffix == ".parquet":
        return DataLoader.from_parquet(p)
    raise ValueError(f"Unsupported file type '{p.suffix}'. Use .csv, .npy, .npz, or .parquet")


def _run_basic(data1: np.ndarray, data2: np.ndarray) -> dict:
    from src.similarity import SimilarityAnalyzer

    f1, f2 = data1.flatten(), data2.flatten()
    if f1.shape != f2.shape:
        return {
            "basic_metrics": None,
            "basic_metrics_error": (
                f"Basic metrics require equal-length flattened arrays "
                f"({f1.size} vs {f2.size}). Use --metrics advanced to skip them."
            ),
        }
    return SimilarityAnalyzer.compare_all(f1, f2)


def _run_advanced(analyzer, data1: np.ndarray, data2: np.ndarray, args) -> dict:
    return analyzer.compare_all(
        data1,
        data2,
        n_components=args.n_components,
        n_topics=args.n_topics,
        n_clusters=args.n_clusters,
    )


def _print_table(label: str, results: dict) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  {'Metric':<42} {'Value':>14}")
    print(f"  {'-' * 42} {'-' * 14}")
    for k, v in results.items():
        if k.endswith("_error"):
            continue
        if v is None:
            val_str = "N/A"
        elif isinstance(v, float):
            val_str = f"{v:.6f}"
        elif isinstance(v, int):
            val_str = str(v)
        else:
            val_str = str(v)
        print(f"  {k:<42} {val_str:>14}")

    errors = {k: v for k, v in results.items() if k.endswith("_error")}
    if errors:
        print(f"\n  Errors:")
        for k, v in errors.items():
            print(f"    {k}: {v}")


def _build_results(analyzer, ref_data, query_data, args) -> dict:
    results: dict = {}
    if args.metrics in ("basic", "all"):
        results.update(_run_basic(ref_data, query_data))
    if args.metrics in ("advanced", "all"):
        results.update(_run_advanced(analyzer, ref_data, query_data, args))
    return results


def cmd_compare(args) -> None:
    from src.advanced_metrics import AdvancedSimilarityAnalyzer

    data1 = _load(args.file1)
    data2 = _load(args.file2)
    label = f"{Path(args.file1).name}  vs  {Path(args.file2).name}"

    analyzer = AdvancedSimilarityAnalyzer()
    results = _build_results(analyzer, data1, data2, args)

    if args.format == "json":
        print(json.dumps({label: results}, indent=2, default=str))
    else:
        _print_table(label, results)


def cmd_compare_ref(args) -> None:
    from src.advanced_metrics import AdvancedSimilarityAnalyzer

    ref_data = _load(args.reference)
    ref_name = Path(args.reference).name

    analyzer = AdvancedSimilarityAnalyzer()

    if args.metrics in ("advanced", "all"):
        print(
            f"Fitting models on reference '{ref_name}' ...",
            file=sys.stderr,
        )
        analyzer.fit(
            ref_data,
            n_components=args.n_components,
            n_topics=args.n_topics,
            n_clusters=args.n_clusters,
        )
        print("Done. Reusing cached models for all queries.\n", file=sys.stderr)

    all_results: dict = {}
    for qpath in args.queries:
        query_data = _load(qpath)
        label = f"{ref_name}  vs  {Path(qpath).name}"
        results = _build_results(analyzer, ref_data, query_data, args)
        all_results[label] = results
        if args.format == "table":
            _print_table(label, results)

    if args.format == "json":
        print(json.dumps(all_results, indent=2, default=str))


def cmd_plot(args) -> None:
    from src.visualizer import Visualizer

    output = Path(args.output)

    if args.plot_type == "pca":
        data1 = _load(args.file1)
        data2 = _load(args.file2)
        labels = (Path(args.file1).name, Path(args.file2).name)
        fig = Visualizer.pca_scatter(
            data1, data2, labels=labels, n_components=args.n_components
        )

    elif args.plot_type == "dist":
        data1 = _load(args.file1)
        data2 = _load(args.file2)
        labels = (Path(args.file1).name, Path(args.file2).name)
        fig = Visualizer.feature_distributions(
            data1, data2, labels=labels, max_features=args.max_features
        )

    elif args.plot_type == "heatmap":
        from src.advanced_metrics import AdvancedSimilarityAnalyzer

        ref_data = _load(args.reference)
        ref_name = Path(args.reference).name
        analyzer = AdvancedSimilarityAnalyzer()
        analyzer.fit(ref_data, n_components=args.n_components,
                     n_topics=args.n_topics, n_clusters=args.n_clusters)

        all_results: dict = {}
        for qpath in args.queries:
            label = f"{ref_name} vs {Path(qpath).name}"
            all_results[label] = analyzer.compare_all(
                ref_data, _load(qpath),
                n_components=args.n_components,
                n_topics=args.n_topics,
                n_clusters=args.n_clusters,
            )
        fig = Visualizer.metrics_heatmap(all_results)

    else:
        raise ValueError(f"Unknown plot type: {args.plot_type}")

    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output}", file=sys.stderr)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics",
        choices=["basic", "advanced", "all"],
        default="all",
        help="Which metric group to compute (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        metavar="N",
        help="PCA/SVD embedding dimensions (default: 2)",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=5,
        metavar="N",
        help="Number of LDA topics (default: 5)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        metavar="N",
        help="Number of KMeans clusters (default: 3)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="similarity",
        description=(
            "Compare numerical datasets with basic and advanced similarity metrics.\n\n"
            "Supported file types: .csv, .npy, .npz"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- compare ---
    p_compare = sub.add_parser(
        "compare",
        help="Compare two files head-to-head",
        description="Compare two datasets and print similarity metrics.",
    )
    p_compare.add_argument("file1", help="First dataset (.csv / .npy / .npz)")
    p_compare.add_argument("file2", help="Second dataset (.csv / .npy / .npz)")
    _add_common_args(p_compare)

    # --- compare-ref ---
    p_ref = sub.add_parser(
        "compare-ref",
        help="Compare one reference against many queries (fits models once)",
        description=(
            "Fit expensive models (PCA, SVD, LDA, KMeans) on the reference dataset\n"
            "once, then reuse the cached models for every query file. Much faster\n"
            "than calling 'compare' repeatedly when the reference stays the same."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ref.add_argument("reference", help="Reference dataset (.csv / .npy / .npz)")
    p_ref.add_argument(
        "queries",
        nargs="+",
        help="One or more query datasets to compare against the reference",
    )
    _add_common_args(p_ref)

    # --- plot ---
    _plot_common = argparse.ArgumentParser(add_help=False)
    _plot_common.add_argument(
        "--output", "-o", default="similarity_plot.png",
        help="Output file path (default: similarity_plot.png)",
    )
    _plot_common.add_argument(
        "--dpi", type=int, default=150,
        help="Resolution in dots per inch (default: 150)",
    )

    p_plot = sub.add_parser(
        "plot",
        help="Generate visualisation plots",
        description="Save similarity visualisations to image files.",
    )
    plot_sub = p_plot.add_subparsers(dest="plot_type", required=True)

    # plot pca
    pp = plot_sub.add_parser("pca", parents=[_plot_common],
                             help="PCA scatter of two datasets")
    pp.add_argument("file1", help="First dataset")
    pp.add_argument("file2", help="Second dataset")
    pp.add_argument("--n-components", type=int, default=2, metavar="N",
                    help="PCA components (2 or 3, default: 2)")

    # plot dist
    pd_ = plot_sub.add_parser("dist", parents=[_plot_common],
                              help="Per-feature distribution overlay")
    pd_.add_argument("file1", help="First dataset")
    pd_.add_argument("file2", help="Second dataset")
    pd_.add_argument("--max-features", type=int, default=8, metavar="N",
                     help="Maximum number of features to plot (default: 8)")

    # plot heatmap
    ph = plot_sub.add_parser("heatmap", parents=[_plot_common],
                             help="Metrics heatmap across multiple queries")
    ph.add_argument("reference", help="Reference dataset")
    ph.add_argument("queries", nargs="+", help="Query datasets")
    ph.add_argument("--n-components", type=int, default=2, metavar="N")
    ph.add_argument("--n-topics", type=int, default=5, metavar="N")
    ph.add_argument("--n-clusters", type=int, default=3, metavar="N")

    args = parser.parse_args()

    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "compare-ref":
        cmd_compare_ref(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()
