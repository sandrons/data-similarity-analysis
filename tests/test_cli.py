"""Tests for src/cli.py — CLI subcommands via subprocess."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cli(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def write_csv(path: Path, data: np.ndarray) -> None:
    np.savetxt(path, data, delimiter=",", header=",".join(f"f{i}" for i in range(data.shape[1])), comments="")


@pytest.fixture()
def csv_pair(tmp_path):
    """Two same-shape CSV files."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(20, 3))
    b = rng.normal(size=(20, 3))
    fa, fb = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(fa, a)
    write_csv(fb, b)
    return str(fa), str(fb)


@pytest.fixture()
def ref_and_queries(tmp_path):
    """One reference CSV and two query CSVs of different sizes."""
    rng = np.random.default_rng(99)
    ref = rng.normal(size=(30, 3))
    q1  = rng.normal(size=(25, 3))
    q2  = rng.normal(loc=5.0, size=(15, 3))
    fref = tmp_path / "ref.csv"
    fq1  = tmp_path / "q1.csv"
    fq2  = tmp_path / "q2.csv"
    write_csv(fref, ref)
    write_csv(fq1, q1)
    write_csv(fq2, q2)
    return str(fref), str(fq1), str(fq2)


# ---------------------------------------------------------------------------
# General CLI behaviour
# ---------------------------------------------------------------------------

class TestCLIHelp:
    def test_top_level_help_exits_zero(self):
        proc = run_cli("--help")
        assert proc.returncode == 0

    def test_compare_help_exits_zero(self):
        proc = run_cli("compare", "--help")
        assert proc.returncode == 0

    def test_compare_ref_help_exits_zero(self):
        proc = run_cli("compare-ref", "--help")
        assert proc.returncode == 0

    def test_no_subcommand_exits_nonzero(self):
        proc = run_cli()
        assert proc.returncode != 0

    def test_unsupported_file_type_exits_nonzero(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("1,2,3\n")
        proc = run_cli("compare", str(bad), str(bad))
        assert proc.returncode != 0


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------

class TestCompare:
    def test_table_output_contains_metric_names(self, csv_pair):
        fa, fb = csv_pair
        proc = run_cli("compare", fa, fb, "--metrics", "basic")
        assert proc.returncode == 0
        assert "euclidean_distance" in proc.stdout

    def test_advanced_metrics_appear(self, csv_pair):
        fa, fb = csv_pair
        proc = run_cli("compare", fa, fb, "--metrics", "advanced",
                       "--n-topics", "3", "--n-clusters", "3")
        assert proc.returncode == 0
        assert "pca_embedding_similarity" in proc.stdout

    def test_json_output_is_valid(self, csv_pair):
        fa, fb = csv_pair
        proc = run_cli("compare", fa, fb, "--metrics", "basic", "--format", "json")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert isinstance(data, dict)
        # The single top-level key is the label "a.csv  vs  b.csv"
        (result,) = data.values()
        assert "euclidean_distance" in result

    def test_json_all_metrics(self, csv_pair):
        fa, fb = csv_pair
        proc = run_cli("compare", fa, fb, "--metrics", "all", "--format", "json",
                       "--n-topics", "3", "--n-clusters", "3")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        (result,) = data.values()
        assert "pca_embedding_similarity" in result
        assert "euclidean_distance" in result

    def test_label_contains_filenames(self, csv_pair):
        fa, fb = csv_pair
        proc = run_cli("compare", fa, fb, "--metrics", "basic")
        assert "a.csv" in proc.stdout
        assert "b.csv" in proc.stdout

    def test_npy_files_accepted(self, tmp_path):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(15, 2))
        b = rng.normal(size=(15, 2))
        fa, fb = tmp_path / "a.npy", tmp_path / "b.npy"
        np.save(fa, a)
        np.save(fb, b)
        proc = run_cli("compare", str(fa), str(fb), "--metrics", "advanced",
                       "--n-topics", "2", "--n-clusters", "2")
        assert proc.returncode == 0


# ---------------------------------------------------------------------------
# compare-ref subcommand
# ---------------------------------------------------------------------------

class TestCompareRef:
    def test_produces_output_for_each_query(self, ref_and_queries):
        fref, fq1, fq2 = ref_and_queries
        proc = run_cli("compare-ref", fref, fq1, fq2, "--metrics", "advanced",
                       "--n-topics", "3", "--n-clusters", "3")
        assert proc.returncode == 0
        assert "q1.csv" in proc.stdout
        assert "q2.csv" in proc.stdout

    def test_json_output_has_entry_per_query(self, ref_and_queries):
        fref, fq1, fq2 = ref_and_queries
        proc = run_cli("compare-ref", fref, fq1, fq2,
                       "--metrics", "advanced", "--format", "json",
                       "--n-topics", "3", "--n-clusters", "3")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert len(data) == 2

    def test_fitting_message_on_stderr(self, ref_and_queries):
        fref, fq1, fq2 = ref_and_queries
        proc = run_cli("compare-ref", fref, fq1, fq2,
                       "--metrics", "advanced", "--n-topics", "3", "--n-clusters", "3")
        assert "Fitting" in proc.stderr

    def test_single_query(self, ref_and_queries):
        fref, fq1, _ = ref_and_queries
        proc = run_cli("compare-ref", fref, fq1, "--metrics", "advanced",
                       "--n-topics", "3", "--n-clusters", "3")
        assert proc.returncode == 0
        assert "pca_embedding_similarity" in proc.stdout

    def test_basic_only_no_fitting_message(self, ref_and_queries):
        fref, fq1, fq2 = ref_and_queries
        proc = run_cli("compare-ref", fref, fq1, fq2, "--metrics", "basic")
        assert "Fitting" not in proc.stderr

    def test_different_sized_queries_no_crash(self, ref_and_queries):
        # q1 (25 rows) and q2 (15 rows) differ from ref (30 rows)
        fref, fq1, fq2 = ref_and_queries
        proc = run_cli("compare-ref", fref, fq1, fq2,
                       "--metrics", "advanced", "--n-topics", "3", "--n-clusters", "3")
        assert proc.returncode == 0
