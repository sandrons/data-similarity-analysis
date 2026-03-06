"""Tests for src/data_loader.py."""
import numpy as np
import pytest

from src.data_loader import DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(5)
DATA_2D = RNG.normal(size=(30, 4))


# ---------------------------------------------------------------------------
# from_csv
# ---------------------------------------------------------------------------

class TestFromCSV:
    def test_loads_numeric_columns(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": ["x", "y"]})
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        result = DataLoader.from_csv(path)
        assert result.shape == (2, 2)  # only numeric columns
        assert result.dtype == float

    def test_returns_2d_float_array(self, tmp_path):
        path = tmp_path / "data.csv"
        np.savetxt(path, DATA_2D, delimiter=",",
                   header=",".join(f"f{i}" for i in range(DATA_2D.shape[1])), comments="")
        result = DataLoader.from_csv(path)
        assert result.ndim == 2
        assert result.dtype == float

    def test_shape_preserved(self, tmp_path):
        path = tmp_path / "data.csv"
        np.savetxt(path, DATA_2D, delimiter=",",
                   header=",".join(f"f{i}" for i in range(DATA_2D.shape[1])), comments="")
        result = DataLoader.from_csv(path)
        assert result.shape == DATA_2D.shape


# ---------------------------------------------------------------------------
# from_numpy
# ---------------------------------------------------------------------------

class TestFromNumpy:
    def test_loads_npy_file(self, tmp_path):
        path = tmp_path / "data.npy"
        np.save(path, DATA_2D)
        result = DataLoader.from_numpy(path)
        assert result.shape == DATA_2D.shape
        np.testing.assert_allclose(result, DATA_2D)

    def test_loads_npz_file_first_array(self, tmp_path):
        path = tmp_path / "data.npz"
        np.savez(path, arr=DATA_2D)
        result = DataLoader.from_numpy(path)
        assert result.shape == DATA_2D.shape

    def test_returns_float_dtype(self, tmp_path):
        path = tmp_path / "int_data.npy"
        np.save(path, np.array([[1, 2], [3, 4]], dtype=int))
        result = DataLoader.from_numpy(path)
        assert result.dtype == float


# ---------------------------------------------------------------------------
# from_array
# ---------------------------------------------------------------------------

class TestFromArray:
    def test_list_input_converted_to_2d(self):
        result = DataLoader.from_array([1, 2, 3])
        assert result.ndim == 2
        assert result.shape == (1, 3)

    def test_1d_array_promoted_to_2d(self):
        result = DataLoader.from_array(np.array([1.0, 2.0]))
        assert result.shape == (1, 2)

    def test_2d_array_preserved(self):
        result = DataLoader.from_array(DATA_2D)
        assert result.shape == DATA_2D.shape

    def test_dtype_is_float(self):
        result = DataLoader.from_array([[1, 2], [3, 4]])
        assert result.dtype == float


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_minmax_range(self):
        result = DataLoader.normalize(DATA_2D, method="minmax")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_standard_zero_mean(self):
        result = DataLoader.normalize(DATA_2D, method="standard")
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-10)

    def test_standard_unit_variance(self):
        result = DataLoader.normalize(DATA_2D, method="standard")
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-10)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            DataLoader.normalize(DATA_2D, method="l2")

    def test_output_shape_preserved(self):
        result = DataLoader.normalize(DATA_2D, method="minmax")
        assert result.shape == DATA_2D.shape


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_default_ratio_half_half(self):
        part1, part2 = DataLoader.split(DATA_2D, ratio=0.5)
        assert len(part1) == 15
        assert len(part2) == 15

    def test_parts_cover_all_rows(self):
        part1, part2 = DataLoader.split(DATA_2D)
        assert len(part1) + len(part2) == len(DATA_2D)

    def test_no_overlap_without_shuffle(self):
        part1, part2 = DataLoader.split(DATA_2D, shuffle=False)
        # Without shuffle the split is deterministic: part1 = DATA_2D[:n]
        n = int(len(DATA_2D) * 0.5)
        np.testing.assert_array_equal(part1, DATA_2D[:n])
        np.testing.assert_array_equal(part2, DATA_2D[n:])

    def test_shuffle_changes_order(self):
        part1_no_shuffle, _ = DataLoader.split(DATA_2D, shuffle=False)
        part1_shuffle, _ = DataLoader.split(DATA_2D, shuffle=True, random_state=0)
        # With high probability the shuffled first half differs from the unshuffled one
        assert not np.array_equal(part1_no_shuffle, part1_shuffle)

    def test_ratio_70_30(self):
        part1, part2 = DataLoader.split(DATA_2D, ratio=0.7)
        assert len(part1) == int(len(DATA_2D) * 0.7)
        assert len(part2) == len(DATA_2D) - int(len(DATA_2D) * 0.7)
