"""
Data loading and preprocessing utilities for the similarity analysis package.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union


class DataLoader:
    """Utilities for loading and preprocessing numerical datasets."""

    @staticmethod
    def from_csv(path: Union[str, Path], **kwargs) -> np.ndarray:
        """
        Load a dataset from a CSV file, retaining only numeric columns.

        Args:
            path: Path to the CSV file
            **kwargs: Extra arguments forwarded to ``pandas.read_csv``

        Returns:
            2D float array (n_samples x n_numeric_features)
        """
        df = pd.read_csv(path, **kwargs)
        return df.select_dtypes(include=[np.number]).values.astype(float)

    @staticmethod
    def from_numpy(path: Union[str, Path]) -> np.ndarray:
        """
        Load a dataset from a ``.npy`` or ``.npz`` file.

        For ``.npz`` archives the first array is returned.

        Args:
            path: Path to the NumPy file

        Returns:
            Loaded array as float
        """
        path = Path(path)
        if path.suffix == ".npz":
            archive = np.load(path)
            return archive[list(archive.keys())[0]].astype(float)
        return np.load(path).astype(float)

    @staticmethod
    def from_array(data: Union[list, np.ndarray]) -> np.ndarray:
        """
        Convert a Python list or NumPy array to a 2D float array.

        Args:
            data: Input data

        Returns:
            2D float array
        """
        return np.atleast_2d(np.asarray(data, dtype=float))

    @staticmethod
    def normalize(data: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize a dataset in-place along columns (features).

        Args:
            data: Input array (n_samples x n_features)
            method: ``"minmax"`` scales each feature to [0, 1];
                    ``"standard"`` applies z-score normalization

        Returns:
            Normalized array (same shape as input)

        Raises:
            ValueError: If an unknown method is given
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        data = np.atleast_2d(np.asarray(data, dtype=float))
        if method == "minmax":
            return MinMaxScaler().fit_transform(data)
        if method == "standard":
            return StandardScaler().fit_transform(data)
        raise ValueError(f"Unknown normalization method '{method}'. Use 'minmax' or 'standard'.")

    @staticmethod
    def split(
        data: np.ndarray,
        ratio: float = 0.5,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset into two non-overlapping subsets.

        Args:
            data: Input array (n_samples x n_features)
            ratio: Fraction of rows assigned to the first subset
            shuffle: Randomly permute rows before splitting
            random_state: Seed for the random permutation

        Returns:
            Tuple ``(part1, part2)`` where ``len(part1) = floor(n * ratio)``
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        if shuffle:
            rng = np.random.default_rng(random_state)
            data = data[rng.permutation(len(data))]
        split_idx = int(len(data) * ratio)
        return data[:split_idx], data[split_idx:]
