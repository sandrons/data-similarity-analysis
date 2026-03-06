"""Basic similarity metrics for numerical data comparison"""
import numpy as np
from scipy.spatial.distance import euclidean, cosine, cityblock
from scipy.stats import pearsonr
from typing import Tuple, Union


class SimilarityAnalyzer:
    """
    Computes basic similarity metrics between numerical datasets.
    
    Supported metrics:
    - Euclidean Distance
    - Cosine Similarity
    - Manhattan Distance
    - Pearson Correlation
    """
    
    @staticmethod
    def euclidean_distance(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two datasets.
        
        Args:
            data1: First dataset (1D or 2D array)
            data2: Second dataset (same shape as data1)
            
        Returns:
            float: Euclidean distance
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return euclidean(data1, data2)
    
    @staticmethod
    def cosine_similarity(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two datasets.
        Returns value between 0 (dissimilar) and 1 (identical).
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            float: Cosine similarity (1 - cosine distance)
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return 1 - cosine(data1, data2)
    
    @staticmethod
    def manhattan_distance(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Manhattan distance between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            float: Manhattan distance
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return cityblock(data1, data2)
    
    @staticmethod
    def pearson_correlation(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Pearson correlation coefficient between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Tuple[float, float]: Correlation coefficient and p-value
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        corr, pvalue = pearsonr(data1, data2)
        return corr, pvalue
    
    @staticmethod
    def minkowski_distance(data1: np.ndarray, data2: np.ndarray, p: float = 2) -> float:
        """
        Calculate Minkowski distance (generalization of Euclidean and Manhattan).
        
        Args:
            data1: First dataset
            data2: Second dataset
            p: Order of the norm (p=1 is Manhattan, p=2 is Euclidean)
            
        Returns:
            float: Minkowski distance
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return np.linalg.norm(data1 - data2, ord=p)
    
    @staticmethod
    def jaccard_similarity(data1: np.ndarray, data2: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate Jaccard similarity for binary or thresholded data.
        
        Args:
            data1: First dataset
            data2: Second dataset
            threshold: Threshold for converting to binary
            
        Returns:
            float: Jaccard similarity (0 to 1)
        """
        data1 = (np.asarray(data1).flatten() > threshold).astype(int)
        data2 = (np.asarray(data2).flatten() > threshold).astype(int)
        
        intersection = np.sum(data1 & data2)
        union = np.sum(data1 | data2)
        
        if union == 0:
            return 1.0
        return intersection / union