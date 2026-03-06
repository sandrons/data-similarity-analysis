"""
Core similarity metrics for numerical data comparison.
Includes basic distance and similarity measures.
"""
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr, spearmanr
from typing import Union, Tuple


class SimilarityAnalyzer:
    """Basic similarity metrics for numerical datasets."""
    
    def __init__(self):
        """Initialize the SimilarityAnalyzer."""
        pass
    
    @staticmethod
    def euclidean_distance(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two datasets.
        Lower values indicate higher similarity.
        
        Args:
            data1: First dataset (1D or 2D array)
            data2: Second dataset (same shape as data1)
            
        Returns:
            Euclidean distance as float
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return float(euclidean(data1, data2))
    
    @staticmethod
    def manhattan_distance(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Manhattan distance (L1 norm) between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Manhattan distance as float
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return float(cityblock(data1, data2))
    
    @staticmethod
    def cosine_similarity(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two datasets.
        Range: -1 to 1 (1 = identical direction, 0 = orthogonal, -1 = opposite)
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Cosine similarity score (1 - cosine_distance)
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        # cosine returns distance, so we convert to similarity
        return float(1 - cosine(data1, data2))
    
    @staticmethod
    def pearson_correlation(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Pearson correlation coefficient between two datasets.
        Measures linear relationship between variables.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        corr, pval = pearsonr(data1, data2)
        return float(corr), float(pval)
    
    @staticmethod
    def spearman_correlation(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Spearman correlation coefficient (rank-based).
        More robust to outliers than Pearson.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        corr, pval = spearmanr(data1, data2)
        return float(corr), float(pval)
    
    @staticmethod
    def minkowski_distance(data1: np.ndarray, data2: np.ndarray, p: int = 3) -> float:
        """
        Calculate Minkowski distance (generalized distance metric).
        p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev
        
        Args:
            data1: First dataset
            data2: Second dataset
            p: Order of the norm (default: 3)
            
        Returns:
            Minkowski distance as float
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return float(np.linalg.norm(data1 - data2, ord=p))
    
    @staticmethod
    def chebyshev_distance(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Chebyshev distance (maximum absolute difference).
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Chebyshev distance as float
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        return float(np.max(np.abs(data1 - data2)))
    
    @staticmethod
    def jaccard_similarity(data1: np.ndarray, data2: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate Jaccard similarity using binarized data.
        
        Args:
            data1: First dataset
            data2: Second dataset
            threshold: Threshold for binarization (default: 0.5)
            
        Returns:
            Jaccard similarity (0 to 1)
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        
        bin1 = (data1 > threshold).astype(int)
        bin2 = (data2 > threshold).astype(int)
        
        intersection = np.sum(bin1 & bin2)
        union = np.sum(bin1 | bin2)
        
        return float(intersection / union) if union > 0 else 0.0
    
    @staticmethod
    def hellinger_distance(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Hellinger distance (for probability distributions).
        
        Args:
            data1: First probability distribution
            data2: Second probability distribution
            
        Returns:
            Hellinger distance
        """
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        
        # Normalize to probability distributions
        data1 = data1 / np.sum(data1)
        data2 = data2 / np.sum(data2)
        
        return float(np.sqrt(0.5 * np.sum((np.sqrt(data1) - np.sqrt(data2)) ** 2)))
    
    @staticmethod
    def compare_all(data1: np.ndarray, data2: np.ndarray) -> dict:
        """
        Compare two datasets using all available metrics.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Dictionary with all similarity/distance metrics
        """
        analyzer = SimilarityAnalyzer()
        results = {
            'euclidean_distance': analyzer.euclidean_distance(data1, data2),
            'manhattan_distance': analyzer.manhattan_distance(data1, data2),
            'cosine_similarity': analyzer.cosine_similarity(data1, data2),
            'chebyshev_distance': analyzer.chebyshev_distance(data1, data2),
            'jaccard_similarity': analyzer.jaccard_similarity(data1, data2),
            'hellinger_distance': analyzer.hellinger_distance(data1, data2),
        }
        
        try:
            corr, pval = analyzer.pearson_correlation(data1, data2)
            results['pearson_correlation'] = corr
            results['pearson_pvalue'] = pval
        except Exception as e:
            results['pearson_correlation'] = None
            results['pearson_pvalue'] = None
        
        try:
            corr, pval = analyzer.spearman_correlation(data1, data2)
            results['spearman_correlation'] = corr
            results['spearman_pvalue'] = pval
        except Exception as e:
            results['spearman_correlation'] = None
            results['spearman_pvalue'] = None
        
        return results
