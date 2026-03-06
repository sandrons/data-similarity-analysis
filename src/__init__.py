"""Data Similarity Analysis Package"""
from .similarity import SimilarityAnalyzer
from .advanced_metrics import AdvancedSimilarityAnalyzer
from .data_loader import DataLoader

__version__ = "1.0.0"
__all__ = ["SimilarityAnalyzer", "AdvancedSimilarityAnalyzer", "DataLoader"]