import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]
