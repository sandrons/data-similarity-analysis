# Data Similarity Analysis

## Overview
This project provides a comprehensive toolkit for assessing similarity between numerical datasets.
It covers classic distance metrics, advanced embedding-based measures, LDA topic similarity,
and clustering-based comparison — useful for data deduplication, clustering, anomaly detection,
and recommendation systems.

## Features
- **Advanced Similarity Measures**: Embedding-based (PCA/SVD), kernel MMD, LDA topic similarity, KMeans and DBSCAN clustering comparison.
- **Basic Similarity Measures**: Euclidean, Manhattan, Minkowski, Chebyshev, Pearson, Spearman, Cosine Similarity, Jaccard Index, Hellinger distance.
- **Data Loading**: CSV, NumPy file, and array loading with normalization and splitting utilities.
- **Performance**: Optimized algorithms built on NumPy, SciPy, and scikit-learn.

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/sandrons/data-similarity-analysis.git
cd data-similarity-analysis
pip install -r requirements.txt
```

## Package Structure

```
src/
├── __init__.py          # Exports SimilarityAnalyzer, AdvancedSimilarityAnalyzer, DataLoader
├── similarity.py        # Basic distance and correlation metrics
├── advanced_metrics.py  # Embedding, LDA, and clustering metrics
└── data_loader.py       # Data loading and preprocessing utilities
```

---

## Usage Examples

### Example 1: PCA Embedding Similarity

Projects both datasets into a shared PCA space and computes cosine similarity between
their mean embeddings. Useful for comparing high-dimensional datasets in a reduced space.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(50, 10)
data2 = np.random.rand(50, 10)

analyzer = AdvancedSimilarityAnalyzer()
score = analyzer.pca_embedding_similarity(data1, data2, n_components=3)
print(f"PCA Embedding Similarity: {score:.4f}")
# Range: [-1, 1] — higher means more similar in PCA space
```

### Example 2: LDA Topic Similarity

Treats dataset features as pseudo-word frequencies, fits a Latent Dirichlet Allocation model,
and compares the resulting topic distributions via Jensen-Shannon divergence.
Returns a score in [0, 1] where 1 means identical topic mixtures.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.abs(np.random.rand(30, 8))  # LDA requires non-negative values
data2 = np.abs(np.random.rand(30, 8))

analyzer = AdvancedSimilarityAnalyzer()
score = analyzer.lda_topic_similarity(data1, data2, n_topics=5)
print(f"LDA Topic Similarity: {score:.4f}")
# Range: [0, 1] — 1 = identical topic distribution
```

### Example 3: KMeans Cluster Similarity

Fits KMeans on the combined data and compares how the two datasets are distributed
across clusters. Returns three metrics: cosine similarity of cluster-frequency vectors,
cluster overlap (IoU proxy), and Normalized Mutual Information.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(40, 5)
data2 = np.random.rand(40, 5)

analyzer = AdvancedSimilarityAnalyzer()
results = analyzer.kmeans_cluster_similarity(data1, data2, n_clusters=4)
print(f"Cluster Cosine Similarity : {results['cluster_cosine_similarity']:.4f}")
print(f"Cluster Overlap           : {results['cluster_overlap']:.4f}")
print(f"Cluster NMI               : {results['cluster_nmi']:.4f}")
```

### Example 4: DBSCAN Structure Similarity

Runs DBSCAN independently on each standardized dataset and compares structural
properties: number of discovered clusters and fraction of noise points.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(60, 4)
data2 = np.random.rand(60, 4)

analyzer = AdvancedSimilarityAnalyzer()
results = analyzer.dbscan_structure_similarity(data1, data2, eps=0.5, min_samples=3)
print(f"Cluster Count Similarity : {results['cluster_count_similarity']:.4f}")
print(f"Noise Ratio Similarity   : {results['noise_ratio_similarity']:.4f}")
print(f"Clusters in data1        : {results['n_clusters_data1']}")
print(f"Clusters in data2        : {results['n_clusters_data2']}")
```

### Example 5: SVD Embedding Similarity

Uses TruncatedSVD (more efficient than PCA for sparse or very wide data) to project
both datasets into a latent space, then compares mean embeddings via cosine similarity.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(100, 50)
data2 = np.random.rand(100, 50)

analyzer = AdvancedSimilarityAnalyzer()
score = analyzer.svd_embedding_similarity(data1, data2, n_components=5)
print(f"SVD Embedding Similarity: {score:.4f}")
```

### Example 6: Kernel MMD Similarity

Computes the RBF-kernel Maximum Mean Discrepancy between two datasets and converts
it to a similarity via `exp(-MMD²)`. Values close to 1 indicate similar distributions.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(50, 6)
data2 = np.random.rand(50, 6)

analyzer = AdvancedSimilarityAnalyzer()
score = analyzer.kernel_mmd_similarity(data1, data2)
print(f"Kernel MMD Similarity: {score:.4f}")
# Range: [0, 1] — 1 = distributions are indistinguishable
```

### Example 7: Adjusted Rand Index

Clusters each dataset independently with KMeans and computes the Adjusted Rand Index
between the two cluster-label assignments (row-by-row). Requires equal sample counts.

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(40, 4)
data2 = np.random.rand(40, 4)

analyzer = AdvancedSimilarityAnalyzer()
score = analyzer.adjusted_rand_similarity(data1, data2, n_clusters=3)
print(f"Adjusted Rand Index: {score:.4f}")
# Range: [-1, 1] — 1 = perfect cluster agreement, 0 = random
```

### Example 8: Run All Advanced Metrics at Once

```python
import numpy as np
from src import AdvancedSimilarityAnalyzer

data1 = np.random.rand(30, 6)
data2 = np.random.rand(30, 6)

results = AdvancedSimilarityAnalyzer.compare_all(
    data1, data2, n_components=2, n_topics=4, n_clusters=3
)
for metric, value in results.items():
    print(f"{metric}: {value}")
```

---

## Basic Metrics

The `SimilarityAnalyzer` class provides classic distance and correlation measures.

### Example 9: Cosine Similarity

Measures the angle between two vectors. Returns 1 for identical direction, 0 for
orthogonal, and -1 for opposite directions. Independent of magnitude.

```python
import numpy as np
from src import SimilarityAnalyzer

vector1 = np.array([1, 0, 0, 1], dtype=float)
vector2 = np.array([1, 1, 0, 0], dtype=float)

analyzer = SimilarityAnalyzer()
score = analyzer.cosine_similarity(vector1, vector2)
print(f"Cosine Similarity: {score:.4f}")
# Range: [-1, 1]
```

### Example 10: Jaccard Index

Binarizes both arrays using a threshold and computes the ratio of intersection
to union. Useful for comparing presence/absence patterns.

```python
import numpy as np
from src import SimilarityAnalyzer

data1 = np.array([0.8, 0.2, 0.9, 0.1])
data2 = np.array([0.7, 0.6, 0.4, 0.3])

analyzer = SimilarityAnalyzer()
score = analyzer.jaccard_similarity(data1, data2, threshold=0.5)
print(f"Jaccard Index: {score:.4f}")
# Range: [0, 1] — 1 = identical binary patterns
```

### All Basic Metrics at a Glance

| Method | Description | Range |
|---|---|---|
| `euclidean_distance` | Straight-line distance | [0, ∞) |
| `manhattan_distance` | Sum of absolute differences | [0, ∞) |
| `minkowski_distance` | Generalized p-norm distance | [0, ∞) |
| `chebyshev_distance` | Maximum absolute difference | [0, ∞) |
| `cosine_similarity` | Angular similarity | [-1, 1] |
| `pearson_correlation` | Linear correlation + p-value | [-1, 1] |
| `spearman_correlation` | Rank-based correlation + p-value | [-1, 1] |
| `jaccard_similarity` | Binary set overlap | [0, 1] |
| `hellinger_distance` | Distance between distributions | [0, 1] |

```python
from src import SimilarityAnalyzer
import numpy as np

data1 = np.random.rand(20)
data2 = np.random.rand(20)

results = SimilarityAnalyzer.compare_all(data1, data2)
for metric, value in results.items():
    print(f"{metric}: {value}")
```

---

## Data Loading

```python
from src import DataLoader
import numpy as np

# From CSV
data = DataLoader.from_csv("dataset.csv")

# From NumPy file
data = DataLoader.from_numpy("dataset.npy")

# From a Python list or array
data = DataLoader.from_array([[1, 2, 3], [4, 5, 6]])

# Normalize
normalized = DataLoader.normalize(data, method="minmax")    # or "standard"

# Split into two subsets
part1, part2 = DataLoader.split(data, ratio=0.7, shuffle=True)
```

---

## Contributing
Fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
