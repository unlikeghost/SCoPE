import ot
from scipy.spatial.distance import cdist
import numpy as np


def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    # dist = np.linalg.norm(x1 - x2)
    dist = cdist(x1, x2, metric='euclidean').mean()
    return float(dist)

def cosine(x1: np.ndarray, x2: np.ndarray) -> float:
    dist = cdist(x1, x2, metric='cosine')
    return dist.mean()

def wasserstein(x1: np.ndarray, x2: np.ndarray) -> float:

    cluster_weights = np.ones(x2.shape[0]) / x2.shape[0]
    sample_weights = np.ones(x1.shape[0]) / x1.shape[0]

    cost_matrix_values = ot.dist(x2, x1, metric='euclidean')

    return ot.emd2(cluster_weights, sample_weights, cost_matrix_values)
