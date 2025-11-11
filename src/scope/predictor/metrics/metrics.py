import ot
import numpy as np


def squared_euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    diff = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    dist = np.linalg.norm(diff) ** 2
    return float(dist)

def cosine(x1: np.ndarray, x2: np.ndarray) -> float:
    dot_product = np.sum(x1 * x2, axis=-1)
    norm_x1 = np.linalg.norm(x1, axis=-1)
    norm_x2 = np.linalg.norm(x2, axis=-1)

    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    cosine_distance = 1 - cosine_similarity

    return np.mean(cosine_distance).item()

def wasserstein(x1: np.ndarray, x2: np.ndarray) -> float:

    cluster_weights = np.ones(x2.shape[0]) / x2.shape[0]
    sample_weights = np.ones(x1.shape[0]) / x1.shape[0]

    cost_matrix_values = ot.dist(x2, x1, metric='euclidean')

    return ot.emd2(cluster_weights, sample_weights, cost_matrix_values)
