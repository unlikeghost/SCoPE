import numpy as np
from typing import List, Union

from .base import _BasePredictor
from .metrics import *


class SCoPEPredictorV1(_BasePredictor):
    _supported_metrics = {
        "squared_euclidean": lambda x1, x2: squared_euclidean(x1, x2),
        "cosine": lambda x1, x2: cosine(x1, x2),
        "wasserstein": lambda x1, x2: wasserstein(x1, x2),
    }

    def __init__(self,
                 evaluation_metrics: Union[str, List[str]] = 'squared_euclidean',
                 **kwargs) -> None:
        super().__init__(
            **kwargs
        )

        if isinstance(evaluation_metrics, str):
            evaluation_metrics = [evaluation_metrics]

        valid_metrics = [
            evaluation_metric in self._supported_metrics
            for evaluation_metric in evaluation_metrics
        ]

        if not all(valid_metrics):
            raise ValueError(
                f"Invalid evaluation metric(s): {', '.join(evaluation_metrics)}. "
                f"Valid options are: {', '.join(self._supported_metrics.keys())}"
            )

        self.evaluation_metrics = evaluation_metrics

    def _forward(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:

        scores_ = []
        for evaluation_metric in self.evaluation_metrics:
            if evaluation_metric == 'wasserstein':
                # Make a 2D matrix from 3D to use OT
                batch_size, n_samples, metrics = current_cluster.shape
                current_cluster = current_cluster.reshape(batch_size, n_samples * metrics)
                current_sample = current_sample.reshape(1, n_samples * metrics)

            # We append the score for each distance metric, not for each dissimilarity metric
            scores_.append(
                self._supported_metrics[evaluation_metric](
                    current_cluster, current_sample
                )
            )

        # sum all scores
        score_ = sum(scores_)

        return score_

