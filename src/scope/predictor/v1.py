import numpy as np
from typing import List, Union

from .base_ import _BasePredictor
from .metrics import *


class SCoPEPredictorV1(_BasePredictor):
    _supported_metrics = {
        "euclidean": lambda x1, x2: euclidean(x1, x2),
        "cosine": lambda x1, x2: cosine(x1, x2),
        "wasserstein": lambda x1, x2: wasserstein(x1, x2),
    }

    def __init__(self,
                 distance_metrics: Union[str, List[str]] = 'euclidean',
                 **kwargs) -> None:
        super().__init__(
            **kwargs
        )

        if isinstance(distance_metrics, str):
            distance_metrics = [distance_metrics]

        valid_metrics = [
            distance_metric in self._supported_metrics
            for distance_metric in distance_metrics
        ]

        if not all(valid_metrics):
            raise ValueError(
                f"Invalid distance metric(s): {', '.join(distance_metrics)}. "
                f"Valid options are: {', '.join(self._supported_metrics.keys())}"
            )

        self.distance_metrics = distance_metrics

    def _forward(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:

        scores_ = []
        for distance_metric in self.distance_metrics:
            # We append the score for each distance metric, not for each dissimilarity metric
            scores_.append(
                self._supported_metrics[distance_metric](
                    current_cluster, current_sample
                )
            )

        # sum all scores
        score_ = sum(scores_)

        return score_

