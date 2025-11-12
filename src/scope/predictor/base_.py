import numpy as np
from scipy.stats import gmean
from collections import Counter
from typing import Dict, Union, Any, List, Optional
from abc import ABC


class _BasePredictor(ABC):
    start_key_value_cluster: str = 'SCoPE_Cluster_'
    start_key_value_sample: str = 'SCoPE_Sample_'
    start_key_sigma: str = 'SCoPE_Sigma'

    _prototype_methods = {
        "mean": lambda data: np.mean(data, axis=1),
        "median": lambda data: np.median(data, axis=1),
        "sum": lambda data: np.sum(data, axis=1),
        "gmean": lambda data: gmean(data, axis=1)
    }

    def __init__(self, prototype_method: Optional[str] = None, epsilon: Optional[float] = None):

        if prototype_method is not None and prototype_method not in self._prototype_methods:
            raise ValueError(
                f"Invalid prototype method: {prototype_method}. "
                f"Valid options: "
                f"{self._prototype_methods} or None for no prototype"
            )

        self.prototype_method = prototype_method
        self.epsilon = epsilon if epsilon and epsilon >= 0.0 else 1e-12

    def _compute_aggregated_prototype(self, data: np.ndarray) -> np.ndarray:
        """Compute prototype using specified prototype method"""
        if self.prototype_method is None:
            raise ValueError("Aggregation method cannot be None when computing prototype")

        prototype = self._prototype_methods[self.prototype_method](
            data
        )
        prototype = np.expand_dims(prototype, axis=1)

        return prototype

    def _compute_proba(self, dists: np.ndarray) -> np.ndarray:
        dists = np.array(dists, dtype=float)
        scores = 1.0 / (dists + self.epsilon) ** 2
        proba = scores / np.sum(scores)
        return proba

    def _forward(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        raise NotImplementedError()

    def forward(self, list_of_cm: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
        """
            Here we receive a list of dictionaries containing data matrices.
            This data comes from CompressionMatrix class
            the intput should be:
                list of:
                    SCoPE_Cluster_n: an array of dissimilarity compression metrics: (samples, samples+1, n_compressors, n_metrics)
                    SCoPE_Sample_n: an array of dissimilarity compression metrics: (1, samples+1, n_compressors, n_metrics) -> sample to classify
                    SCoPE_Sigma_n: inf, not using at the moment
        """
        if not isinstance(list_of_cm, list):
            raise ValueError("Input should be a list of dictionaries containing data matrices.")

        if not list_of_cm:
            raise ValueError("Input list is empty.")

        result: List[Dict[str, Any]] = []

        for data_matrix in list_of_cm:
            cluster_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_cluster),
                    data_matrix.keys()
                )
            )
            sample_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_sample),
                    data_matrix.keys()
                )
            )

            iteration_output: Dict[str, Any] = {
                'scores': {
                    cluster_key[len(self.start_key_value_cluster):]: []
                    for cluster_key in cluster_keys
                },
                'proba': {
                    cluster_key[len(self.start_key_value_cluster):]: float('inf')
                    for cluster_key in cluster_keys
                },
                'predicted_class': None,
            }

            for cluster_key in cluster_keys:
                cluster_name: str = cluster_key[len(self.start_key_value_cluster):]
                sample_key: str = list(
                    filter(
                        lambda x: x.endswith(cluster_name),
                        sample_keys)
                )[0]

                cluster_: np.ndarray = data_matrix[cluster_key] # samples we know of class "cluster_key"
                sample_: np.ndarray = data_matrix[sample_key] # sample to predict based on cluster "cluster_key"

                if self.prototype_method is not None:
                    # We calculate the prototype if the prototype method is specified
                    cluster_ = self._compute_aggregated_prototype(cluster_)

                compressor_metric, _, _ = cluster_.shape

                for index in range(compressor_metric):
                    current_score = self._forward(
                        current_cluster=cluster_[index, :, :], # shape: (samples, samples+1)
                        current_sample=sample_[index, :, :] # shape: (1, samples+1)
                    )
                    iteration_output['scores'][cluster_name].append(current_score)

            dict_scores: Dict[str, List[float]] = iteration_output['scores']

            class_keys: List[str]  = list(dict_scores.keys())

            if any(len(s) > 1 for s in dict_scores.values()):
                # Hard voting case: multiple compressors contribute scores
                distances_values = np.array(
                    [dict_scores[key] for key in class_keys])  # shape: (n_classes, n_compressors)
                min_indices = np.argmin(distances_values, axis=0)  # winner class index per compressor
                winning_classes = [class_keys[idx] for idx in min_indices]

                votes = Counter(winning_classes)
                max_votes = max(votes.values())
                candidates = [cls for cls, count in votes.items() if count == max_votes]
                total_voters = len(winning_classes)  # total number of votes cast

                all_proba = np.zeros(len(class_keys))

                if len(candidates) > 1:
                    # Tie: distribute probabilities based on inverse average distances for tied classes
                    avg_distances = np.array([np.mean(distances_values[class_keys.index(cls)]) for cls in candidates])
                    inv_distances = 1 / (avg_distances + 1e-12)  # avoid division by zero
                    probs = inv_distances / np.sum(inv_distances)  # normalize to probabilities

                    for idx, cls in enumerate(candidates):
                        all_proba[class_keys.index(cls)] = probs[idx]

                else:
                    for cls, count in votes.items():
                        all_proba[class_keys.index(cls)] = count / total_voters

            else:
                # Single compressor case: compute probabilities directly from distances for each class
                distances_values = np.array([dict_scores[cls] for cls in class_keys])
                all_proba = np.apply_along_axis(self._compute_proba, 0, distances_values)

            iteration_output['proba'] = {
                class_keys[index]: all_proba[index].item()
                for index in range(len(class_keys))
            }
            iteration_output['predicted_class'] = class_keys[np.argmax(all_proba)]

            result.append(iteration_output)

        return result

    def __call__(self,
                 list_of_cm: Union[
                     List[
                         Dict[str, np.ndarray]
                     ],
                     Dict[str, np.ndarray]
                 ]) -> List[Dict[str, Any]]:

        if isinstance(list_of_cm, dict):
            list_of_cm = [list_of_cm]

        return self.forward(list_of_cm)