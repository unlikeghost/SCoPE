# -*- coding: utf-8 -*-
"""
    SCoPE
    Compression Matrix
    Jesus Alan Hernandez Galvan
"""
import warnings
import numpy as np
import multiprocessing
from copy import  deepcopy
import concurrent.futures
from itertools import product
from concurrent.futures import as_completed

from typing import Union, List, Dict, Tuple

from .compressors import compute_compression
from .dissimilarity import compute_compression_metric


class CompressionMatrix:
    epsilon: float = 1e-4

    compressors = {
        'bz2': 0,
        'zlib': 1,
        'rle': 2,
        'lz77': 3,
        'gzip': 4,
        'smilez': 5,
        'smaz': 6
    }

    compression_metrics = {
        'ncd': 0,
        'cdm': 1,
        'cd': 2,
        'ucd': 3,
        'ncc': 4,
        'nccd': 5,
        'clm': 6,
    }

    def _validate_compressor_names(self, compressor_names):
        invalid_compressors = [c for c in compressor_names if c not in self.compressors]
        if invalid_compressors:
            raise ValueError(
                f"Invalid compressor(s): {', '.join(invalid_compressors)}. "
                f"Valid options are: {', '.join(self.compressors.keys())}"
            )

    def _validate_metric_names(self, compression_metric_names):
        invalid_metrics = [m for m in compression_metric_names if m not in self.compression_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid compression metric(s): {', '.join(invalid_metrics)}. "
                f"Valid options are: {', '.join(self.compression_metrics)}"
            )

    def _validate_join_str_value(self, join_string):
        if not isinstance(join_string, str):
            raise ValueError(
                f"Invalid join string: {join_string}. join_string must be a string object."
            )

    def _validate_args(self, compressor_names, compression_metric_names, join_string):
        self._validate_compressor_names(compressor_names)
        self._validate_metric_names(compression_metric_names)
        self._validate_join_str_value(join_string)

    def __init__(self,
                 compressor_names: Union[str, List[str]] = 'gzip',
                 compression_metric_names: Union[str, List[str]] = 'ncd',
                 compression_level: int = 9,
                 join_string: str = ' ',
                 # get_sigma: bool = True,
                 n_jobs: int = 1
                 ):

        if isinstance(compressor_names, str):
            compressor_names = [compressor_names]

        if isinstance(compression_metric_names, str):
            compression_metric_names = [compression_metric_names]

        self.compressor_names = set(compressor_names)
        self.compression_metric_names = set(compression_metric_names)
        self._validate_args(compressor_names, compression_metric_names, join_string)
        self.join_string = join_string

        self.compression_level = compression_level

        self._total_compressors = len(self.compressors)
        self._total_metrics = len(self.compression_metrics)

        self._n_compressors = len(self.compressor_names)
        self._n_metrics = len(self.compression_metric_names)

        if n_jobs == -1:
           self.workers = multiprocessing.cpu_count()
        else:
            self.workers = min(multiprocessing.cpu_count(), n_jobs)

    @staticmethod
    def _compute_dissimilarity_metric(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: float, metric: str) -> float:

        _score = compute_compression_metric(
            c_x1=c_x1,
            c_x2=c_x2,
            c_x1x2=c_x1x2,
            c_x2x1=c_x2x1,
            metric=metric
        )

        if _score < 0:
            warnings.warn(
                f"Expected dissimilarity score < 0, but got {_score}"
                f"with metric {metric}",
                category=UserWarning
            )

        return _score

    def __get_compression_size__(self, sequence: Union[str, bytes], compressor: str) -> int:
        if len(sequence) == 0:
            raise ValueError(f"WARNING: Empty sequence for compression with {compressor}")

        compressed_sequence = compute_compression(
            sequence=sequence,
            compressor=compressor,
            compression_level=self.compression_level,
        )
        return len(compressed_sequence)


    def compute_ovo(self, x1: str, x2: str) -> np.ndarray:
        """
        This function returns an array of [C, M]
        C: Compressor
        M: Metric (Current value for x1, x2 where the value is defined by the current metric)
        """
        matrix_values = np.full(
            shape=(
                self._total_compressors,
                self._total_metrics,
            ),
            fill_value=np.nan
        )

        # Some metrics needs x1x2 x2x1, to normalize the value.
        x1x2 = self.join_string.join([x1, x2])
        x2x1 = self.join_string.join([x2, x1])

        sequences = [x1, x2, x1x2, x2x1]

        for compressor in self.compressor_names:
            compressor_index = self.compressors[compressor]

            # here we got the compress length for each sequence, using the current compressor
            compressed_sizes = [
                len(compute_compression(seq, compressor, self.compression_level))
                for seq in sequences
            ]

            c_x1, c_x2, c_x1x2, c_x2x1 = compressed_sizes

            for metric in self.compression_metric_names:
                metric_index = self.compression_metrics[metric]
                _score = self._compute_dissimilarity_metric(
                    c_x1=c_x1,
                    c_x2=c_x2,
                    c_x1x2=c_x1x2,
                    c_x2x1=c_x2x1,
                    metric=metric
                )

                matrix_values[compressor_index, metric_index] = _score

        # Remove nan values. This is just to keep the same order in every run, the same place for compressor, metric
        nan_mask = ~np.isnan(matrix_values)

        result = matrix_values[nan_mask].reshape(
            self._n_compressors,
            self._n_metrics
        )

        return result

    def _ova_threads(self, sample: str, cluster: List[str]) -> Tuple[np.ndarray, np.ndarray]:

        cluster_and_x_ = deepcopy(cluster)
        cluster_and_x_.append(sample)

        combinations = list(product(cluster_and_x_, repeat=2))
        cluster_v_cluster = combinations[:-len(cluster)-1]

        # Shape: 1 unknow sample, n know samples + 1 unknow sample, n compressors, n metrics
        sample_matrix = np.zeros((1, len(cluster) + 1, self._n_compressors, self._n_metrics))

        # Shape: n know sample, n know samples + 1 unknow sample, n compressors, n metrics
        cluster_matrix = np.zeros((len(cluster), len(cluster) + 1, self._n_compressors, self._n_metrics))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Sample comparisons one by one where x1 = unknow sample and x2 = cluster_and_x[index]
            sample_futures = {
                executor.submit(self.compute_ovo, sample, kw_cluster_sample): index
                for index, kw_cluster_sample in enumerate(cluster_and_x_)
            }

            # Sample comparisons one by one where x1 = know sample and x2 = cluster_v_cluster[index]
            cluster_futures = {
                executor.submit(self.compute_ovo, x1, x2): index
                for index, (x1, x2) in enumerate(cluster_v_cluster)
            }

            # Collects data after the thread finished
            for future in as_completed(sample_futures):
                index = sample_futures[future]
                try:
                    sample_matrix[0, index, :, :] = future.result()
                except Exception as e:
                    raise e

            for future in as_completed(cluster_futures):
                index = cluster_futures[future]
                try:
                    kw_sample_index = index % len(cluster)  # current sample
                    kw_sample_vs_index = index // len(cluster)  # current sample vs index sample

                    cluster_matrix[kw_sample_index, kw_sample_vs_index, :, :] = future.result()
                except Exception as e:
                    raise e

        return sample_matrix, cluster_matrix

    def _ova_serie(self, sample: str, cluster: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        cluster_and_x_ = deepcopy(cluster)
        cluster_and_x_.append(sample)

        combinations = list(product(cluster_and_x_, repeat=2))
        cluster_v_cluster = combinations[:-len(cluster) - 1]

        # Shape: 1 unknow sample, n know samples + 1 unknow sample, n compressors, n metrics
        sample_matrix = np.zeros((1, len(cluster) + 1, self._n_compressors, self._n_metrics))

        # Shape: n know sample, n know samples + 1 unknow sample, n compressors, n metrics
        cluster_matrix = np.zeros((len(cluster), len(cluster) + 1, self._n_compressors, self._n_metrics))


        for index, kw_cluster_sample in enumerate(cluster_and_x_):
            sample_matrix[0, index, :, :,] = self.compute_ovo(
                x1=sample,
                x2=kw_cluster_sample
            )

        for index, (x1, x2) in enumerate(cluster_v_cluster):
            kw_sample_index = index % len(cluster) # current sample
            kw_sample_vs_index = index // len(cluster) # current sample vs index sample

            cluster_matrix[kw_sample_index, kw_sample_vs_index, :, :] = self.compute_ovo(
                x1=x1,
                x2=x2
            )

        return sample_matrix, cluster_matrix

    def _compute_ova(self, sample: str, cluster: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
            This functions works as helper if user wants to use threading
        """

        cluster_ = sorted(cluster)
        sigma = float('inf')

        if self.workers > 1:
            sample_matrix,  cluster_matrix = self._ova_threads(sample, cluster_)

        else:
            sample_matrix,  cluster_matrix = self._ova_serie(sample, cluster_)

        # if self.get_sigma:
        #     sigma = self.compute_sigma(samples=cluster)

        # Shape: n_compressors * n_metrics, 1, n_samples + 1
        sample_matrix = np.transpose(sample_matrix, (2, 3, 0, 1)).reshape(self._n_compressors * self._n_metrics, 1, len(cluster) + 1)

        # Shape: n_compressors * n_metrics, n_samples, n_samples + 1
        cluster_matrix = np.transpose(cluster_matrix, (2, 3, 0, 1)).reshape(self._n_compressors * self._n_metrics, len(cluster), len(cluster) + 1)

        return sample_matrix, cluster_matrix, sigma

    def get_one_compression_matrix(self, sample: str, kw_samples: Dict[Union[int, str], List[str]]) ->  Dict[str, np.ndarray]:

        if not isinstance(kw_samples, dict):
            raise ValueError(
                "kw_samples must be a dictionary."
            )

        str_keys = [k for k in kw_samples.keys() if isinstance(k, str)]

        if str_keys:
            warnings.warn(
                "String-type keys were detected in kw_samples. "
                "This may affect evaluation metrics. "
                "It is recommended to encode the keys as integers.",
                UserWarning
            )

        seq_lengths = [len(sample)]
        for v in kw_samples.values():
            if isinstance(v, str):
                seq_lengths.append(len(v))
            elif isinstance(v, list):
                seq_lengths.extend(len(s) for s in v if isinstance(s, str))

        results = {}

        for cluster_key, cluster_values in kw_samples.items():
            sample_matrix, cluster_matrix, sigma = self._compute_ova(
                sample=sample,
                cluster=cluster_values
            )

            results[f"SCoPE_Cluster_{cluster_key}"] = cluster_matrix
            results[f"SCoPE_Sample_{cluster_key}"] = sample_matrix
            results[f'SCoPE_Sigma_{cluster_key}'] = sigma

        return results

    def get_multiple_compression_matrix(self, samples: List[str], kw_samples: List[Dict[Union[int, str], List[str]]]) ->  List[Dict[str, np.ndarray]]:

        if len(samples) != len(kw_samples):
            raise ValueError(
                f"'samples' and 'kw_samples' must have the same length "
                f"(got {len(samples)} and {len(kw_samples)})."
            )

        return [
            self.get_one_compression_matrix(sample, kw_sample)
            for sample, kw_sample in zip(samples, kw_samples)
        ]

    def __call__(self,
                 samples: Union[List[str], str],
                 kw_samples: Union[Dict[Union[int, str], List[str]],
                                   List[Dict[Union[int, str], List[str]]]]) ->  List[Dict[str, np.ndarray]]:

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(kw_samples, dict):
            kw_samples = [kw_samples]

        if len(samples) != len(kw_samples):
            raise ValueError(
                f"'samples' and 'kw_samples' must have the same length "
                f"(got {len(samples)} and {len(kw_samples)})."
            )

        if len(samples) == 1:
            return [self.get_one_compression_matrix(
                sample=samples[0],
                kw_samples=kw_samples[0]
            )]

        return self.get_multiple_compression_matrix(
                samples=samples,
                kw_samples=kw_samples
            )