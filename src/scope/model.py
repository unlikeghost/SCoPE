# -*- coding: utf-8 -*-
"""
    SCoPE
    SCoPE Model
    Jesus Alan Hernandez Galvan
"""
import numpy as np
from typing import Union, List, Optional, Dict, Any

from .compression import CompressionMatrix
from .predictor import _BasePredictor, SCoPEPredictorV1


class SCoPE:

    def to_dict(self) -> dict:
        params = {
            'compressor_names': self._compressor_names,
            'compression_metric_names': self._compression_metric_names,
            'compression_level': self._compression_level,
            'join_string': self._join_string,
            'distance_metrics': self._distance_metrics,
            'prototype_method': self._prototype_method,
        }

        return params

    def __init__(self,
                 distance_metrics: str = 'euclidean',
                 prototype_method: Optional[str] = None,
                 compressor_names: Union[str, List[str]] = 'gzip',
                 compression_metric_names: Union[str, List[str]] = 'ncd',
                 compression_level: int = 9,
                 join_string: str = '',
                 n_jobs: int = 2,
                 ):

        self.predictor: _BasePredictor = SCoPEPredictorV1(
            distance_metrics=distance_metrics,
            prototype_method=prototype_method,
        )

        self.compression_matrix: CompressionMatrix = CompressionMatrix(
            compressor_names=compressor_names,
            compression_metric_names=compression_metric_names,
            compression_level=compression_level,
            join_string=join_string,
            n_jobs=n_jobs
        )

        self._distance_metrics = distance_metrics
        self._prototype_method = prototype_method

        self._compressor_names = compressor_names
        self._compression_metric_names = compression_metric_names
        self._compression_level = compression_level
        self._join_string = join_string

    def _pre_forward(self,
                        samples: Union[List[str], str],
                        kw_samples: Union[Dict[Union[int, str], List[str]],
                                   List[Dict[Union[int, str], List[str]]]]) -> List[Dict[str, np.ndarray]]:

        list_of_cm = self.compression_matrix(
            samples=samples,
            kw_samples=kw_samples
        )
        return list_of_cm

    def forward(self, list_of_cm: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:

        if len(list_of_cm) == 0:
            raise ValueError("list_of_cm must not be empty.")

        predictions = self.predictor(list_of_cm=list_of_cm)

        return predictions

    def __call__(self, samples: Union[List[str], str],
                 kw_samples: Union[Dict[Union[int, str], List[str]],
                 List[Dict[Union[int, str], List[str]]]]) -> List[Dict[str, np.ndarray]]:

        list_of_cm = self._pre_forward(
            samples=samples,
            kw_samples=kw_samples
        )

        return self.forward(list_of_cm=list_of_cm)