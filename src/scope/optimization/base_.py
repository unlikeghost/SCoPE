import os
import pickle
import optuna
import numpy as np
from math import prod
from optuna.storages import RDBStorage
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..model import SCoPE
from .params_ import ParameterSpace
from .eval_ import evaluate_single_fold


class ScOPEOptimizer(ABC):
    """Updated ScOPE optimizer for the unified ScOPE class."""

    def __init__(self,
                 parameter_space: Optional[ParameterSpace] = None,
                 n_jobs: int = 1,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_optimization",
                 output_path: str = "./results",
                 n_trials: int = 50,
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 ):

        self.parameter_space = parameter_space or ParameterSpace()

        self.n_jobs = n_jobs
        self.random_seed: int = random_seed
        self.cv_folds = cv_folds
        self.study_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_name = study_name
        self.output_path = output_path

        # Store results
        self.study = None
        self.best_params = None
        self.best_model = None

        self.n_trials = n_trials

        if isinstance(target_metric, str):
            self.target_metric_name = target_metric
            self.target_metric_weights = None
            self.is_combined = False

        elif isinstance(target_metric, dict):
            self.target_metric_name = 'combined'
            self.target_metric_weights = target_metric
            self.is_combined = True

            total = sum(target_metric.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Metric weights must sum to 1.0, got {total}")
        else:
            raise ValueError("target_metric must be str or dict")

    def print_parameter_space(self):
        """Print detailed parameter space information."""
        print("Parameter space includes:")
        print("=" * 60)

        compressor_count = len(self.parameter_space.compressor_names_options)
        compression_metric_count = len(self.parameter_space.compression_metric_names_options)
        concat_value_count = len(self.parameter_space.concat_value_options)
        prototype_method_count = len(self.parameter_space.prototype_method_options)
        distance_metrics_count = len(self.parameter_space.distance_metrics_options)

        print("BASIC PARAMETERS:")
        print(f"  • Compressor combinations ({compressor_count})")
        print(f"  • Compression metric combinations ({compression_metric_count})")
        print(
            f"  • Join string options ({concat_value_count}): {[repr(s) for s in self.parameter_space.concat_value_options]}")
        print(
            f"  • Aggregation methods ({prototype_method_count}): {self.parameter_space.prototype_method_options}")
        print(f"  • Evaluation Metrics ({distance_metrics_count})")

        total_space = prod([
            compressor_count,
            compression_metric_count,
            concat_value_count,
            prototype_method_count,
            distance_metrics_count,
        ])

        print("=" * 60)
        print(f"Total size of search space: {total_space}")
        print("=" * 60)

    @staticmethod
    def create_model_from_params(params: Dict[str, Any]) -> SCoPE:
        """Create a SCoPE model based on parameters."""

        compressor_names = params['compressor_names']
        if isinstance(compressor_names, str):
            compressor_names = compressor_names.split(',')

        compression_metric_names = params['compression_metric_names']
        if isinstance(compression_metric_names, str):
            compression_metric_names = compression_metric_names.split(',')

        evaluation_metric_names = params['distance_metrics']
        if isinstance(evaluation_metric_names, str):
            evaluation_metric_names = evaluation_metric_names.split(',')

        prototype_method = params.get('prototype_method')
        if prototype_method == 'None' or prototype_method == 'null' or prototype_method == '':
            prototype_method = None

        base_params = {
            'prototype_method': prototype_method,
            'distance_metrics': evaluation_metric_names,
            'compressor_names': compressor_names,
            'compression_metric_names': compression_metric_names,
            'join_string': params['join_string'],
            'n_jobs': 2
        }

        return SCoPE(**base_params)

    @abstractmethod
    def optimize(self,
                 x_validation: List[str],
                 y_validation: List[str],
                 kw_samples_validation: List[Dict[str, Any]]) -> Any:
        """Main optimization method - must be implemented by subclasses."""
        pass

    @abstractmethod
    def analyze_results(self) -> Any:
        """Results analysis - must be implemented by subclasses."""
        pass

    def suggest_categorical_params(self, trial) -> Dict[str, Any]:
        """Suggest categorical parameters."""
        prototype_method = trial.suggest_categorical(
            'prototype_method',
            self.parameter_space.prototype_method_options
        )

        compressor_choices = [','.join(combo) for combo in self.parameter_space.compressor_names_options]
        metric_choices = [','.join(combo) for combo in self.parameter_space.compression_metric_names_options]
        eval_metric_choices = [','.join(combo) for combo in self.parameter_space.distance_metrics_options]

        return {
            'join_string': trial.suggest_categorical(
                'join_string',
                self.parameter_space.concat_value_options
            ),
            'distance_metrics': trial.suggest_categorical(
                'distance_metrics',
                eval_metric_choices
            ),
            'prototype_method': None if prototype_method == '' else prototype_method,
            'compressor_names': trial.suggest_categorical(
                'compressor_names',
                compressor_choices
            ),
            'compression_metric_names': trial.suggest_categorical(
                'compression_metric_names',
                metric_choices
            )

        }

    # def suggest_boolean_params(self, trial) -> Dict[str, Any]:
    #     """Suggest boolean parameters."""
    #     return {}
    #
    # def suggest_integer_params(self, trial) -> Dict[str, Any]:
    #     """Suggest integer parameters using ranges."""
    #     return {}

    def suggest_all_params(self, trial) -> Dict[str, Any]:
        """Combine all parameter suggestions."""
        params = {}

        params.update(self.suggest_categorical_params(trial))
        # params.update(self.suggest_boolean_params(trial))
        # params.update(self.suggest_integer_params(trial))

        return params

    def get_optimization_direction(self) -> str:
        """Determine optimization direction for Optuna."""
        if self.is_combined:
            return 'maximize'
        elif self.target_metric_name == 'log_loss':
            return 'minimize'
        elif self.target_metric_name == 'mcc':
            return 'maximize'
        else:
            return 'maximize'

    def print_best_configuration(self):
        """Print best configuration."""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")

        print("Best configuration:")
        for param, value in self.best_params.items():
            if param == 'join_string':
                value = repr(value)
            print(f"  {param}: {value}")

    def print_target_metric_info(self):
        """Print information about the configured target metric."""
        print("Target metric configuration:")
        if self.is_combined:
            print("  Type: Combined metric")
            print("  Weights:")
            for metric, weight in self.target_metric_weights.items():
                print(f"    {metric}: {weight:.3f}")
            print(f"  Optimization direction: {self.get_optimization_direction()}")
        else:
            print("  Type: Single metric")
            print(f"  Metric: {self.target_metric_name}")
            print(f"  Optimization direction: {self.get_optimization_direction()}")

    def get_best_model(self) -> SCoPE:
        """Get the best optimized model."""
        if self.best_model is None:
            raise ValueError("No optimized model found. Run optimize() first.")
        return self.best_model

    def calculate_objective_score(self, scores: Dict[str, float]) -> float:
        """Calculate objective score."""

        if self.is_combined:
            combined_score = 0.0
            for metric, weight in self.target_metric_weights.items():
                if metric in scores:
                    if metric == 'log_loss':
                        combined_score += (1 - scores[metric]) * weight
                    elif metric == 'mcc':
                        normalized_score = max(scores[metric], 0.0)
                        combined_score += normalized_score * weight
                    else:
                        combined_score += scores[metric] * weight

            return combined_score

        elif self.target_metric_name == 'log_loss':
            return scores['log_loss']

        elif self.target_metric_name == 'mcc':
            return max(scores['mcc'], 0.0)

        else:
            return scores[self.target_metric_name]

    def create_objective_function(self,
                                   x_validation: List[str],
                                   y_validation: List[str],
                                   kw_samples_validation: List[Dict[str, Any]]):
        """Objective function for Optuna."""

        def objective(trial):
            params = self.suggest_all_params(trial)

            study_trials = trial.study.get_trials(deepcopy=False)
            for t in study_trials:
                if t.state and t.params == params:
                    print(f"Skipping duplicate trial with params: {params}")
                    return t.value

            try:
                model = self.create_model_from_params(params)

                scores = self.evaluate_model(
                    model=model,
                    x_samples=x_validation,
                    y_true=y_validation,
                    kw_samples_list=kw_samples_validation
                )

                return self.calculate_objective_score(scores)

            except Exception as e:
                print(f"Error in trial {trial.number}: {e}")
                print(f"Trial params: {params if 'params' in locals() else 'Not available'}")
                import traceback
                traceback.print_exc()
                return 0.0 if self.get_optimization_direction() == 'maximize' else 10.0

        return objective

    def evaluate_model(self,
                       model: SCoPE,
                       x_samples: List[str],
                       y_true: List[str],
                       kw_samples_list: List[Dict[str, Any]]
                       ) -> Dict[str, float]:

        indices = np.arange(len(x_samples))
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        unique_classes = sorted(list(set(y_true)))
        if len(unique_classes) != 2:
            raise ValueError(f"Expected exactly 2 classes, but found {len(unique_classes)}: {unique_classes}")

        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        model_params = model.to_dict() if hasattr(model, "to_dict") else {}
        fold_data_list = []
        for fold_idx, (_, val_idx) in enumerate(skf.split(indices, y_true)):
            x_val = [x_samples[i] for i in val_idx]
            y_val = [y_true[i] for i in val_idx]
            kw_val = [kw_samples_list[i] for i in val_idx]
            fold_data_list.append((fold_idx, x_val, y_val, kw_val))

        results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(evaluate_single_fold, fold_data, model_params, class_to_idx, unique_classes):
                    fold_data[0]
                for fold_data in fold_data_list
            }
            for future in as_completed(futures):
                fold_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Fold {fold_idx} failed: {e}")
                    results.append({
                        'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1': 0.0, 'f2': 0.0,
                        'auc_roc': 0.5, 'auc_pr': 0.5, 'log_loss': 1.0, 'mcc': -1.0
                    })

        final_scores = {
            metric: np.mean([res[metric] for res in results]).item()
            for metric in results[0]
        }

        return final_scores

    def load_results(self, study_name: Optional[str] = None):
        """Load previous results from the SQLite database."""

        if study_name:
            storage = RDBStorage(f"sqlite:///{self.output_path}/optuna_{study_name}.sqlite3")
            target_study_name = study_name
        else:
            # Assume storage is already set up in a subclass
            storage = getattr(self, 'storage', None)
            target_study_name = self.study_name

            if storage is None:
                raise ValueError("No storage configured. Cannot load results.")

        try:
            self.study = optuna.load_study(
                study_name=target_study_name,
                storage=storage
            )

            self.best_params = self.study.best_params
            self.best_model = self.create_model_from_params(self.best_params)

            print(f"Study loaded from SQLite: {len(self.study.trials)} trials")
            print(f"Best value: {self.study.best_value}")

        except Exception as e:
            raise ValueError(f"Could not load study: {e}")

    def save_results(self, filename: Optional[str] = None):
        """Save only metadata - SQLite has the full study data."""
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        if filename is None:
            filename = f"{self.study_name}_metadata_{self.study_date}.pkl"

        os.makedirs(self.output_path, exist_ok=True)
        filepath = os.path.join(self.output_path, filename)

        # Only save metadata
        results = {
            'study_name': self.study_name,
            'study_date': self.study_date,
            'n_trials_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'target_metric_info': {
                'is_combined': self.is_combined,
                'target_metric_name': self.target_metric_name,
                'target_metric_weights': self.target_metric_weights
            },
            'parameter_space_config': {
                'compressor_names_options': self.parameter_space.compressor_names_options,
                'compression_metric_names_options': self.parameter_space.compression_metric_names_options,
                'model_distance_metrics': self.parameter_space.distance_metrics_options
            },
            'sqlite_path': f"{self.output_path}/optuna_{self.study_name}.sqlite3"
        }

        with open(filepath, 'wb') as f:
            pickle.dump(results, f) # noqa

        print(f"Metadata saved to {filepath}")
        print(f"Full study data in SQLite: {results['sqlite_path']}")

