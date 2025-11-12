import os
import warnings
import pandas as pd
from typing import List, Dict, Optional, Any, Union

import optuna
from optuna.storages import RDBStorage
import optunahub

from .base_ import ScOPEOptimizer
from .params_ import ParameterSpace

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


class ScOPEOptimizerAuto(ScOPEOptimizer):
    """Automatic sampler selection optimization for ScOPE models using AutoSampler."""

    def __init__(self,
                 parameter_space: Optional[ParameterSpace] = None,
                 n_jobs: int = 1,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_auto_optimization",
                 output_path: str = "./results",
                 n_trials: int = 75,
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 ):

        super().__init__(
            parameter_space=parameter_space,
            n_jobs=n_jobs,
            random_seed=random_seed,
            cv_folds=cv_folds,
            study_name=study_name,
            output_path=output_path,
            n_trials=n_trials,
            target_metric=target_metric,
        )

        try:
            self.auto_sampler_module = optunahub.load_module("samplers/auto_sampler")
            print("AutoSampler loaded successfully from OptunaHub")

        except Exception as e:
            print(f"Failed to load AutoSampler: {e}")
            print("Make sure you have installed: pip install optunahub cmaes scipy torch")
            raise ImportError("AutoSampler dependencies not available. Please install: pip install optunahub cmaes scipy torch")


        os.makedirs(self.output_path, exist_ok=True)

        self.storage = RDBStorage(
            f"sqlite:///{self.output_path}/optuna_{self.study_name}.sqlite3",
            engine_kwargs={"connect_args": {"timeout": 20.0}}
        )

    def optimize(self,
                 x_validation: List[str],
                 y_validation: List[str],
                 kw_samples_validation: List[Dict[str, Any]]) -> optuna.Study:
        """Run automatic sampler selection optimization"""

        print("=== AUTO SAMPLER OPTIMIZATION SCOPE ===\n")
        print(f"Validation data: {len(x_validation)} samples")
        print(f"Classes: {sorted(set(y_validation))}")
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.cv_folds}")
        print()

        self.print_target_metric_info()
        print()

        self.print_parameter_space()

        objective_func = self.create_objective_function(
            x_validation, y_validation, kw_samples_validation
        )

        direction = self.get_optimization_direction()

        auto_sampler_kwargs = {
            'seed': self.random_seed,
        }

        self.study = optuna.create_study(
            direction=direction,
            sampler=self.auto_sampler_module.AutoSampler(**auto_sampler_kwargs),
            study_name=f"{self.study_name}_{self.study_date}",
            storage=self.storage,
            load_if_exists=True
        )

        print("\nStarting AutoSampler optimization...")
        print("AutoSampler will automatically switch between algorithms as needed...")

        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )

        self.best_params = self.study.best_params
        self.best_model = self.create_model_from_params(self.best_params)

        print("\n=== AUTO SAMPLER OPTIMIZATION RESULTS ===")
        print(f"Best score: {self.study.best_value:.4f}")

        compressor_names = self.best_params['compressor_names']
        compression_metric_names = self.best_params['compression_metric_names']
        distance_metrics_names = self.best_params['distance_metrics']

        is_ensemble = len(compressor_names) > 1 or len(compression_metric_names) > 1

        print(f"Compressors: {compressor_names}")
        print(f"Compression metrics: {compression_metric_names}")
        print(f"Evaluation metrics: {distance_metrics_names}")

        if is_ensemble:
            print("Ensemble configuration detected")

        aggregation_method = self.best_params.get('aggregation_method')
        if aggregation_method is not None:
            print(f"Aggregation method: {aggregation_method}")

        self._analyze_auto_sampler_performance()

        self.print_best_configuration()

        return self.study

    def _analyze_auto_sampler_performance(self):
        """Analyze AutoSampler's performance and decisions"""
        if not self.study or not self.study.trials:
            return

        print("\nAutoSampler Analysis:")

        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) >= 10:
            values = [t.value for t in completed_trials]

            early_trials = values[:len(values) // 3] if len(values) >= 9 else values[:3]
            late_trials = values[-len(values) // 3:] if len(values) >= 9 else values[-3:]

            if self.get_optimization_direction() == 'maximize':
                early_best = max(early_trials)
                late_best = max(late_trials)
                improvement = late_best - early_best
            else:
                early_best = min(early_trials)
                late_best = min(late_trials)
                improvement = early_best - late_best

            print(f"   Early stage best: {early_best:.4f}")
            print(f"   Late stage best: {late_best:.4f}")
            print(f"   Improvement: {improvement:.4f}")

            if len(values) >= 20:
                recent_std = pd.Series(values[-10:]).std()
                print(f"   Recent convergence (std): {recent_std:.4f}")

    def analyze_results(self):
        """Analyze AutoSampler optimization results"""

        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        print("\n=== DETAILED AUTO SAMPLER ANALYSIS ===")

        completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])

        print(f"Completed trials: {completed_trials}")
        print(f"Pruned trials: {pruned_trials}")
        print(f"Failed trials: {failed_trials}")

        print("\nAutoSampler Configuration:")
        print(f"  • Sampler: {type(self.study.sampler).__name__}")
        print("  • Adaptive strategy: Automatic (GPSampler + TPESampler)")

        if len(self.study.trials) >= 10:
            values = [t.value for t in self.study.trials if t.value is not None]
            if values:
                print("\nPerformance Evolution:")

                phase_size = len(values) // 3
                if phase_size > 0:
                    phase1 = values[:phase_size]
                    phase2 = values[phase_size:2 * phase_size]
                    phase3 = values[2 * phase_size:]

                    print(f"  • Phase 1 (Exploration) avg: {sum(phase1) / len(phase1):.4f}")
                    print(f"  • Phase 2 (Exploitation) avg: {sum(phase2) / len(phase2):.4f}")
                    print(f"  • Phase 3 (Convergence) avg: {sum(phase3) / len(phase3):.4f}")

                    if self.get_optimization_direction() == 'maximize':
                        print(f"  • Best in phase 1: {max(phase1):.4f}")
                        print(f"  • Best in phase 2: {max(phase2):.4f}")
                        print(f"  • Best in phase 3: {max(phase3):.4f}")
                    else:
                        print(f"  • Best in phase 1: {min(phase1):.4f}")
                        print(f"  • Best in phase 2: {min(phase2):.4f}")
                        print(f"  • Best in phase 3: {min(phase3):.4f}")

        df_results = self.study.trials_dataframe()
        if not df_results.empty:
            print(f"\nTotal unique parameter combinations evaluated: {len(df_results)}")

            ensemble_trials = []
            individual_trials = []

            for idx, row in df_results.iterrows():
                compressor_names = str(row.get('params_compressor_names', '')).split(',')
                compression_metric_names = str(row.get('params_compression_metric_names', '')).split(',')

                is_ensemble = len(compressor_names) > 1 or len(compression_metric_names) > 1

                if is_ensemble:
                    ensemble_trials.append(row)
                else:
                    individual_trials.append(row)

            print("\nEnsemble vs Individual Analysis:")
            print(f"  Ensemble trials: {len(ensemble_trials)}")
            print(f"  Individual trials: {len(individual_trials)}")

            if ensemble_trials:
                ensemble_df = pd.DataFrame(ensemble_trials)
                print(f"  Best ensemble score: {ensemble_df['value'].max():.4f}")

            if individual_trials:
                individual_df = pd.DataFrame(individual_trials)
                print(f"  Best individual score: {individual_df['value'].max():.4f}")

        try:
            importance = optuna.importance.get_param_importances(self.study)
            print("\nParameter importance (AutoSampler insights):")
            print("-" * 50)

            basic_params = {}

            if basic_params:
                print("Basic Parameters:")
                for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()

            print("Overall Ranking (Top 10 - AutoSampler prioritized):")
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for param, importance in top_params:
                print(f"  {param}: {importance:.6f}")

        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")

        return df_results

    def save_analysis_report(self, filename: Optional[str] = None):
        """Save detailed AutoSampler analysis report to a text file"""

        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        if filename is None:
            filename = f"{self.study_name}_auto_analysis_{self.study_date}.txt"

        filepath = os.path.join(self.output_path, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCOPE AUTOSAMPLER OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("STUDY INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Study name: {self.study_name}\n")
            f.write(f"Study date: {self.study_date}\n")
            f.write(f"Output path: {self.output_path}\n")
            f.write("Sampler: AutoSampler (adaptive algorithm selection)\n\n")

            f.write("AUTOSAMPLER CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write("Adaptive strategy: GPSampler (early) + TPESampler (categorical) + dynamic switching\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write("Automatic algorithm selection: Enabled\n\n")

            f.write("OPTIMIZATION CONFIGURATION:\n")
            f.write("-" * 30 + "\n")

            if self.is_combined:
                f.write("Target metric: Combined\n")
                f.write("Metric weights:\n")
                for metric, weight in self.target_metric_weights.items():
                    f.write(f"  {metric}: {weight:.3f}\n")
            else:
                f.write(f"Target metric: {self.target_metric_name}\n")

            f.write(f"Optimization direction: {self.get_optimization_direction()}\n")
            f.write(f"Number of trials: {self.n_trials}\n")
            f.write(f"CV folds: {self.cv_folds}\n")
            f.write(f"Best score achieved: {self.study.best_value:.6f}\n\n")

            f.write("AUTOSAMPLER STRATEGY:\n")
            f.write("-" * 30 + "\n")

            f.write("PARAMETER SPACE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Compressor combinations: {len(self.parameter_space.compressor_names_options)}\n")
            f.write(f"Compression metric combinations: {len(self.parameter_space.compression_metric_names_options)}\n")
            f.write(f"Join string options: {self.parameter_space.concat_value_options}\n")
            f.write(f"Model evaluation metrics: {self.parameter_space.distance_metrics_options}\n")
            f.write(f"Aggregation methods: {self.parameter_space.prototype_method_options}\n")

        print(f"AutoSampler analysis report saved to {filepath}")

    def save_complete_analysis(self):
        """Save only essential analysis files - SQLite has the full data"""

        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")

        self.save_results()
        self.save_analysis_report()

        print("\nAnalysis saved - no data duplication:")
        print(f"Directory: {self.output_path}/")
        print(f"  optuna_{self.study_name}.sqlite3 (FULL DATA - use optuna-dashboard)")
        print(f"  {self.study_name}_metadata_{self.study_date}.pkl (config only)")
        print(f"  {self.study_name}_analysis_{self.study_date}.txt (human readable report)")
        print(f"\nTo view results: optuna-dashboard sqlite:///{self.output_path}/optuna_{self.study_name}.sqlite3")

