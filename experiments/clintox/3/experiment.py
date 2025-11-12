import os
import random
import tomllib
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from collections import OrderedDict

from scope.optimization import ParameterSpace, all_subsets, ScOPEOptimizerAuto
from scope.utils import make_report, SampleGenerator


def setup_parameters():
    """Setup parameter space for optimization"""
    compressors = all_subsets(
        ['gzip', 'bz2', 'smilez', 'lz77']
    )

    dissimilarity_metrics = all_subsets(
        ['ncd', 'cdm', 'clm']
    )

    distance_metrics = all_subsets(
        ['euclidean', 'wasserstein', 'cosine']
    )

    prototype_method = ['mean', 'gmean', 'median', '']

    params = ParameterSpace(
        compressor_names_options=compressors,
        compression_metric_names_options=dissimilarity_metrics,
        concat_value_options=[' '],
        prototype_method_options=prototype_method,
        distance_metrics_options=distance_metrics

    )

    return params


def load_config():
    """Load and process configuration from TOML file"""
    with open("settings.toml", "rb") as f:
        config = tomllib.load(f)

    # Extract config values
    TEST_SAMPLES = config["experiment"]["test_samples"]
    TRIALS = config["experiment"]["trials"]
    CVFOLDS = config["experiment"]["cvfolds"]
    TARGET_METRIC = config["experiment"]["target_metric"]
    STUDY_NAME = f"{config["experiment"]["study_name"]}"
    RANDOM_SEED = config["experiment"]["random_seed"]

    SMILES_COLUMN = config["dataset"]["smiles_column"]
    LABEL_COLUMN = config["dataset"]["label_column"]

    RESULTS_PATH = config["paths"]["results_path"]
    ANALYSIS_RESULTS_PATH = RESULTS_PATH
    EVALUATION_RESULTS_PATH = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), "Evaluation")

    # Set random seeds
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    return {
        'TEST_SAMPLES': TEST_SAMPLES,
        'TRIALS': TRIALS,
        'CVFOLDS': CVFOLDS,
        'TARGET_METRIC': TARGET_METRIC,
        'STUDY_NAME': STUDY_NAME,
        'RANDOM_SEED': RANDOM_SEED,
        'SMILES_COLUMN': SMILES_COLUMN,
        'LABEL_COLUMN': LABEL_COLUMN,
        'RESULTS_PATH': RESULTS_PATH,
        'ANALYSIS_RESULTS_PATH': ANALYSIS_RESULTS_PATH,
        'EVALUATION_RESULTS_PATH': EVALUATION_RESULTS_PATH
    }


def preprocess_smiles(smiles: str, min_str_length: int = 10) -> str | None:
    """Canonicalize and clean a SMILES string. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None


        representations = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )

        if len(representations) < min_str_length:
            return None

        return representations

    except:
        return None


def load_and_process_datasets(min_str_length: int = 10):
    """Load Clintox dataset and process into DataFrames"""
    # Load Clintox dataset (raw SMILES)
    tasks, datasets, _ = dc.molnet.load_clintox(
        splitter="stratified", reload=True, featurizer=dc.feat.DummyFeaturizer()
    )
    train_dataset, valid_dataset, test_dataset = datasets

    search_dataset = dc.data.DiskDataset.merge([valid_dataset, test_dataset])

    # Convert to DataFrames with preprocessing
    def build_dataframe(dataset):
        smiles_clean = [
            preprocess_smiles(s, min_str_length=min_str_length)
            for s in dataset.X
        ]
        mask = [s is not None for s in smiles_clean]  # drop invalids
        return pd.DataFrame({
            "smiles": np.array(smiles_clean)[mask],
            "fda_approved": dataset.y[mask, 0].astype(int),
            "ct_tox": dataset.y[mask, 1].astype(int)
        })

    df_test = build_dataframe(test_dataset)
    df_search = build_dataframe(search_dataset)

    return df_test, df_search

def main():

    # Setup parameters and configuration
    params = setup_parameters()
    config = load_config()
    TEST_SAMPLES = config['TEST_SAMPLES']
    TRIALS = config['TRIALS']
    CVFOLDS = config['CVFOLDS']
    TARGET_METRIC = config['TARGET_METRIC']
    STUDY_NAME = config['STUDY_NAME']
    RANDOM_SEED = config['RANDOM_SEED']
    SMILES_COLUMN = config['SMILES_COLUMN']
    LABEL_COLUMN = config['LABEL_COLUMN']
    RESULTS_PATH = config['RESULTS_PATH']
    ANALYSIS_RESULTS_PATH = config['ANALYSIS_RESULTS_PATH']
    EVALUATION_RESULTS_PATH = config['EVALUATION_RESULTS_PATH']

    # Load and process datasets
    df_test, df_search = load_and_process_datasets(min_str_length=10)

    print(df_test.shape, df_search.shape)
    x_test, y_test = df_test[SMILES_COLUMN].values, df_test[LABEL_COLUMN].values
    x_search, y_search = df_search[SMILES_COLUMN].values, df_search[LABEL_COLUMN].values

    search_generator = SampleGenerator(
        data=x_search,
        labels=y_search,
        seed=RANDOM_SEED,
    )

    optimizer = ScOPEOptimizerAuto(
        n_jobs=1,
        n_trials=TRIALS,
        random_seed=RANDOM_SEED,
        target_metric=TARGET_METRIC,
        study_name=STUDY_NAME,
        output_path=ANALYSIS_RESULTS_PATH,
        cv_folds=CVFOLDS,
        parameter_space=params
    )

    search_all_x = []
    search_all_y = []
    search_all_kw = []

    for x_search_i, y_search_i, search_kw_samples_i in search_generator.generate(num_samples=TEST_SAMPLES):
        search_all_x.append(x_search_i)
        search_all_y.append(y_search_i)
        search_all_kw.append(search_kw_samples_i)

    study = optimizer.optimize(
        search_all_x,
        search_all_y,
        search_all_kw
    )

    optimizer.save_complete_analysis()

    best_model = optimizer.get_best_model()

    test_generator = SampleGenerator(
        data=x_test,
        labels=y_test,
        seed=RANDOM_SEED,
    )

    test_all_y_true = []
    test_all_y_predicted = []
    test_all_y_proba = []

    for x_test_i, y_test_i, test_kw_samples_i in test_generator.generate(num_samples=TEST_SAMPLES):
        pred_x = best_model(
            samples=x_test_i,
            kw_samples=test_kw_samples_i
        )[0]

        prediction: dict = pred_x['proba']

        sorted_dict = OrderedDict(sorted(prediction.items()))

        pred_key = max(sorted_dict, key=sorted_dict.get)

        predicted_class = int(pred_key.replace("sample_", ""))

        test_all_y_predicted.append(
            predicted_class
        )

        test_all_y_proba.append(
            list(sorted_dict.values())
        )

        test_all_y_true.append(
            y_test_i
        )

    results = make_report(
        y_true=test_all_y_true,
        y_pred=test_all_y_predicted,
        y_pred_proba=test_all_y_proba,
        save_path=EVALUATION_RESULTS_PATH
    )

    print(results)


if __name__ == "__main__":
    main()