import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict


from ..model import SCoPE
from ..utils.report_generation import make_report


def evaluate_single_fold(fold_data: Tuple, model_params: Dict, class_to_idx: Dict, unique_classes: List) -> Dict[str, float]:
    """
        Evaluate a single fold in parallel - handles batch processing.
        This function will be executed in a separate process/thread.
    """
    fold_idx, x_val, y_val, kw_val = fold_data
    model = SCoPE(**model_params)
    fold_predictions = []
    fold_proba = []
    predictions_batch = model(samples=x_val, kw_samples=kw_val)
    try:

        for i, predictions in enumerate(predictions_batch):
            try:
                if isinstance(predictions, dict):
                    prediction = predictions
                elif isinstance(predictions, list) and len(predictions) > 0:
                    prediction = predictions[0]
                else:
                    raise ValueError("Unexpected prediction format")

                predicted_class_str = prediction.get('predicted_class', '0')
                try:
                    predicted_class = int(predicted_class_str)
                except (ValueError, TypeError):
                    predicted_class = unique_classes[0]

                probs: dict = prediction.get('proba', {})
                probs = OrderedDict(sorted(probs.items()))
                proba_values = list(probs.values())

                if len(proba_values) != 2:
                    raise ValueError(f"Expected 2 class probabilities, got {len(proba_values)}")

                fold_predictions.append(predicted_class)
                fold_proba.append(proba_values)

            except Exception as e:
                print(f"Warning: Prediction failed for fold {fold_idx}, sample {i}: {e}")
                fold_predictions.append(unique_classes[0])
                fold_proba.append([0.5, 0.5])

        # Process results
        y_val_numeric = [class_to_idx[y] for y in y_val[:len(fold_predictions)]]
        y_pred_numeric = [class_to_idx.get(pred, 0) for pred in fold_predictions]
        y_pred_proba_array = np.array(fold_proba)

        fold_scores = {
            'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1': 0.0, 'f2': 0.0,
            'auc_roc': 0.5, 'auc_pr': 0.5, 'log_loss': 1.0, 'mcc': -1.0
        }

        if len(set(y_pred_numeric)) > 1 and len(set(y_val_numeric)) > 1:
            try:
                report = make_report(y_val_numeric, y_pred_numeric, y_pred_proba_array)
                fold_scores = {
                    'accuracy': report['accuracy'],
                    'balanced_accuracy': report['balanced_accuracy'],
                    'f1': report['f1'],
                    'f2': report['f2'],
                    'auc_roc': report['auc_roc'],
                    'auc_pr': report['auc_pr'],
                    'log_loss': report['log_loss'],
                    'mcc': report['mcc']
                }
            except Exception as e:
                print(f"Warning: Metric computation failed for fold {fold_idx}: {e}")

        return fold_scores

    except Exception as e:
        print(f"Error in fold {fold_idx}: {e}")
        return {
            'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1': 0.0, 'f2': 0.0,
            'auc_roc': 0.5, 'auc_pr': 0.5, 'log_loss': 1.0, 'mcc': -1.0
        }