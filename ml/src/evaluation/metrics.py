"""
Evaluation metrics module.
Computes accuracy, precision, recall, F1 (macro and per-class).
"""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: Optional[list[str]] = None,
    average: str = 'macro',
) -> dict:
    """
    Compute comprehensive classification metrics.

    Returns:
        dict with accuracy, macro/weighted F1, precision, recall,
        per-class metrics, and classification report.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'weighted_recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, f1_val in enumerate(per_class_f1):
        name = class_names[i] if class_names and i < len(class_names) else str(i)
        per_class[name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(f1_val),
            'support': int(np.sum(y_true == i)),
        }

    metrics['per_class'] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report string
    if class_names:
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0,
        )
    else:
        report = classification_report(y_true, y_pred, zero_division=0)
    metrics['classification_report'] = report

    return metrics


def compute_top_k_accuracy(
    probabilities: list[list[float]],
    y_true: list[int],
    k: int = 3,
) -> float:
    """Compute top-k accuracy from probability distributions."""
    correct = 0
    for probs, true_label in zip(probabilities, y_true):
        top_k_preds = np.argsort(probs)[-k:]
        if true_label in top_k_preds:
            correct += 1
    return correct / len(y_true) if y_true else 0.0
