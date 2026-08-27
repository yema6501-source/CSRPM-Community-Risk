from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from .uncertainty import (
    expected_calibration_error,
    multiclass_brier,
    negative_log_likelihood,
    predictive_entropy,
)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, int)
    prob = np.asarray(prob, float)
    pred = prob.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "ece": expected_calibration_error(y_true, prob),
        "brier": multiclass_brier(y_true, prob),
        "nll": negative_log_likelihood(y_true, prob),
        "mean_predictive_entropy": float(predictive_entropy(prob).mean()),
    }
    one_hot = np.eye(prob.shape[1])[y_true]
    try:
        out["roc_auc_ovr_macro"] = roc_auc_score(one_hot, prob, multi_class="ovr", average="macro")
    except ValueError:
        out["roc_auc_ovr_macro"] = np.nan
    try:
        out["pr_auc_macro"] = average_precision_score(one_hot, prob, average="macro")
    except ValueError:
        out["pr_auc_macro"] = np.nan
    return {k: float(v) for k, v in out.items()}
