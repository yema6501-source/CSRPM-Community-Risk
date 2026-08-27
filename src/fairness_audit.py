from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def subgroup_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: Sequence,
    high_risk_class: int = 2,
) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    for g in pd.unique(group):
        m = group == g
        yt, yp = y_true[m], y_pred[m]
        if len(yt) == 0:
            continue
        true_pos = ((yt == high_risk_class) & (yp == high_risk_class)).sum()
        false_neg = ((yt == high_risk_class) & (yp != high_risk_class)).sum()
        false_pos = ((yt != high_risk_class) & (yp == high_risk_class)).sum()
        true_neg = ((yt != high_risk_class) & (yp != high_risk_class)).sum()
        rows.append({
            "group": g,
            "n": int(len(yt)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "high_risk_tpr": _safe_rate(true_pos, true_pos + false_neg),
            "high_risk_fpr": _safe_rate(false_pos, false_pos + true_neg),
            "high_risk_fnr": _safe_rate(false_neg, true_pos + false_neg),
        })
    return pd.DataFrame(rows)


def vulnerability_quartiles(values: Sequence[float]) -> np.ndarray:
    s = pd.Series(values)
    return pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str).to_numpy()
