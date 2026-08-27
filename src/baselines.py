from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def classical_baselines(seed: int = 41) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.035, max_depth=3, random_state=seed
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3,
            max_iter=600, early_stopping=True, random_state=seed
        ),
    }


def fit_predict_classical(model, X_train, y_train, X_eval):
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_eval)
    else:
        decision = model.decision_function(X_eval)
        decision = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(decision)
        prob = exp / exp.sum(axis=1, keepdims=True)
    return pred, prob
