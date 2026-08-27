from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import brier_score_loss


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def predictive_entropy(probabilities: np.ndarray, normalized: bool = True, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, float), eps, 1.0)
    h = -(p * np.log(p)).sum(axis=-1)
    if normalized:
        h = h / np.log(p.shape[-1])
    return h


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    p = np.asarray(prob)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correctness = (prediction == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(correctness[m].mean() - confidence[m].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, float)
    one_hot = np.eye(p.shape[1])[y_true]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def negative_log_likelihood(y_true: np.ndarray, prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(prob, float), eps, 1.0)
    return float(-np.log(p[np.arange(len(y_true)), np.asarray(y_true, int)]).mean())


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, adjacency: torch.Tensor, passes: int = 30) -> Dict[str, np.ndarray]:
    """Monte-Carlo dropout decomposition into entropy and mutual-information uncertainty."""
    model.train()
    samples = []
    for _ in range(passes):
        logits = model(x, adjacency)
        samples.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    stack = np.stack(samples, axis=0)
    mean_prob = stack.mean(axis=0)
    pred_ent = predictive_entropy(mean_prob, normalized=False)
    expected_ent = predictive_entropy(stack, normalized=False).mean(axis=0)
    mutual_info = pred_ent - expected_ent
    return {
        "mean_probability": mean_prob,
        "predictive_entropy": pred_ent,
        "expected_entropy": expected_ent,
        "mutual_information": mutual_info,
    }


class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)
        criterion = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
