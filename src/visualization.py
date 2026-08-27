from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve, auc


def save_confusion(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, values_format="d")
    ax.set_title("CSRPM Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_calibration(y_true, prob, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    n_classes = prob.shape[1]
    for k in range(n_classes):
        yk = (np.asarray(y_true) == k).astype(int)
        frac, mean = calibration_curve(yk, prob[:, k], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=f"Class {k}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_summary(table: pd.DataFrame, metric: str, path):
    order = table.groupby("model")[metric].mean().sort_values(ascending=False).index
    mean = table.groupby("model")[metric].mean().reindex(order)
    std = table.groupby("model")[metric].std().reindex(order)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(mean.index, mean.values, yerr=std.values, capsize=3)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
