from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_mean_ci(values: Sequence[float], confidence: float = 0.95, n_boot: int = 10000, seed: int = 41):
    x = np.asarray(values, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    alpha = (1 - confidence) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(i > j for i in a for j in b)
    lt = sum(i < j for i in a for j in b)
    return float((gt - lt) / (len(a) * len(b)))


def paired_wilcoxon(proposed: Sequence[float], baseline: Sequence[float]) -> Dict[str, float]:
    a, b = np.asarray(proposed, float), np.asarray(baseline, float)
    if len(a) != len(b):
        raise ValueError("Paired tests require the same seeds/runs.")
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"statistic": float(stat), "p_value": float(p)}


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm family-wise-error correction."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        q = min(1.0, (m - rank) * p)
        running = max(running, q)
        adjusted[name] = running
    return adjusted


def summarize_runs(values: Sequence[float], seed: int = 41) -> Dict[str, float]:
    x = np.asarray(values, float)
    mean, lo, hi = bootstrap_mean_ci(x, seed=seed)
    return {
        "mean": mean,
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_runs": int(len(x)),
    }
