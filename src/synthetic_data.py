from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_communities: int = 12
    n_days: int = 720
    seed: int = 41
    missing_rate: float = 0.03
    shock_probability: float = 0.015
    start_date: str = "2023-01-01"


FEATURES = [
    "sentiment_negativity",
    "mobility_variance",
    "service_pressure",
    "air_quality_stress",
    "temperature_anomaly",
    "unemployment_proxy",
    "poverty_proxy",
    "digital_activity",
    "community_engagement",
    "infrastructure_reports",
]


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float, baseline: float) -> np.ndarray:
    x = np.empty(n, dtype=float)
    x[0] = baseline + rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = baseline + phi * (x[t - 1] - baseline) + rng.normal(0, sigma)
    return x


def generate_synthetic_panel(cfg: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Generate a controlled panel used only for sensitivity and ablation tests.

    The synthetic target is intentionally not presented as external validation.
    """
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    rows = []

    for c in range(cfg.n_communities):
        socioeconomic = rng.beta(2.2, 3.0)
        engagement = rng.beta(3.0, 2.0)
        base = {
            "sentiment_negativity": 0.25 + 0.35 * socioeconomic,
            "mobility_variance": 0.35 + 0.20 * rng.random(),
            "service_pressure": 0.20 + 0.45 * socioeconomic,
            "air_quality_stress": 0.20 + 0.25 * rng.random(),
            "temperature_anomaly": 0.25,
            "unemployment_proxy": 0.15 + 0.55 * socioeconomic,
            "poverty_proxy": 0.20 + 0.60 * socioeconomic,
            "digital_activity": 0.35 + 0.50 * rng.random(),
            "community_engagement": engagement,
            "infrastructure_reports": 0.15 + 0.35 * socioeconomic,
        }

        series: Dict[str, np.ndarray] = {}
        for f in FEATURES:
            series[f] = _ar1(
                rng, cfg.n_days,
                phi=rng.uniform(0.72, 0.94),
                sigma=rng.uniform(0.025, 0.07),
                baseline=base[f],
            )

        shock = np.zeros(cfg.n_days)
        for t in range(5, cfg.n_days):
            if rng.random() < cfg.shock_probability:
                length = int(rng.integers(3, 12))
                shock[t:min(cfg.n_days, t + length)] += np.linspace(
                    0.5, 0.05, min(length, cfg.n_days - t)
                )
        series["sentiment_negativity"] += 0.45 * shock
        series["service_pressure"] += 0.25 * shock
        series["mobility_variance"] += 0.18 * shock

        latent = (
            0.26 * series["sentiment_negativity"]
            + 0.18 * series["service_pressure"]
            + 0.14 * series["mobility_variance"]
            + 0.11 * series["air_quality_stress"]
            + 0.10 * series["unemployment_proxy"]
            + 0.08 * series["poverty_proxy"]
            + 0.06 * series["infrastructure_reports"]
            - 0.08 * series["community_engagement"]
            + rng.normal(0, 0.035, cfg.n_days)
        )
        latent = 1 / (1 + np.exp(-5 * (latent - 0.35)))

        # Future event burden is stochastic rather than a deterministic copy of latent risk.
        rate = np.clip(0.4 + 8.5 * latent, 0.1, None)
        event_count = rng.poisson(rate)

        for t, d in enumerate(dates):
            row = {
                "date": d,
                "community_id": c + 1,
                **{f: float(np.clip(series[f][t], 0, 1.5)) for f in FEATURES},
                "observed_event_count": int(event_count[t]),
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["date", "community_id"]).reset_index(drop=True)

    # Simulate missing observations only in covariates.
    cov_mask = rng.random((len(df), len(FEATURES))) < cfg.missing_rate
    for j, f in enumerate(FEATURES):
        df.loc[cov_mask[:, j], f] = np.nan
    return df
