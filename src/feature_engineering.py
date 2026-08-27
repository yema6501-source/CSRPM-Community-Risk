from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


@dataclass
class TemporalSplit:
    train_end: pd.Timestamp
    valid_end: pd.Timestamp


class LeakageSafePreprocessor:
    """Fit imputation and scaling on training rows only."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.fitted = False

    def fit(self, x: np.ndarray) -> "LeakageSafePreprocessor":
        x_imp = self.imputer.fit_transform(x)
        self.scaler.fit(x_imp)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on training data first.")
        return self.scaler.transform(self.imputer.transform(x)).astype(np.float32)


def add_temporal_features(
    panel: pd.DataFrame,
    count_features: Sequence[str],
    lags: Sequence[int] = (1, 2, 4),
    rolling_windows: Sequence[int] = (4, 8),
) -> pd.DataFrame:
    df = panel.sort_values(["community_id", "date"]).copy()
    grouped = df.groupby("community_id", group_keys=False)

    # Previous event burden is valid as a historical feature; never use current/future burden.
    if "observed_event_count" in df.columns:
        df["past_event_count"] = grouped["observed_event_count"].shift(1)

    for col in count_features:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = grouped[col].shift(lag)
        for w in rolling_windows:
            df[f"{col}_rollmean{w}"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{col}_rollstd{w}"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=2).std()
            )
    df["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["date"]).dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["date"]).dt.month / 12.0)
    return df


def chronological_boundaries(dates: pd.Series, train_fraction: float = 0.70, valid_fraction: float = 0.15) -> TemporalSplit:
    unique_dates = np.array(sorted(pd.to_datetime(dates).unique()))
    if len(unique_dates) < 10:
        raise ValueError("Too few time points for chronological evaluation.")
    i_train = max(1, int(len(unique_dates) * train_fraction)) - 1
    i_valid = max(i_train + 1, int(len(unique_dates) * (train_fraction + valid_fraction))) - 1
    i_valid = min(i_valid, len(unique_dates) - 2)
    return TemporalSplit(pd.Timestamp(unique_dates[i_train]), pd.Timestamp(unique_dates[i_valid]))


def split_mask(dates: pd.Series, split: TemporalSplit) -> Dict[str, np.ndarray]:
    d = pd.to_datetime(dates)
    return {
        "train": (d <= split.train_end).to_numpy(),
        "valid": ((d > split.train_end) & (d <= split.valid_end)).to_numpy(),
        "test": (d > split.valid_end).to_numpy(),
    }


def build_future_target(
    df: pd.DataFrame,
    horizon: int = 1,
    event_col: str = "observed_event_count",
) -> pd.DataFrame:
    """Forecast future community event burden at t+h using information available through t."""
    out = df.sort_values(["community_id", "date"]).copy()
    out["future_event_count"] = (
        out.groupby("community_id")[event_col].shift(-horizon)
    )
    return out.dropna(subset=["future_event_count"]).copy()


def fit_risk_thresholds(y_train_count: np.ndarray) -> Tuple[float, float]:
    q1, q2 = np.quantile(np.asarray(y_train_count, float), [1 / 3, 2 / 3])
    if q2 <= q1:
        q2 = q1 + 1e-8
    return float(q1), float(q2)


def discretize_risk(counts: np.ndarray, thresholds: Tuple[float, float]) -> np.ndarray:
    q1, q2 = thresholds
    return np.digitize(np.asarray(counts, float), bins=[q1, q2], right=True).astype(np.int64)


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date", "community_id", "community_area", "community_area_name",
        "observed_event_count", "future_event_count", "risk_class",
        "primary_type", "status",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def panel_to_tensor(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    value_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert row-wise panel features into [time, community, feature] tensor."""
    work = df[["date", "community_id"]].copy()
    work["_row"] = np.arange(len(work))
    dates = np.array(sorted(pd.to_datetime(work["date"]).unique()))
    communities = np.array(sorted(work["community_id"].astype(int).unique()))
    dmap = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    cmap = {int(c): i for i, c in enumerate(communities)}

    tensor = np.full((len(dates), len(communities), value_array.shape[1]), np.nan, dtype=np.float32)
    mask = np.zeros((len(dates), len(communities)), dtype=bool)
    for r, (date, cid) in enumerate(zip(pd.to_datetime(work["date"]), work["community_id"].astype(int))):
        i, j = dmap[pd.Timestamp(date)], cmap[cid]
        tensor[i, j] = value_array[r]
        mask[i, j] = True
    return tensor, mask, communities
