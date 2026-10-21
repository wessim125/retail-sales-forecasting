from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def rmspe(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute RMSPE on positive actual sales only."""
    mask = y_true > 0
    diff = (y_pred[mask] - y_true[mask]) / y_true[mask]
    return np.sqrt(np.mean(diff ** 2))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def time_series_cv_splits(df: pd.DataFrame, n_splits: int = 3, min_train_size: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return temporal train/test splits using TimeSeriesSplit."""
    if min_train_size is None:
        min_train_size = max(1, len(df) // (n_splits + 1))

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
    splits = []
    for train_idx, test_idx in tscv.split(df):
        if len(train_idx) >= min_train_size:
            splits.append((train_idx, test_idx))
    return splits


def evaluation_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'RMSPE': rmspe(y_true, y_pred),
    }
