"""Metric helpers for the M1 baseline evaluation.

Implements the primary metrics from the experiment plan: MAE, RMSE, R² and
MAPE. Following the plan (§4.1 / §5.1), MAPE is computed only on records
with ``fuel_rate > 1`` L/h (engine-off / idle rows have near-zero target and
would make MAPE undefined or wildly inflated); the coverage and count of rows
used are reported alongside it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAPE_LOWER_BOUND = 1.0  # L/h, per plan §4.1


def mean_absolute_percentage_error(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    lower_bound: float = MAPE_LOWER_BOUND,
) -> dict[str, float]:
    """MAPE over records where ``y_true > lower_bound`` (plan §4.1).

    Returns a dict with the MAPE value, the row count used and the coverage
    (fraction of all records that satisfied the filter).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt > lower_bound
    n_total = int(len(yt))
    n_used = int(mask.sum())
    if n_used == 0:
        return {"mape": float("nan"), "n": 0, "coverage": 0.0}
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)
    return {"mape": mape, "n": n_used, "coverage": n_used / n_total}


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return the standard regression metric set (MAE, RMSE, R², MAPE)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(math.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    mape = mean_absolute_percentage_error(yt, yp)
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "mape": mape["mape"],
        "mape_n": mape["n"],
        "mape_coverage": round(mape["coverage"], 6),
    }
