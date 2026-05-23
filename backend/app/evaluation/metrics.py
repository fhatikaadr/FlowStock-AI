import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
)

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE with epsilon guard to prevent division by zero or infinity
    when actual sales are 0 or near-zero.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    epsilon = 1.0   # floor of 1 unit to avoid inflated % on tiny values
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE — bounded [0, 2], handles zero actuals gracefully.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    try:
        mae   = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        mae   = 0.0
    try:
        mse   = float(mean_squared_error(y_true, y_pred))
    except Exception:
        mse   = 0.0
    rmse  = float(np.sqrt(mse))
    try:
        medae = float(median_absolute_error(y_true, y_pred))
    except Exception:
        medae = 0.0
    try:
        mape  = _safe_mape(y_true, y_pred)
    except Exception:
        mape  = 0.0
    try:
        smape = _smape(y_true, y_pred)
    except Exception:
        smape = 0.0
    try:
        r2    = float(r2_score(y_true, y_pred))
    except Exception:
        r2    = 0.0
    try:
        evs   = float(explained_variance_score(y_true, y_pred))
    except Exception:
        evs   = 0.0

    def clean(v: float) -> float:
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v

    return {
        "mae":   round(clean(mae),   4),
        "mse":   round(clean(mse),   4),
        "rmse":  round(clean(rmse),  4),
        "medae": round(clean(medae), 4),
        "mape":  round(clean(mape),  4),
        "smape": round(clean(smape), 4),
        "r2":    round(clean(r2),    4),
        "evs":   round(clean(evs),   4),
    }
