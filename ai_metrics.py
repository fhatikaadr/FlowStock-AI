from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OPTIMIZATION_ROOT = PROJECT_ROOT / "Inventory Optimization Management"
BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "inventory_forecast_bundle.joblib"


def _ensure_optimization_path() -> None:
    optimization_path = str(OPTIMIZATION_ROOT)
    if optimization_path not in sys.path:
        sys.path.insert(0, optimization_path)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def _safe_evs(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var_true = float(np.var(y_true))
    if var_true == 0:
        return 1.0 if float(np.var(y_true - y_pred)) == 0 else 0.0
    return float(1.0 - (float(np.var(y_true - y_pred)) / var_true))


def _safe_medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


@lru_cache(maxsize=1)
def load_inventory_forecast_metrics() -> dict[str, Any]:
    if not BUNDLE_PATH.exists():
        return {}

    _ensure_optimization_path()

    from src.data import load_store_sales
    from src.features import add_horizon_target, build_feature_frame, feature_columns

    bundle = joblib.load(BUNDLE_PATH)
    models = bundle.get("models", {})
    model = models.get(1) if isinstance(models, dict) else None
    if model is None:
        return {}

    source_dataset = bundle.get("source_dataset")
    if not source_dataset:
        return {}

    raw = load_store_sales(source_dataset)
    features = build_feature_frame(raw)
    columns = feature_columns()

    train_cutoff = pd.to_datetime(bundle.get("train_cutoff"), errors="coerce")
    if pd.isna(train_cutoff):
        latest_date = features["date"].max()
        train_cutoff = latest_date - pd.Timedelta(days=90)

    horizon_frame = add_horizon_target(features, 1)
    horizon_frame = horizon_frame.dropna(subset=["predicted_demand"]).copy()
    valid_frame = horizon_frame.loc[horizon_frame["date"] > train_cutoff].copy()
    valid_frame = valid_frame.dropna(subset=columns)
    if valid_frame.empty:
        return {}

    y_true = np.asarray(valid_frame["predicted_demand"], dtype=float)
    y_pred = np.asarray(model.predict(valid_frame[columns]), dtype=float)
    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(math.sqrt(mse))
    medae = _safe_medae(y_true, y_pred)
    mape = _safe_mape(y_true, y_pred)
    smape = _safe_smape(y_true, y_pred)
    r2 = _safe_r2(y_true, y_pred)
    evs = _safe_evs(y_true, y_pred)

    return {
        "forecast_model": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "medae": medae,
            "mape": mape,
            "smape": smape,
            "r2": r2,
            "evs": evs,
            "train_rows": int(len(horizon_frame.loc[horizon_frame["date"] <= train_cutoff])),
            "valid_rows": int(len(valid_frame)),
            "source_dataset": str(source_dataset),
        }
    }