from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from .data import DEFAULT_DATA_PATH, load_store_sales
from .features import add_horizon_target, build_feature_frame, feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
BUNDLE_PATH = ARTIFACT_DIR / "inventory_forecast_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "inventory_forecast_metrics.json"
VALIDATION_DAYS = 90


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))


def train_inventory_forecast_model(csv_path: str | Path | None = None) -> Path:
    raw = load_store_sales(csv_path)
    features = build_feature_frame(raw)
    columns = feature_columns()

    latest_date = features["date"].max()
    train_cutoff = latest_date - pd.Timedelta(days=VALIDATION_DAYS)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    models: dict[int, lgb.LGBMRegressor] = {}
    metrics: dict[str, dict[str, float]] = {}

    horizon = 1
    horizon_frame = add_horizon_target(features, horizon)
    horizon_frame = horizon_frame.dropna(subset=["predicted_demand"]).copy()

    train_mask = horizon_frame["date"] <= train_cutoff
    valid_mask = horizon_frame["date"] > train_cutoff

    train_frame = horizon_frame.loc[train_mask].copy()
    valid_frame = horizon_frame.loc[valid_mask].copy()

    train_frame = train_frame.dropna(subset=columns)
    valid_frame = valid_frame.dropna(subset=columns)

    if train_frame.empty or valid_frame.empty:
        raise ValueError("Insufficient data to train the inventory forecast model")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )

    model.fit(
        train_frame[columns],
        train_frame["predicted_demand"],
        eval_set=[(valid_frame[columns], valid_frame["predicted_demand"])],
        eval_metric="rmse",
        categorical_feature=["store_code", "item_code"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    valid_pred = pd.Series(model.predict(valid_frame[columns]), index=valid_frame.index)
    valid_target = valid_frame["predicted_demand"]

    rmse = float(np.sqrt(mean_squared_error(valid_target, valid_pred)))
    mape = _safe_mape(valid_target, valid_pred)

    models[horizon] = model
    metrics[str(horizon)] = {
        "rmse": rmse,
        "mape": mape,
        "train_rows": int(len(train_frame)),
        "valid_rows": int(len(valid_frame)),
    }

    bundle = {
        "models": models,
        "feature_columns": columns,
        "max_horizon": 14,
        "train_cutoff": str(train_cutoff.date()),
        "source_dataset": str(Path(csv_path)) if csv_path is not None else str(DEFAULT_DATA_PATH),
        "store_categories": list(pd.Categorical(features["store_id"]).categories.astype(str)),
        "item_categories": list(pd.Categorical(features["item_id"]).categories.astype(str)),
    }
    joblib.dump(bundle, BUNDLE_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return BUNDLE_PATH


if __name__ == "__main__":
    path = train_inventory_forecast_model()
    print(f"Saved model bundle to {path}")
