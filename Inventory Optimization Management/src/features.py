from __future__ import annotations

import numpy as np
import pandas as pd


LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]


def _complete_daily_series(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    full_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
    group = group.set_index("date").reindex(full_index)
    group.index.name = "date"
    group["store_id"] = group["store_id"].ffill().bfill()
    group["item_id"] = group["item_id"].ffill().bfill()
    group["sales"] = group["sales"].fillna(0.0)
    return group.reset_index()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (store_id, item_id), group in df.groupby(["store_id", "item_id"], sort=False):
        completed = _complete_daily_series(group)
        completed["store_id"] = str(store_id)
        completed["item_id"] = str(item_id)
        completed["series_id"] = f"{store_id}__{item_id}"

        completed = completed.sort_values("date").reset_index(drop=True)

        for lag in LAGS:
            completed[f"lag_{lag}"] = completed["sales"].shift(lag)

        lagged_sales = completed["sales"].shift(1)
        for window in ROLL_WINDOWS:
            completed[f"roll_mean_{window}"] = lagged_sales.rolling(window=window, min_periods=1).mean()
            completed[f"roll_std_{window}"] = lagged_sales.rolling(window=window, min_periods=2).std().fillna(0.0)

        completed["ewm_mean_7"] = lagged_sales.ewm(span=7, adjust=False).mean()
        completed["ewm_mean_14"] = lagged_sales.ewm(span=14, adjust=False).mean()

        completed["day_of_week"] = completed["date"].dt.dayofweek
        completed["week_of_year"] = completed["date"].dt.isocalendar().week.astype(int)
        completed["month"] = completed["date"].dt.month
        completed["quarter"] = completed["date"].dt.quarter
        completed["day"] = completed["date"].dt.day
        completed["day_of_year"] = completed["date"].dt.dayofyear
        completed["is_weekend"] = completed["day_of_week"].isin([5, 6]).astype(int)
        completed["is_month_start"] = completed["date"].dt.is_month_start.astype(int)
        completed["is_month_end"] = completed["date"].dt.is_month_end.astype(int)
        completed["time_idx"] = (completed["date"] - completed["date"].min()).dt.days

        parts.append(completed)

    features = pd.concat(parts, ignore_index=True)
    features["store_code"] = pd.Categorical(features["store_id"]).codes.astype(int)
    features["item_code"] = pd.Categorical(features["item_id"]).codes.astype(int)
    return features.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def add_horizon_target(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = features.copy()
    frame["horizon"] = horizon
    frame["predicted_demand"] = frame.groupby(["store_id", "item_id"], sort=False)["sales"].shift(-horizon)
    return frame


def latest_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    latest = features.sort_values("date").groupby(["store_id", "item_id"], sort=False).tail(1)
    return latest.reset_index(drop=True)


def build_recursive_feature_row(
    history_values: list[float],
    current_date: pd.Timestamp,
    series_start_date: pd.Timestamp,
    store_id: str,
    item_id: str,
    store_code: int,
    item_code: int,
) -> pd.DataFrame:
    history = pd.Series(history_values, dtype=float)

    row = {
        "lag_1": history.iloc[-1] if len(history) >= 1 else np.nan,
        "lag_7": history.iloc[-7] if len(history) >= 7 else np.nan,
        "lag_14": history.iloc[-14] if len(history) >= 14 else np.nan,
        "lag_28": history.iloc[-28] if len(history) >= 28 else np.nan,
        "roll_mean_7": float(history.iloc[-7:].mean()) if len(history) >= 1 else np.nan,
        "roll_mean_14": float(history.iloc[-14:].mean()) if len(history) >= 1 else np.nan,
        "roll_mean_28": float(history.iloc[-28:].mean()) if len(history) >= 1 else np.nan,
        "roll_std_7": float(history.iloc[-7:].std(ddof=0)) if len(history) >= 2 else 0.0,
        "roll_std_14": float(history.iloc[-14:].std(ddof=0)) if len(history) >= 2 else 0.0,
        "roll_std_28": float(history.iloc[-28:].std(ddof=0)) if len(history) >= 2 else 0.0,
        "ewm_mean_7": float(history.ewm(span=7, adjust=False).mean().iloc[-1]) if len(history) >= 1 else np.nan,
        "ewm_mean_14": float(history.ewm(span=14, adjust=False).mean().iloc[-1]) if len(history) >= 1 else np.nan,
        "day_of_week": current_date.dayofweek,
        "week_of_year": int(current_date.isocalendar().week),
        "month": current_date.month,
        "quarter": current_date.quarter,
        "day": current_date.day,
        "day_of_year": current_date.dayofyear,
        "is_weekend": int(current_date.dayofweek in [5, 6]),
        "is_month_start": int(current_date.is_month_start),
        "is_month_end": int(current_date.is_month_end),
        "time_idx": (current_date - series_start_date).days,
        "store_code": store_code,
        "item_code": item_code,
        "horizon": 1,
    }
    frame = pd.DataFrame([row])
    frame["store_id"] = store_id
    frame["item_id"] = item_id
    frame["date"] = current_date
    return frame


def feature_columns() -> list[str]:
    return [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "roll_mean_7",
        "roll_mean_14",
        "roll_mean_28",
        "roll_std_7",
        "roll_std_14",
        "roll_std_28",
        "ewm_mean_7",
        "ewm_mean_14",
        "day_of_week",
        "week_of_year",
        "month",
        "quarter",
        "day",
        "day_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "time_idx",
        "store_code",
        "item_code",
        "horizon",
    ]
