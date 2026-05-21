from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import load_store_sales
from .features import build_recursive_feature_row, feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "inventory_forecast_bundle.joblib"


def load_bundle(bundle_path: str | Path | None = None) -> dict:
    path = Path(bundle_path) if bundle_path is not None else BUNDLE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)


def forecast_inventory_demand(csv_path: str | Path | None = None, bundle_path: str | Path | None = None) -> pd.DataFrame:
    raw = load_store_sales(csv_path)

    bundle = load_bundle(bundle_path)
    models: dict[int, object] = bundle["models"]
    columns: list[str] = bundle["feature_columns"]
    store_categories: list[str] = bundle["store_categories"]
    item_categories: list[str] = bundle["item_categories"]

    if columns != feature_columns():
        raise ValueError("Feature columns in bundle do not match current feature definition")

    results = []
    model = models[1]

    for (store_id, item_id), group in raw.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date")
        history = list(group["sales"].astype(float).values)
        current_date = pd.to_datetime(group["date"].max())
        series_start_date = pd.to_datetime(group["date"].min())

        store_code = store_categories.index(str(store_id)) if str(store_id) in store_categories else -1
        item_code = item_categories.index(str(item_id)) if str(item_id) in item_categories else -1

        for horizon in range(1, 15):
            frame = build_recursive_feature_row(
                history_values=history,
                current_date=current_date,
                series_start_date=series_start_date,
                store_id=str(store_id),
                item_id=str(item_id),
                store_code=store_code,
                item_code=item_code,
            )
            prediction = float(max(model.predict(frame[columns])[0], 0.0))
            forecast_date = current_date + pd.Timedelta(days=1)

            results.append(
                pd.DataFrame(
                    [
                        {
                            "store_id": str(store_id),
                            "item_id": str(item_id),
                            "forecast_date": forecast_date,
                            "horizon": horizon,
                            "predicted_demand": prediction,
                        }
                    ]
                )
            )

            history.append(prediction)
            current_date = forecast_date

    return pd.concat(results, ignore_index=True).sort_values(["store_id", "item_id", "forecast_date"]).reset_index(drop=True)


if __name__ == "__main__":
    forecast = forecast_inventory_demand()
    print(forecast.head(20).to_string(index=False))
