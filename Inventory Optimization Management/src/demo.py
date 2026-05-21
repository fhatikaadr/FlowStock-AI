from __future__ import annotations

from pathlib import Path

from .inventory_recommendation import build_inventory_recommendations
from .predict import forecast_inventory_demand
from .train import BUNDLE_PATH, train_inventory_forecast_model


def run_demo() -> None:
    if not Path(BUNDLE_PATH).exists():
        print("Model bundle not found, training first...")
        train_inventory_forecast_model()

    forecast = forecast_inventory_demand()
    print(forecast.head(30).to_string(index=False))

    recommendations = build_inventory_recommendations()
    print("\nInventory AI recommendations:")
    print(recommendations.head(20).to_string(index=False))


if __name__ == "__main__":
    run_demo()
