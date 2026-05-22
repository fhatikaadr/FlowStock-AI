from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT, load_store_sales
from .policy import PolicyConfig, classify_status, reorder_point
from .predict import forecast_inventory_demand

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supabase_utils import load_table_df, supabase_enabled


DEFAULT_INVENTORY_PATH = PROJECT_ROOT / "dataset" / "inventory.csv"
DEFAULT_PRODUCTS_PATH = PROJECT_ROOT / "dataset" / "products.csv"
DEFAULT_WAREHOUSES_PATH = PROJECT_ROOT / "dataset" / "warehouses.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "inventory_ai_recommendations.csv"


def _daily_stats(raw_sales: pd.DataFrame) -> pd.DataFrame:
    stats = (
        raw_sales.groupby(["store_id", "item_id"], as_index=False)["sales"]
        .agg(avg_daily_demand="mean", std_daily_demand="std")
        .fillna(0.0)
    )
    stats["store_id"] = stats["store_id"].astype(str)
    stats["item_id"] = stats["item_id"].astype(str)
    return stats


def _best_transfer_source(df: pd.DataFrame, product_id: str, target_warehouse_id: str, needed_units: float) -> tuple[str, float] | None:
    donor_pool = df[(df["product_id"] == product_id) & (df["warehouse_id"] != target_warehouse_id)].copy()
    donor_pool["transferable"] = (donor_pool["current_stock"] - donor_pool["target_stock"]).clip(lower=0)
    donor_pool = donor_pool[donor_pool["transferable"] > 0]
    if donor_pool.empty:
        return None
    best = donor_pool.sort_values("transferable", ascending=False).iloc[0]
    transfer_units = float(min(float(needed_units), float(best["transferable"])))
    return str(best["warehouse_name"]), transfer_units


def build_inventory_recommendations(
    inventory_path: str | Path | None = None,
    products_path: str | Path | None = None,
    warehouses_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cfg: PolicyConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or PolicyConfig()

    inventory_path = Path(inventory_path) if inventory_path is not None else DEFAULT_INVENTORY_PATH
    products_path = Path(products_path) if products_path is not None else DEFAULT_PRODUCTS_PATH
    warehouses_path = Path(warehouses_path) if warehouses_path is not None else DEFAULT_WAREHOUSES_PATH
    output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH

    if supabase_enabled():
        inv = load_table_df("inventory")
        products = load_table_df("products")
        warehouses = load_table_df("warehouses")
    else:
        inv = pd.read_csv(inventory_path)
        products = pd.read_csv(products_path)
        warehouses = pd.read_csv(warehouses_path)

    inv["product_id"] = inv["product_id"].astype(str)
    inv["warehouse_id"] = inv["warehouse_id"].astype(str)
    inv["current_stock"] = pd.to_numeric(inv["current_stock"], errors="coerce").fillna(0.0)

    raw_sales = load_store_sales()
    raw_sales["store_id"] = raw_sales["store_id"].astype(str)
    raw_sales["item_id"] = raw_sales["item_id"].astype(str)

    forecast = forecast_inventory_demand()
    forecast["store_id"] = forecast["store_id"].astype(str)
    forecast["item_id"] = forecast["item_id"].astype(str)
    forecast_14 = (
        forecast.groupby(["store_id", "item_id"], as_index=False)["predicted_demand"]
        .sum()
        .rename(columns={"store_id": "warehouse_id", "item_id": "product_id", "predicted_demand": "predicted_demand_14d"})
    )

    stats = _daily_stats(raw_sales).rename(columns={"store_id": "warehouse_id", "item_id": "product_id"})

    merged = inv.merge(forecast_14, on=["warehouse_id", "product_id"], how="left")
    merged = merged.merge(stats, on=["warehouse_id", "product_id"], how="left")
    merged["predicted_demand_14d"] = merged["predicted_demand_14d"].fillna(0.0)
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.0)

    # Business request: all key inventory quantity outputs must be rounded down.
    merged["predicted_demand_14d"] = np.floor(merged["predicted_demand_14d"]).astype(int)

    merged["reorder_point"] = merged.apply(
        lambda r: reorder_point(float(r["avg_daily_demand"]), float(r["std_daily_demand"]), cfg), axis=1
    )
    merged["reorder_point"] = np.floor(merged["reorder_point"]).astype(int)
    merged["target_stock"] = merged[["predicted_demand_14d", "reorder_point"]].max(axis=1)
    merged["target_stock"] = np.floor(merged["target_stock"]).astype(int)
    merged["shortage"] = (merged["target_stock"] - merged["current_stock"]).clip(lower=0)
    merged["shortage"] = np.floor(merged["shortage"]).astype(int)
    merged["status"] = merged.apply(
        lambda r: classify_status(float(r["current_stock"]), float(r["predicted_demand_14d"]), float(r["reorder_point"]), cfg),
        axis=1,
    )

    warehouses_map = warehouses.rename(columns={"id": "warehouse_id", "name": "warehouse_name"})
    warehouses_map["warehouse_id"] = warehouses_map["warehouse_id"].astype(str)
    merged = merged.merge(warehouses_map[["warehouse_id", "warehouse_name"]], on="warehouse_id", how="left")

    # Normalize recommended actions to canonical set: None, Transfer, Discount, Order
    actions: list[str] = []
    for _, row in merged.iterrows():
        if row["status"] == "Overstock":
            actions.append("Discount")
            continue
        if row["status"] == "Healthy":
            actions.append("None")
            continue

        # status is Critical (needs supply)
        needed_units = float(row["shortage"])
        transfer = _best_transfer_source(merged, str(row["product_id"]), str(row["warehouse_id"]), needed_units)
        if transfer is None:
            actions.append("Order")
        else:
            actions.append("Transfer")

    merged["recommended_action"] = actions

    products_map = products.rename(columns={"id": "product_id", "name": "product_name", "category": "product_category"})
    products_map["product_id"] = products_map["product_id"].astype(str)
    merged = merged.merge(products_map[["product_id", "sku", "product_name", "product_category"]], on="product_id", how="left")

    out_cols = [
        "id",
        "product_id",
        "product_name",
        "sku",
        "product_category",
        "warehouse_id",
        "warehouse_name",
        "current_stock",
        "predicted_demand_14d",
        "reorder_point",
        "target_stock",
        "shortage",
        "status",
        "recommended_action",
    ]
    output = merged[out_cols].sort_values(["product_id", "warehouse_id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


if __name__ == "__main__":
    df = build_inventory_recommendations()
    print(df.head(20).to_string(index=False))
    print(f"\nSaved to: {DEFAULT_OUTPUT_PATH}")
