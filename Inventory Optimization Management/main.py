from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "inventory_forecast_metrics.json"

# Ensure local src package and project root are importable
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_inventory_forecast_model  # type: ignore
from src.inventory_recommendation import build_inventory_recommendations  # type: ignore
from ai_metrics import load_inventory_forecast_metrics  # type: ignore
from supabase_utils import supabase_enabled, upsert_rows  # type: ignore

app = FastAPI(title="Inventory Optimization Management API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "inventory-optimization-management"}


@app.post("/api/train")
def api_train(dataset_path: Optional[str] = Query(default=None)) -> dict:
    try:
        bundle_path = train_inventory_forecast_model(dataset_path)
        return {"status": "ok", "bundle_path": str(bundle_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/sync-inventory")
def api_sync_inventory() -> dict:
    try:
        recommendations = build_inventory_recommendations()

        if not supabase_enabled():
            return {
                "status": "ok",
                "message": "Supabase not configured, only local recommendations were generated.",
                "rows": int(len(recommendations)),
            }

        inventory_rows = []
        recommendation_rows = []

        for _, row in recommendations.iterrows():
            row_dict = row.to_dict()
            recommendation_rows.append({
                "id": int(row_dict.get("id", 0) or 0),
                "product_id": str(row_dict.get("product_id", "")),
                "product_name": row_dict.get("product_name"),
                "sku": row_dict.get("sku"),
                "product_category": row_dict.get("product_category"),
                "warehouse_id": str(row_dict.get("warehouse_id", "")),
                "warehouse_name": row_dict.get("warehouse_name"),
                "current_stock": int(row_dict.get("current_stock", 0) or 0),
                "predicted_demand_14d": int(row_dict.get("predicted_demand_14d", 0) or 0),
                "reorder_point": int(row_dict.get("reorder_point", 0) or 0),
                "target_stock": int(row_dict.get("target_stock", 0) or 0),
                "shortage": int(row_dict.get("shortage", 0) or 0),
                "status": row_dict.get("status"),
                "recommended_action": row_dict.get("recommended_action"),
            })

            inventory_rows.append({
                "product_id": str(row_dict.get("product_id", "")),
                "warehouse_id": str(row_dict.get("warehouse_id", "")),
                "predicted_demand": int(row_dict.get("predicted_demand_14d", 0) or 0),
                "status": row_dict.get("status"),
                "recommended_action": row_dict.get("recommended_action"),
                "target_stock": int(row_dict.get("target_stock", 0) or 0),
                "shortage": int(row_dict.get("shortage", 0) or 0),
            })

        upsert_rows("inventory_ai_recommendations", recommendation_rows, on_conflict="id")
        upsert_rows("inventory", inventory_rows, on_conflict="product_id,warehouse_id")

        return {
            "status": "ok",
            "message": "Inventory predictions and recommendations synced to Supabase.",
            "rows": int(len(recommendations)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/metrics")
def api_metrics() -> dict:
    # Prefer computed full metrics from the model bundle (includes mae, mse, medae, smape, r2, evs)
    metrics = load_inventory_forecast_metrics()
    if metrics:
        return metrics

    # Fallback to saved metrics file (legacy training output which may be partial)
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                pass

    raise HTTPException(status_code=404, detail="No metrics available")


@app.get("/")
def root() -> dict:
    return {
        "name": "Inventory Optimization Management",
        "endpoints": ["/health", "/api/train", "/api/sync-inventory", "/api/metrics"],
    }
