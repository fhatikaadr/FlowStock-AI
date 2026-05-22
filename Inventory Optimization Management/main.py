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
from ai_metrics import load_inventory_forecast_metrics  # type: ignore

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
        "endpoints": ["/health", "/api/train", "/api/metrics"],
    }
