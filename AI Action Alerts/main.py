from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASET_DIR = PROJECT_ROOT / "dataset"
RECOMMENDATIONS_CSV = ARTIFACTS_DIR / "inventory_ai_recommendations.csv"
PRODUCTS_CSV = DATASET_DIR / "products.csv"
WAREHOUSES_CSV = DATASET_DIR / "warehouses.csv"
INVENTORY_CSV = DATASET_DIR / "inventory.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="FlowStock AI Action Alerts API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AIActionAlert(BaseModel):
    id: str
    severity: Literal["critical", "warning", "success"]
    title: str
    body: str
    timeLabel: str
    productName: Optional[str] = None
    sku: Optional[str] = None
    warehouseName: Optional[str] = None
    currentStock: Optional[int] = None
    predictedDemand14d: Optional[int] = None
    targetStock: Optional[int] = None
    shortage: Optional[int] = None
    recommendedAction: Optional[Literal["None", "Transfer", "Discount", "Order"]] = None
    ctaLabel: Optional[str] = None


class AIActionAlertsResponse(BaseModel):
    data: list[AIActionAlert]
    total: int


# Metrics endpoint removed from this service to avoid duplicate metrics.


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV) if PRODUCTS_CSV.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def load_warehouses() -> pd.DataFrame:
    return pd.read_csv(WAREHOUSES_CSV) if WAREHOUSES_CSV.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def load_inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_CSV) if INVENTORY_CSV.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(RECOMMENDATIONS_CSV) if RECOMMENDATIONS_CSV.exists() else pd.DataFrame()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_action(action: str) -> str:
    if action == "Restock Order":
        return "Order"
    if action == "⚡ Transfer":
        return "Transfer"
    return action or "None"


def priority_score(row: pd.Series) -> float:
    status_weight = {"Critical": 120.0, "Overstock": 75.0, "Healthy": 20.0}
    action_weight = {"Order": 45.0, "Transfer": 40.0, "Discount": 25.0, "None": 0.0}

    current_stock = max(float(row.get("current_stock", 0) or 0), 0.0)
    target_stock = max(float(row.get("target_stock", 0) or 0), 1.0)
    predicted_demand = max(float(row.get("predicted_demand_14d", 0) or 0), 0.0)
    shortage = max(float(row.get("shortage", 0) or 0), 0.0)
    cost_order = max(float(row.get("cost_order_idr", 0) or 0), 0.0)
    cost_transfer = max(float(row.get("cost_transfer_idr", 0) or 0), 0.0)
    action = str(row.get("recommended_action", "None"))
    status = str(row.get("status", "Healthy"))

    shortage_ratio = shortage / target_stock
    demand_pressure = max(predicted_demand - current_stock, 0.0) / max(current_stock, 1.0)
    cost_pressure = min(cost_order / 1_000_000.0, 35.0) if action == "Order" else min(cost_transfer / 1_000_000.0, 25.0)

    return (
        status_weight.get(status, 0.0)
        + action_weight.get(action, 0.0)
        + shortage_ratio * 140.0
        + demand_pressure * 55.0
        + cost_pressure
    )


def category_from_row(row: pd.Series) -> Optional[str]:
    action = str(row.get("recommended_action", "None"))
    status = str(row.get("status", "Healthy"))
    if action == "Transfer":
        return "Transfer Opportunity"
    if action == "Discount" or status == "Overstock":
        return "Overstock Alert"
    if action == "Order" or status == "Critical":
        return "Impending Stockout"
    return None


def time_label_for_index(index: int) -> str:
    if index == 0:
        return "2m ago"
    if index == 1:
        return "1h ago"
    if index == 2:
        return "3h ago"
    return f"{index + 1}h ago"


def format_idr(value: float | None) -> str:
    safe_value = 0 if value is None or pd.isna(value) else value
    return f"Rp {int(round(float(safe_value))):,}"


def build_lookup_maps():
    products = load_products()
    warehouses = load_warehouses()
    inventory = load_inventory()

    product_by_id: dict[int, dict] = {}
    product_by_name: dict[str, dict] = {}
    warehouse_by_id: dict[int, dict] = {}
    warehouse_by_name: dict[str, dict] = {}
    inventory_by_product: dict[int, list[dict]] = {}

    for _, row in products.iterrows():
        product = {
            "id": int(row.get("id", 0) or 0),
            "sku": str(row.get("sku", "")),
            "name": str(row.get("name", "")),
            "category": str(row.get("category", "")),
            "price": float(row.get("price", 0) or 0),
            "weight": float(row.get("weight", 0) or 0),
        }
        product_by_id[product["id"]] = product
        product_by_name[product["name"].lower()] = product

    for _, row in warehouses.iterrows():
        warehouse = {
            "id": int(row.get("id", 0) or 0),
            "name": str(row.get("name", "")),
            "latitude": float(row.get("latitude", 0) or 0),
            "longitude": float(row.get("longitude", 0) or 0),
        }
        warehouse_by_id[warehouse["id"]] = warehouse
        warehouse_by_name[warehouse["name"].lower()] = warehouse

    for _, row in inventory.iterrows():
        item = {
            "product_id": int(row.get("product_id", 0) or 0),
            "warehouse_id": int(row.get("warehouse_id", 0) or 0),
            "current_stock": float(row.get("current_stock", 0) or 0),
            "predicted_demand": float(row.get("predicted_demand", 0) or 0),
        }
        inventory_by_product.setdefault(item["product_id"], []).append(item)

    return product_by_id, product_by_name, warehouse_by_id, warehouse_by_name, inventory_by_product


def best_transfer_source(product_id: int, destination_id: int, warehouse_by_id: dict[int, dict], inventory_by_product: dict[int, list[dict]]):
    destination = warehouse_by_id.get(destination_id)
    if not destination:
        return None

    best = None
    best_score = -1e18
    for row in inventory_by_product.get(product_id, []):
        surplus = row["current_stock"] - row["predicted_demand"]
        if surplus <= 0:
            continue
        source = warehouse_by_id.get(int(row["warehouse_id"]))
        if not source:
            continue
        distance = haversine_km(source["latitude"], source["longitude"], destination["latitude"], destination["longitude"])
        score = surplus * 1000 - distance
        if score > best_score:
            best_score = score
            best = {"source": source, "distance": distance, "surplus": surplus}
    return best


def build_cost_inputs(row: pd.Series, product_by_id: dict[int, dict], product_by_name: dict[str, dict], warehouse_by_id: dict[int, dict], warehouse_by_name: dict[str, dict], inventory_by_product: dict[int, list[dict]]):
    product = product_by_id.get(int(row.get("product_id", 0) or 0)) or product_by_name.get(str(row.get("product_name", "")).lower())
    warehouse = warehouse_by_id.get(int(row.get("warehouse_id", 0) or 0)) or warehouse_by_name.get(str(row.get("warehouse_name", "")).lower())
    quantity = max(int(row.get("shortage", 0) or 0), int(row.get("predicted_demand_14d", 0) or 0), 1)

    product_price = float(product.get("price", 0) if product else 0)
    product_weight_kg = float(product.get("weight", 0) if product else 0) / 1000.0
    transfer_source = best_transfer_source(int(product.get("id", 0) or 0), int(warehouse.get("id", 0) or 0), warehouse_by_id, inventory_by_product) if product and warehouse else None
    transfer_distance_km = transfer_source["distance"] if transfer_source else None

    order_item_cost = product_price * quantity
    order_shipping_cost = max(15000.0, product_weight_kg * quantity * 3200.0)
    order_other_costs = max(5000.0, order_item_cost * 0.015)
    estimated_order_total = order_item_cost + order_shipping_cost + order_other_costs

    transfer_shipping_cost = max(8000.0, product_weight_kg * quantity * 1800.0)
    transfer_other_costs = max(5000.0, product_price * quantity * 0.01)
    estimated_transfer_total = transfer_shipping_cost + transfer_other_costs

    overstock_qty = max(int(row.get("current_stock", 0) or 0) - int(row.get("target_stock", 0) or 0), 0)
    discount_qty = max(overstock_qty, 1)
    estimated_discount_total = product_price * discount_qty * 0.15 + max(5000.0, discount_qty * 2500.0)

    return {
        "product": product,
        "warehouse": warehouse,
        "transfer_source": transfer_source,
        "transfer_distance_km": transfer_distance_km,
        "estimated_order_total": estimated_order_total,
        "estimated_transfer_total": estimated_transfer_total,
        "estimated_discount_total": estimated_discount_total,
    }


def build_alert(row: pd.Series, index: int, cost_inputs: dict) -> AIActionAlert:
    action = str(row.get("recommended_action", "None"))
    status = str(row.get("status", "Healthy"))
    product_name = str(row.get("product_name", "Unknown Product"))
    warehouse_name = str(row.get("warehouse_name", "Unknown Warehouse"))
    sku = str(row.get("sku", "")) or None
    current_stock = int(row.get("current_stock", 0) or 0)
    target_stock = int(row.get("target_stock", 0) or 0)
    shortage = int(row.get("shortage", 0) or 0)

    cost_order = format_idr(cost_inputs["estimated_order_total"])
    cost_transfer = format_idr(cost_inputs["estimated_transfer_total"])
    cost_discount = format_idr(cost_inputs["estimated_discount_total"])

    if action == "Transfer":
        severity = "critical" if shortage > 0 or status == "Critical" else "warning"
        return AIActionAlert(
            id=f"alert-{row.get('id', index)}",
            severity=severity,
            title="Impending Stockout" if severity == "critical" else "Transfer Opportunity",
            body=f"{warehouse_name} should receive {shortage} units of {product_name} (SKU {sku or '-'}) to avoid a stockout. Estimated transfer cost {cost_transfer} versus order cost {cost_order}.",
            timeLabel=time_label_for_index(index),
            productName=product_name,
            sku=sku,
            warehouseName=warehouse_name,
            currentStock=current_stock,
            predictedDemand14d=int(row.get("predicted_demand_14d", 0) or 0),
            targetStock=target_stock,
            shortage=shortage,
            recommendedAction="Transfer",
            ctaLabel="Execute Transfer Now",
        )

    if action == "Order":
        return AIActionAlert(
            id=f"alert-{row.get('id', index)}",
            severity="critical",
            title="Impending Stockout",
            body=f"{warehouse_name} is projected to be short {shortage} units of {product_name} (SKU {sku or '-'}) within 14 days. Estimated order cost {cost_order}; transfer fallback {cost_transfer} if donor stock exists.",
            timeLabel=time_label_for_index(index),
            productName=product_name,
            sku=sku,
            warehouseName=warehouse_name,
            currentStock=current_stock,
            predictedDemand14d=int(row.get("predicted_demand_14d", 0) or 0),
            targetStock=target_stock,
            shortage=shortage,
            recommendedAction="Order",
            ctaLabel="Place Order Now",
        )

    if action == "Discount":
        return AIActionAlert(
            id=f"alert-{row.get('id', index)}",
            severity="warning",
            title="Overstock Alert",
            body=f"{warehouse_name} is holding {current_stock} units of {product_name} — above target {target_stock}. Estimated markdown cost {cost_discount} to move excess stock faster.",
            timeLabel=time_label_for_index(index),
            productName=product_name,
            sku=sku,
            warehouseName=warehouse_name,
            currentStock=current_stock,
            predictedDemand14d=int(row.get("predicted_demand_14d", 0) or 0),
            targetStock=target_stock,
            shortage=shortage,
            recommendedAction="Discount",
            ctaLabel="Review Simulation",
        )

    return AIActionAlert(
        id=f"alert-{row.get('id', index)}",
        severity="success",
        title="Stock Health Check",
        body=f"{product_name} at {warehouse_name} is aligned with target stock. Keep monitoring demand and replenishment costs.",
        timeLabel=time_label_for_index(index),
        productName=product_name,
        sku=sku,
        warehouseName=warehouse_name,
        currentStock=current_stock,
        predictedDemand14d=int(row.get("predicted_demand_14d", 0) or 0),
        targetStock=target_stock,
        shortage=shortage,
        recommendedAction="None",
        ctaLabel="View Simulation",
    )


def build_prompt(candidates: list[dict], category_name: str) -> str:
    return f"""You are an inventory control assistant for a dashboard.

Select exactly one best alert for this category: {category_name}.
Return valid JSON only using this exact shape:
{{"selected_id": 123}}

Rules:
- Select only an id from the provided candidates.
- Prioritize the most critical item within this category.
- Do not include explanations or extra keys.

Candidates:
{json.dumps(candidates, ensure_ascii=False)}
"""


async def select_single_alert_id(candidates: list[dict], category_name: str) -> Optional[int]:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key or not candidates:
        return None

    prompt = build_prompt(candidates, category_name)
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
                },
            )
        if response.status_code >= 400:
            return None
        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = json.loads(text)
        selected_id = parsed.get("selected_id")
        return int(selected_id) if selected_id is not None else None
    except Exception:
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "flowstock-ai-action-alerts-api"}


# /api/metrics removed from this service to centralize metrics in Inventory Optimization Management.


@app.get("/api/action-alerts")
async def action_alerts(limit: int = Query(default=3, ge=1, le=3)) -> AIActionAlertsResponse:
    if not isinstance(limit, int):
        limit = 3

    if not RECOMMENDATIONS_CSV.exists():
        raise HTTPException(status_code=404, detail=f"Recommendation file not found: {RECOMMENDATIONS_CSV}")

    recommendations = load_recommendations()
    if recommendations.empty:
        raise HTTPException(status_code=404, detail=f"Recommendation file not found: {RECOMMENDATIONS_CSV}")

    product_by_id, product_by_name, warehouse_by_id, warehouse_by_name, inventory_by_product = build_lookup_maps()

    rows = []
    for _, row in recommendations.iterrows():
        normalized_action = normalize_action(str(row.get("recommended_action", "None")))
        current = {
            **row.to_dict(),
            "id": int(row.get("id", 0) or 0),
            "product_id": int(row.get("product_id", 0) or 0),
            "warehouse_id": int(row.get("warehouse_id", 0) or 0),
            "current_stock": int(row.get("current_stock", 0) or 0),
            "predicted_demand_14d": int(row.get("predicted_demand_14d", 0) or 0),
            "target_stock": int(row.get("target_stock", 0) or 0),
            "shortage": int(row.get("shortage", 0) or 0),
            "status": str(row.get("status", "Healthy")),
            "recommended_action": normalized_action,
            "alert_category": category_from_row(row),
        }
        current["priority_score"] = priority_score(pd.Series(current))
        rows.append(current)

    category_order = ["Impending Stockout", "Transfer Opportunity", "Overstock Alert"]
    selected_rows: list[pd.Series] = []
    used_ids: set[int] = set()

    for category_name in category_order:
        if len(selected_rows) >= limit:
            break

        category_rows = [row for row in rows if row["alert_category"] == category_name and row["id"] not in used_ids]
        category_rows = sorted(category_rows, key=lambda row: (row["priority_score"], row["shortage"], row["predicted_demand_14d"]), reverse=True)
        if not category_rows:
            continue

        candidates = [
            {
                "id": row["id"],
                "product_name": row.get("product_name"),
                "sku": row.get("sku"),
                "warehouse_name": row.get("warehouse_name"),
                "current_stock": row.get("current_stock"),
                "predicted_demand_14d": row.get("predicted_demand_14d"),
                "target_stock": row.get("target_stock"),
                "shortage": row.get("shortage"),
                "status": row.get("status"),
                "recommended_action": row.get("recommended_action"),
                "cost_transfer_idr": row.get("cost_transfer_idr", 0),
                "cost_order_idr": row.get("cost_order_idr", 0),
                "priority_score": row.get("priority_score", 0),
            }
            for row in category_rows[:25]
        ]

        selected_id = await select_single_alert_id(candidates, category_name)
        chosen = None
        if selected_id is not None:
            chosen = next((row for row in category_rows if row["id"] == selected_id), None)

        if chosen is None:
            chosen = category_rows[0]

        used_ids.add(chosen["id"])
        selected_rows.append(pd.Series(chosen))

    if len(selected_rows) < limit:
        fallback_sorted = sorted(rows, key=lambda row: (row["priority_score"], row["shortage"], row["predicted_demand_14d"]), reverse=True)
        for row in fallback_sorted:
            if row["id"] in used_ids:
                continue
            used_ids.add(row["id"])
            selected_rows.append(pd.Series(row))
            if len(selected_rows) >= limit:
                break

    alerts = []
    for index, row in enumerate(selected_rows[:limit]):
        cost_inputs = build_cost_inputs(row, product_by_id, product_by_name, warehouse_by_id, warehouse_by_name, inventory_by_product)
        alerts.append(build_alert(row, index, cost_inputs))

    return AIActionAlertsResponse(data=alerts, total=len(alerts))


@app.get("/")
def root() -> dict:
    return {
        "name": "FlowStock AI Action Alerts API",
        "endpoints": ["/health", "/api/action-alerts"],
    }
