from __future__ import annotations

import json
import os
import math
import sys
from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional

import pandas as pd
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RECOMMENDATIONS_CSV = ARTIFACTS_DIR / "inventory_ai_recommendations.csv"
DATASET_DIR = PROJECT_ROOT / "dataset"
PRODUCTS_CSV = DATASET_DIR / "products.csv"
WAREHOUSES_CSV = DATASET_DIR / "warehouses.csv"
INVENTORY_CSV = DATASET_DIR / "inventory.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supabase_utils import load_table_df, supabase_enabled, update_row

app = FastAPI(title="FlowStock AI Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    product_name: str
    warehouse_name: str
    current_stock: int = Field(ge=0)
    predicted_demand_14d: int = Field(ge=0)
    target_stock: int = Field(ge=0)
    shortage: int = Field(ge=0)
    status: Literal["Healthy", "Critical", "Overstock"]
    recommended_action: Literal["None", "Transfer", "Discount", "Order"]


class SolutionOption(BaseModel):
    title: str
    description: str
    costImpact: str
    riskLevel: Literal["Low", "Medium", "High"]
    feasibility: Literal["Low", "Medium", "High"]
    cost_breakdown: Optional[dict] = None


class RecommendationExplanation(BaseModel):
    recommended_action: Literal["None", "Transfer", "Discount", "Order"]
    best_option: SolutionOption
    alternative_option: SolutionOption


class ActionAlert(BaseModel):
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


class ActionAlertsResponse(BaseModel):
    data: list[ActionAlert]
    total: int


# Metrics endpoint removed — metrics are served by Inventory Optimization Management service


class CostInput(BaseModel):
    product_price: float = 0.0
    product_weight_grams: float = 0.0
    product_weight_kg: float = 0.0
    quantity: int = 0
    source_warehouse: Optional[str] = None
    source_distance_km: Optional[float] = None
    destination_distance_km: Optional[float] = None
    transfer_distance_km: Optional[float] = None
    estimated_order_total: Optional[float] = None
    estimated_transfer_total: Optional[float] = None
    estimated_discount_total: Optional[float] = None
    estimated_monitor_total: Optional[float] = None


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    if supabase_enabled():
        df = load_table_df("products")
        if not df.empty:
            return df

    if not PRODUCTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PRODUCTS_CSV)


@lru_cache(maxsize=1)
def load_warehouses() -> pd.DataFrame:
    if supabase_enabled():
        df = load_table_df("warehouses")
        if not df.empty:
            return df

    if not WAREHOUSES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(WAREHOUSES_CSV)


@lru_cache(maxsize=1)
def load_inventory() -> pd.DataFrame:
    if supabase_enabled():
        df = load_table_df("inventory")
        if not df.empty:
            return df

    if not INVENTORY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(INVENTORY_CSV)


@lru_cache(maxsize=1)
def load_recommendations() -> pd.DataFrame:
    if supabase_enabled():
        df = load_table_df("inventory_ai_recommendations")
        if not df.empty:
            return df

    if not RECOMMENDATIONS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RECOMMENDATIONS_CSV)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize(text: str) -> str:
    return str(text).strip().lower()


def lookup_product(payload: RecommendationRequest) -> dict:
    products = load_products()
    if products.empty:
        return {}

    match = products[
        products.get("name", pd.Series(dtype=str)).astype(str).str.lower() == payload.product_name.lower()
    ]
    if match.empty and "sku" in products.columns:
        match = products[products["sku"].astype(str).str.lower() == payload.product_name.lower()]
    if match.empty:
        return {}

    row = match.iloc[0].to_dict()
    price = float(row.get("price", 0.0) or 0.0)
    weight_grams = float(row.get("weight", 0.0) or 0.0)
    return {
        "id": row.get("id"),
        "sku": row.get("sku"),
        "name": row.get("name", payload.product_name),
        "category": row.get("category"),
        "price": price,
        "weight_grams": weight_grams,
        "weight_kg": weight_grams / 1000.0 if weight_grams else 0.0,
        "seasonality_group": row.get("seasonality_group"),
    }


def lookup_warehouse(name: str) -> dict:
    warehouses = load_warehouses()
    if warehouses.empty:
        return {}

    match = warehouses[warehouses.get("name", pd.Series(dtype=str)).astype(str).str.lower() == name.lower()]
    if match.empty:
        return {}

    row = match.iloc[0].to_dict()
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "latitude": float(row.get("latitude", 0.0) or 0.0),
        "longitude": float(row.get("longitude", 0.0) or 0.0),
    }


def find_transfer_source(payload: RecommendationRequest, product: dict, destination: dict) -> dict:
    inventory = load_inventory()
    warehouses = load_warehouses()
    if inventory.empty or warehouses.empty or not product:
        return {}

    product_id = product.get("id")
    if pd.isna(product_id):
        return {}

    subset = inventory[inventory["product_id"] == int(product_id)].copy()
    if subset.empty:
        return {}

    subset["surplus"] = pd.to_numeric(subset.get("current_stock", 0), errors="coerce").fillna(0) - pd.to_numeric(
        subset.get("predicted_demand", 0), errors="coerce"
    ).fillna(0)
    subset = subset[subset["surplus"] > 0]
    if subset.empty:
        return {}

    warehouse_rows = warehouses.set_index("id") if "id" in warehouses.columns else warehouses
    best_source = None
    best_score = None

    for _, row in subset.iterrows():
        source_id = row.get("warehouse_id")
        if pd.isna(source_id):
            continue
        source_id = int(source_id)
        if source_id not in warehouse_rows.index:
            continue
        source_row = warehouse_rows.loc[source_id]
        source_lat = float(source_row.get("latitude", 0.0) or 0.0)
        source_lon = float(source_row.get("longitude", 0.0) or 0.0)
        dest_lat = destination.get("latitude", 0.0)
        dest_lon = destination.get("longitude", 0.0)
        distance_km = haversine_km(source_lat, source_lon, dest_lat, dest_lon)
        surplus = float(row.get("surplus", 0.0) or 0.0)
        score = (surplus, -distance_km)
        if best_score is None or score > best_score:
            best_score = score
            best_source = {
                "warehouse_id": source_id,
                "warehouse_name": source_row.get("name"),
                "distance_km": distance_km,
                "surplus": surplus,
            }

    return best_source or {}


def build_cost_inputs(payload: RecommendationRequest) -> CostInput:
    product = lookup_product(payload)
    destination = lookup_warehouse(payload.warehouse_name)
    quantity = max(int(payload.shortage or 0), int(payload.predicted_demand_14d or 0), 1)

    transfer_source = find_transfer_source(payload, product, destination)
    transfer_distance_km = transfer_source.get("distance_km")
    source_distance_km = transfer_distance_km

    product_price = float(product.get("price", 0.0) or 0.0)
    product_weight_kg = float(product.get("weight_kg", 0.0) or 0.0)
    transfer_weight_cost = max(8000.0, product_weight_kg * quantity * 1800.0)
    order_shipping_cost = max(15000.0, product_weight_kg * quantity * 3200.0)

    order_item_cost = product_price * quantity
    order_other_costs = max(5000.0, order_item_cost * 0.015)
    estimated_order_total = order_item_cost + order_shipping_cost + order_other_costs

    transfer_item_cost = 0.0
    transfer_other_costs = max(5000.0, product_price * quantity * 0.01)
    estimated_transfer_total = transfer_item_cost + transfer_weight_cost + transfer_other_costs

    overstock_qty = max(int(payload.current_stock - payload.target_stock), 0)
    discount_qty = max(overstock_qty, 1)
    markdown_rate = 0.15
    estimated_discount_total = product_price * discount_qty * markdown_rate + max(5000.0, discount_qty * 2500.0)

    estimated_monitor_total = max(2500.0, product_price * 0.002)

    return CostInput(
        product_price=product_price,
        product_weight_grams=float(product.get("weight_grams", 0.0) or 0.0),
        product_weight_kg=product_weight_kg,
        quantity=quantity,
        source_warehouse=transfer_source.get("warehouse_name"),
        source_distance_km=source_distance_km,
        destination_distance_km=0.0 if destination else None,
        transfer_distance_km=transfer_distance_km,
        estimated_order_total=estimated_order_total,
        estimated_transfer_total=estimated_transfer_total,
        estimated_discount_total=estimated_discount_total,
        estimated_monitor_total=estimated_monitor_total,
    )


def cost_breakdown_from_total(total_cost: Optional[float], item_cost: float = 0.0, shipping_cost: float = 0.0, other_costs: float = 0.0) -> Optional[dict]:
    if total_cost is None:
        return None
    return {
        "currency": "IDR",
        "item_cost": float(item_cost),
        "shipping_cost": float(shipping_cost),
        "other_costs": float(other_costs),
        "total_cost": float(total_cost),
    }


def time_label_for_index(index: int) -> str:
    if index == 0:
        return "2m ago"
    if index == 1:
        return "1h ago"
    if index == 2:
        return "3h ago"
    return f"{index + 1}h ago"


def format_idr(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        value = 0
    return f"Rp {int(round(float(value))):,}"


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


def build_action_alert_payload(row: pd.Series, index: int) -> ActionAlert:
    payload = RecommendationRequest(
        product_name=str(row.get("product_name", "Unknown Product")),
        warehouse_name=str(row.get("warehouse_name", "Unknown Warehouse")),
        current_stock=int(float(row.get("current_stock", 0) or 0)),
        predicted_demand_14d=int(float(row.get("predicted_demand_14d", 0) or 0)),
        target_stock=int(float(row.get("target_stock", 0) or 0)),
        shortage=int(float(row.get("shortage", 0) or 0)),
        status=str(row.get("status", "Healthy")),
        recommended_action=str(row.get("recommended_action", "None")),
    )
    cost_inputs = build_cost_inputs(payload)
    sku = str(row.get("sku", ""))
    action = payload.recommended_action
    cost_order = format_idr(cost_inputs.estimated_order_total)
    cost_transfer = format_idr(cost_inputs.estimated_transfer_total)
    cost_discount = format_idr(cost_inputs.estimated_discount_total)
    current_stock = payload.current_stock
    target_stock = payload.target_stock
    shortage = payload.shortage
    product_name = payload.product_name
    warehouse_name = payload.warehouse_name

    if action == "Transfer":
        severity = "critical" if shortage > 0 or payload.status == "Critical" else "warning"
        title = "Impending Stockout" if severity == "critical" else "Transfer Opportunity"
        body = (
            f"{warehouse_name} should receive {shortage} units of {product_name} (SKU {sku}) to avoid a stockout. "
            f"Estimated transfer cost {cost_transfer} versus order cost {cost_order}."
        )
        cta = "Execute Transfer Now"
    elif action == "Order":
        severity = "critical"
        title = "Impending Stockout"
        body = (
            f"{warehouse_name} is projected to be short {shortage} units of {product_name} (SKU {sku}) within 14 days. "
            f"Estimated order cost {cost_order}; transfer fallback {cost_transfer} if donor stock exists."
        )
        cta = "Place Order Now"
    elif action == "Discount":
        severity = "warning"
        title = "Overstock Alert"
        body = (
            f"{warehouse_name} is holding {current_stock} units of {product_name} — above target {target_stock}. "
            f"Estimated discount cost {cost_discount} to move excess stock faster."
        )
        cta = "Review Simulation"
    else:
        severity = "success"
        title = "Stock Health Check"
        body = (
            f"{product_name} at {warehouse_name} is aligned with target stock. "
            f"Keep monitoring demand and replenishment costs."
        )
        cta = "View Simulation"

    return ActionAlert(
        id=f"alert-{row.get('id', index)}",
        severity=severity,
        title=title,
        body=body,
        timeLabel=time_label_for_index(index),
        productName=product_name,
        sku=sku or None,
        warehouseName=warehouse_name,
        currentStock=current_stock,
        predictedDemand14d=payload.predicted_demand_14d,
        targetStock=target_stock,
        shortage=shortage,
        recommendedAction=action, 
        ctaLabel=cta,
    )


def build_action_alert_candidates(df: pd.DataFrame, limit: int) -> list[dict]:
    candidates: list[dict] = []
    ranked = df.copy()
    ranked["priority_score"] = ranked.apply(priority_score, axis=1)
    ranked = ranked.sort_values(["priority_score", "shortage", "predicted_demand_14d"], ascending=[False, False, False])

    for _, row in ranked.head(limit).iterrows():
        candidates.append(
            {
                "id": int(row["id"]),
                "product_name": row.get("product_name"),
                "sku": row.get("sku"),
                "warehouse_name": row.get("warehouse_name"),
                "current_stock": int(row.get("current_stock", 0) or 0),
                "predicted_demand_14d": int(row.get("predicted_demand_14d", 0) or 0),
                "target_stock": int(row.get("target_stock", 0) or 0),
                "shortage": int(row.get("shortage", 0) or 0),
                "status": row.get("status"),
                "recommended_action": row.get("recommended_action"),
                "cost_transfer_idr": float(row.get("cost_transfer_idr", 0) or 0),
                "cost_order_idr": float(row.get("cost_order_idr", 0) or 0),
                "priority_score": float(row.get("priority_score", 0) or 0),
            }
        )

    return candidates


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


def build_category_candidates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ranked = df.copy()
    ranked["priority_score"] = ranked.apply(priority_score, axis=1)
    ranked["alert_category"] = ranked.apply(category_from_row, axis=1)
    return {
        "Impending Stockout": ranked[ranked["alert_category"] == "Impending Stockout"].copy(),
        "Transfer Opportunity": ranked[ranked["alert_category"] == "Transfer Opportunity"].copy(),
        "Overstock Alert": ranked[ranked["alert_category"] == "Overstock Alert"].copy(),
    }


async def select_single_alert_id_with_gemini(candidates: list[dict], category_name: str) -> Optional[int]:
    if not candidates:
        return None

    selected_ids = await select_alert_ids_with_gemini(candidates, 1)
    if selected_ids:
        return selected_ids[0]

    return int(max(candidates, key=lambda item: item.get("priority_score", 0)).get("id"))


def build_action_alert_selection_prompt(candidates: list[dict], limit: int) -> str:
    return f"""You are an inventory control assistant for a dashboard.

Select the {limit} most critical alerts from the candidates below, ordered from most to least urgent.
Return valid JSON only using this exact shape:
{{"selected_ids":[1,2,3]}}

Rules:
- Select only ids from the provided candidates.
- Prioritize imminent stockout, high shortage ratio, high demand pressure, transfer opportunities when donor stock is available, and severe overstock capital lock.
- Prefer actionable items over healthy ones.
- Do not include explanations or extra keys.

Candidates:
{json.dumps(candidates, ensure_ascii=False)}
"""


async def select_alert_ids_with_gemini(candidates: list[dict], limit: int) -> list[int]:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key or not candidates:
        return []

    prompt = build_action_alert_selection_prompt(candidates, limit)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "responseMimeType": "application/json",
                    },
                },
            )

        if response.status_code >= 400:
            return []

        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = json.loads(text)
        selected_ids = parsed.get("selected_ids", [])

        return [int(value) for value in selected_ids if str(value).isdigit() or isinstance(value, int)]
    except Exception:
        return []


def build_action_alerts_from_df(df: pd.DataFrame, limit: int, selected_ids: Optional[list[int]] = None) -> ActionAlertsResponse:
    ranked = df.copy()
    ranked["priority_score"] = ranked.apply(priority_score, axis=1)
    ranked = ranked.sort_values(["priority_score", "shortage", "predicted_demand_14d"], ascending=[False, False, False])

    selected_rows: list[pd.Series] = []
    seen_ids: set[int] = set()

    if selected_ids:
        row_map = {int(row["id"]): row for _, row in ranked.iterrows()}
        for alert_id in selected_ids:
            if alert_id in row_map and alert_id not in seen_ids:
                selected_rows.append(row_map[alert_id])
                seen_ids.add(alert_id)

    if len(selected_rows) < limit:
        for _, row in ranked.iterrows():
            row_id = int(row["id"])
            if row_id in seen_ids:
                continue
            selected_rows.append(row)
            seen_ids.add(row_id)
            if len(selected_rows) >= limit:
                break

    alerts = [build_action_alert_payload(row, index) for index, row in enumerate(selected_rows[:limit])]
    return ActionAlertsResponse(data=alerts, total=len(alerts))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "flowstock-ai-recommendation-api"}


# /api/metrics removed from this service to avoid duplicate metric sources.


@app.get("/api/inventory-recommendations")
def inventory_recommendations(
    warehouse: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict:
    if not RECOMMENDATIONS_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation file not found: {RECOMMENDATIONS_CSV}",
        )

    df = pd.read_csv(RECOMMENDATIONS_CSV)
    for column in [
        "id",
        "product_id",
        "warehouse_id",
        "current_stock",
        "predicted_demand_14d",
        "reorder_point",
        "target_stock",
        "shortage",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    if warehouse and warehouse != "All Warehouses" and "warehouse_name" in df.columns:
        df = df[df["warehouse_name"].str.lower() == warehouse.lower()]
    if category and category != "All Categories" and "product_category" in df.columns:
        df = df[df["product_category"].str.lower() == category.lower()]
    if status and status != "All Statuses" and "status" in df.columns:
        df = df[df["status"] == status]
    if action and action != "All Actions" and "recommended_action" in df.columns:
        df = df[df["recommended_action"] == action]

    df = df.head(limit)
    return {"data": df.to_dict(orient="records"), "total": int(len(df))}


@app.get("/api/action-alerts")
async def action_alerts(
    limit: int = Query(default=3, ge=1, le=3),
) -> ActionAlertsResponse:
    if not isinstance(limit, int):
        limit = 3

    if not RECOMMENDATIONS_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation file not found: {RECOMMENDATIONS_CSV}",
        )

    df = pd.read_csv(RECOMMENDATIONS_CSV)
    for column in [
        "id",
        "product_id",
        "warehouse_id",
        "current_stock",
        "predicted_demand_14d",
        "reorder_point",
        "target_stock",
        "shortage",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    category_frames = build_category_candidates(df)
    alert_order = ["Impending Stockout", "Transfer Opportunity", "Overstock Alert"]
    selected_rows: list[pd.Series] = []

    for category_name in alert_order:
        if len(selected_rows) >= limit:
            break

        category_frame = category_frames.get(category_name, pd.DataFrame())
        if category_frame.empty:
            continue

        candidates = build_action_alert_candidates(category_frame, 25)
        selected_id = await select_single_alert_id_with_gemini(candidates, category_name)

        if selected_id is not None:
            matched = category_frame[category_frame["id"] == selected_id]
            if not matched.empty:
                selected_rows.append(matched.iloc[0])
                continue

        selected_rows.append(category_frame.sort_values(["priority_score", "shortage", "predicted_demand_14d"], ascending=[False, False, False]).iloc[0])

    if len(selected_rows) < limit:
        remaining = df.copy()
        remaining["priority_score"] = remaining.apply(priority_score, axis=1)
        remaining = remaining.sort_values(["priority_score", "shortage", "predicted_demand_14d"], ascending=[False, False, False])
        seen_ids = {int(row.get("id")) for row in selected_rows}
        for _, row in remaining.iterrows():
            row_id = int(row.get("id", 0) or 0)
            if row_id in seen_ids:
                continue
            selected_rows.append(row)
            seen_ids.add(row_id)
            if len(selected_rows) >= limit:
                break

    alerts = [build_action_alert_payload(row, index) for index, row in enumerate(selected_rows[:limit])]
    return ActionAlertsResponse(data=alerts, total=len(alerts))


@app.post("/api/generate-recommendation-explanation")
async def generate_recommendation_explanation(payload: RecommendationRequest) -> RecommendationExplanation:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    cost_inputs = build_cost_inputs(payload)

    if not api_key:
        return build_fallback_explanation(payload, cost_inputs)

    prompt = build_prompt(payload, cost_inputs)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.4,
                        "responseMimeType": "application/json",
                    },
                },
            )

        if response.status_code >= 400:
            return build_fallback_explanation(payload, cost_inputs)

        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = json.loads(text)

        # compute local breakdowns to use as fallback if Gemini output omits them
        product = lookup_product(payload)
        transfer_total = cost_inputs.estimated_transfer_total
        order_total = cost_inputs.estimated_order_total
        discount_total = cost_inputs.estimated_discount_total
        monitor_total = cost_inputs.estimated_monitor_total

        transfer_breakdown = cost_breakdown_from_total(
            transfer_total,
            item_cost=0.0,
            shipping_cost=max(0.0, (transfer_total or 0.0) * 0.72),
            other_costs=max(0.0, (transfer_total or 0.0) * 0.28),
        )
        order_breakdown = cost_breakdown_from_total(
            order_total,
            item_cost=float(product.get("price", 0.0) or 0.0) * cost_inputs.quantity,
            shipping_cost=max(15000.0, (cost_inputs.product_weight_kg or 0.0) * cost_inputs.quantity * 3200.0),
            other_costs=max(5000.0, float(product.get("price", 0.0) or 0.0) * cost_inputs.quantity * 0.015),
        )
        discount_breakdown = cost_breakdown_from_total(
            discount_total,
            item_cost=0.0,
            shipping_cost=max(0.0, (discount_total or 0.0) * 0.15),
            other_costs=max(5000.0, (discount_total or 0.0) * 0.85),
        )
        monitor_breakdown = cost_breakdown_from_total(
            monitor_total,
            item_cost=0.0,
            shipping_cost=0.0,
            other_costs=float(monitor_total or 0.0),
        )

        def _ensure_cost_cb(option: dict) -> dict:
            if not isinstance(option, dict):
                return option
            cb = option.get("cost_breakdown")
            if cb and isinstance(cb, dict) and cb.get("total_cost") is not None:
                return option
            title = (option.get("title", "") or "").lower()
            if "transfer" in title:
                option["cost_breakdown"] = transfer_breakdown
            elif "order" in title or "supplier" in title:
                option["cost_breakdown"] = order_breakdown
            elif "discount" in title or "markdown" in title:
                option["cost_breakdown"] = discount_breakdown
            else:
                option["cost_breakdown"] = monitor_breakdown
            return option

        best = _ensure_cost_cb(parsed.get("best_option", {}))
        alt = _ensure_cost_cb(parsed.get("alternative_option", {}))

        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(**best),
            alternative_option=SolutionOption(**alt),
        )
    except Exception:
        return build_fallback_explanation(payload, cost_inputs)


@app.get("/")
def root() -> dict:
    return {
        "name": "FlowStock AI Recommendation API",
        "endpoints": [
            "/health",
            "/api/inventory-recommendations",
            "/api/generate-recommendation-explanation",
        ],
    }


def build_fallback_explanation(payload: RecommendationRequest, cost_inputs: CostInput) -> RecommendationExplanation:
    product = lookup_product(payload)
    transfer_total = cost_inputs.estimated_transfer_total
    order_total = cost_inputs.estimated_order_total
    discount_total = cost_inputs.estimated_discount_total
    monitor_total = cost_inputs.estimated_monitor_total

    transfer_breakdown = cost_breakdown_from_total(
        transfer_total,
        item_cost=0.0,
        shipping_cost=max(0.0, (transfer_total or 0.0) * 0.72),
        other_costs=max(0.0, (transfer_total or 0.0) * 0.28),
    )
    order_breakdown = cost_breakdown_from_total(
        order_total,
        item_cost=float(product.get("price", 0.0) or 0.0) * cost_inputs.quantity,
        shipping_cost=max(15000.0, (cost_inputs.product_weight_kg or 0.0) * cost_inputs.quantity * 3200.0),
        other_costs=max(5000.0, float(product.get("price", 0.0) or 0.0) * cost_inputs.quantity * 0.015),
    )
    discount_breakdown = cost_breakdown_from_total(
        discount_total,
        item_cost=0.0,
        shipping_cost=max(0.0, (discount_total or 0.0) * 0.15),
        other_costs=max(5000.0, (discount_total or 0.0) * 0.85),
    )
    monitor_breakdown = cost_breakdown_from_total(
        monitor_total,
        item_cost=0.0,
        shipping_cost=0.0,
        other_costs=float(monitor_total or 0.0),
    )

    if payload.recommended_action == "Transfer":
        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(
                title=f"Transfer stock from {cost_inputs.source_warehouse or 'surplus warehouse'}",
                description=f"Transfer stock to {payload.warehouse_name} to cover the shortage of {payload.shortage} units for {payload.product_name}.",
                costImpact="Lower cost because it only incurs inter-warehouse shipping and handling.",
                riskLevel="Low",
                feasibility="High",
                cost_breakdown=transfer_breakdown,
            ),
            alternative_option=SolutionOption(
                title="Order from supplier",
                description="If donor stock is not available, place an order with the supplier.",
                costImpact="Higher cost due to item price, shipping, and supplier handling fees.",
                riskLevel="Medium",
                feasibility="Medium",
                cost_breakdown=order_breakdown,
            ),
        )

    if payload.recommended_action == "Order":
        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(
                title="Order from supplier",
                description=f"Stock at {payload.warehouse_name} is insufficient to meet 14-day demand.",
                costImpact="Prevents lost sales but increases procurement and shipping costs.",
                riskLevel="Low",
                feasibility="High",
                cost_breakdown=order_breakdown,
            ),
            alternative_option=SolutionOption(
                title=f"Transfer stock from {cost_inputs.source_warehouse or 'surplus warehouse'}",
                description="Search for donor warehouses before creating a new order.",
                costImpact="Cheaper if donor stock is available.",
                riskLevel="Medium",
                feasibility="Medium",
                cost_breakdown=transfer_breakdown,
            ),
        )

    if payload.recommended_action == "Discount":
        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(
                title="Apply discount",
                description=f"Stock of {payload.product_name} is above the target {payload.target_stock}; applying a discount can speed up turnover.",
                costImpact="Reduces holding cost but incurs markdown expense.",
                riskLevel="Low",
                feasibility="High",
                cost_breakdown=discount_breakdown,
            ),
            alternative_option=SolutionOption(
                title="Hold and monitor",
                description="Monitor sales and avoid discounting if demand starts to increase.",
                costImpact="Low immediate cost, but holding cost remains.",
                riskLevel="Medium",
                feasibility="High",
                cost_breakdown=monitor_breakdown,
            ),
        )

    return RecommendationExplanation(
        recommended_action="None",
        best_option=SolutionOption(
            title="No action needed",
            description=f"Stock of {payload.product_name} is within target levels at {payload.warehouse_name}.",
            costImpact="No additional cost.",
            riskLevel="Low",
            feasibility="High",
            cost_breakdown=monitor_breakdown,
        ),
        alternative_option=SolutionOption(
            title="Monitor daily",
            description="Monitor daily demand movements to stay prepared.",
            costImpact="Minimal cost.",
            riskLevel="Low",
            feasibility="High",
            cost_breakdown=monitor_breakdown,
        ),
    )


def build_prompt(payload: RecommendationRequest, cost_inputs: CostInput) -> str:
    transfer_source = cost_inputs.source_warehouse or "surplus warehouse"
    return f"""You are an inventory optimization assistant. Return JSON only with best_option and alternative_option.

Product: {payload.product_name}
Warehouse: {payload.warehouse_name}
Current stock: {payload.current_stock}
Predicted demand 14d: {payload.predicted_demand_14d}
Target stock: {payload.target_stock}
Shortage: {payload.shortage}
Status: {payload.status}
Recommended action: {payload.recommended_action}
Product price: {cost_inputs.product_price:.0f} IDR/unit
Product weight: {cost_inputs.product_weight_grams:.0f} grams ({cost_inputs.product_weight_kg:.2f} kg)
Estimated order total: {cost_inputs.estimated_order_total:.0f} IDR
Estimated transfer total: {cost_inputs.estimated_transfer_total:.0f} IDR
Estimated discount total: {cost_inputs.estimated_discount_total:.0f} IDR
Estimated monitor total: {cost_inputs.estimated_monitor_total:.0f} IDR
Transfer source: {transfer_source}
Transfer distance km: {0.0 if cost_inputs.transfer_distance_km is None else cost_inputs.transfer_distance_km:.2f}

Rules:
- best_option and alternative_option must each contain title, description, costImpact, riskLevel, feasibility, and cost_breakdown.
- cost_breakdown must include currency, item_cost, shipping_cost, other_costs, total_cost.
- Use the provided product price, weight, and estimated totals to explain the cost comparison clearly.
- riskLevel must be one of Low, Medium, High.
- feasibility must be one of Low, Medium, High.
- Output valid JSON only.
"""
