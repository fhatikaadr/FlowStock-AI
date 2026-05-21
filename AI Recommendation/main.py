from __future__ import annotations

import json
import os
from pathlib import Path
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


class RecommendationExplanation(BaseModel):
    recommended_action: Literal["None", "Transfer", "Discount", "Order"]
    best_option: SolutionOption
    alternative_option: SolutionOption


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "flowstock-ai-recommendation-api"}


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


@app.post("/api/generate-recommendation-explanation")
async def generate_recommendation_explanation(payload: RecommendationRequest) -> RecommendationExplanation:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key:
        return build_fallback_explanation(payload)

    prompt = build_prompt(payload)

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
            return build_fallback_explanation(payload)

        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = json.loads(text)

        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(**parsed["best_option"]),
            alternative_option=SolutionOption(**parsed["alternative_option"]),
        )
    except Exception:
        return build_fallback_explanation(payload)


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


def build_fallback_explanation(payload: RecommendationRequest) -> RecommendationExplanation:
    if payload.recommended_action == "Transfer":
        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(
                title="Transfer stock from surplus warehouse",
                description=f"Pindahkan stok ke {payload.warehouse_name} untuk menutup shortage {payload.shortage} unit pada {payload.product_name}.",
                costImpact="Biaya lebih rendah dibanding order baru.",
                riskLevel="Low",
                feasibility="High",
            ),
            alternative_option=SolutionOption(
                title="Order from supplier",
                description="Jika donor stock tidak tersedia, lakukan order ke supplier.",
                costImpact="Biaya lebih tinggi karena purchase dan lead time.",
                riskLevel="Medium",
                feasibility="Medium",
            ),
        )

    if payload.recommended_action == "Order":
        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(
                title="Order from supplier",
                description=f"Stok di {payload.warehouse_name} tidak cukup untuk demand 14 hari.",
                costImpact="Mencegah lost sales tetapi menambah biaya pembelian.",
                riskLevel="Low",
                feasibility="High",
            ),
            alternative_option=SolutionOption(
                title="Transfer stock if possible",
                description="Cari warehouse donor sebelum order baru.",
                costImpact="Lebih murah jika stok donor tersedia.",
                riskLevel="Medium",
                feasibility="Medium",
            ),
        )

    if payload.recommended_action == "Discount":
        return RecommendationExplanation(
            recommended_action=payload.recommended_action,
            best_option=SolutionOption(
                title="Apply discount",
                description=f"Stok {payload.product_name} lebih tinggi dari target {payload.target_stock}; diskon bisa mempercepat perputaran.",
                costImpact="Menurunkan holding cost.",
                riskLevel="Low",
                feasibility="High",
            ),
            alternative_option=SolutionOption(
                title="Hold and monitor",
                description="Pantau penjualan dan hindari diskon jika demand mulai naik.",
                costImpact="Biaya rendah sekarang, tetapi holding cost tetap berjalan.",
                riskLevel="Medium",
                feasibility="High",
            ),
        )

    return RecommendationExplanation(
        recommended_action="None",
        best_option=SolutionOption(
            title="No action needed",
            description=f"Stok {payload.product_name} masih sesuai target di {payload.warehouse_name}.",
            costImpact="No additional cost.",
            riskLevel="Low",
            feasibility="High",
        ),
        alternative_option=SolutionOption(
            title="Monitor daily",
            description="Pantau pergerakan demand harian untuk menjaga kesiapan.",
            costImpact="Minimal cost.",
            riskLevel="Low",
            feasibility="High",
        ),
    )


def build_prompt(payload: RecommendationRequest) -> str:
    return f"""You are an inventory optimization assistant. Return JSON only with best_option and alternative_option.

Product: {payload.product_name}
Warehouse: {payload.warehouse_name}
Current stock: {payload.current_stock}
Predicted demand 14d: {payload.predicted_demand_14d}
Target stock: {payload.target_stock}
Shortage: {payload.shortage}
Status: {payload.status}
Recommended action: {payload.recommended_action}

Rules:
- best_option and alternative_option must each contain title, description, costImpact, riskLevel, feasibility.
- riskLevel must be one of Low, Medium, High.
- feasibility must be one of Low, Medium, High.
- Output valid JSON only.
"""
