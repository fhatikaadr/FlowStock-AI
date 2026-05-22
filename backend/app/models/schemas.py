from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ── Campaign name → promotional_effect multiplier mapping ───────────────────
CAMPAIGN_MULTIPLIERS: Dict[str, float] = {
    "none":           1.00,
    "payday sale":    1.15,
    "flash sale":     1.25,
    "year end sale":  1.30,
    "holiday promo":  1.20,
    "clearance":      0.90,
}

# ── Quarter label → quarter code mapping ────────────────────────────────────
QUARTER_MAP: Dict[str, str] = {
    # Frontend label    backend quarter code
    "jan - mar":        "Q1",
    "jan-mar":          "Q1",
    "q1":               "Q1",
    "apr - jun":        "Q2",
    "apr-jun":          "Q2",
    "q2":               "Q2",
    "jul - sep":        "Q3",
    "jul-sep":          "Q3",
    "q3":               "Q3",
    "oct - des":        "Q4",
    "oct-des":          "Q4",
    "oct - dec":        "Q4",
    "oct-dec":          "Q4",
    "q4":               "Q4",
}


class ForecastRequest(BaseModel):
    store: Optional[int] = None
    item: Optional[int] = None
    model: str = Field(
        default="xgboost",
        description="Model: xgboost | prophet | sarima | mlp | lstm"
    )
    forecast_period_days: int = Field(default=90, ge=1, le=365)

    # Quarter accepts both "Q4" and "Oct - Des" style labels
    quarter: Optional[str] = Field(
        default=None,
        description="Q1/Q2/Q3/Q4 or 'Jan - Mar' / 'Oct - Des' style labels"
    )
    seasonality_impact: Optional[str] = Field(
        default=None,
        description="Low | Medium | High"
    )
    demand_multiplier: Optional[float] = Field(default=1.0, ge=0.0)

    # Named campaign support — mapped to promotional_effect internally
    campaign: Optional[str] = Field(
        default=None,
        description="Campaign name: 'Payday Sale', 'Flash Sale', 'Year End Sale', etc."
    )
    promotional_effect: Optional[float] = Field(default=1.0, ge=0.0)

    def resolved_quarter(self) -> Optional[str]:
        """Normalise quarter label to Q1-Q4."""
        if not self.quarter:
            return None
        return QUARTER_MAP.get(self.quarter.strip().lower(), self.quarter.upper())

    def resolved_promotional_effect(self) -> float:
        """Return campaign multiplier if a campaign is named, else raw promotional_effect."""
        if self.campaign:
            return CAMPAIGN_MULTIPLIERS.get(self.campaign.strip().lower(), 1.0)
        return self.promotional_effect or 1.0


class Metrics(BaseModel):
    mae:   float
    mse:   float
    rmse:  float
    medae: float
    mape:  float
    smape: float
    r2:    float
    evs:   float


class ForecastDataPoint(BaseModel):
    date:            str
    predicted_sales: float


class WeeklyForecastPoint(BaseModel):
    week:            str   # e.g. "Oct W1"
    predicted_sales: float


class HistoricalDataPoint(BaseModel):
    date:  str
    sales: float


class Insight(BaseModel):
    summary:                  str
    stockout_risk:            str
    peak_week:                Optional[str]   # e.g. "Nov W4"
    peak_sales:               Optional[float]
    recommended_safety_stock: str
    recommended_action:       str


class ForecastResponse(BaseModel):
    status:         str
    selected_model: str
    metrics:        Metrics
    # Daily forecast (for raw data consumers / tables)
    forecast:         List[ForecastDataPoint]
    # Weekly aggregated forecast (for the chart)
    weekly_forecast:  List[WeeklyForecastPoint]
    # Historical data for the chart's blue line
    historical:       List[HistoricalDataPoint]
    insight:          Insight


class SavedScenarioCreate(BaseModel):
    name:        str
    description: Optional[str] = None
    parameters:  Dict[str, Any]


class ScenarioResponse(BaseModel):
    id:          int
    name:        str
    description: Optional[str]
    parameters:  Dict[str, Any]
    created_at:  datetime

    class Config:
        from_attributes = True


class ModelMetadataResponse(BaseModel):
    model_name:   str
    description:  str
    last_trained: Optional[datetime]

    class Config:
        from_attributes = True
