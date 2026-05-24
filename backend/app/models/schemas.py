from pydantic import BaseModel, Field, ConfigDict
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

# ── Month label → month number mapping ──────────────────────────────────────
MONTH_MAP: Dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "okt": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "des": 12,
    "december": 12,
}


class ForecastRequest(BaseModel):
    store: Optional[int] = Field(default=None, examples=[1])
    item: Optional[int] = Field(default=None, examples=[23])
    model: str = Field(
        default="xgboost",
        description="Model: xgboost | prophet | sarima | mlp | lstm",
        examples=["xgboost"],
    )

    # Month filter for frontend chart (e.g. "March")
    month: Optional[str] = Field(
        default=None,
        description="Month name, e.g. March",
        examples=["March"],
    )

    def resolved_month(self) -> Optional[int]:
        """Normalise month label to month number (1-12)."""
        if not self.month:
            return None
        key = self.month.strip().lower()
        if key.isdigit():
            value = int(key)
            return value if 1 <= value <= 12 else None
        return MONTH_MAP.get(key)


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
    bullets:                  Optional[List[str]] = []   # LLM-generated bullet points


# ── AI Recommendation endpoint schemas (AIaaS) ───────────────────────────────

class AIInsightRequest(BaseModel):
    """Input payload for POST /forecast/ai-insight"""
    store:              Optional[int]   = None
    item:               Optional[int]   = None
    # Accepts EITHER daily forecast (preferred) OR pre-aggregated weekly_forecast.
    # If both are supplied, daily_forecast takes precedence.
    forecast:           Optional[List[ForecastDataPoint]]       = None  # daily
    weekly_forecast:    Optional[List[WeeklyForecastPoint]]     = []    # weekly (legacy / chart)
    historical:         Optional[List[HistoricalDataPoint]] = []
    metrics:            Optional[Metrics]  = None
    peak_week:          Optional[str]   = None
    peak_sales:         Optional[float] = None


class AIInsightResponse(BaseModel):
    """Response from the AI Recommendation endpoint"""
    summary:                  str
    stockout_risk:            str
    peak_week:                Optional[str]
    peak_sales:               Optional[float]
    recommended_safety_stock: str
    recommended_action:       str
    bullets:                  List[str] = []
    source:                   str = "llm"   # 'llm' or 'rule-based'


class ForecastResponse(BaseModel):
    status:         str
    selected_model: str
    target_year:    int                    # e.g. 2018 — the year being forecast
    forecast_month: Optional[str]          # e.g. "March" — the requested month, if any
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
    # parameters should include store, item, model, month used in the forecast
    parameters:  Dict[str, Any]
    month:       Optional[str] = Field(
        default=None,
        description="Month the scenario was generated for, e.g. March",
    )


class ScenarioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:          int
    name:        str
    description: Optional[str]
    parameters:  Dict[str, Any]
    month:       Optional[str]
    created_at:  datetime


class ForecastHistoryItem(BaseModel):
    """One row returned by GET /forecast/history/{store}/{item}"""
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id:              int
    store:           Optional[int]
    item:            Optional[int]
    model_used:      str
    forecast_date:   datetime
    predictions:     List[ForecastDataPoint]
    metrics:         Optional[Dict[str, Any]]
    insight_summary: Optional[str]


class MetricsHistoryItem(BaseModel):
    """One row returned by GET /forecast/metrics"""
    model:   str
    metrics: Optional[Dict[str, Any]]
    date:    Optional[datetime]


class ModelMetadataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model_name:   str
    description:  str
    last_trained: Optional[datetime]
