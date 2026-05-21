from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime

class ForecastRequest(BaseModel):
    store: Optional[int] = None
    item: Optional[int] = None
    model: str = Field(default="xgboost", description="Model to use: xgboost, prophet, sarima, mlp, lstm")
    forecast_period_days: int = Field(default=90, ge=1, le=365)
    quarter: Optional[str] = Field(default=None, description="Q1, Q2, Q3, Q4")
    seasonality_impact: Optional[str] = Field(default=None, description="Low, Medium, High")
    demand_multiplier: Optional[float] = 1.0
    promotional_effect: Optional[float] = 1.0

class Metrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    medae: float
    mape: float
    r2: float
    evs: float

class ForecastDataPoint(BaseModel):
    date: str
    predicted_sales: float

class Insight(BaseModel):
    summary: str
    stockout_risk: str
    recommended_safety_stock: str
    recommended_action: str

class ForecastResponse(BaseModel):
    status: str
    selected_model: str
    metrics: Metrics
    forecast: List[ForecastDataPoint]
    insight: Insight

class SavedScenarioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class ScenarioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ModelMetadataResponse(BaseModel):
    model_name: str
    description: str
    last_trained: Optional[datetime]
    
    class Config:
        from_attributes = True
