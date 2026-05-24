from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Date
from datetime import datetime
from app.database import Base

class HistoricalSales(Base):
    __tablename__ = "historical_sales"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    store = Column(Integer, index=True)
    item = Column(Integer, index=True)
    sales = Column(Float)

class SavedScenario(Base):
    __tablename__ = "saved_scenarios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    parameters = Column(JSON) # e.g., {"store": 1, "item": 23, "model": "xgboost", "month": "March"}
    month = Column(String, nullable=True)  # e.g. "March"
    created_at = Column(DateTime, default=datetime.utcnow)

class ForecastHistory(Base):
    __tablename__ = "forecast_history"
    
    id = Column(Integer, primary_key=True, index=True)
    store = Column(Integer, index=True)
    item = Column(Integer, index=True)
    model_used = Column(String)
    forecast_date = Column(DateTime, default=datetime.utcnow)
    predictions = Column(JSON) # e.g., [{"date": "...", "predicted_sales": ...}]
    metrics = Column(JSON)
    insight_summary = Column(String, nullable=True)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, index=True)
    description = Column(String)
    last_trained = Column(DateTime, nullable=True)
    hyperparameters = Column(JSON)
