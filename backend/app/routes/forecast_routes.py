from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.schemas import ForecastRequest, ForecastResponse, SavedScenarioCreate, ScenarioResponse, ModelMetadataResponse, ForecastDataPoint
from app.models import db_models
from app.services.data_loader import load_and_preprocess_data
from app.forecasting.model_manager import ModelManager
from app.services.simulation_engine import apply_what_if_scenario
from app.services.ai_insight import generate_insights
from app.evaluation.metrics import calculate_metrics
from app.visualization.charts import generate_prediction_chart

router = APIRouter()
model_manager = ModelManager()

@router.post("/forecast/run", response_model=ForecastResponse)
def run_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    try:
        # 1. Load Data
        df = load_and_preprocess_data(store=request.store, item=request.item)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given store/item combination")

        # 2. Train and Predict (In a real production system, models would be pre-trained and loaded)
        # Here we train dynamically for simplicity and to preserve the existing behavior.
        # We split the last N days as a validation set to calculate metrics.
        # N = min(len(df) * 0.2, 30) days for quick metric calculation
        val_size = min(int(len(df) * 0.2), 30)
        train_df = df.iloc[:-val_size]
        val_df = df.iloc[-val_size:]
        
        # Train on train_df to get metrics
        model = model_manager.get_model(request.model)
        model.train(train_df)
        val_predictions = model.predict(val_df)
        metrics = calculate_metrics(val_df['sales'].values, val_predictions)
        
        # Now train on full data to forecast future
        future_dates, raw_predictions = model_manager.train_and_predict(request.model, df, request.forecast_period_days)
        
        # 3. Apply What-If Simulation
        adjusted_predictions = apply_what_if_scenario(
            predictions=raw_predictions,
            dates=future_dates,
            quarter=request.quarter,
            seasonality_impact=request.seasonality_impact,
            demand_multiplier=request.demand_multiplier,
            promotional_effect=request.promotional_effect
        )
        
        # 4. Generate Insights
        insights = generate_insights(df['sales'].values, adjusted_predictions)
        
        # 5. Format Response
        forecast_points = []
        for d, p in zip(future_dates, adjusted_predictions):
            forecast_points.append(ForecastDataPoint(date=d.strftime("%Y-%m-%d"), predicted_sales=float(p)))
            
        # Optional: Save history to DB
        history = db_models.ForecastHistory(
            store=request.store,
            item=request.item,
            model_used=request.model,
            predictions=[p.model_dump() for p in forecast_points],
            metrics=metrics,
            insight_summary=insights["summary"]
        )
        db.add(history)
        db.commit()

        # Generate Chart locally (Optional in background)
        generate_prediction_chart(future_dates, [], adjusted_predictions, request.model)

        return ForecastResponse(
            status="success",
            selected_model=request.model,
            metrics=metrics,
            forecast=forecast_points,
            insight=insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/history/{store}/{item}")
def get_forecast_history(store: int, item: int, db: Session = Depends(get_db)):
    history = db.query(db_models.ForecastHistory).filter(
        db_models.ForecastHistory.store == store,
        db_models.ForecastHistory.item == item
    ).order_by(db_models.ForecastHistory.forecast_date.desc()).all()
    return history

@router.post("/forecast/scenario/save", response_model=ScenarioResponse)
def save_scenario(scenario: SavedScenarioCreate, db: Session = Depends(get_db)):
    db_scenario = db_models.SavedScenario(
        name=scenario.name,
        description=scenario.description,
        parameters=scenario.parameters
    )
    db.add(db_scenario)
    db.commit()
    db.refresh(db_scenario)
    return db_scenario

@router.get("/forecast/models")
def get_models():
    return {"supported_models": model_manager.get_supported_models()}

@router.get("/forecast/metrics")
def get_metrics(db: Session = Depends(get_db)):
    # Simple aggregate of recent metrics
    history = db.query(db_models.ForecastHistory).order_by(db_models.ForecastHistory.forecast_date.desc()).limit(10).all()
    return [{"model": h.model_used, "metrics": h.metrics, "date": h.forecast_date} for h in history]
