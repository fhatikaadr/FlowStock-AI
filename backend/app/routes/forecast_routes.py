import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.schemas import (
    ForecastRequest, ForecastResponse, SavedScenarioCreate,
    ScenarioResponse, ForecastDataPoint, Metrics
)
from app.models import db_models
from app.services.data_loader import load_and_preprocess_data
from app.forecasting.model_manager import ModelManager
from app.services.simulation_engine import apply_what_if_scenario
from app.services.ai_insight import generate_insights
from app.evaluation.metrics import calculate_metrics
from app.visualization.charts import generate_prediction_chart

logger = logging.getLogger(__name__)

router = APIRouter()
model_manager = ModelManager()


@router.post("/forecast/run", response_model=ForecastResponse)
def run_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    try:
        # 1. Load & preprocess data
        df = load_and_preprocess_data(store=request.store, item=request.item)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No data found for the given store/item combination."
            )

        # 2. Chronological train/validation split for metric calculation
        #    Use the last 20% of data (max 30 days) as validation
        val_size = min(int(len(df) * 0.2), 30)
        if val_size < 1:
            raise HTTPException(
                status_code=400,
                detail="Not enough data to create a validation split."
            )

        train_df = df.iloc[:-val_size].copy()
        val_df   = df.iloc[-val_size:].copy()

        # --- Train one model instance and evaluate on validation set ---
        model = model_manager.get_model(request.model)
        model.train(train_df)

        # Get validation predictions using the model's own predict interface.
        # For feature-based models (XGBoost) we pass val_df directly.
        # For sequence/univariate models (MLP, LSTM, SARIMA, Prophet) we
        # generate a future dataframe of the same length from train_end.
        val_future_df = model_manager.generate_future_dataframe(
            last_date=train_df["date"].max(),
            steps=val_size
        )
        val_predictions = model.predict(val_future_df)

        # Compute metrics vs true validation values
        metrics_dict = calculate_metrics(val_df["sales"].values, val_predictions)
        metrics_obj  = Metrics(**metrics_dict)

        # 3. Retrain on FULL data and generate the actual future forecast
        full_model = model_manager.get_model(request.model)
        full_model.train(df)

        future_df = model_manager.generate_future_dataframe(
            last_date=df["date"].max(),
            steps=request.forecast_period_days
        )
        raw_predictions = full_model.predict(future_df)
        future_dates    = future_df["date"]

        # 4. Apply what-if simulation adjustments
        adjusted_predictions = apply_what_if_scenario(
            predictions=raw_predictions,
            dates=future_dates,
            quarter=request.seasonality_impact and request.quarter,
            seasonality_impact=request.seasonality_impact,
            demand_multiplier=request.demand_multiplier or 1.0,
            promotional_effect=request.promotional_effect or 1.0,
        )

        # 5. Generate AI insights
        insights = generate_insights(df["sales"].values, adjusted_predictions)

        # 6. Format forecast data points
        forecast_points = [
            ForecastDataPoint(
                date=d.strftime("%Y-%m-%d"),
                predicted_sales=round(float(p), 4)
            )
            for d, p in zip(future_dates, adjusted_predictions)
        ]

        # 7. Persist forecast history to DB
        history_record = db_models.ForecastHistory(
            store=request.store,
            item=request.item,
            model_used=request.model,
            predictions=[pt.model_dump() for pt in forecast_points],
            metrics=metrics_dict,
            insight_summary=insights["summary"],
        )
        db.add(history_record)
        db.commit()

        # 8. Save chart (actual vs predicted over validation window)
        generate_prediction_chart(
            dates=val_df["date"].tolist(),
            actual_sales=val_df["sales"].tolist(),
            predicted_sales=val_predictions.tolist(),
            model_name=request.model,
            output_path="prediction_chart.png",
        )

        return ForecastResponse(
            status="success",
            selected_model=request.model,
            metrics=metrics_obj,
            forecast=forecast_points,
            insight=insights,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Forecast pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/history/{store}/{item}")
def get_forecast_history(store: int, item: int, db: Session = Depends(get_db)):
    history = (
        db.query(db_models.ForecastHistory)
        .filter(
            db_models.ForecastHistory.store == store,
            db_models.ForecastHistory.item  == item,
        )
        .order_by(db_models.ForecastHistory.forecast_date.desc())
        .all()
    )
    return history


@router.post("/forecast/scenario/save", response_model=ScenarioResponse)
def save_scenario(scenario: SavedScenarioCreate, db: Session = Depends(get_db)):
    db_scenario = db_models.SavedScenario(
        name=scenario.name,
        description=scenario.description,
        parameters=scenario.parameters,
    )
    db.add(db_scenario)
    db.commit()
    db.refresh(db_scenario)
    return db_scenario


@router.get("/forecast/models")
def get_models():
    return {"supported_models": model_manager.get_supported_models()}


@router.get("/forecast/metrics")
def get_recent_metrics(db: Session = Depends(get_db)):
    history = (
        db.query(db_models.ForecastHistory)
        .order_by(db_models.ForecastHistory.forecast_date.desc())
        .limit(10)
        .all()
    )
    return [
        {"model": h.model_used, "metrics": h.metrics, "date": h.forecast_date}
        for h in history
    ]
