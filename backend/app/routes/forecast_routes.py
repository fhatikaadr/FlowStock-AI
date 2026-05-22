import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.schemas import (
    ForecastRequest, ForecastResponse,
    SavedScenarioCreate, ScenarioResponse,
    ForecastDataPoint, WeeklyForecastPoint,
    HistoricalDataPoint, Metrics,
)
from app.models import db_models
from app.services.data_loader import load_and_preprocess_data
from app.forecasting.model_manager import ModelManager
from app.services.simulation_engine import apply_what_if_scenario
from app.services.ai_insight import generate_insights, aggregate_to_weekly
from app.evaluation.metrics import calculate_metrics
from app.visualization.charts import generate_prediction_chart

logger = logging.getLogger(__name__)
router = APIRouter()
model_manager = ModelManager()


@router.post("/forecast/run", response_model=ForecastResponse)
def run_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    try:
        # ── 1. Load & preprocess data ────────────────────────────────────────
        df = load_and_preprocess_data(store=request.store, item=request.item)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No data found for the given store/item combination."
            )

        # ── 2. Resolve frontend-friendly inputs ──────────────────────────────
        resolved_quarter    = request.resolved_quarter()
        resolved_promo      = request.resolved_promotional_effect()
        seasonality_impact  = (request.seasonality_impact or "").capitalize() or None

        # ── 3. Chronological year-based train/val split ──────────────────────
        last_year = df["year"].max()
        train_df  = df[df["year"] < last_year].copy()
        val_df    = df[df["year"] == last_year].copy()

        if train_df.empty or val_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough yearly data to create a train/validation split."
            )

        # ── 4. Train and evaluate on validation year ─────────────────────────
        model = model_manager.get_model(request.model)
        model.train(train_df)

        # XGBoost: pass val_df directly (has real lag features)
        # Other models: auto-regressive via generate_future_dataframe
        try:
            val_predictions = model.predict(val_df)
        except Exception:
            val_future_df   = model_manager.generate_future_dataframe(
                last_date=train_df["date"].max(),
                steps=len(val_df),
            )
            val_predictions = model.predict(val_future_df)

        metrics_dict = calculate_metrics(val_df["sales"].values, val_predictions)
        metrics_obj  = Metrics(**metrics_dict)

        # ── 5. Retrain on full data → real future forecast ────────────────────
        full_model = model_manager.get_model(request.model)
        full_model.train(df)

        future_df       = model_manager.generate_future_dataframe(
            last_date=df["date"].max(),
            steps=request.forecast_period_days,
        )
        raw_predictions = full_model.predict(future_df)
        future_dates    = future_df["date"]

        # ── 6. What-if simulation ─────────────────────────────────────────────
        adjusted_predictions = apply_what_if_scenario(
            predictions=raw_predictions,
            dates=future_dates,
            quarter=resolved_quarter,
            seasonality_impact=seasonality_impact,
            demand_multiplier=request.demand_multiplier or 1.0,
            promotional_effect=resolved_promo,
        )

        # ── 7. Build historical data (last 90 days for the chart blue line) ───
        hist_window = df.tail(90)
        historical_points = [
            HistoricalDataPoint(
                date=row["date"].strftime("%Y-%m-%d"),
                sales=float(row["sales"]),
            )
            for _, row in hist_window.iterrows()
        ]

        # ── 8. Daily forecast data points ─────────────────────────────────────
        forecast_points = [
            ForecastDataPoint(
                date=d.strftime("%Y-%m-%d"),
                predicted_sales=round(float(p), 4),
            )
            for d, p in zip(future_dates, adjusted_predictions)
        ]

        # ── 9. Weekly aggregated forecast (for the chart) ─────────────────────
        weekly_raw = aggregate_to_weekly(future_dates.tolist(), adjusted_predictions)
        weekly_points = [
            WeeklyForecastPoint(week=w["week"], predicted_sales=w["predicted_sales"])
            for w in weekly_raw
        ]

        # ── 10. AI Insights with peak week detection ───────────────────────────
        insights_dict = generate_insights(
            historical_sales=df["sales"].values,
            predicted_sales=adjusted_predictions,
            future_dates=future_dates.tolist(),
        )

        # ── 11. Persist to DB ─────────────────────────────────────────────────
        history_record = db_models.ForecastHistory(
            store=request.store,
            item=request.item,
            model_used=request.model,
            predictions=[p.model_dump() for p in forecast_points],
            metrics=metrics_dict,
            insight_summary=insights_dict["summary"],
        )
        db.add(history_record)
        db.commit()

        # ── 12. Save chart ────────────────────────────────────────────────────
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
            weekly_forecast=weekly_points,
            historical=historical_points,
            insight=insights_dict,
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
    return {
        "supported_models": model_manager.get_supported_models(),
        "supported_campaigns": list(__import__("app.models.schemas", fromlist=["CAMPAIGN_MULTIPLIERS"]).CAMPAIGN_MULTIPLIERS.keys()),
        "supported_quarters": ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Des)"],
    }


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
