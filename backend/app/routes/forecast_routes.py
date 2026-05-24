import logging
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.schemas import (
    ForecastRequest, ForecastResponse,
    SavedScenarioCreate, ScenarioResponse,
    ForecastDataPoint, WeeklyForecastPoint,
    HistoricalDataPoint, Metrics,
    AIInsightRequest, AIInsightResponse,
    ForecastHistoryItem, MetricsHistoryItem,
)
from app.models import db_models
from app.services.data_loader import load_and_preprocess_data, load_raw_daily_data
from app.forecasting.model_manager import ModelManager
from app.services.simulation_engine import apply_what_if_scenario
from app.services.ai_insight import aggregate_to_weekly
from app.services.ai_recommendation import generate_ai_recommendation
from app.evaluation.metrics import calculate_metrics
from app.visualization.charts import generate_prediction_chart

import numpy as np
from datetime import timedelta
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()
model_manager = ModelManager()


# NOTE: _distribute_weekly_to_daily has been removed.
# The pipeline now works at daily resolution end-to-end, so each
# prediction already corresponds to a single calendar day.


# ─────────────────────────────────────────────────────────────────────────────
# POST /forecast/run
# Runs the full forecasting pipeline for a store/item/month combination.
# Always forecasts a full 52-week window into the next year, then slices the
# requested month from those results.
# ─────────────────────────────────────────────────────────────────────────────
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

        logger.info(
            "Loaded %d rows for store=%s item=%s (min_date=%s max_date=%s)",
            len(df),
            request.store,
            request.item,
            df["date"].min().date(),
            df["date"].max().date(),
        )

        # ── 2. Resolve frontend-friendly inputs ──────────────────────────────
        resolved_month     = request.resolved_month()
        seasonality_impact = None

        # ── 3. Chronological year-based train/val split ──────────────────────
        # Data is now at daily resolution. Detect a stray artifact year at the
        # boundary by checking if the max year has ≤ 7 rows (less than a week).
        last_year = int(df["date"].dt.year.max())
        if (df["date"].dt.year == last_year).sum() <= 7:
            logger.info(
                "Daily boundary artifact detected: year %d has ≤7 rows. Rolling back to %d.",
                last_year, last_year - 1,
            )
            last_year -= 1

        train_df = df[df["date"].dt.year < last_year].copy()
        val_df   = df[df["date"].dt.year == last_year].copy()

        if train_df.empty or val_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough yearly data to create a train/validation split."
            )

        # ── 4. Train and evaluate on validation year ─────────────────────────
        model = model_manager.get_model(request.model)
        model.train(train_df)

        # XGBoost: pass val_df directly (has real lag features from the loader)
        # Other models: auto-regressive via generate_future_dataframe
        try:
            val_predictions = model.predict(val_df)
        except Exception:
            val_future_df = model_manager.generate_future_dataframe(
                last_date=train_df["date"].max(),
                steps=len(val_df),
            )
            val_predictions = model.predict(val_future_df)

        val_predictions = np.nan_to_num(val_predictions, nan=0.0, posinf=0.0, neginf=0.0)

        metrics_dict = calculate_metrics(val_df["sales"].values, val_predictions)
        metrics_obj  = Metrics(**metrics_dict)

        # ── 5. Retrain on full data → real future forecast ────────────────────
        full_model = model_manager.get_model(request.model)
        full_model.train(df)

        # Generate a full 365-day (one year) daily forecast ahead.
        # Working at daily resolution means each prediction maps to exactly
        # one calendar day — no weekly distribution needed.
        target_year          = int(last_year + 1)
        forecast_period_days = 365

        future_df = model_manager.generate_future_dataframe(
            last_date=df["date"].max(),
            steps=forecast_period_days,
        )
        raw_predictions = full_model.predict(future_df)
        raw_predictions = np.nan_to_num(raw_predictions, nan=0.0, posinf=0.0, neginf=0.0)
        future_dates    = future_df["date"]

        # ── 6. What-if simulation ─────────────────────────────────────────────
        adjusted_predictions = apply_what_if_scenario(
            predictions=raw_predictions,
            dates=future_dates,
            quarter=None,
            seasonality_impact=seasonality_impact,
            demand_multiplier=1.0,
            promotional_effect=1.0,
        )
        adjusted_predictions = np.nan_to_num(adjusted_predictions, nan=0.0, posinf=0.0, neginf=0.0)

        # ── 6b. Slice to the requested month (daily resolution) ───────────────
        if resolved_month:
            mask = (
                (future_dates.dt.year  == target_year) &
                (future_dates.dt.month == resolved_month)
            )
            daily_dates_month = future_dates[mask]
            daily_preds_month = adjusted_predictions[mask.to_numpy()]
            if len(daily_dates_month) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No forecast points for {target_year}-{resolved_month:02d}. "
                        "Check that the dataset covers dates close to year-end."
                    ),
                )
        else:
            daily_dates_month = future_dates
            daily_preds_month = adjusted_predictions

        # ── 7. Daily historical data — raw Supabase/CSV ───────────────────────
        raw_daily = load_raw_daily_data(store=request.store, item=request.item)
        if resolved_month:
            hist_daily = raw_daily[
                (raw_daily["date"].dt.year  == last_year) &
                (raw_daily["date"].dt.month == resolved_month)
            ]
        else:
            hist_daily = raw_daily[raw_daily["date"].dt.year == last_year]

        historical_points = [
            HistoricalDataPoint(
                date=row["date"].strftime("%Y-%m-%d"),
                sales=float(row["sales"]),
            )
            for _, row in hist_daily.iterrows()
        ]

        # ── 8. Daily forecast data points ─────────────────────────────────────
        # Predictions are already at daily resolution — emit them directly.
        forecast_points = [
            ForecastDataPoint(
                date=d.strftime("%Y-%m-%d"),
                predicted_sales=round(float(p), 4),
            )
            for d, p in zip(daily_dates_month, daily_preds_month)
        ]

        # ── 9. Weekly aggregated forecast (for the chart summary / ai-insight) ──
        # Aggregate the daily predictions to weekly buckets for the summary chart.
        daily_dates_list = list(daily_dates_month)
        daily_preds_arr  = np.array(daily_preds_month)
        weekly_raw = aggregate_to_weekly(daily_dates_list, daily_preds_arr)
        weekly_points = [
            WeeklyForecastPoint(week=w["week"], predicted_sales=w["predicted_sales"])
            for w in weekly_raw
        ]

        # ── 10. AI Insight — infer per month × item ───────────────────────────
        # Run LLM (or rule-based fallback) scoped to the selected month & item.
        # Historical baseline: same month in the previous year (daily resolution).
        if resolved_month:
            hist_window = df[
                (df["date"].dt.year  == last_year) &
                (df["date"].dt.month == resolved_month)
            ].copy()
        else:
            hist_window = df.tail(90)  # ~3 months of daily data

        # Compute peak_week / peak_sales from the monthly weekly aggregation.
        if weekly_raw:
            _peak = max(weekly_raw, key=lambda w: w["predicted_sales"])
            peak_week  = _peak["week"]
            peak_sales = round(float(_peak["predicted_sales"]), 2)
        else:
            peak_week  = None
            peak_sales = None

        insights_dict = generate_ai_recommendation(
            weekly_forecast=weekly_raw,
            historical_sales_values=hist_window["sales"].tolist(),
            metrics=metrics_dict,
            peak_week=peak_week,
            peak_sales=peak_sales,
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
            target_year=target_year,
            forecast_month=request.month,
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


# ─────────────────────────────────────────────────────────────────────────────
# GET /forecast/history/{store}/{item}
# Returns past forecast runs for a given store+item pair, newest first.
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/forecast/history/{store}/{item}",
    response_model=list[ForecastHistoryItem],
    summary="Get forecast history for a store/item pair",
)
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
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast history found for store={store}, item={item}.",
        )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# POST /forecast/scenario/save
# Saves a named forecast scenario (store, item, model, month) to the DB.
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/forecast/scenario/save",
    response_model=ScenarioResponse,
    summary="Save a forecast scenario",
)
def save_scenario(scenario: SavedScenarioCreate, db: Session = Depends(get_db)):
    db_scenario = db_models.SavedScenario(
        name=scenario.name,
        description=scenario.description,
        parameters=scenario.parameters,
        month=scenario.month,
    )
    db.add(db_scenario)
    db.commit()
    db.refresh(db_scenario)
    return db_scenario


# ─────────────────────────────────────────────────────────────────────────────
# GET /forecast/models
# Returns the list of supported forecasting models and valid month names.
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/forecast/models",
    summary="List supported forecasting models and valid month options",
)
def get_models():
    from app.models.schemas import MONTH_MAP
    return {
        "supported_models": model_manager.get_supported_models(),
        "supported_months": sorted(
            {k.capitalize() for k in MONTH_MAP if not k.isdigit() and len(k) > 3},
            key=lambda m: MONTH_MAP[m.lower()],
        ),
        "note": (
            "Pass 'month' in the /forecast/run request body (e.g. 'March'). "
            "The backend always forecasts a full year ahead and slices the requested month."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /forecast/metrics
# Returns the 10 most recent forecast metric summaries from the DB.
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/forecast/metrics",
    response_model=list[MetricsHistoryItem],
    summary="Get recent forecast performance metrics",
)
def get_recent_metrics(db: Session = Depends(get_db)):
    history = (
        db.query(db_models.ForecastHistory)
        .order_by(db_models.ForecastHistory.forecast_date.desc())
        .limit(10)
        .all()
    )
    return [
        MetricsHistoryItem(
            model=h.model_used,
            metrics=h.metrics,
            date=h.forecast_date,
        )
        for h in history
    ]


# ─────────────────────────────────────────────────────────────────────────────
# POST /forecast/ai-insight
# Generates AI-powered business insights from already-computed forecast data.
# Accepts weekly_forecast + historical + metrics from a previous /forecast/run
# call and returns natural-language insights and bullet points.
# Falls back to rule-based insight when HF_TOKEN is not configured.
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/forecast/ai-insight",
    response_model=AIInsightResponse,
    summary="Generate AI business insights from forecast results",
)
def get_ai_insight(request: AIInsightRequest):
    try:
        metrics_dict = request.metrics.model_dump() if request.metrics else {}

        historical_values = [
            float(pt.sales) for pt in (request.historical or [])
        ]

        # ── Resolve weekly_forecast from daily data or use pre-aggregated ──────
        # Priority: daily forecast (request.forecast) > weekly_forecast supplied
        if request.forecast:
            # Aggregate daily → weekly so the LLM prompt stays concise
            daily_dates = [pd.Timestamp(pt.date) for pt in request.forecast]
            daily_preds = np.array([pt.predicted_sales for pt in request.forecast])
            weekly_forecast_dicts = aggregate_to_weekly(daily_dates, daily_preds)
        elif request.weekly_forecast:
            weekly_forecast_dicts = [
                {"week": pt.week, "predicted_sales": pt.predicted_sales}
                for pt in request.weekly_forecast
            ]
        else:
            weekly_forecast_dicts = []

        if not weekly_forecast_dicts:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'forecast' (daily) or 'weekly_forecast' in the request body.",
            )

        # ── Auto-compute peak week / peak sales from the weekly summary ─────────
        peak_week  = request.peak_week
        peak_sales = request.peak_sales
        if (peak_week is None or peak_sales is None) and weekly_forecast_dicts:
            best       = max(weekly_forecast_dicts, key=lambda w: w["predicted_sales"])
            peak_week  = peak_week  or best["week"]
            peak_sales = peak_sales or round(float(best["predicted_sales"]), 2)

        result = generate_ai_recommendation(
            weekly_forecast=weekly_forecast_dicts,
            historical_sales_values=historical_values,
            metrics=metrics_dict,
            peak_week=peak_week,
            peak_sales=peak_sales,
        )

        hf_token = os.getenv("HF_TOKEN", "")
        source = "rule-based" if not hf_token or hf_token in ("xx", "your_token_here", "") else "llm"

        return AIInsightResponse(
            summary=result["summary"],
            stockout_risk=result["stockout_risk"],
            peak_week=result.get("peak_week"),
            peak_sales=result.get("peak_sales"),
            recommended_safety_stock=result["recommended_safety_stock"],
            recommended_action=result["recommended_action"],
            bullets=result.get("bullets", []),
            source=source,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI Insight endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

