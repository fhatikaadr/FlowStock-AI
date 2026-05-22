import numpy as np
import pandas as pd
from typing import List, Dict


def _week_label(date: pd.Timestamp) -> str:
    """
    Convert a date to a frontend-friendly week label, e.g. 'Oct W1', 'Nov W3'.
    Week number within the month is calculated as ceil(day / 7).
    """
    month_abbr = date.strftime("%b")          # 'Jan', 'Feb', ...
    week_of_month = (date.day - 1) // 7 + 1  # 1-5
    return f"{month_abbr} W{week_of_month}"


def aggregate_to_weekly(dates: List, predicted_sales: np.ndarray) -> List[Dict]:
    """
    Aggregate daily predictions into weekly totals grouped by week label.
    Returns list of {"week": "Oct W1", "predicted_sales": float}.
    """
    records = []
    week_totals: Dict[str, float] = {}
    week_order: List[str] = []

    for date, pred in zip(dates, predicted_sales):
        dt = pd.Timestamp(date)
        label = _week_label(dt)
        if label not in week_totals:
            week_totals[label] = 0.0
            week_order.append(label)
        week_totals[label] += float(pred)

    for label in week_order:
        records.append({"week": label, "predicted_sales": round(week_totals[label], 2)})

    return records


def generate_insights(
    historical_sales: np.ndarray,
    predicted_sales: np.ndarray,
    future_dates=None,
) -> dict:
    """
    Generates AI business insights comparing the last 90 days of historical data
    to the upcoming forecast period — avoiding the distortion of comparing against
    ALL historical years.
    """
    # Compare against the most recent 90 days of history (same forecast window)
    recent_hist = historical_sales[-90:] if len(historical_sales) >= 90 else historical_sales
    hist_mean = float(np.mean(recent_hist))
    pred_mean = float(np.mean(predicted_sales))

    if hist_mean == 0:
        hist_mean = 1e-5

    diff_pct = ((pred_mean - hist_mean) / hist_mean) * 100

    # Trend direction based on linear slope of predictions
    if len(predicted_sales) >= 2:
        slope = float(np.polyfit(range(len(predicted_sales)), predicted_sales, 1)[0])
        trend = "increase" if slope > 0 else "decrease"
        trend_pct = abs(diff_pct)
    else:
        trend = "increase" if diff_pct > 0 else "decrease"
        trend_pct = abs(diff_pct)

    summary = f"Demand expected to {trend} by {trend_pct:.0f}% over the forecast period"

    # Stockout risk based on forecast trend vs recent history
    if diff_pct > 15:
        stockout_risk = "High"
        recommended_safety_stock = f"+{int(abs(diff_pct))}%"
        recommended_action = "Increase inventory before week 2 of the forecast period"
    elif diff_pct < -15:
        stockout_risk = "Low"
        recommended_safety_stock = f"-{int(abs(diff_pct))}%"
        recommended_action = "Delay replenishment — demand is trending below average"
    else:
        stockout_risk = "Medium"
        recommended_safety_stock = "Maintain current stock levels"
        recommended_action = "Monitor weekly demand closely for any spikes"

    # Peak week detection (weekly aggregation)
    peak_week = None
    peak_sales = None
    if future_dates is not None and len(future_dates) > 0:
        weekly = aggregate_to_weekly(future_dates, predicted_sales)
        if weekly:
            peak_entry = max(weekly, key=lambda x: x["predicted_sales"])
            peak_week  = peak_entry["week"]
            peak_sales = round(peak_entry["predicted_sales"], 2)

    return {
        "summary":                  summary,
        "stockout_risk":            stockout_risk,
        "peak_week":                peak_week,
        "peak_sales":               peak_sales,
        "recommended_safety_stock": recommended_safety_stock,
        "recommended_action":       recommended_action,
    }
