"""
ai_recommendation.py
--------------------
AIaaS integration with HuggingFace OpenAI-compatible router.
Uses Qwen/Qwen2.5-7B-Instruct to generate executive business insights
from sales forecast data.

The LLM is prompted to return structured JSON so the response maps
directly to the frontend AI Insight panel layout.
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BASE_URL = "https://router.huggingface.co/v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    weekly_forecast: List[Dict],
    historical_summary: Dict,
    metrics: Dict,
    peak_week: Optional[str],
    peak_sales: Optional[float],
    product_name: Optional[str] = None,
    month_label: Optional[str] = None,
) -> str:
    """
    Construct the user-facing prompt with aggregated forecast data.
    We keep numbers compact so it fits inside a small token window.
    """

    # Context header — tells the LLM which product & period it's analyzing
    product_ctx = product_name or "Unknown product"
    period_ctx  = month_label  or "Full year"

    # Summarise weekly forecast as a simple table (max 13 weeks shown)
    max_weeks = min(len(weekly_forecast), 13)
    weekly_rows = "\n".join(
        f"  {row['week']}: {row['predicted_sales']:.1f} units"
        for row in weekly_forecast[:max_weeks]
    )

    return f"""
You are an AI business analytics assistant reviewing sales forecast data.

CONTEXT:
  Product : {product_ctx}
  Period  : {period_ctx}

Generate a structured JSON response with EXACTLY these fields:
{{
  "summary": "<one sentence summarizing the key forecast trend for {product_ctx} in {period_ctx}>",
  "stockout_risk": "<Low | Medium | High>",
  "recommended_safety_stock": "<concise safety stock recommendation>",
  "recommended_action": "<one actionable recommendation specific to {product_ctx}>",
  "bullets": [
    "<insight 1: trend analysis for {product_ctx}>",
    "<insight 2: peak or risk in {period_ctx}>",
    "<insight 3: inventory action>"
  ]
}}

RULES:
- Maximum 3 bullets, each under 20 words
- Use simple executive-friendly language
- Mention the product name and period naturally in your response
- Do NOT include technical ML jargon
- stockout_risk must be exactly one of: Low, Medium, High
- bullets must be in plain text, start with "•"
- Return ONLY the JSON object — no extra text

FORECAST DATA ({period_ctx}):
Peak Day  : {peak_week or 'N/A'} — {f"{peak_sales:.1f} units/day" if peak_sales else 'N/A'}
Weekly totals (sum of daily predictions per week):
{weekly_rows}

METRICS (model evaluation on prior year):
  MAE: {metrics.get('mae', 0):.2f} units/day
  RMSE: {metrics.get('rmse', 0):.2f} units/day
  R²: {metrics.get('r2', 0):.2f}
  sMAPE: {metrics.get('smape', 0):.2%}

HISTORICAL BASELINE (same period, prior year): {historical_summary.get('avg_sales', 0):.1f} units/day (daily avg)
TREND vs HISTORY: {historical_summary.get('trend_pct', 0):+.0f}%
""".strip()


# ── LLM caller ────────────────────────────────────────────────────────────────

def _call_hf_llm(prompt: str) -> str:
    """Call HuggingFace router and return raw content string."""
    client = OpenAI(
        base_url=HF_BASE_URL,
        api_key=HF_TOKEN,
    )

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a business intelligence analyst. "
                    "Always respond ONLY with valid JSON as instructed. "
                    "Do not add markdown, code fences, or explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=300,
    )

    return completion.choices[0].message.content.strip()


# ── JSON extractor ────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """
    Safely extract a JSON object from the LLM response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown fences like ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Final fallback — return minimal structure
    logger.warning("LLM response could not be parsed as JSON. Raw: %s", raw[:200])
    return {}


# ── Main public function ───────────────────────────────────────────────────────

def generate_ai_recommendation(
    weekly_forecast: List[Dict],
    historical_sales_values: List[float],
    metrics: Dict,
    peak_week: Optional[str] = None,
    peak_sales: Optional[float] = None,
    product_name: Optional[str] = None,
    month_label: Optional[str] = None,
) -> Dict:
    """
    Generate AI-powered business insights using the HuggingFace LLM.

    Falls back gracefully to rule-based insight if the API is unavailable
    or the token is not configured.

    Returns a dict matching the frontend AI Insight panel fields:
        summary, stockout_risk, peak_week, peak_sales,
        recommended_safety_stock, recommended_action, bullets
    """

    # ── Compute historical summary stats ──────────────────────────────────────
    # historical_sales_values are DAILY values.
    # weekly_forecast values are WEEKLY SUMS (7-day aggregates).
    # We must convert to the same unit (daily avg) before computing trend_pct.
    if historical_sales_values:
        avg_hist = float(sum(historical_sales_values) / len(historical_sales_values))  # daily avg
    else:
        avg_hist = 0.0

    if weekly_forecast:
        total_weekly_sum = sum(w["predicted_sales"] for w in weekly_forecast)
        # Approximate number of forecast days (7 days/week × number of weeks)
        n_forecast_days = len(weekly_forecast) * 7
        forecast_daily_avg = total_weekly_sum / n_forecast_days
    else:
        forecast_daily_avg = 0.0

    trend_pct = ((forecast_daily_avg - avg_hist) / avg_hist * 100) if avg_hist > 0 else 0.0

    historical_summary = {
        "avg_sales": avg_hist,
        "trend_pct": trend_pct,
    }

    # ── Check if HF token is configured ──────────────────────────────────────
    if not HF_TOKEN or HF_TOKEN in ("xx", "your_token_here", ""):
        logger.warning("HF_TOKEN not configured. Using rule-based fallback insight.")
        return _rule_based_fallback(trend_pct, peak_week, peak_sales, product_name, month_label)

    # ── Build prompt and call LLM ─────────────────────────────────────────────
    prompt = _build_prompt(
        weekly_forecast=weekly_forecast,
        historical_summary=historical_summary,
        metrics=metrics,
        peak_week=peak_week,
        peak_sales=peak_sales,
        product_name=product_name,
        month_label=month_label,
    )

    try:
        raw_response = _call_hf_llm(prompt)
        parsed = _parse_llm_json(raw_response)

        # Validate and normalise critical fields
        stockout_risk = parsed.get("stockout_risk", "Medium")
        if stockout_risk not in ("Low", "Medium", "High"):
            stockout_risk = "Medium"

        bullets = parsed.get("bullets", [])
        if isinstance(bullets, list):
            bullets = [str(b).lstrip("•").strip() for b in bullets[:3]]
        else:
            bullets = []

        return {
            "summary": parsed.get("summary", f"Demand expected to {'increase' if trend_pct >= 0 else 'decrease'} by {abs(trend_pct):.0f}% over the forecast period"),
            "stockout_risk": stockout_risk,
            "peak_week": peak_week,
            "peak_sales": peak_sales,
            "recommended_safety_stock": parsed.get("recommended_safety_stock", "Maintain current stock levels"),
            "recommended_action": parsed.get("recommended_action", "Monitor weekly demand closely for any spikes"),
            "bullets": bullets,
        }

    except Exception as exc:
        logger.error("HuggingFace API call failed: %s. Using rule-based fallback.", exc)
        return _rule_based_fallback(trend_pct, peak_week, peak_sales, product_name, month_label)


# ── Rule-based fallback (when token not set / API unreachable) ────────────────

def _rule_based_fallback(
    trend_pct: float,
    peak_week: Optional[str],
    peak_sales: Optional[float],
    product_name: Optional[str] = None,
    month_label: Optional[str] = None,
) -> Dict:
    """Rule-based insight with product and month context."""
    item_ctx   = product_name or "this item"
    period_ctx = month_label  or "the forecast period"
    trend_word = "increase" if trend_pct >= 0 else "decrease"
    summary = (
        f"Demand for {item_ctx} expected to {trend_word} by "
        f"{abs(trend_pct):.0f}% in {period_ctx}"
    )

    if trend_pct > 15:
        risk = "High"
        safety_stock = f"+{int(abs(trend_pct))}%"
        action = f"Increase inventory of {item_ctx} before week 2 of {period_ctx}"
        bullets = [
            f"• {item_ctx} demand trending up {trend_pct:.0f}% vs prior year — plan ahead",
            f"• Peak expected at {peak_week} with ~{peak_sales:.0f} units" if peak_week else f"• Demand spike detected in {period_ctx}",
            "• Increase safety stock to prevent stockouts during demand surge",
        ]
    elif trend_pct < -15:
        risk = "Low"
        safety_stock = f"-{int(abs(trend_pct))}%"
        action = f"Delay replenishment of {item_ctx} — demand is trending below average in {period_ctx}"
        bullets = [
            f"• {item_ctx} demand declining {abs(trend_pct):.0f}% vs prior year in {period_ctx}",
            "• Risk of overstock if current replenishment rate is maintained",
            "• Consider promotions or bundle deals to stimulate demand",
        ]
    else:
        risk = "Medium"
        safety_stock = "Maintain current stock levels"
        action = f"Monitor weekly demand for {item_ctx} in {period_ctx} closely for any spikes"
        bullets = [
            f"• {item_ctx} demand stable in {period_ctx} — within ±15% of prior year",
            f"• Peak week: {peak_week} (~{peak_sales:.0f} units)" if peak_week else "• No significant demand spikes detected",
            "• Maintain current reorder schedule; review weekly",
        ]

    return {
        "summary": summary,
        "stockout_risk": risk,
        "peak_week": peak_week,
        "peak_sales": peak_sales,
        "recommended_safety_stock": safety_stock,
        "recommended_action": action,
        "bullets": bullets,
    }
