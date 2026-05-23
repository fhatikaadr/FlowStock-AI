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
) -> str:
    """
    Construct the user-facing prompt with aggregated forecast data.
    We keep numbers compact so it fits inside a small token window.
    """

    # Summarise weekly forecast as a simple table (max 13 weeks shown)
    max_weeks = min(len(weekly_forecast), 13)
    weekly_rows = "\n".join(
        f"  {row['week']}: {row['predicted_sales']:.1f} units"
        for row in weekly_forecast[:max_weeks]
    )

    return f"""
You are an AI business analytics assistant reviewing sales forecast data.

Generate a structured JSON response with EXACTLY these fields:
{{
  "summary": "<one sentence summarizing the key forecast trend>",
  "stockout_risk": "<Low | Medium | High>",
  "recommended_safety_stock": "<concise safety stock recommendation>",
  "recommended_action": "<one actionable recommendation>",
  "bullets": [
    "<insight 1: trend analysis>",
    "<insight 2: peak or risk>",
    "<insight 3: inventory action>"
  ]
}}

RULES:
- Maximum 3 bullets, each under 20 words
- Use simple executive-friendly language
- Do NOT include technical ML jargon
- stockout_risk must be exactly one of: Low, Medium, High
- bullets must be in plain text, start with "•"
- Return ONLY the JSON object — no extra text

FORECAST DATA:
Peak Week: {peak_week or 'N/A'}
Peak Sales: {peak_sales or 'N/A'} units
Weekly Forecast:
{weekly_rows}

METRICS (evaluation on last year):
  MAE: {metrics.get('mae', 0):.2f}
  RMSE: {metrics.get('rmse', 0):.2f}
  R²: {metrics.get('r2', 0):.2f}
  sMAPE: {metrics.get('smape', 0):.2%}

HISTORICAL BASELINE (recent 13 weeks avg): {historical_summary.get('avg_sales', 0):.1f} units/week
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
    if historical_sales_values:
        recent = historical_sales_values[-13:] if len(historical_sales_values) >= 13 else historical_sales_values
        avg_hist = float(sum(recent) / len(recent))
    else:
        avg_hist = 0.0

    total_forecast_avg = (
        float(sum(w["predicted_sales"] for w in weekly_forecast) / len(weekly_forecast))
        if weekly_forecast else 0.0
    )
    trend_pct = ((total_forecast_avg - avg_hist) / avg_hist * 100) if avg_hist > 0 else 0.0

    historical_summary = {
        "avg_sales": avg_hist,
        "trend_pct": trend_pct,
    }

    # ── Check if HF token is configured ──────────────────────────────────────
    if not HF_TOKEN or HF_TOKEN in ("xx", "your_token_here", ""):
        logger.warning("HF_TOKEN not configured. Using rule-based fallback insight.")
        return _rule_based_fallback(trend_pct, peak_week, peak_sales)

    # ── Build prompt and call LLM ─────────────────────────────────────────────
    prompt = _build_prompt(
        weekly_forecast=weekly_forecast,
        historical_summary=historical_summary,
        metrics=metrics,
        peak_week=peak_week,
        peak_sales=peak_sales,
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
        return _rule_based_fallback(trend_pct, peak_week, peak_sales)


# ── Rule-based fallback (when token not set / API unreachable) ────────────────

def _rule_based_fallback(
    trend_pct: float,
    peak_week: Optional[str],
    peak_sales: Optional[float],
) -> Dict:
    """Minimal rule-based insight mirroring the original ai_insight logic."""
    trend_word = "increase" if trend_pct >= 0 else "decrease"
    summary = f"Demand expected to {trend_word} by {abs(trend_pct):.0f}% over the forecast period"

    if trend_pct > 15:
        risk = "High"
        safety_stock = f"+{int(abs(trend_pct))}%"
        action = "Increase inventory before week 2 of the forecast period"
        bullets = [
            f"• Demand trending up {trend_pct:.0f}% vs recent history — plan ahead",
            f"• Peak expected at {peak_week} with ~{peak_sales:.0f} units" if peak_week else "• Peak detected in forecast period",
            "• Increase safety stock to prevent stockouts during demand surge",
        ]
    elif trend_pct < -15:
        risk = "Low"
        safety_stock = f"-{int(abs(trend_pct))}%"
        action = "Delay replenishment — demand is trending below average"
        bullets = [
            f"• Demand declining {abs(trend_pct):.0f}% vs recent history",
            "• Risk of overstock if current replenishment rate is maintained",
            "• Consider promotions or bundle deals to stimulate demand",
        ]
    else:
        risk = "Medium"
        safety_stock = "Maintain current stock levels"
        action = "Monitor weekly demand closely for any spikes"
        bullets = [
            f"• Demand stable — within ±15% of recent baseline",
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
