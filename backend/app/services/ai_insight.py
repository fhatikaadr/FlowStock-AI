import numpy as np

def generate_insights(historical_sales: np.ndarray, predicted_sales: np.ndarray) -> dict:
    """
    Generates AI business insights based on historical averages and forecasted predictions.
    """
    hist_mean = np.mean(historical_sales)
    pred_mean = np.mean(predicted_sales)
    
    if hist_mean == 0:
        hist_mean = 1e-5 # avoid div zero
        
    diff_pct = ((pred_mean - hist_mean) / hist_mean) * 100
    
    summary = f"Demand expected to {'increase' if diff_pct > 0 else 'decrease'} by {abs(diff_pct):.0f}%"
    
    if diff_pct > 15:
        stockout_risk = "High"
        recommended_safety_stock = f"+{int(abs(diff_pct))}%"
        recommended_action = "Increase inventory before week 2"
    elif diff_pct < -15:
        stockout_risk = "Low"
        recommended_safety_stock = f"-{int(abs(diff_pct))}%"
        recommended_action = "Delay replenishment to avoid overstock"
    else:
        stockout_risk = "Medium"
        recommended_safety_stock = "Maintain current levels"
        recommended_action = "Monitor demand closely"

    return {
        "summary": summary,
        "stockout_risk": stockout_risk,
        "recommended_safety_stock": recommended_safety_stock,
        "recommended_action": recommended_action
    }
