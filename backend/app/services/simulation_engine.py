import pandas as pd
import numpy as np

def apply_what_if_scenario(predictions: np.ndarray, dates: pd.Series, quarter: str = None, seasonality_impact: str = None, demand_multiplier: float = 1.0, promotional_effect: float = 1.0) -> np.ndarray:
    """
    Applies what-if simulation scenarios on the raw model predictions.
    """
    adjusted_predictions = predictions.copy()
    
    # Apply baseline multipliers
    adjusted_predictions = adjusted_predictions * demand_multiplier * promotional_effect
    
    # Convert dates to quarters if we need quarter-specific logic
    if quarter or seasonality_impact:
        date_dt = pd.to_datetime(dates)
        
        for i, dt in enumerate(date_dt):
            multiplier = 1.0
            
            # Quarter specific rules
            if quarter:
                # E.g., if we only want to boost Q4
                q = f"Q{dt.quarter}"
                if q == quarter:
                    multiplier *= 1.15
                    
            # Seasonality impact
            if seasonality_impact == "High":
                multiplier *= 1.20
            elif seasonality_impact == "Low":
                multiplier *= 0.80
                
            adjusted_predictions[i] *= multiplier
            
    return adjusted_predictions
