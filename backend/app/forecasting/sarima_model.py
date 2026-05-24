import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class SARIMAForecaster:
    def __init__(self):
        self._res = None
        self._trained_steps = 0

    def train(self, df: pd.DataFrame):
        """Fit SARIMA on the 'sales' column.

        Uses a weekly seasonal period (m=7) since the training data is now
        at daily resolution. SARIMA(1,1,1)(1,0,1,7) captures:
          - (1,1,1): short-term AR/MA with differencing
          - (1,0,1,7): weekly seasonality without seasonal differencing
            (daily data already trends gently; over-differencing hurts here)
        """
        series = df['sales'].values
        self._trained_steps = len(series)
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),   # weekly seasonal period
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._res = model.fit(disp=False)

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Forecast `len(future_df)` steps ahead from the end of the training data.
        The future_df argument is used only to determine how many steps to forecast.

        Note: SARIMA is slow for large step counts. Cap at 366 steps
        (one year of daily data) to avoid excessive runtime.
        """
        if self._res is None:
            raise RuntimeError("SARIMAForecaster has not been trained yet.")
        steps = min(len(future_df), 366)
        preds = np.array(self._res.forecast(steps=steps))
        # Pad with last value if future_df requested more steps than the cap
        if len(future_df) > steps:
            pad = np.full(len(future_df) - steps, preds[-1])
            preds = np.concatenate([preds, pad])
        return np.maximum(preds, 0.0)
