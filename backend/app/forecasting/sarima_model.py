import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

class SARIMAForecaster:
    def __init__(self):
        self._res = None

    def train(self, df: pd.DataFrame):
        """Fit SARIMA on the 'sales' column."""
        series = df['sales'].values
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
        )
        self._res = model.fit(disp=False)

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Forecast `len(future_df)` steps ahead from the end of the training data.
        The future_df argument is used only to determine how many steps to forecast.
        """
        if self._res is None:
            raise RuntimeError("SARIMAForecaster has not been trained yet.")
        steps = len(future_df)
        return np.array(self._res.forecast(steps=steps))
