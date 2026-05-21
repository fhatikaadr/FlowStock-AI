from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import warnings

class SARIMAForecaster:
    def __init__(self):
        self.model = None
        self.res = None

    def train(self, df: pd.DataFrame):
        warnings.filterwarnings("ignore")
        # Use weekly seasonality for performance as in the original code
        sarima_train = df['sales'].values
        self.model = SARIMAX(sarima_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        self.res = self.model.fit(disp=False)

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        steps = len(future_df)
        forecast = self.res.forecast(steps=steps)
        return forecast
