from prophet import Prophet
import pandas as pd
import numpy as np

class ProphetForecaster:
    def __init__(self):
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    def train(self, df: pd.DataFrame):
        prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        self.model.fit(prophet_df)

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        future_dates = future_df[['date']].rename(columns={'date': 'ds'})
        forecast = self.model.predict(future_dates)
        return forecast['yhat'].values
