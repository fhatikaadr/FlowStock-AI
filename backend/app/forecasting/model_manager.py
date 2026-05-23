from app.forecasting.xgboost_model import XGBoostForecaster
from app.forecasting.prophet_model import ProphetForecaster
from app.forecasting.sarima_model import SARIMAForecaster
from app.forecasting.mlp_model import MLPForecaster
from app.forecasting.lstm_model import LSTMForecaster
import pandas as pd
import numpy as np
from datetime import timedelta

class ModelManager:
    def __init__(self):
        self.registry = {
            "xgboost": XGBoostForecaster,
            "prophet": ProphetForecaster,
            "sarima": SARIMAForecaster,
            "mlp": MLPForecaster,
            "lstm": LSTMForecaster
        }
        
    def get_supported_models(self):
        return list(self.registry.keys())

    def get_model(self, model_name: str):
        if model_name.lower() not in self.registry:
            raise ValueError(f"Model {model_name} not supported.")
        return self.registry[model_name.lower()]()

    def generate_future_dataframe(self, last_date, steps: int) -> pd.DataFrame:
        future_dates = [last_date + timedelta(days=7 * i) for i in range(1, steps + 1)]
        df = pd.DataFrame({'date': pd.to_datetime(future_dates)})
        df['year']       = df['date'].dt.year
        df['month']      = df['date'].dt.month
        df['quarter']    = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        return df

    def train_and_predict(self, model_name: str, historical_df: pd.DataFrame, forecast_period_days: int):
        model = self.get_model(model_name)
        
        # Train on all available data for the forecast request
        model.train(historical_df)
        
        last_date = historical_df['date'].max()
        future_df = self.generate_future_dataframe(last_date, forecast_period_days)
        
        predictions = model.predict(future_df)
        return future_df['date'], predictions
