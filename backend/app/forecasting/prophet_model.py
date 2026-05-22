import logging
import warnings
from prophet import Prophet
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

class ProphetForecaster:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        self._fitted = False

    def train(self, df: pd.DataFrame):
        """Train on a dataframe that has 'date' and 'sales' columns."""
        prophet_df = (
            df[['date', 'sales']]
            .rename(columns={'date': 'ds', 'sales': 'y'})
        )
        self.model.fit(prophet_df)
        self._fitted = True

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Predict from any dataframe that contains a 'date' column.
        Works for both validation slices and generate_future_dataframe output.
        """
        if not self._fitted:
            raise RuntimeError("ProphetForecaster has not been trained yet.")
        future = future_df[['date']].rename(columns={'date': 'ds'})
        forecast = self.model.predict(future)
        return forecast['yhat'].values
