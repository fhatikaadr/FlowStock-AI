import xgboost as xgb
import pandas as pd
import numpy as np

FEATURES = ['year', 'month', 'dayofweek', 'dayofyear', 'quarter', 'is_weekend']

class XGBoostForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.trained_features = None

    def train(self, df: pd.DataFrame):
        # Only use features that exist in the dataframe
        self.trained_features = [f for f in FEATURES if f in df.columns]
        X = df[self.trained_features]
        y = df['sales']
        self.model.fit(X, y)

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        if self.trained_features is None:
            raise RuntimeError("Model has not been trained yet.")
        # future_df may come from generate_future_dataframe (has all feature cols)
        # or from a raw val_df (also has all cols after feature engineering)
        available = [f for f in self.trained_features if f in future_df.columns]
        X = future_df[available]
        return self.model.predict(X)
