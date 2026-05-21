import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)
        self.features = ['year', 'month', 'dayofweek', 'dayofyear', 'quarter', 'is_weekend']

    def train(self, df: pd.DataFrame):
        # We need to make sure features exist in df
        available_features = [f for f in self.features if f in df.columns]
        X_train = df[available_features]
        y_train = df['sales']
        self.model.fit(X_train, y_train)
        self.trained_features = available_features

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        X_test = future_df[self.trained_features]
        return self.model.predict(X_test)
