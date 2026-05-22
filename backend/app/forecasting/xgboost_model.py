import xgboost as xgb
import pandas as pd
import numpy as np
from collections import deque

# Calendar features that can be computed for any future date
CALENDAR_FEATURES = [
    'year', 'month', 'dayofweek', 'dayofyear',
    'quarter', 'is_weekend', 'weekofyear',
]

# Lag/rolling features that must be propagated iteratively
LAG_FEATURES = [
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7',
]


class XGBoostForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,            # reduced from 8 — less overfitting
            subsample=0.8,          # row sampling for regularization
            colsample_bytree=0.8,   # feature sampling for regularization
            min_child_weight=5,     # prevents splits on tiny leaf nodes
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            random_state=42,
            n_jobs=-1,
        )
        self.trained_features: list[str] = []
        # Circular buffer of recent actual/predicted values for lag propagation
        self._history: deque | None = None

    # ─────────────────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        available_cal = [f for f in CALENDAR_FEATURES if f in df.columns]
        available_lag = [f for f in LAG_FEATURES      if f in df.columns]
        self.trained_features = available_cal + available_lag

        X = df[self.trained_features]
        y = df['sales']
        self.model.fit(X, y)

        # Keep a rolling buffer of the last 30 actual sales values.
        # This lets us compute all lag / rolling features during future prediction.
        self._history = deque(df['sales'].values[-30:], maxlen=30)

    # ─────────────────────────────────────────────────────────────────────────
    def _lag_features_from_history(self) -> dict:
        """Derive all lag/rolling features from the current history buffer."""
        h = list(self._history)    # oldest → newest, length ≤ 30
        n = len(h)

        def _safe(idx: int) -> float:
            return h[idx] if abs(idx) <= n else h[0]

        return {
            'lag_1':            _safe(-1),
            'lag_7':            _safe(-7),
            'lag_14':           _safe(-14),
            'lag_30':           _safe(-30),
            'rolling_mean_7':   float(np.mean(h[-7:])),
            'rolling_mean_14':  float(np.mean(h[-14:])),
            'rolling_mean_30':  float(np.mean(h)),
            'rolling_std_7':    float(np.std(h[-7:]) if len(h) >= 2 else 0.0),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Two modes:
        - If `future_df` contains actual lag feature columns (e.g. validation slice
          from load_and_preprocess_data), use them directly — no propagation error.
        - If `future_df` comes from generate_future_dataframe (no lag cols), use
          auto-regressive lag propagation step-by-step.
        """
        if not self.trained_features:
            raise RuntimeError("XGBoostForecaster has not been trained yet.")

        lag_cols_present = any(f in future_df.columns for f in LAG_FEATURES)

        if lag_cols_present:
            # ── Direct prediction (validation on historical data) ─────────────
            available = [f for f in self.trained_features if f in future_df.columns]
            X = future_df[available]
            return np.maximum(self.model.predict(X), 0.0)

        # ── Auto-regressive prediction (true future dates) ────────────────────
        predictions: list[float] = []
        for _, row in future_df.iterrows():
            lag_feats = self._lag_features_from_history()
            feat_row: dict = {}
            for f in self.trained_features:
                if f in CALENDAR_FEATURES:
                    feat_row[f] = row[f]
                else:
                    feat_row[f] = lag_feats.get(f, 0.0)

            X = pd.DataFrame([feat_row])[self.trained_features]
            pred = float(max(self.model.predict(X)[0], 0.0))
            predictions.append(pred)
            self._history.append(pred)

        return np.array(predictions)
