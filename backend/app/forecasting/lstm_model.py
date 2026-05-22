import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

class LSTMForecaster:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self._model = None
        self._last_window = None   # shape: (window_size,) — scaled 1-D

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.window_size, 1)),
            LSTM(50, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _make_sequences(self, scaled: np.ndarray):
        """Create (X, y) sliding-window sequences for LSTM input."""
        X, y = [], []
        for i in range(len(scaled) - self.window_size):
            X.append(scaled[i : i + self.window_size])
            y.append(scaled[i + self.window_size])
        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)
        return X, y

    def train(self, df: pd.DataFrame):
        data = df['sales'].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(data).flatten()   # 1-D

        X, y = self._make_sequences(scaled)
        self._model = self._build_model()
        self._model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        self._last_window = scaled[-self.window_size:].copy()  # 1-D

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LSTMForecaster has not been trained yet.")

        steps = len(future_df)
        predictions_scaled = []
        window = self._last_window.copy()  # 1-D, length window_size

        for _ in range(steps):
            X_pred = window.reshape(1, self.window_size, 1)
            pred = self._model.predict(X_pred, verbose=0)[0, 0]
            predictions_scaled.append(pred)
            # Slide window
            window = np.append(window[1:], pred)

        preds = np.array(predictions_scaled).reshape(-1, 1)
        return self.scaler.inverse_transform(preds).flatten()
