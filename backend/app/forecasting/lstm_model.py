import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from collections import deque


class LSTMForecaster:
    def __init__(self, window_size: int = 14):
        self.window_size = window_size
        self.scaler      = MinMaxScaler(feature_range=(0, 1))
        self._model      = None
        self._history: deque | None = None

    def _build_model(self) -> Sequential:
        model = Sequential([
            Input(shape=(self.window_size, 1)),
            Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='huber')
        return model

    def _make_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(len(scaled) - self.window_size):
            X.append(scaled[i : i + self.window_size])
            y.append(scaled[i + self.window_size])
        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)
        return X, y

    def train(self, df: pd.DataFrame):
        data   = df['sales'].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(data).flatten()

        X, y = self._make_sequences(scaled)

        self._model = self._build_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, verbose=0),
        ]
        self._model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        self._history = deque(scaled[-self.window_size:], maxlen=self.window_size)

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LSTMForecaster has not been trained yet.")

        steps = len(future_df)
        preds_scaled: list[float] = []
        window = list(self._history)   # length == window_size

        for _ in range(steps):
            X_pred = np.array(window).reshape(1, self.window_size, 1)
            pred   = float(self._model.predict(X_pred, verbose=0)[0, 0])
            pred   = max(0.0, min(pred, 1.0))
            preds_scaled.append(pred)
            window.pop(0)
            window.append(pred)

        result = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(result, 0.0)
