import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class MLPForecaster:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = Sequential([
            Input(shape=(self.window_size,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.last_window = None

    def create_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.window_size):
            X.append(dataset[i:(i + self.window_size), 0])
            Y.append(dataset[i + self.window_size, 0])
        return np.array(X), np.array(Y)

    def train(self, df: pd.DataFrame):
        data = df[['sales']].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self.create_dataset(scaled_data)
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0) # using lower epochs for faster backend tests
        
        # Save the last window to predict future sequences
        self.last_window = scaled_data[-self.window_size:]

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        steps = len(future_df)
        predictions = []
        current_window = self.last_window.copy()
        
        for _ in range(steps):
            X_pred = current_window.reshape((1, self.window_size))
            pred_scaled = self.model.predict(X_pred, verbose=0)
            predictions.append(pred_scaled[0, 0])
            
            current_window = np.append(current_window[1:], pred_scaled, axis=0)
            
        preds = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return preds.flatten()
