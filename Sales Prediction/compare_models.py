import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

def print_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    medae = median_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    print(f"--- {model_name} Metrics ---")
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")
    print(f"MedAE: {medae:.4f} | MAPE: {mape:.4f} | R2: {r2:.4f} | EVS: {evs:.4f}\n")

def main():
    print("Loading and aggregating data...")
    df = pd.read_csv('../dataset/store_sales.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregating to global daily sales
    daily_df = df.groupby('date')['sales'].sum().reset_index()
    daily_df.sort_values('date', inplace=True)
    
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['dayofweek'] = daily_df['date'].dt.dayofweek
    daily_df['dayofyear'] = daily_df['date'].dt.dayofyear
    
    train_df = daily_df[daily_df['year'] < 2017].copy()
    test_df = daily_df[daily_df['year'] == 2017].copy()
    
    print(f"Train size: {len(train_df)} days, Test size: {len(test_df)} days\n")
    
    results = {'date': test_df['date'], 'actual_sales': test_df['sales'].values}
    y_test_true = test_df['sales'].values
    
    # ------------------ XGBoost ------------------
    print("Training XGBoost...")
    features = ['year', 'month', 'dayofweek', 'dayofyear']
    X_train_xgb = train_df[features]
    y_train_xgb = train_df['sales']
    X_test_xgb = test_df[features]
    
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_preds = xgb_model.predict(X_test_xgb)
    results['XGBoost'] = xgb_preds
    print_metrics("XGBoost", y_test_true, xgb_preds)
    
    # ------------------ Prophet ------------------
    print("Training Prophet...")
    prophet_train = train_df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(prophet_train)
    
    prophet_future = test_df[['date']].rename(columns={'date': 'ds'})
    prophet_forecast = m.predict(prophet_future)
    prophet_preds = prophet_forecast['yhat'].values
    results['Prophet'] = prophet_preds
    print_metrics("Prophet", y_test_true, prophet_preds)
    
    # ------------------ SARIMA ------------------
    print("Training SARIMA...")
    # SARIMA can be very slow for daily data with yearly seasonality. Using weekly (s=7).
    sarima_train = train_df['sales'].values
    sarima_model = SARIMAX(sarima_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_res = sarima_model.fit(disp=False)
    sarima_preds = sarima_res.forecast(steps=len(test_df))
    results['SARIMA'] = sarima_preds
    print_metrics("SARIMA", y_test_true, sarima_preds)
    
    # ------------------ MLP & LSTM Preprocessing ------------------
    print("Preprocessing for Neural Networks...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[['sales']])
    test_scaled = scaler.transform(test_df[['sales']])
    
    window_size = 30
    def create_dataset(dataset, window_size):
        X, Y = [], []
        for i in range(len(dataset) - window_size):
            X.append(dataset[i:(i + window_size), 0])
            Y.append(dataset[i + window_size, 0])
        return np.array(X), np.array(Y)
        
    full_scaled = np.vstack((train_scaled, test_scaled))
    X_nn, y_nn = create_dataset(full_scaled, window_size)
    
    # The last len(test_df) items belong to test
    X_train_nn = X_nn[:-len(test_df)]
    y_train_nn = y_nn[:-len(test_df)]
    X_test_nn = X_nn[-len(test_df):]
    
    # ------------------ MLP ------------------
    print("Training MLP...")
    mlp_model = Sequential([
        Input(shape=(window_size,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    mlp_model.compile(optimizer='adam', loss='mse')
    mlp_model.fit(X_train_nn, y_train_nn, epochs=30, batch_size=32, verbose=0)
    
    mlp_preds_scaled = mlp_model.predict(X_test_nn, verbose=0)
    mlp_preds = scaler.inverse_transform(mlp_preds_scaled).flatten()
    results['MLP'] = mlp_preds
    print_metrics("MLP", y_test_true, mlp_preds)
    
    # ------------------ LSTM ------------------
    print("Training LSTM...")
    X_train_lstm = X_train_nn.reshape((X_train_nn.shape[0], X_train_nn.shape[1], 1))
    X_test_lstm = X_test_nn.reshape((X_test_nn.shape[0], X_test_nn.shape[1], 1))
    
    lstm_model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_nn, epochs=30, batch_size=32, verbose=0)
    
    lstm_preds_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
    results['LSTM'] = lstm_preds
    print_metrics("LSTM", y_test_true, lstm_preds)
    
    # ------------------ Visualization ------------------
    print("Generating visualization...")
    plt.figure(figsize=(16, 8))
    plt.plot(results['date'], results['actual_sales'], label='Actual Sales', color='black', linewidth=2)
    plt.plot(results['date'], results['XGBoost'], label='XGBoost', alpha=0.8)
    plt.plot(results['date'], results['Prophet'], label='Prophet', alpha=0.8)
    plt.plot(results['date'], results['SARIMA'], label='SARIMA', alpha=0.8)
    plt.plot(results['date'], results['MLP'], label='MLP', alpha=0.8)
    plt.plot(results['date'], results['LSTM'], label='LSTM', alpha=0.8)
    
    plt.title('Model Comparison: Total Daily Sales (2017)')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Saved chart to model_comparison.png")

if __name__ == "__main__":
    main()
