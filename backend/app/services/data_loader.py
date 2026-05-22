import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH", "../dataset/store_sales.csv")

def load_and_preprocess_data(store: int = None, item: int = None) -> pd.DataFrame:
    """
    Loads historical sales data and applies feature engineering.
    Optionally filters by store and/or item.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df['date'] = pd.to_datetime(df['date'])

    if store is not None:
        df = df[df['store'] == store]
    if item is not None:
        df = df[df['item'] == item]

    # Aggregate to one row per day
    daily_df = df.groupby('date')['sales'].sum().reset_index()
    daily_df.sort_values('date', inplace=True)
    daily_df.reset_index(drop=True, inplace=True)

    # ── Calendar features ────────────────────────────────────────────────────
    daily_df['year']      = daily_df['date'].dt.year
    daily_df['month']     = daily_df['date'].dt.month
    daily_df['dayofweek'] = daily_df['date'].dt.dayofweek
    daily_df['dayofyear'] = daily_df['date'].dt.dayofyear
    daily_df['quarter']   = daily_df['date'].dt.quarter
    daily_df['is_weekend']= daily_df['dayofweek'].isin([5, 6]).astype(int)
    daily_df['weekofyear']= daily_df['date'].dt.isocalendar().week.astype(int)

    # ── Lag features ─────────────────────────────────────────────────────────
    daily_df['lag_1']  = daily_df['sales'].shift(1)
    daily_df['lag_7']  = daily_df['sales'].shift(7)
    daily_df['lag_14'] = daily_df['sales'].shift(14)
    daily_df['lag_30'] = daily_df['sales'].shift(30)

    # ── Rolling statistics ───────────────────────────────────────────────────
    daily_df['rolling_mean_7']  = daily_df['sales'].rolling(7,  min_periods=1).mean()
    daily_df['rolling_mean_14'] = daily_df['sales'].rolling(14, min_periods=1).mean()
    daily_df['rolling_mean_30'] = daily_df['sales'].rolling(30, min_periods=1).mean()
    daily_df['rolling_std_7']   = daily_df['sales'].rolling(7,  min_periods=1).std().fillna(0)

    # Back-fill any remaining NaNs from initial lag periods
    daily_df.bfill(inplace=True)
    daily_df.reset_index(drop=True, inplace=True)

    return daily_df
