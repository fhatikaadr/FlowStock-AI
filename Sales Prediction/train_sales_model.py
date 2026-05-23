import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score, 
    explained_variance_score,
    median_absolute_error
)
import matplotlib.pyplot as plt

def main():
    print("Loading data...")
    df = pd.read_csv('../dataset/store_sales.csv')
    
    # 2. Data Loading & Feature Engineering
    print("Feature engineering...")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    
    # Extract dates for later visualization before dropping the column from features
    dates_col = df['date'].copy()
    
    # 3. Chronological Split
    train_df = df[df['year'] < 2017].copy()
    test_df = df[df['year'] == 2017].copy()
    
    train_dates = dates_col[df['year'] < 2017]
    test_dates = dates_col[df['year'] == 2017]
    
    features = ['store', 'item', 'year', 'month', 'dayofweek', 'dayofyear']
    target = 'sales'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    # 4. Model Training
    print("Training model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    medae = median_absolute_error(y_test, predictions)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Median Absolute Error (MedAE): {medae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
    
    # 6. Visualization
    print("Generating visualization...")
    vis_df = pd.DataFrame({
        'date': test_dates,
        'actual_sales': y_test,
        'predicted_sales': predictions
    })
    
    # Group by day
    daily_sales = vis_df.groupby('date')[['actual_sales', 'predicted_sales']].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales['date'], daily_sales['actual_sales'], label='Actual Sales', alpha=0.7)
    plt.plot(daily_sales['date'], daily_sales['predicted_sales'], label='Predicted Sales', alpha=0.7)
    plt.title('Total Daily Actual Sales vs Predicted Sales (2017)')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_vs_actuals.png')
    print("Saved chart to predictions_vs_actuals.png")

if __name__ == "__main__":
    main()
