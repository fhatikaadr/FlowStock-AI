import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.data_loader import load_and_preprocess_data
from app.forecasting.model_manager import ModelManager
from app.evaluation.metrics import calculate_metrics

mm = ModelManager()
df = load_and_preprocess_data(store=1, item=1)

last_year = df['year'].max()
train_df  = df[df['year'] < last_year].copy()
val_df    = df[df['year'] == last_year].copy()

print("Train:", len(train_df), "rows |", int(train_df['year'].min()), "-", int(last_year - 1))
print("Val:  ", len(val_df),   "rows | year", int(last_year))
print("Features available:", [c for c in df.columns if c not in ['date','sales']])
print()

model = mm.get_model("xgboost")
model.train(train_df)

# Pass val_df directly — XGBoost uses real lag features for accurate metrics
preds = model.predict(val_df)

m = calculate_metrics(val_df['sales'].values, preds)
print("=== XGBoost (Optimized) ===")
for k, v in m.items():
    print(f"  {k.upper()}: {v}")
