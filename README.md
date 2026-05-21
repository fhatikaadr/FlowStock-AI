FlowStock-AI
===============

Inventory Management Optimization — ML forecasting + inventory optimizer

Contents
- `src/data_ingest.py` — download dataset
- `src/preprocess.py` — preprocessing & feature engineering
- `src/train.py` — train LightGBM global forecasting model
- `src/forecast_infer.py` — load model & generate forecasts
- `src/inventory.py` — safety stock and reorder point calculations
- `requirements.txt` — Python deps

Usage (example)

1. Create virtual env and install deps:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Download dataset and inspect:

```bash
python -m src.data_ingest
```

3. Train model (this will create `model.joblib`):

```bash
python -m src.train
```

4. Run inference and inventory optimization (example):

```bash
python -m src.forecast_infer --horizon 14 --service-level 0.95 --lead-time 7
```
# FlowStock-AI
Tugas Besar 4012
