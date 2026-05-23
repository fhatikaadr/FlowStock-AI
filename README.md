FlowStock-AI
===============

Inventory Management Optimization — ML forecasting + inventory optimizer

<<<<<<< HEAD
=======
## Hugging Face deploy

The repo now includes a root `Dockerfile` for Hugging Face Spaces.

By default it runs the `AI Recommendation` FastAPI app on port `7860`.

If you want to deploy the inventory optimizer instead, change `APP_DIR` in the `Dockerfile` to `Inventory Optimization Management`.

>>>>>>> 521348a31de2d03e3fa03a0a52b9b7a1c16316dd
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
