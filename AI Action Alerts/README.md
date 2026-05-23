# AI Action Alerts

<<<<<<< HEAD
Folder ini disiapkan untuk modul AI alert berdasarkan kondisi stok dan risiko stockout.

Status: placeholder
=======
Standalone FastAPI service for ranking and serving the dashboard's AI Action Alerts.

## Endpoints

- `GET /health`
- `GET /api/metrics` - full validation metrics for the inventory forecast model
- `GET /api/action-alerts`

## Run

```powershell
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Environment

Set `GEMINI_API_KEY` to enable Gemini ranking. Without it, the service uses deterministic fallback ranking from `inventory_ai_recommendations.csv` and the dataset files in the parent `FlowStock-AI` folder.
>>>>>>> 521348a31de2d03e3fa03a0a52b9b7a1c16316dd
