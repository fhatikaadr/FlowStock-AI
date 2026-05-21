# AI Recommendation Service

Self-contained API service for FlowStock AI inventory recommendations and Gemini-powered explanations.

## Endpoints

- `GET /health` - health check
- `GET /api/inventory-recommendations` - returns recommendation rows from `FlowStock-AI/artifacts/inventory_ai_recommendations.csv`
- `POST /api/generate-recommendation-explanation` - returns a Gemini explanation or a fallback explanation

## Run

```bash
cd "c:\Users\tika\Downloads\FlowStock-AI\AI Recommendation"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

## Web usage

Point the web app to `http://localhost:8000` and call the API with `fetch()`.
