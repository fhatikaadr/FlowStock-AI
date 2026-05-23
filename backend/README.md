---
title: FlowStock AI Forecasting Backend
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# FlowStock AI — Forecasting Backend API

REST API untuk sales forecasting berbasis AI menggunakan XGBoost, Prophet, SARIMA, MLP, dan LSTM.

## Endpoints

| Method | Path | Keterangan |
|--------|------|------------|
| `GET` | `/` | Welcome message |
| `GET` | `/docs` | Swagger UI (interactive docs) |
| `POST` | `/forecast/run` | Jalankan forecasting |
| `GET` | `/forecast/models` | Daftar model & opsi yang tersedia |
| `GET` | `/forecast/metrics` | Metrics forecast terbaru dari DB |
| `GET` | `/forecast/history/{store}/{item}` | Riwayat forecast per store/item |
| `POST` | `/forecast/scenario/save` | Simpan skenario what-if |
| `POST` | `/forecast/ai-insight` | Generate AI insight via Qwen LLM |

## Environment Variables (Secrets)

Set secrets berikut di HF Spaces → Settings → Variables and Secrets:

| Variable | Keterangan |
|----------|------------|
| `SUPABASE_URL` | URL project Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key Supabase |
| `DATABASE_URL` | PostgreSQL connection string Supabase |
| `HF_TOKEN` | HuggingFace token untuk Qwen LLM |
