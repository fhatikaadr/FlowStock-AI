# Inventory Optimization Management

Model AI khusus untuk memprediksi demand 14 hari ke depan dari data historis `store_sales.csv`.

## Tujuan

- `predicted_demand` untuk inventory
- `status` inventory berbasis policy stok
- `recommended_action` otomatis: transfer, restock, discount, atau none

## Struktur

- `src/data.py` - loader dataset
- `src/features.py` - feature engineering time series panel
- `src/train.py` - training model 1-step-ahead
- `src/predict.py` - inference recursive forecast 14 hari
- `src/policy.py` - safety stock, reorder point, status rule
- `src/inventory_recommendation.py` - hasil akhir predicted demand + status + recommended action
- `src/demo.py` - contoh eksekusi cepat

## Cara pakai

```bash
cd "Inventory Optimization Management"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
python -m src.inventory_recommendation
```

Output akan tersimpan di `../artifacts/inventory_ai_recommendations.csv`.

## Supabase (opsional)

Kalau mau data diambil dari Supabase dan hasil prediksi ditulis balik ke Supabase, isi `.env` dengan:

```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

Table yang dipakai:
- `store_sales` untuk data historis penjualan
- `inventory` untuk update hasil prediksi
- `products` dan `warehouses` untuk enrichment data
