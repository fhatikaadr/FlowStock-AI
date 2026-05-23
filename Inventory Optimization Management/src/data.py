from __future__ import annotations

<<<<<<< HEAD
=======
import sys
>>>>>>> 521348a31de2d03e3fa03a0a52b9b7a1c16316dd
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "store_sales.csv"

<<<<<<< HEAD

def load_store_sales(csv_path: str | Path | None = None) -> pd.DataFrame:
=======
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supabase_utils import load_table_df, supabase_enabled


def load_store_sales(csv_path: str | Path | None = None) -> pd.DataFrame:
    if supabase_enabled():
        df = load_table_df("store_sales", "date,store_id,item_id,sales")
        if not df.empty:
            df.columns = [str(col).strip().lower() for col in df.columns]
            required = {"date", "store_id", "item_id", "sales"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns from Supabase store_sales: {sorted(missing)}")

            df = df[["date", "store_id", "item_id", "sales"]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["store_id"] = df["store_id"].astype(str)
            df["item_id"] = df["item_id"].astype(str)
            df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)

            return (
                df.groupby(["store_id", "item_id", "date"], as_index=False)["sales"]
                .sum()
                .sort_values(["store_id", "item_id", "date"])
                .reset_index(drop=True)
            )

>>>>>>> 521348a31de2d03e3fa03a0a52b9b7a1c16316dd
    path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {}
    for col in df.columns:
        if col == "date":
            rename_map[col] = "date"
<<<<<<< HEAD
        elif col == "store":
=======
        elif col in {"store", "warehouse"}:
>>>>>>> 521348a31de2d03e3fa03a0a52b9b7a1c16316dd
            rename_map[col] = "store_id"
        elif col == "item":
            rename_map[col] = "item_id"
        elif col == "sales":
            rename_map[col] = "sales"

    df = df.rename(columns=rename_map)

    required = {"date", "store_id", "item_id", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["date", "store_id", "item_id", "sales"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["store_id"] = df["store_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)

    df = (
        df.groupby(["store_id", "item_id", "date"], as_index=False)["sales"]
        .sum()
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )
    return df
