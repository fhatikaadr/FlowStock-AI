from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "store_sales.csv"


def load_store_sales(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {}
    for col in df.columns:
        if col == "date":
            rename_map[col] = "date"
        elif col == "store":
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
