from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_KEY", "")

# ── Fallback CSV path (for local development only) ────────────────────────────
from pathlib import Path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent   # → backend/
_PROJECT_ROOT = _BACKEND_ROOT.parent                             # → FlowStock-AI/
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    str(_PROJECT_ROOT / "dataset" / "store_sales.csv"),
)

# ── Supabase REST batch size (max rows per request) ───────────────────────────
_SUPABASE_BATCH = 10_000


def _supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _load_from_supabase(warehouse: Optional[int], item: Optional[int]) -> pd.DataFrame:
    """
    Fetch store_sales from Supabase via REST API in batches.
    Table columns: date, warehouse, item, sales
    (column 'warehouse' maps to the former 'store' column in the CSV)
    """
    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        logger.warning("supabase-py not installed. Falling back to CSV.")
        return pd.DataFrame()

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        all_rows: list[dict] = []
        offset = 0

        while True:
            query = client.table("store_sales").select("date,warehouse,item,sales")

            if warehouse is not None:
                query = query.eq("warehouse", warehouse)
            if item is not None:
                query = query.eq("item", item)

            response = (
                query
                .order("date", desc=False)
                .range(offset, offset + _SUPABASE_BATCH - 1)
                .execute()
            )

            batch = response.data or []
            all_rows.extend(batch)

            if len(batch) < _SUPABASE_BATCH:
                break  # last page
            offset += _SUPABASE_BATCH

        if not all_rows:
            logger.warning("Supabase store_sales returned 0 rows.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        # Normalise column: 'warehouse' → 'store' so rest of pipeline stays unchanged
        if "warehouse" in df.columns and "store" not in df.columns:
            df = df.rename(columns={"warehouse": "store"})

        logger.info("Loaded %d rows from Supabase store_sales.", len(df))
        return df

    except Exception as exc:
        logger.error("Supabase fetch failed: %s", exc)
        return pd.DataFrame()


def _load_from_csv(store: Optional[int], item: Optional[int]) -> pd.DataFrame:
    """Fallback: load from local CSV file."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY env vars, "
            "or provide a valid DATASET_PATH."
        )

    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    if store is not None:
        col = "warehouse" if "warehouse" in df.columns else "store"
        df = df[df[col] == store]
    if item is not None:
        df = df[df["item"] == item]

    return df


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common feature engineering applied to either Supabase or CSV data.
    Input must have columns: date, sales (optionally store/item filtered).
    Aggregates to weekly (W-MON) resolution.
    """
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate to weekly level (W-MON)
    weekly_df = df.set_index("date").resample("W-MON")["sales"].sum().reset_index()
    weekly_df.sort_values("date", inplace=True)
    weekly_df.reset_index(drop=True, inplace=True)

    # ── Calendar features ─────────────────────────────────────────────────────
    weekly_df["year"]       = weekly_df["date"].dt.year
    weekly_df["month"]      = weekly_df["date"].dt.month
    weekly_df["quarter"]    = weekly_df["date"].dt.quarter
    weekly_df["weekofyear"] = weekly_df["date"].dt.isocalendar().week.astype(int)

    # ── Lag features (Weekly) ──────────────────────────────────────────────────
    weekly_df["lag_1"] = weekly_df["sales"].shift(1)
    weekly_df["lag_2"] = weekly_df["sales"].shift(2)
    weekly_df["lag_3"] = weekly_df["sales"].shift(3)
    weekly_df["lag_4"] = weekly_df["sales"].shift(4)

    # ── Rolling statistics (Weekly) ────────────────────────────────────────────
    weekly_df["rolling_mean_4"] = weekly_df["sales"].rolling(4, min_periods=1).mean()
    weekly_df["rolling_mean_8"] = weekly_df["sales"].rolling(8, min_periods=1).mean()
    weekly_df["rolling_std_4"]  = weekly_df["sales"].rolling(4, min_periods=1).std().fillna(0)

    # Back-fill any remaining NaNs from initial lag periods
    weekly_df.bfill(inplace=True)
    weekly_df.reset_index(drop=True, inplace=True)

    return weekly_df


def load_and_preprocess_data(
    store: Optional[int] = None,
    item: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads historical sales data and applies feature engineering.

    Source priority:
      1. Supabase (if SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set)
      2. Local CSV at DATASET_PATH (development fallback)

    The 'store' parameter maps to the 'warehouse' column in Supabase.
    """
    if _supabase_enabled():
        raw = _load_from_supabase(warehouse=store, item=item)
        if not raw.empty:
            return _apply_feature_engineering(raw)
        logger.warning("Supabase returned empty data — falling back to CSV.")

    # Fallback to CSV
    raw = _load_from_csv(store=store, item=item)
    return _apply_feature_engineering(raw)
