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

# ── Supabase REST batch size ───────────────────────────────────────────────────
# Supabase REST API caps each response at 1 000 rows by default.
# The pagination loop exits when len(batch) < _SUPABASE_BATCH, so this value
# MUST match Supabase's real limit — otherwise the loop exits after the first
# page and only ~1 000 rows (e.g. 2013–mid-2015) are loaded instead of all
# 1 826 rows (2013–2017).
_SUPABASE_BATCH = 1_000


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
    Operates at DAILY resolution so the model can learn intra-week and
    intra-month patterns (including the payday spike on the 25th).
    """
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate to daily level (sum across any store/item duplicates for same date)
    daily_df = df.groupby("date", as_index=False)["sales"].sum()
    daily_df.sort_values("date", inplace=True)
    daily_df.reset_index(drop=True, inplace=True)

    # ── Calendar features ─────────────────────────────────────────────────────
    daily_df["year"]        = daily_df["date"].dt.year
    daily_df["month"]       = daily_df["date"].dt.month
    daily_df["quarter"]     = daily_df["date"].dt.quarter
    daily_df["weekofyear"]  = daily_df["date"].dt.isocalendar().week.astype(int)
    daily_df["day_of_month"] = daily_df["date"].dt.day
    daily_df["day_of_week"]  = daily_df["date"].dt.dayofweek   # 0=Mon … 6=Sun
    daily_df["week_of_month"] = (daily_df["date"].dt.day - 1) // 7 + 1

    # ── Payday features ───────────────────────────────────────────────────────
    # The 25th of each month is payday → strong sales spike
    daily_df["is_payday"]        = (daily_df["day_of_month"] == 25).astype(int)
    daily_df["is_payday_window"] = daily_df["day_of_month"].isin([24, 25, 26]).astype(int)

    # ── Double-date features ───────────────────────────────────────────────────
    # Shopping events where day == month: 1/1, 2/2, 3/3 … 12/12
    # (e.g. 11.11 Singles' Day, 12.12 Double Twelve, etc.)
    daily_df["is_double_date"] = (
        daily_df["day_of_month"] == daily_df["month"]
    ).astype(int)
    # ±1 day halo around the double-date event
    daily_df["is_double_date_window"] = (
        (daily_df["day_of_month"] - daily_df["month"]).abs() <= 1
    ).astype(int)

    # ── Lag features (Daily) ──────────────────────────────────────────────────
    for lag in range(1, 8):
        daily_df[f"lag_{lag}"] = daily_df["sales"].shift(lag)

    # ── Rolling statistics (Daily) ────────────────────────────────────────────
    daily_df["rolling_mean_7"]  = daily_df["sales"].rolling(7,  min_periods=1).mean()
    daily_df["rolling_mean_14"] = daily_df["sales"].rolling(14, min_periods=1).mean()
    daily_df["rolling_std_7"]   = daily_df["sales"].rolling(7,  min_periods=1).std().fillna(0)

    # Back-fill any remaining NaNs from initial lag periods
    daily_df.bfill(inplace=True)
    daily_df.reset_index(drop=True, inplace=True)

    return daily_df


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


def load_raw_daily_data(
    store: Optional[int] = None,
    item: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load raw **daily** sales data aggregated to one row per date.

    Used for building the daily historical chart line and for computing
    the historical baseline in AI insight. Sales are SUMMED across all
    warehouses for each date so the scale matches the forecast model,
    which also trains on date-aggregated totals.

    Returns a DataFrame with columns: date (datetime64), sales (float).
    """
    if _supabase_enabled():
        raw = _load_from_supabase(warehouse=store, item=item)
        if not raw.empty:
            raw["date"] = pd.to_datetime(raw["date"])
            # Aggregate to one row per date — same as _apply_feature_engineering
            raw = (
                raw.groupby("date", as_index=False)["sales"]
                .sum()
                .sort_values("date")
                .reset_index(drop=True)
            )
            logger.info("Loaded %d aggregated daily rows for display.", len(raw))
            return raw[["date", "sales"]]
        logger.warning("Supabase returned empty data for raw daily load — falling back to CSV.")

    raw = _load_from_csv(store=store, item=item)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = (
        raw.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return raw[["date", "sales"]]
