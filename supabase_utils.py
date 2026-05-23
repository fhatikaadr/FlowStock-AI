from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pandas as pd


def _has_supabase_config() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


@lru_cache(maxsize=1)
def get_supabase_client():
    if not _has_supabase_config():
        return None

    from supabase import Client, create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def supabase_enabled() -> bool:
    return get_supabase_client() is not None


def load_table_df(table_name: str, columns: str = "*") -> pd.DataFrame:
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    response = client.table(table_name).select(columns).execute()
    rows: list[dict[str, Any]] = response.data or []
    return pd.DataFrame(rows)


def upsert_rows(table_name: str, rows: list[dict[str, Any]], on_conflict: str | None = None) -> None:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    try:
        if on_conflict:
            client.table(table_name).upsert(rows, on_conflict=on_conflict).execute()
        else:
            client.table(table_name).upsert(rows).execute()
    except TypeError:
        client.table(table_name).upsert(rows).execute()


def update_row(table_name: str, values: dict[str, Any], filters: dict[str, Any]) -> None:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    query = client.table(table_name).update(values)
    for key, value in filters.items():
        query = query.eq(key, value)
    query.execute()