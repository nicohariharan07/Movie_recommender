"""
Cleaning & normalization.

The TMDB CSVs store `genres`, `keywords`, `cast`, `crew` as stringified JSON,
e.g.  "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
This module parses those into plain Python lists of strings and handles
missing/duplicate rows before feature engineering.
"""

from __future__ import annotations

import ast

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Columns that come in as stringified list-of-dicts in the raw TMDB CSVs
JSON_LIST_COLUMNS = ["genres", "keywords", "cast", "crew"]


def _safe_literal_eval(value):
    """Parse a stringified Python list/dict; return [] if it's not parseable."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_json_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert stringified JSON list columns into real Python lists of dicts."""
    df = df.copy()
    columns = columns or [c for c in JSON_LIST_COLUMNS if c in df.columns]
    for col in columns:
        df[col] = df[col].apply(_safe_literal_eval)
    logger.info(f"Parsed JSON-like columns: {columns}")
    return df


def extract_names(entries: list[dict], key: str = "name", top_n: int | None = None) -> list[str]:
    """Pull out the 'name' field from a list of dicts, e.g. genre/keyword objects."""
    names = [e.get(key, "") for e in entries if isinstance(e, dict) and e.get(key)]
    return names[:top_n] if top_n else names


def extract_director(crew: list[dict]) -> str:
    """Find the Director from the crew list (job == 'Director')."""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def clean_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    dedupe_on: str = "title",
) -> pd.DataFrame:
    """
    Drop rows missing any required column, drop duplicate titles, reset index.
    Keeping a clean, contiguous 0..n-1 index matters — the similarity matrix
    is positionally indexed and must line up 1:1 with the DataFrame rows.
    """
    before = len(df)

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame")

    df = df.dropna(subset=required_cols)
    df = df.drop_duplicates(subset=[dedupe_on], keep="first")
    df = df.reset_index(drop=True)

    logger.info(f"Cleaned dataset: {before} -> {len(df)} rows " f"(dropped {before - len(df)})")
    return df
