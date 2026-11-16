"""
Feature engineering: build the single `tags` text column that becomes each
item's profile — this is what gets vectorized in src/models/vectorizer.py.

tags = overview + genres + keywords + top cast + director
"""

from __future__ import annotations

import pandas as pd

from src.data.preprocess import extract_director, extract_names
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _collapse_names(names: list[str], strip_spaces: bool = True) -> list[str]:
    """
    'Sam Worthington' -> 'samworthington'
    Removing spaces inside multi-word names keeps them as a single TF-IDF token,
    so the model doesn't conflate "Chris Evans" and "Chris Pratt" via a shared "chris" token.
    """
    if strip_spaces:
        return [n.replace(" ", "").lower() for n in names]
    return [n.lower() for n in names]


def build_tags_column(
    df: pd.DataFrame,
    overview_col: str = "overview",
    genres_col: str = "genres",
    keywords_col: str = "keywords",
    cast_col: str = "cast",
    crew_col: str = "crew",
    top_cast_n: int = 3,
    include_director: bool = True,
    strip_spaces_in_names: bool = True,
) -> pd.DataFrame:
    """
    Build the combined `tags` column used as each item's content profile.
    Gracefully skips any columns that aren't present (e.g. a books dataset
    with no `crew` column).
    """
    df = df.copy()

    def row_to_tags(row) -> str:
        parts: list[str] = []

        if overview_col in df.columns and pd.notna(row.get(overview_col)):
            parts.append(str(row[overview_col]))

        if genres_col in df.columns:
            genre_names = extract_names(row.get(genres_col, []))
            parts.extend(_collapse_names(genre_names, strip_spaces_in_names))

        if keywords_col in df.columns:
            kw_names = extract_names(row.get(keywords_col, []))
            parts.extend(_collapse_names(kw_names, strip_spaces_in_names))

        if cast_col in df.columns:
            cast_names = extract_names(row.get(cast_col, []), top_n=top_cast_n)
            parts.extend(_collapse_names(cast_names, strip_spaces_in_names))

        if include_director and crew_col in df.columns:
            director = extract_director(row.get(crew_col, []))
            if director:
                parts.append(_collapse_names([director], strip_spaces_in_names)[0])

        return " ".join(parts).lower()

    df["tags"] = df.apply(row_to_tags, axis=1)
    empty_tags = (df["tags"].str.strip() == "").sum()
    if empty_tags:
        logger.warning(f"{empty_tags} rows have an empty tags string (cold-start candidates)")

    logger.info("Built `tags` column (overview + genres + keywords + cast + director)")
    return df


def build_tags_from_generic_text(
    df: pd.DataFrame, text_columns: list[str], sep: str = " "
) -> pd.DataFrame:
    """
    Simpler variant for datasets where each attribute is already a plain string
    column (e.g. a books CSV with `description`, `genre`, `author`).
    """
    df = df.copy()
    df["tags"] = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(sep.join, axis=1)
        .str.lower()
    )
    logger.info(f"Built `tags` column from columns: {text_columns}")
    return df
