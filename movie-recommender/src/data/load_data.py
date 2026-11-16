"""
Load raw dataset(s) from disk and merge into a single working DataFrame.

Designed around the TMDB 5000 Movies + Credits Kaggle dataset, but any dataset
with `overview`/`description`, `genres`, and a cast/author-like column will work —
see src/features/build_features.py for the only place column names matter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_movies(movies_csv: str | Path, credits_csv: str | Path | None = None) -> pd.DataFrame:
    """
    Load the TMDB movies CSV and (optionally) merge in the credits CSV on title.

    Parameters
    ----------
    movies_csv : path to tmdb_5000_movies.csv
    credits_csv : optional path to tmdb_5000_credits.csv (adds cast/crew columns)

    Returns
    -------
    Merged DataFrame with one row per movie.
    """
    movies_csv = Path(movies_csv)
    if not movies_csv.exists():
        raise FileNotFoundError(
            f"No dataset found at {movies_csv}. "
            "Download the TMDB dataset from Kaggle and place it in data/raw/, "
            "or run `python scripts/make_sample_dataset.py` to generate a demo dataset."
        )

    movies = pd.read_csv(movies_csv)
    logger.info(f"Loaded {len(movies)} rows from {movies_csv.name}")

    if credits_csv is not None:
        credits_csv = Path(credits_csv)
        if credits_csv.exists():
            credits = pd.read_csv(credits_csv)
            # TMDB credits file uses `movie_id`; movies file uses `id`
            credits = credits.rename(columns={"movie_id": "id"})
            overlap = [c for c in credits.columns if c in movies.columns and c != "id"]
            credits = credits.drop(columns=overlap)
            movies = movies.merge(credits, on="id", how="left")
            logger.info(f"Merged credits — now {movies.shape[1]} columns")
        else:
            logger.warning(f"Credits file {credits_csv} not found; continuing without cast/crew")

    return movies


def load_generic_csv(path: str | Path) -> pd.DataFrame:
    """Generic loader for any single-file dataset (e.g. a books dataset)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset found at {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    return df
