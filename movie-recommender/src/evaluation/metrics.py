"""
Proxy evaluation metrics for a content-based recommender.

There's no user-rating ground truth here, so we can't compute RMSE/precision
against real feedback. Instead we measure whether the recommendations are
*thematically coherent* (genre overlap), *not just near-duplicates*
(intra-list diversity), and *not always the same handful of popular items*
(catalog coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.recommender import Recommender


def genre_overlap_at_k(
    recommender: Recommender, df: pd.DataFrame, genre_col: str = "genre_names", k: int = 5, sample_size: int = 200
) -> float:
    """
    For a sample of titles, what fraction of their top-k recommendations
    share at least one genre with the source title? Higher = more coherent.
    Requires a `genre_names` column of list[str] on df (aligned with recommender.titles).
    """
    if genre_col not in df.columns:
        raise KeyError(f"'{genre_col}' not found — add a parsed genre-names list column first")

    sample_titles = df["title"].sample(min(sample_size, len(df)), random_state=42)
    genre_lookup = dict(zip(df["title"], df[genre_col]))

    overlaps = []
    for title in sample_titles:
        try:
            recs = recommender.recommend(title, k=k)
        except Exception:
            continue
        source_genres = set(genre_lookup.get(title, []))
        if not source_genres:
            continue
        for rec_title, _ in recs:
            rec_genres = set(genre_lookup.get(rec_title, []))
            overlaps.append(1.0 if source_genres & rec_genres else 0.0)

    return float(np.mean(overlaps)) if overlaps else 0.0


def intra_list_diversity(recommender: Recommender, title: str, k: int = 5) -> float:
    """
    Average pairwise dissimilarity (1 - cosine similarity) among a single
    recommendation list. Low diversity (~0) means the list is full of
    near-duplicates of each other, not just of the query title.
    """
    recs = recommender.recommend(title, k=k)
    rec_titles = [t for t, _ in recs]
    indices = [recommender._title_to_index[t] for t in rec_titles if t in recommender._title_to_index]

    if len(indices) < 2:
        return 0.0

    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sim = recommender.similarity_matrix[indices[i], indices[j]]
            pairs.append(1 - sim)
    return float(np.mean(pairs))


def catalog_coverage(recommender: Recommender, sample_titles: list[str], k: int = 5) -> float:
    """
    Fraction of the entire catalog that shows up in *any* top-k list across a
    sample of query titles. Low coverage suggests the recommender collapses
    to recommending the same popular subset regardless of the query.
    """
    recommended_titles = set()
    for title in sample_titles:
        try:
            recs = recommender.recommend(title, k=k)
            recommended_titles.update(t for t, _ in recs)
        except Exception:
            continue

    return len(recommended_titles) / len(recommender.titles) if len(recommender.titles) else 0.0
