"""
Thin wrapper around sklearn's TfidfVectorizer / CountVectorizer so the rest
of the pipeline doesn't care which one is configured.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.utils.helpers import get_logger, timeit

logger = get_logger(__name__)


class TagVectorizer:
    """Fits a TF-IDF or Count vectorizer over the `tags` column."""

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ):
        if method not in {"tfidf", "count"}:
            raise ValueError("method must be 'tfidf' or 'count'")
        self.method = method
        vec_cls = TfidfVectorizer if method == "tfidf" else CountVectorizer
        self.vectorizer = vec_cls(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=tuple(ngram_range),
            min_df=min_df,
        )
        self._fitted = False

    @timeit
    def fit_transform(self, tags: list[str]) -> spmatrix:
        matrix = self.vectorizer.fit_transform(tags)
        self._fitted = True
        logger.info(
            f"Vectorized {matrix.shape[0]} items into {matrix.shape[1]} "
            f"features using {self.method}"
        )
        return matrix

    def transform(self, tags: list[str]) -> spmatrix:
        """Vectorize new/unseen text using the already-fitted vocabulary
        (used for scoring a brand-new item against the existing catalog)."""
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fit before calling transform()")
        return self.vectorizer.transform(tags)

    def get_feature_names(self) -> list[str]:
        return list(self.vectorizer.get_feature_names_out())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved vectorizer to {path}")

    @staticmethod
    def load(path: str | Path) -> "TagVectorizer":
        return joblib.load(path)
