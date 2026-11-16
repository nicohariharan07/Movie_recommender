"""
The user-facing Recommender: title in -> top-K similar titles out.

This is the object main.py / app/streamlit_app.py / app/api.py all talk to.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.models.similarity import build_similarity_matrix
from src.models.vectorizer import TagVectorizer
from src.utils.helpers import fuzzy_find_title, get_logger

logger = get_logger(__name__)


class TitleNotFoundError(Exception):
    """Raised when a title can't be found or fuzzy-matched in the catalog."""


class Recommender:
    """
    Content-based recommender over a fitted TF-IDF/cosine similarity matrix.

    Parameters
    ----------
    titles : the `title` column, aligned positionally with the similarity matrix
    similarity_matrix : (n_items, n_items) cosine similarity matrix
    vectorizer : fitted TagVectorizer (needed to score brand-new items on the fly)
    metadata : optional full DataFrame for enriching output (genre, year, etc.)
    """

    def __init__(
        self,
        titles: pd.Series,
        similarity_matrix: np.ndarray,
        vectorizer: TagVectorizer | None = None,
        metadata: pd.DataFrame | None = None,
        feature_matrix=None,
    ):
        self.titles = titles.reset_index(drop=True)
        self.similarity_matrix = similarity_matrix
        self.vectorizer = vectorizer
        self.metadata = metadata
        self.feature_matrix = feature_matrix  # fitted catalog TF-IDF matrix, needed for cold-start scoring
        self._title_to_index = {t: i for i, t in enumerate(self.titles)}

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def recommend(
        self, title: str, k: int = 5, fuzzy_threshold: float = 0.6
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar titles to `title`, as (title, score) pairs,
        sorted by descending similarity. Raises TitleNotFoundError if no close
        match exists in the catalog.
        """
        resolved = self._resolve_title(title, fuzzy_threshold)
        idx = self._title_to_index[resolved]

        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:k]  # exclude itself

        return [(self.titles.iloc[i], float(score)) for i, score in scores]

    def recommend_for_new_item(self, tags_text: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Score a brand-new item (not in the fitted catalog) against the existing
        catalog using its raw tags text. This is the cold-start mitigation:
        as soon as *some* text/genre/cast info exists for the new item, it can
        be recommended-against without retraining the whole pipeline.

        Note: this finds existing catalog titles similar to the new item — it
        does NOT retroactively add the new item into other items' recommendation
        lists (that still requires a periodic full refit).
        """
        if self.vectorizer is None or self.feature_matrix is None:
            raise RuntimeError(
                "Recommender needs a fitted vectorizer + feature_matrix to score new items "
                "(fit with Recommender.fit(), which stores both automatically)"
            )

        new_vec = self.vectorizer.transform([tags_text])
        sims = cosine_similarity(new_vec, self.feature_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self.titles.iloc[i], float(sims[i])) for i in top_idx]

    def explain(self, title: str, recommended_title: str, top_n_terms: int = 8) -> list[str]:
        """
        Best-effort explainability: which shared vocabulary terms likely drove
        the similarity between two titles. Requires the vectorizer + feature names.
        """
        if self.vectorizer is None or self.metadata is None or "tags" not in self.metadata:
            return []

        idx_a = self._title_to_index[self._resolve_title(title)]
        idx_b = self._title_to_index[self._resolve_title(recommended_title)]

        tags_a = set(self.metadata["tags"].iloc[idx_a].split())
        tags_b = set(self.metadata["tags"].iloc[idx_b].split())
        shared = tags_a & tags_b
        return sorted(shared)[:top_n_terms]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, dir_path: str) -> None:
        import os

        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.titles, f"{dir_path}/titles.pkl")
        np.save(f"{dir_path}/similarity.npy", self.similarity_matrix)
        if self.vectorizer is not None:
            self.vectorizer.save(f"{dir_path}/vectorizer.pkl")
        if self.metadata is not None:
            joblib.dump(self.metadata, f"{dir_path}/metadata.pkl")
        if self.feature_matrix is not None:
            sparse.save_npz(f"{dir_path}/feature_matrix.npz", sparse.csr_matrix(self.feature_matrix))
        logger.info(f"Saved recommender artifacts to {dir_path}")

    @classmethod
    def load(cls, dir_path: str) -> "Recommender":
        import os

        titles = joblib.load(f"{dir_path}/titles.pkl")
        similarity_matrix = np.load(f"{dir_path}/similarity.npy")
        vectorizer = None
        vec_path = f"{dir_path}/vectorizer.pkl"
        if os.path.exists(vec_path):
            vectorizer = TagVectorizer.load(vec_path)
        metadata = None
        meta_path = f"{dir_path}/metadata.pkl"
        if os.path.exists(meta_path):
            metadata = joblib.load(meta_path)
        feature_matrix = None
        fm_path = f"{dir_path}/feature_matrix.npz"
        if os.path.exists(fm_path):
            feature_matrix = sparse.load_npz(fm_path)
        logger.info(f"Loaded recommender artifacts from {dir_path}")
        return cls(titles, similarity_matrix, vectorizer, metadata, feature_matrix)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        title_col: str = "title",
        tags_col: str = "tags",
        vectorizer: TagVectorizer | None = None,
    ) -> "Recommender":
        """Convenience constructor: fit vectorizer + similarity in one call."""
        vectorizer = vectorizer or TagVectorizer()
        feature_matrix = vectorizer.fit_transform(df[tags_col].tolist())
        similarity_matrix = build_similarity_matrix(feature_matrix)
        return cls(df[title_col], similarity_matrix, vectorizer, metadata=df, feature_matrix=feature_matrix)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _resolve_title(self, title: str, fuzzy_threshold: float = 0.6) -> str:
        if title in self._title_to_index:
            return title
        match = fuzzy_find_title(title, list(self._title_to_index.keys()), fuzzy_threshold)
        if match is None:
            raise TitleNotFoundError(
                f"'{title}' not found in catalog and no close match above threshold "
                f"{fuzzy_threshold}. Try checking spelling or listing available titles."
            )
        logger.info(f"Fuzzy-matched '{title}' -> '{match}'")
        return match
