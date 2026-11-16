"""
Build and persist the item x item cosine similarity matrix.

For catalogs beyond a few tens of thousands of items, materializing the full
dense n x n matrix stops being memory-friendly (10k items -> 100M floats -> 800MB).
`top_k_similarity` below offers a memory-safe alternative that only keeps each
item's top-K nearest neighbors, which is what a production system would use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.helpers import get_logger, timeit

logger = get_logger(__name__)


@timeit
def build_similarity_matrix(feature_matrix: spmatrix) -> np.ndarray:
    """Full dense item x item cosine similarity matrix. Fine up to ~10-20k items."""
    sim = cosine_similarity(feature_matrix)
    logger.info(f"Built dense similarity matrix of shape {sim.shape}")
    return sim


@timeit
def top_k_similarity(feature_matrix: spmatrix, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-safe alternative: for each item, keep only its top-K most similar
    neighbors instead of the full n x n matrix.

    Returns
    -------
    neighbor_indices : (n_items, k) array of neighbor row indices
    neighbor_scores   : (n_items, k) array of matching similarity scores
    """
    sim = cosine_similarity(feature_matrix)
    n = sim.shape[0]
    k = min(k, n - 1)

    # argsort is O(n log n); argpartition is O(n) and sufficient since we
    # only need the top-k unordered, then we sort just those k
    neighbor_indices = np.zeros((n, k), dtype=int)
    neighbor_scores = np.zeros((n, k), dtype=float)

    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        neighbor_indices[i] = top_idx
        neighbor_scores[i] = row[top_idx]

    logger.info(f"Built top-{k} neighbor index for {n} items")
    return neighbor_indices, neighbor_scores


def save_similarity_matrix(sim: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, sparse.csr_matrix(sim))
    logger.info(f"Saved similarity matrix to {path}")


def load_similarity_matrix(path: str | Path) -> np.ndarray:
    return sparse.load_npz(path).toarray()
