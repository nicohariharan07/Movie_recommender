import numpy as np

from src.models.similarity import build_similarity_matrix, top_k_similarity
from src.models.vectorizer import TagVectorizer

TAGS = [
    "space war rebellion empire starship",   # similar to #1
    "space war rebellion pilot superweapon", # similar to #0
    "farm sisters small town inheritance",   # similar to #3
    "farm family generations small town",    # similar to #2
]


def _fitted_matrix():
    vec = TagVectorizer(stop_words=None)
    return vec.fit_transform(TAGS)


def test_similarity_matrix_is_square_and_symmetric():
    feature_matrix = _fitted_matrix()
    sim = build_similarity_matrix(feature_matrix)
    assert sim.shape == (4, 4)
    assert np.allclose(sim, sim.T, atol=1e-6)


def test_diagonal_is_self_similarity_of_one():
    feature_matrix = _fitted_matrix()
    sim = build_similarity_matrix(feature_matrix)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-6)


def test_similar_items_score_higher_than_unrelated():
    feature_matrix = _fitted_matrix()
    sim = build_similarity_matrix(feature_matrix)
    # item 0 (space) should be more similar to item 1 (space) than to item 2 (farm)
    assert sim[0, 1] > sim[0, 2]


def test_top_k_similarity_shapes_and_ranking():
    feature_matrix = _fitted_matrix()
    neighbor_idx, neighbor_scores = top_k_similarity(feature_matrix, k=2)
    assert neighbor_idx.shape == (4, 2)
    assert neighbor_scores.shape == (4, 2)
    # scores per row should be in descending order
    for row in neighbor_scores:
        assert list(row) == sorted(row, reverse=True)
    # item 0's top neighbor should be item 1 (both space-themed)
    assert neighbor_idx[0, 0] == 1
