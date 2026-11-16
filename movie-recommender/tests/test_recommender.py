import pandas as pd
import pytest

from src.models.recommender import Recommender, TitleNotFoundError

DF = pd.DataFrame(
    {
        "title": ["Galactic Reckoning", "Starlight Rebellion", "The Last Harvest", "Harvest Moon"],
        "tags": [
            "space war rebellion empire starship",
            "space war rebellion pilot superweapon",
            "farm sisters small town inheritance",
            "farm family generations small town",
        ],
    }
)


def test_fit_creates_recommender_with_correct_title_count():
    rec = Recommender.fit(DF)
    assert len(rec.titles) == 4


def test_recommend_returns_top_k_excluding_self():
    rec = Recommender.fit(DF)
    results = rec.recommend("Galactic Reckoning", k=3)
    titles = [t for t, _ in results]
    assert "Galactic Reckoning" not in titles
    assert len(results) == 3


def test_recommend_ranks_thematically_similar_title_first():
    rec = Recommender.fit(DF)
    results = rec.recommend("Galactic Reckoning", k=1)
    assert results[0][0] == "Starlight Rebellion"


def test_recommend_scores_are_descending():
    rec = Recommender.fit(DF)
    results = rec.recommend("Galactic Reckoning", k=3)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_fuzzy_matches_typo():
    rec = Recommender.fit(DF)
    results = rec.recommend("Galactic Rekoning", k=1)  # typo
    assert len(results) == 1


def test_recommend_raises_for_unknown_title():
    rec = Recommender.fit(DF)
    with pytest.raises(TitleNotFoundError):
        rec.recommend("Completely Unrelated Nonexistent Title Xyz", k=1)


def test_recommend_for_new_item_cold_start():
    rec = Recommender.fit(DF)
    results = rec.recommend_for_new_item("space war rebellion empire", k=2)
    assert len(results) == 2
    # should surface the two space-themed catalog titles
    titles = {t for t, _ in results}
    assert titles == {"Galactic Reckoning", "Starlight Rebellion"}


def test_save_and_load_roundtrip(tmp_path):
    rec = Recommender.fit(DF)
    save_dir = tmp_path / "artifacts"
    rec.save(str(save_dir))

    loaded = Recommender.load(str(save_dir))
    results = loaded.recommend("Galactic Reckoning", k=1)
    assert results[0][0] == "Starlight Rebellion"
