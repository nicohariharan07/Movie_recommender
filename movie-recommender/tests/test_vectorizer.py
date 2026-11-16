import pytest

from src.models.vectorizer import TagVectorizer


TAGS = [
    "space war rebellion empire starship",
    "space war rebellion pilot superweapon",
    "farm sisters small town inheritance",
    "farm family generations small town",
]


def test_fit_transform_returns_correct_shape():
    vec = TagVectorizer(method="tfidf", max_features=50)
    matrix = vec.fit_transform(TAGS)
    assert matrix.shape[0] == len(TAGS)
    assert matrix.shape[1] <= 50


def test_transform_requires_fit_first():
    vec = TagVectorizer()
    with pytest.raises(RuntimeError):
        vec.transform(["some new text"])


def test_transform_after_fit_works():
    vec = TagVectorizer()
    vec.fit_transform(TAGS)
    result = vec.transform(["a brand new space war movie"])
    assert result.shape[0] == 1


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        TagVectorizer(method="not_a_real_method")


def test_count_method_gives_integer_counts():
    vec = TagVectorizer(method="count", stop_words=None)
    matrix = vec.fit_transform(["space space war", "farm town"])
    # count vectorizer should count "space" appearing twice in the first doc
    feature_names = vec.get_feature_names()
    space_idx = feature_names.index("space")
    assert matrix[0, space_idx] == 2


def test_save_and_load_roundtrip(tmp_path):
    vec = TagVectorizer()
    vec.fit_transform(TAGS)
    save_path = tmp_path / "vectorizer.pkl"
    vec.save(save_path)

    loaded = TagVectorizer.load(save_path)
    result = loaded.transform(["space war rebellion"])
    assert result.shape[0] == 1
