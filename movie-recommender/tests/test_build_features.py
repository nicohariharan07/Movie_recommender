import pandas as pd

from src.features.build_features import build_tags_column, build_tags_from_generic_text


def test_build_tags_column_combines_all_sources():
    df = pd.DataFrame(
        {
            "overview": ["A hero saves the world"],
            "genres": [[{"name": "Action"}, {"name": "Sci-Fi"}]],
            "keywords": [[{"name": "space war"}]],
            "cast": [[{"name": "Chris Evans"}, {"name": "Sam Worthington"}]],
            "crew": [[{"job": "Director", "name": "Jane Doe"}]],
        }
    )
    result = build_tags_column(df)
    tags = result["tags"].iloc[0]

    assert "a hero saves the world" in tags
    assert "action" in tags
    assert "spacewar" in tags  # multi-word keyword also has spaces stripped, like names
    assert "chrisevans" in tags  # spaces stripped from names
    assert "janedoe" in tags


def test_build_tags_column_flags_empty_tags(caplog):
    df = pd.DataFrame({"overview": [None], "genres": [[]], "keywords": [[]], "cast": [[]], "crew": [[]]})
    result = build_tags_column(df)
    assert result["tags"].iloc[0] == ""


def test_build_tags_column_skips_missing_columns():
    # a books-like dataset with no crew/cast columns shouldn't error
    df = pd.DataFrame({"overview": ["A story"], "genres": [[{"name": "Drama"}]]})
    result = build_tags_column(df)
    assert "a story" in result["tags"].iloc[0]
    assert "drama" in result["tags"].iloc[0]


def test_build_tags_from_generic_text():
    df = pd.DataFrame(
        {
            "description": ["A gripping tale"],
            "genre": ["Thriller"],
            "author": ["J. Smith"],
        }
    )
    result = build_tags_from_generic_text(df, text_columns=["description", "genre", "author"])
    tags = result["tags"].iloc[0]
    assert "a gripping tale" in tags
    assert "thriller" in tags
    assert "j. smith" in tags
