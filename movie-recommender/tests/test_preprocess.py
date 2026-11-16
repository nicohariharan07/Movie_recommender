import pandas as pd
import pytest

from src.data.preprocess import (
    clean_dataframe,
    extract_director,
    extract_names,
    parse_json_columns,
)


def test_parse_json_columns_handles_valid_and_missing():
    df = pd.DataFrame(
        {
            "genres": ["[{'id': 1, 'name': 'Action'}]", None],
            "title": ["Movie A", "Movie B"],
        }
    )
    parsed = parse_json_columns(df, columns=["genres"])
    assert parsed["genres"].iloc[0] == [{"id": 1, "name": "Action"}]
    assert parsed["genres"].iloc[1] == []


def test_parse_json_columns_handles_malformed_string():
    df = pd.DataFrame({"genres": ["not valid json{{{"]})
    parsed = parse_json_columns(df, columns=["genres"])
    assert parsed["genres"].iloc[0] == []


def test_extract_names_returns_name_field():
    entries = [{"id": 1, "name": "Action"}, {"id": 2, "name": "Drama"}]
    assert extract_names(entries) == ["Action", "Drama"]


def test_extract_names_respects_top_n():
    entries = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    assert extract_names(entries, top_n=2) == ["A", "B"]


def test_extract_director_finds_director_job():
    crew = [{"job": "Producer", "name": "Someone"}, {"job": "Director", "name": "Jane Doe"}]
    assert extract_director(crew) == "Jane Doe"


def test_extract_director_returns_empty_if_missing():
    crew = [{"job": "Producer", "name": "Someone"}]
    assert extract_director(crew) == ""


def test_clean_dataframe_drops_nulls_and_dupes():
    df = pd.DataFrame(
        {
            "title": ["A", "A", "B", None],
            "overview": ["x", "x", "y", "z"],
        }
    )
    cleaned = clean_dataframe(df, required_cols=["title", "overview"])
    assert len(cleaned) == 2
    assert list(cleaned["title"]) == ["A", "B"]


def test_clean_dataframe_raises_on_missing_column():
    df = pd.DataFrame({"title": ["A"]})
    with pytest.raises(KeyError):
        clean_dataframe(df, required_cols=["title", "overview"])
