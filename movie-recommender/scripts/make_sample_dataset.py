"""
Generate a small synthetic movie dataset shaped exactly like the TMDB 5000
Movies + Credits CSVs, so the full pipeline can be run and tested offline
without a Kaggle download. Swap in the real CSVs later with zero code changes.

Usage:
    python scripts/make_sample_dataset.py
"""

import json
import random
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

MOVIES = [
    {
        "id": 1,
        "title": "Galactic Reckoning",
        "overview": "A rogue starship captain leads a rebellion against an empire "
        "that has enslaved a thousand worlds, uncovering a conspiracy that threatens "
        "the galaxy's last free colonies.",
        "genres": ["Science Fiction", "Action", "Adventure"],
        "keywords": ["space war", "rebellion", "empire", "starship"],
        "cast": ["Alex Rivera", "Mina Cho", "Jonah Blake"],
        "director": "Priya Anand",
    },
    {
        "id": 2,
        "title": "Starlight Rebellion",
        "overview": "When an empire's fleet threatens the outer colonies, a young "
        "pilot joins a ragtag rebellion to steal the plans for a superweapon and "
        "save her home planet.",
        "genres": ["Science Fiction", "Adventure", "Action"],
        "keywords": ["space war", "rebellion", "pilot", "superweapon"],
        "cast": ["Mina Cho", "Derek Voss", "Alex Rivera"],
        "director": "Priya Anand",
    },
    {
        "id": 3,
        "title": "The Last Harvest",
        "overview": "In a drought-stricken farming town, two estranged sisters must "
        "work together to save their family's land before the bank forecloses.",
        "genres": ["Drama", "Family"],
        "keywords": ["sisters", "farm", "small town", "inheritance"],
        "cast": ["Grace Whitfield", "Elena Marsh"],
        "director": "Tom Callahan",
    },
    {
        "id": 4,
        "title": "Harvest Moon",
        "overview": "A family drama set across three generations of farmers "
        "fighting to keep their land as the town around them changes forever.",
        "genres": ["Drama"],
        "keywords": ["farm", "family", "generations", "small town"],
        "cast": ["Elena Marsh", "William Cho"],
        "director": "Tom Callahan",
    },
    {
        "id": 5,
        "title": "Midnight Heist",
        "overview": "A crew of elite thieves plans the biggest casino heist in "
        "Monaco's history, but a betrayal from within threatens to unravel the plan.",
        "genres": ["Crime", "Thriller", "Action"],
        "keywords": ["heist", "casino", "betrayal", "crew"],
        "cast": ["Derek Voss", "Nadia Farouk", "Jonah Blake"],
        "director": "Ines Beaumont",
    },
    {
        "id": 6,
        "title": "The Vault",
        "overview": "A master thief assembles a crew for one last job: cracking "
        "an unbreakable vault beneath a Monaco casino before dawn.",
        "genres": ["Crime", "Thriller"],
        "keywords": ["heist", "vault", "casino", "one last job"],
        "cast": ["Nadia Farouk", "Derek Voss"],
        "director": "Ines Beaumont",
    },
    {
        "id": 7,
        "title": "Whispers in the Fog",
        "overview": "A detective haunted by an unsolved case returns to her "
        "hometown when a string of disappearances echoes a decades-old murder.",
        "genres": ["Mystery", "Thriller", "Drama"],
        "keywords": ["detective", "cold case", "small town", "murder"],
        "cast": ["Elena Marsh", "William Cho"],
        "director": "Sofia Lindqvist",
    },
    {
        "id": 8,
        "title": "The Silent Coast",
        "overview": "A retired detective is pulled back into the field when a "
        "murder in a fog-bound coastal town mirrors the case that ended her career.",
        "genres": ["Mystery", "Drama"],
        "keywords": ["detective", "murder", "coastal town", "cold case"],
        "cast": ["Grace Whitfield", "Nadia Farouk"],
        "director": "Sofia Lindqvist",
    },
    {
        "id": 9,
        "title": "Love in Lisbon",
        "overview": "A chance encounter between two strangers in Lisbon over one "
        "summer blossoms into a romance neither expected — or is ready for.",
        "genres": ["Romance", "Drama"],
        "keywords": ["summer romance", "strangers", "lisbon", "love"],
        "cast": ["Mina Cho", "Jonah Blake"],
        "director": "Carlos Ferreira",
    },
    {
        "id": 10,
        "title": "A Summer in Porto",
        "overview": "Two travelers meet by accident in Porto and spend one "
        "unforgettable summer falling in love before real life pulls them apart.",
        "genres": ["Romance", "Drama"],
        "keywords": ["summer romance", "travelers", "love", "porto"],
        "cast": ["Grace Whitfield", "Alex Rivera"],
        "director": "Carlos Ferreira",
    },
    {
        "id": 11,
        "title": "Nightmare Frequency",
        "overview": "A radio host starts receiving broadcasts from a killer who "
        "claims to be listening to her show live, forcing her to unmask him on air.",
        "genres": ["Horror", "Thriller"],
        "keywords": ["radio", "killer", "live broadcast", "serial killer"],
        "cast": ["Nadia Farouk", "William Cho"],
        "director": "Marcus Webb",
    },
    {
        "id": 12,
        "title": "Static",
        "overview": "A late-night radio DJ realizes a caller describing a murder "
        "in real time might be the killer himself, broadcasting live to her audience.",
        "genres": ["Horror", "Thriller", "Mystery"],
        "keywords": ["radio", "killer", "live broadcast", "night shift"],
        "cast": ["Elena Marsh", "Derek Voss"],
        "director": "Marcus Webb",
    },
]


def _to_json_list(names: list[str]) -> str:
    """TMDB stores genres/keywords/cast as stringified list of {id, name} dicts."""
    return json.dumps([{"id": random.randint(1, 9999), "name": n} for n in names])


def _crew_json(director: str) -> str:
    return json.dumps(
        [{"id": random.randint(1, 9999), "job": "Director", "department": "Directing", "name": director}]
    )


def build_and_save():
    movies_rows, credits_rows = [], []

    for m in MOVIES:
        movies_rows.append(
            {
                "id": m["id"],
                "title": m["title"],
                "overview": m["overview"],
                "genres": _to_json_list(m["genres"]),
                "keywords": _to_json_list(m["keywords"]),
                "vote_average": round(random.uniform(5.5, 8.8), 1),
                "popularity": round(random.uniform(10, 300), 2),
                "release_date": f"20{random.randint(10, 24)}-0{random.randint(1,9)}-15",
            }
        )
        credits_rows.append(
            {
                "movie_id": m["id"],
                "title": m["title"],
                "cast": _to_json_list(m["cast"]),
                "crew": _crew_json(m["director"]),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(movies_rows).to_csv(OUT_DIR / "tmdb_5000_movies.csv", index=False)
    pd.DataFrame(credits_rows).to_csv(OUT_DIR / "tmdb_5000_credits.csv", index=False)

    print(f"Wrote {len(movies_rows)} synthetic movies to {OUT_DIR}")
    print("Note: titles were deliberately written in similar pairs "
          "(e.g. 'Galactic Reckoning' / 'Starlight Rebellion') so recommendations are easy to sanity-check.")


if __name__ == "__main__":
    build_and_save()
