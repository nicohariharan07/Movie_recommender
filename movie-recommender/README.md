# 🎬 CineMatch — Content-Based Movie/Book Recommender System

A production-style, content-based recommendation engine that suggests similar
movies (or books) using **TF-IDF vectorization** and **cosine similarity**
over a combined text "tag" profile (overview + genres + cast/author + keywords).

Built as a portfolio-grade major project — modular, testable, deployable.

---

## 1. Project Structure

```
movie-recommender/
│
├── README.md                      ← you are here
├── requirements.txt                ← pinned dependencies
├── environment.yml                 ← optional conda env
├── pyproject.toml                  ← packaging + tool config (black/ruff/pytest)
├── .gitignore
├── main.py                         ← CLI entry point (build pipeline end-to-end)
│
├── config/
│   └── config.yaml                 ← paths, column names, TF-IDF params (no hardcoding)
│
├── data/
│   ├── raw/                        ← original TMDB/Books CSVs (untouched, gitignored)
│   ├── external/                   ← reference/lookup data (e.g. genre id maps)
│   └── processed/                  ← cleaned dataset + saved tag column (parquet/csv)
│
├── src/                            ← installable package: `pip install -e .`
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py            ← reads raw CSV(s), merges movies+credits, or generates demo data
│   │   └── preprocess.py           ← cleaning, null handling, text normalization
│   ├── features/
│   │   └── build_features.py       ← builds the combined `tags` column (feature engineering)
│   ├── models/
│   │   ├── vectorizer.py           ← TF-IDF / CountVectorizer wrapper, fit + persist
│   │   ├── similarity.py           ← cosine similarity matrix builder (dense + sparse-safe)
│   │   └── recommender.py          ← Recommender class: title → top-N similar titles
│   ├── evaluation/
│   │   └── metrics.py              ← precision@k proxy, genre-overlap score, diversity, coverage
│   └── utils/
│       └── helpers.py              ← logging, title fuzzy-matching, timers
│
├── app/
│   ├── streamlit_app.py            ← interactive UI (search + poster grid + explanations)
│   └── api.py                      ← FastAPI microservice (/recommend?title=...)
│
├── models/                         ← persisted artifacts: vectorizer.pkl, similarity.npz, indices.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb                ← exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
│
├── tests/                          ← pytest unit tests for every src/ module
│   ├── test_preprocess.py
│   ├── test_build_features.py
│   ├── test_vectorizer.py
│   ├── test_similarity.py
│   └── test_recommender.py
│
├── scripts/
│   └── make_sample_dataset.py      ← generates a small synthetic dataset so the whole
│                                       pipeline runs offline without Kaggle access
│
├── reports/
│   └── figures/                    ← saved EDA plots, similarity heatmaps
│
└── docker/
    └── Dockerfile                  ← containerize the Streamlit/FastAPI app
```

**Why this layout?**
- `src/` is a real Python package (not notebook spaghetti) → reusable, testable, importable.
- `config/config.yaml` centralizes tunables (TF-IDF `max_features`, `ngram_range`, column
  names) so you're not hunting through code to change an experiment.
- `models/` stores fitted artifacts so the app doesn't refit TF-IDF on every request.
- `app/` separates the *demo/deployment* layer from the *ML pipeline* layer.
- `tests/` mirrors `src/` 1:1 — every pipeline stage is independently verifiable.

---

## 2. Dataset

Default target: [TMDB 5000 Movies + Credits](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
(`tmdb_5000_movies.csv` + `tmdb_5000_credits.csv`). Drop both into `data/raw/`.

Relevant columns used to build the item profile:

| Column      | Source file | Role                                  |
|-------------|-------------|----------------------------------------|
| `overview`  | movies      | plot description (main text signal)    |
| `genres`    | movies      | JSON list → flattened genre names      |
| `keywords`  | movies      | JSON list → flattened plot keywords    |
| `cast`      | credits     | top 3 billed actors                    |
| `crew`      | credits     | director extracted from crew list      |

No Kaggle access? Run `python scripts/make_sample_dataset.py` to generate a small
synthetic movie dataset with the same schema so you can run the full pipeline offline,
then swap in the real CSVs later — zero code changes required.

For a **books** version, swap `overview→description`, `cast→author`, `genres→genre`,
and everything downstream (vectorizer, similarity, recommender) works unchanged, since
the pipeline only ever depends on the final `tags` text column and a `title` column.

---

## 3. Pipeline (what `main.py` runs)

```
raw CSV(s)
   │  src/data/load_data.py        (merge, select columns)
   ▼
cleaned DataFrame
   │  src/data/preprocess.py       (drop nulls/dupes, parse JSON-like fields, lowercase)
   ▼
DataFrame + `tags` column
   │  src/features/build_features.py  (overview + genres + keywords + cast + director → one string)
   ▼
TF-IDF matrix (n_items × n_vocab)
   │  src/models/vectorizer.py
   ▼
cosine similarity matrix (n_items × n_items)
   │  src/models/similarity.py
   ▼
Recommender.recommend(title, k=5)
   │  src/models/recommender.py
   ▼
Top-K similar titles + similarity scores
```

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt

# no dataset yet? generate a demo one:
python scripts/make_sample_dataset.py

# run the full pipeline (clean → features → vectorize → similarity → save artifacts)
python main.py --config config/config.yaml

# get recommendations from the CLI
python main.py --recommend "Avatar" --top-k 5

# or launch the interactive app
streamlit run app/streamlit_app.py
```

## 5. Example usage (library API)

```python
from src.models.recommender import Recommender

rec = Recommender.load("models/")          # loads persisted vectorizer + similarity matrix
rec.recommend("The Dark Knight", k=5)
# → [('Batman Begins', 0.41), ('The Dark Knight Rises', 0.38), ...]
```

---

## 6. Evaluation (`src/evaluation/metrics.py`)

Content-based recommenders don't have ground-truth "ratings" to compute RMSE against,
so this project uses proxy metrics instead:

- **Genre overlap@k** — average fraction of shared genres between a title and its top-k
  recommendations (higher = more thematically coherent).
- **Intra-list diversity** — average pairwise dissimilarity within a recommendation list
  (catches an over-narrow recommender that just returns near-duplicates).
- **Catalog coverage** — % of the catalog that ever appears in *any* top-k list (catches
  a recommender that always recommends the same popular handful).

## 7. Limitations & Next Steps (Step 6 discussion)

**Cold start problem:** a brand-new movie/book with no `overview`/genres yet (or a
user with no interaction history) can't be scored — content-based filtering needs
*some* text to vectorize. Mitigation: fall back to metadata-only tags (genre + cast)
until a synopsis is added, or blend with popularity-based defaults.

**Collaborative filtering (the natural next step, Netflix-style):**

| | Content-based (this project) | Collaborative filtering |
|---|---|---|
| Signal used | Item text (overview, genre, cast) | User-item interaction/rating matrix |
| Cold start (new item) | ✅ works immediately | ❌ no ratings yet |
| Cold start (new user) | ✅ works if they like ≥1 item | ❌ no history yet |
| Captures taste patterns beyond text | ❌ (e.g. "people who liked X also liked Y" for unrelated genres) | ✅ |
| Needs a user base | ❌ | ✅ |

Practical upgrade path: build a **hybrid recommender** — use this TF-IDF/cosine engine
to solve cold start for new items, and layer matrix factorization (SVD / ALS) or a
neural two-tower model on top once enough user-item interaction data exists, blending
both scores (e.g. `0.6 * collaborative_score + 0.4 * content_score`).

Other honest limitations of the current approach:
- TF-IDF treats words as independent tokens — no semantic understanding ("space war"
  vs "galactic conflict" are unrelated in TF-IDF space). Swapping in sentence
  embeddings (e.g. `all-MiniLM-L6-v2` via `sentence-transformers`) would fix this.
- No personalization — every user gets the same recommendations for a given title.
- No signal from popularity, recency, or rating quality — a poorly-reviewed but
  textually similar movie ranks the same as a beloved one.

---

## 8. Tech Stack

`pandas` · `numpy` · `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`) ·
`streamlit` (demo UI) · `fastapi` + `uvicorn` (optional API) · `pytest` (tests)
