"""
CLI entry point for the CineMatch pipeline.

    python main.py --config config/config.yaml                 # build pipeline + save artifacts
    python main.py --recommend "Avatar" --top-k 5               # get recommendations
    python main.py --recommend "Avatar" --top-k 5 --explain     # + show shared vocabulary terms
"""

from __future__ import annotations

import argparse
import os

from src.data.load_data import load_movies
from src.data.preprocess import clean_dataframe, extract_names, parse_json_columns
from src.features.build_features import build_tags_column
from src.models.recommender import Recommender, TitleNotFoundError
from src.utils.helpers import get_logger, load_config

logger = get_logger(__name__)

ARTIFACT_DIR = "models"


def build_pipeline(config: dict) -> Recommender:
    paths = config["paths"]
    cols = config["columns"]
    fe = config["feature_engineering"]
    vec_cfg = config["vectorizer"]

    df = load_movies(paths["raw_movies_csv"], paths.get("raw_credits_csv"))
    df = parse_json_columns(df)
    df = clean_dataframe(df, required_cols=[cols["title_col"], cols["overview_col"]])

    df = build_tags_column(
        df,
        overview_col=cols["overview_col"],
        genres_col=cols["genres_col"],
        keywords_col=cols["keywords_col"],
        cast_col=cols["cast_col"],
        crew_col=cols["crew_col"],
        top_cast_n=fe["top_cast_n"],
        include_director=fe["include_director"],
        strip_spaces_in_names=fe["strip_spaces_in_names"],
    )

    # keep a plain genre-name list column around too, useful for evaluation metrics
    if cols["genres_col"] in df.columns:
        df["genre_names"] = df[cols["genres_col"]].apply(extract_names)

    os.makedirs(os.path.dirname(paths["processed_csv"]), exist_ok=True)
    df.to_csv(paths["processed_csv"], index=False)
    logger.info(f"Saved processed dataset to {paths['processed_csv']}")

    from src.models.vectorizer import TagVectorizer

    vectorizer = TagVectorizer(
        method=vec_cfg["method"],
        max_features=vec_cfg["max_features"],
        stop_words=vec_cfg["stop_words"],
        ngram_range=tuple(vec_cfg["ngram_range"]),
        min_df=vec_cfg["min_df"],
    )

    recommender = Recommender.fit(df, title_col=cols["title_col"], tags_col=cols["tags_col"], vectorizer=vectorizer)
    recommender.save(ARTIFACT_DIR)
    return recommender


def main():
    parser = argparse.ArgumentParser(description="CineMatch content-based recommender")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--recommend", type=str, help="Title to get recommendations for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--explain", action="store_true", help="Show shared vocabulary terms")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild pipeline even if artifacts exist")
    args = parser.parse_args()

    config = load_config(args.config)

    artifacts_exist = os.path.exists(f"{ARTIFACT_DIR}/similarity.npy")
    if args.rebuild or not artifacts_exist:
        logger.info("Building pipeline from raw data...")
        recommender = build_pipeline(config)
    else:
        logger.info("Loading existing artifacts (use --rebuild to force a refresh)...")
        recommender = Recommender.load(ARTIFACT_DIR)

    if args.recommend:
        try:
            results = recommender.recommend(args.recommend, k=args.top_k)
        except TitleNotFoundError as e:
            print(f"\n{e}\n")
            return

        print(f"\nTop {len(results)} recommendations for '{args.recommend}':\n")
        for rank, (title, score) in enumerate(results, start=1):
            print(f"  {rank}. {title:<40}  similarity = {score:.3f}")
            if args.explain:
                shared = recommender.explain(args.recommend, title)
                if shared:
                    print(f"       shared terms: {', '.join(shared)}")
        print()
    else:
        print(f"\nPipeline built. {len(recommender.titles)} titles indexed.")
        print("Run again with --recommend \"<title>\" to get recommendations.\n")


if __name__ == "__main__":
    main()
