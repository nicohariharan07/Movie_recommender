"""
Optional FastAPI microservice exposing the recommender over HTTP.

Run with:  uvicorn app.api:app --reload
Then:      GET http://127.0.0.1:8000/recommend?title=Avatar&k=5
"""

import os
import sys

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.recommender import Recommender, TitleNotFoundError

ARTIFACT_DIR = "models"

app = FastAPI(title="CineMatch API", description="Content-based movie recommender", version="0.1.0")
_recommender: Recommender | None = None


class RecommendationItem(BaseModel):
    title: str
    similarity: float


class RecommendationResponse(BaseModel):
    query: str
    results: list[RecommendationItem]


@app.on_event("startup")
def load_model():
    global _recommender
    if os.path.exists(f"{ARTIFACT_DIR}/similarity.npy"):
        _recommender = Recommender.load(ARTIFACT_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _recommender is not None}


@app.get("/titles")
def list_titles():
    if _recommender is None:
        raise HTTPException(503, "Model not loaded — run main.py first to build artifacts")
    return {"count": len(_recommender.titles), "titles": sorted(_recommender.titles.tolist())}


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    title: str = Query(..., description="Title to get recommendations for"),
    k: int = Query(5, ge=1, le=20, description="Number of recommendations"),
):
    if _recommender is None:
        raise HTTPException(503, "Model not loaded — run main.py first to build artifacts")
    try:
        results = _recommender.recommend(title, k=k)
    except TitleNotFoundError as e:
        raise HTTPException(404, str(e))

    return RecommendationResponse(
        query=title,
        results=[RecommendationItem(title=t, similarity=s) for t, s in results],
    )
