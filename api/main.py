"""
FastAPI serving layer for movie recommendations.

Endpoints:
  GET  /          — health check + model info
  POST /recommend — top-K recommendations for a user
  GET  /users     — list valid user IDs (sample)
  GET  /movies/{item_idx} — movie metadata
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, MovieRec, RecommendRequest, RecommendResponse
from src.config import (
    DATA_PROC,
    ITEM_EMBEDDINGS_PATH,
    MODEL_META_PATH,
    RANKER_PATH,
    RETRIEVAL_K,
    USER_EMBEDDINGS_PATH,
)

app = FastAPI(
    title="Movie Recommendation API",
    description="Two-stage recommender: Two-Tower retrieval + LightGBM ranking",
    version="1.0.0",
)

# ── Artifacts loaded at startup ────────────────────────────────────────────
_state: dict = {}


def _load_artifacts() -> None:
    global _state
    movies_path = DATA_PROC / "movies_meta.parquet"
    if movies_path.exists():
        _state["movies"] = pd.read_parquet(movies_path)
    if ITEM_EMBEDDINGS_PATH.exists():
        _state["item_embs"] = np.load(ITEM_EMBEDDINGS_PATH)
    if USER_EMBEDDINGS_PATH.exists():
        _state["user_embs"] = np.load(USER_EMBEDDINGS_PATH)
    if RANKER_PATH.exists():
        with open(RANKER_PATH, "rb") as f:
            _state["ranker"] = pickle.load(f)
    if MODEL_META_PATH.exists():
        _state["meta"] = json.loads(MODEL_META_PATH.read_text())
    train_path = DATA_PROC / "train.parquet"
    if train_path.exists():
        train = pd.read_parquet(train_path)
        _state["train_history"] = (
            train.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )
    n_users = _state.get("user_embs", np.empty((0, 1))).shape[0]
    n_items = _state.get("item_embs", np.empty((0, 1))).shape[0]
    _state["n_users"] = n_users
    _state["n_items"] = n_items


@app.on_event("startup")
async def startup():
    _load_artifacts()


def _top_k(user_idx: int, k: int) -> tuple[list[int], list[float]]:
    item_embs = _state["item_embs"]
    u_emb     = _state["user_embs"][user_idx]
    exclude   = _state.get("train_history", {}).get(user_idx, set())
    scores    = item_embs @ u_emb
    scores[list(exclude)] = -1e9
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top.tolist(), scores[top].tolist()


@app.get("/", response_model=HealthResponse)
def health():
    meta = _state.get("meta", {})
    return HealthResponse(
        status="ok",
        best_model=meta.get("best_model", "Two-Tower+Ranker"),
        n_users=_state.get("n_users", 0),
        n_items=_state.get("n_items", 0),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if "user_embs" not in _state:
        raise HTTPException(503, "Models not loaded — run experiments first.")
    n_users = _state["n_users"]
    if req.user_id < 0 or req.user_id >= n_users:
        raise HTTPException(404, f"user_id must be 0–{n_users - 1}")

    t0 = time.time()
    items, scores = _top_k(req.user_id, req.k if req.model != "ranked" else RETRIEVAL_K)

    if req.model == "ranked" and "ranker" in _state:
        ranker  = _state["ranker"]
        movies  = _state["movies"].set_index("item_idx")
        item_embs = _state["item_embs"]
        u_emb   = _state["user_embs"][req.user_id]
        raw_scores = item_embs @ u_emb

        rows = []
        for rank, item in enumerate(items):
            row = {"retrieval_rank": rank, "retrieval_score": float(raw_scores[item])}
            if item in movies.index:
                m = movies.loc[item]
                genre_cols = [c for c in movies.columns if c.startswith("genre_")]
                row["item_avg_rating"] = float(m.get("item_avg_rating", 3.5) if hasattr(m, "get") else 3.5)
                row["item_n_ratings"]  = int(m.get("item_n_ratings", 0) if hasattr(m, "get") else 0)
                row["item_year"]       = float(m.get("year", 2000) or 2000)
                row["genre_match"]     = 0
                for g in genre_cols:
                    row[g] = int(m.get(g, 0) if hasattr(m, "get") else 0)
            else:
                row.update({"item_avg_rating": 3.5, "item_n_ratings": 0,
                            "item_year": 2000, "genre_match": 0})
            row.update({"user_avg_rating": 3.5, "user_n_ratings": 0, "user_fav_genre": 0})
            rows.append(row)

        feat_df  = pd.DataFrame(rows)
        feat_cols = [c for c in feat_df.columns]
        re_scores = ranker.predict(feat_df[feat_cols])
        order     = np.argsort(re_scores)[::-1]
        items     = [items[i] for i in order[:req.k]]
        scores    = [float(re_scores[order[i]]) for i in range(req.k)]

    latency = (time.time() - t0) * 1000
    movies_df = _state.get("movies", pd.DataFrame())
    id_to_meta = movies_df.set_index("item_idx").to_dict("index") if not movies_df.empty else {}

    recs = []
    for item, sc in zip(items[:req.k], scores[:req.k]):
        meta = id_to_meta.get(item, {})
        recs.append(MovieRec(
            item_idx=item,
            movie_id=int(meta.get("movie_id", item)),
            title=str(meta.get("title", f"Movie {item}")),
            genres=str(meta.get("genres", "Unknown")),
            score=round(float(sc), 4),
        ))

    return RecommendResponse(
        user_id=req.user_id,
        model=req.model,
        recommendations=recs,
        latency_ms=round(latency, 2),
    )


@app.get("/users")
def list_users():
    n = _state.get("n_users", 0)
    return {"n_users": n, "sample_ids": list(range(min(20, n)))}


@app.get("/movies/{item_idx}")
def get_movie(item_idx: int):
    movies = _state.get("movies", pd.DataFrame())
    if movies.empty or item_idx not in movies["item_idx"].values:
        raise HTTPException(404, f"item_idx {item_idx} not found.")
    row = movies[movies["item_idx"] == item_idx].iloc[0].to_dict()
    return row
