"""Pydantic schemas for the recommendation API."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="0-based user index")
    k: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    model: Literal["als", "two_tower", "ranked"] = Field(
        default="ranked", description="Which model to use"
    )

    model_config = {"json_schema_extra": {"example": {
        "user_id": 42, "k": 10, "model": "ranked"
    }}}


class MovieRec(BaseModel):
    item_idx: int
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    model: str
    recommendations: list[MovieRec]
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    best_model: str
    n_users: int
    n_items: int
    version: str = "1.0"
