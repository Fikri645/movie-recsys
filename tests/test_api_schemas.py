"""Unit tests for API Pydantic schemas."""
import pytest
from pydantic import ValidationError

from api.schemas import HealthResponse, MovieRec, RecommendRequest


class TestRecommendRequest:
    def test_defaults(self):
        r = RecommendRequest(user_id=42)
        assert r.k == 10
        assert r.model == "ranked"

    def test_valid_models(self):
        for m in ["als", "two_tower", "ranked"]:
            r = RecommendRequest(user_id=0, model=m)
            assert r.model == m

    def test_invalid_model(self):
        with pytest.raises(ValidationError):
            RecommendRequest(user_id=0, model="bert4rec")

    def test_k_out_of_range(self):
        with pytest.raises(ValidationError):
            RecommendRequest(user_id=0, k=0)
        with pytest.raises(ValidationError):
            RecommendRequest(user_id=0, k=51)


class TestMovieRec:
    def test_valid(self):
        m = MovieRec(item_idx=0, movie_id=1, title="Toy Story (1995)",
                     genres="Animation|Children", score=0.95)
        assert m.score == pytest.approx(0.95)

    def test_score_float(self):
        m = MovieRec(item_idx=0, movie_id=1, title="A", genres="B", score=0)
        assert isinstance(m.score, float)


class TestHealthResponse:
    def test_valid(self):
        h = HealthResponse(status="ok", best_model="Two-Tower+Ranker",
                           n_users=6040, n_items=3706)
        assert h.version == "1.0"
        assert h.status == "ok"
