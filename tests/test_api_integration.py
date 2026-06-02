"""Integration tests for the FastAPI recommendation endpoint.

Uses FastAPI's TestClient — no running server required.
Tests graceful degradation when models are not loaded.
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_schema(self):
        resp = client.get("/")
        data = resp.json()
        assert data["status"] == "ok"
        assert "best_model" in data
        assert "n_users" in data
        assert "n_items" in data
        assert "version" in data


class TestSeriesEndpoint:
    def test_users_returns_200(self):
        resp = client.get("/users")
        assert resp.status_code == 200

    def test_users_schema(self):
        resp = client.get("/users")
        data = resp.json()
        assert "n_users" in data
        assert "sample_ids" in data
        assert isinstance(data["sample_ids"], list)


class TestRecommendEndpoint:
    def test_invalid_user_id_type(self):
        """Non-integer user_id should fail schema validation."""
        resp = client.post("/recommend", json={"user_id": "abc"})
        assert resp.status_code == 422  # pydantic validation error

    def test_valid_request_structure(self):
        """Well-formed request should return 200 or 404/503 (not 500)."""
        resp = client.post("/recommend", json={"user_id": 0, "k": 5, "model": "ranked"})
        assert resp.status_code in (200, 404, 503)

    def test_invalid_model_name(self):
        """Unknown model should fail schema validation."""
        resp = client.post("/recommend", json={"user_id": 0, "model": "bert4rec"})
        assert resp.status_code == 422

    def test_k_out_of_range(self):
        """k=0 should fail validation."""
        resp = client.post("/recommend", json={"user_id": 0, "k": 0})
        assert resp.status_code == 422

    def test_k_max_boundary(self):
        """k=50 is the max allowed."""
        resp = client.post("/recommend", json={"user_id": 0, "k": 50, "model": "ranked"})
        assert resp.status_code in (200, 404, 503)

    def test_k_over_max_rejected(self):
        resp = client.post("/recommend", json={"user_id": 0, "k": 51})
        assert resp.status_code == 422


class TestMovieEndpoint:
    def test_movie_not_found(self):
        """Very large item_idx should return 404, not 500."""
        resp = client.get("/movies/999999")
        assert resp.status_code in (404, 503)
