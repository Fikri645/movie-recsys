"""Unit tests for data_loader utilities."""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data_loader import build_interaction_matrix, get_user_history


def _make_interactions(n_users=3, n_items=5, n_rows=10, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_idx": rng.integers(0, n_users, n_rows),
        "item_idx": rng.integers(0, n_items, n_rows),
        "rating":   rng.integers(4, 6, n_rows),
    })


class TestBuildInteractionMatrix:
    def test_shape(self):
        df = _make_interactions(n_users=3, n_items=5)
        mat = build_interaction_matrix(df, n_users=3, n_items=5)
        assert mat.shape == (3, 5)

    def test_sparse_type(self):
        df = _make_interactions()
        mat = build_interaction_matrix(df, n_users=3, n_items=5)
        assert isinstance(mat, csr_matrix)

    def test_values_are_ones(self):
        df = pd.DataFrame({"user_idx": [0, 1], "item_idx": [2, 3], "rating": [4, 5]})
        mat = build_interaction_matrix(df, n_users=2, n_items=4)
        assert mat[0, 2] == pytest.approx(1.0)
        assert mat[1, 3] == pytest.approx(1.0)
        assert mat[0, 0] == pytest.approx(0.0)

    def test_empty_df(self):
        df = pd.DataFrame({"user_idx": [], "item_idx": [], "rating": []}).astype(int)
        mat = build_interaction_matrix(df, n_users=3, n_items=5)
        assert mat.nnz == 0


class TestGetUserHistory:
    def test_basic(self):
        df = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [2, 3, 4]})
        hist = get_user_history(df)
        assert hist[0] == {2, 3}
        assert hist[1] == {4}

    def test_missing_user(self):
        df = pd.DataFrame({"user_idx": [0], "item_idx": [1]})
        hist = get_user_history(df)
        assert 1 not in hist

    def test_returns_sets(self):
        df = pd.DataFrame({"user_idx": [0, 0], "item_idx": [1, 1]})  # duplicate
        hist = get_user_history(df)
        assert isinstance(hist[0], set)
        assert len(hist[0]) == 1  # deduplicated
