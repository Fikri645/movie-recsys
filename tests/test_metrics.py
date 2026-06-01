"""Unit tests for ranking metrics."""
import pytest
from src.metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_at_k, mrr_at_k


TOP_K   = [0, 1, 2, 5, 8]   # ranked list
RELEVANT = {1, 5}            # ground truth


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k([1, 5], {1, 5}, k=2) == pytest.approx(1.0)

    def test_zero(self):
        assert precision_at_k([3, 4], {1, 2}, k=2) == pytest.approx(0.0)

    def test_partial(self):
        assert precision_at_k([1, 3], {1, 5}, k=2) == pytest.approx(0.5)

    def test_k_larger_than_list(self):
        # only 2 items in top_k but k=5 → count relevant in first 5
        assert precision_at_k([1, 3], {1}, k=5) == pytest.approx(1 / 5)


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k([1, 5, 9], {1, 5}, k=3) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert recall_at_k([3, 4], {1, 2}, k=2) == pytest.approx(0.0)

    def test_partial_recall(self):
        assert recall_at_k([1, 3, 5], {1, 5}, k=2) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert recall_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k([1, 5], {1, 5}, k=2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k([3, 4], {1, 2}, k=2) == pytest.approx(0.0)

    def test_position_matters(self):
        # Relevant item at rank 1 vs rank 2 — first should be higher NDCG
        ndcg_first  = ndcg_at_k([1, 3], {1}, k=2)
        ndcg_second = ndcg_at_k([3, 1], {1}, k=2)
        assert ndcg_first > ndcg_second

    def test_empty_relevant(self):
        assert ndcg_at_k([1, 2], set(), k=2) == pytest.approx(0.0)


class TestHitAtK:
    def test_hit(self):
        assert hit_at_k([3, 1], {1, 5}, k=2) == pytest.approx(1.0)

    def test_miss(self):
        assert hit_at_k([3, 4], {1, 5}, k=2) == pytest.approx(0.0)

    def test_hit_boundary(self):
        assert hit_at_k([3, 4, 1], {1}, k=2) == pytest.approx(0.0)
        assert hit_at_k([3, 4, 1], {1}, k=3) == pytest.approx(1.0)


class TestMRRAtK:
    def test_first_rank(self):
        assert mrr_at_k([1, 3], {1}, k=2) == pytest.approx(1.0)

    def test_second_rank(self):
        assert mrr_at_k([3, 1], {1}, k=2) == pytest.approx(0.5)

    def test_miss(self):
        assert mrr_at_k([3, 4], {1}, k=2) == pytest.approx(0.0)
