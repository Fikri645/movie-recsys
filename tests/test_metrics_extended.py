"""Extended tests for metrics.py — evaluate_model and catalog_coverage."""
import numpy as np
import pytest

from src.metrics import catalog_coverage, evaluate_model, print_metrics

# ── evaluate_model ────────────────────────────────────────────────────────

class TestEvaluateModel:
    def _make_perfect_recommender(self, test_dict):
        """Returns a recommender that always returns exactly the test items."""
        def fn(user_idx, k):
            return list(test_dict.get(user_idx, set()))[:k]
        return fn

    def _make_random_recommender(self, n_items=100):
        rng = np.random.default_rng(42)
        def fn(user_idx, k):
            return rng.choice(n_items, size=k, replace=False).tolist()
        return fn

    def test_perfect_recommender_ndcg_is_1(self):
        test_dict  = {0: {1, 2, 3}, 1: {4, 5}}
        train_dict = {0: set(), 1: set()}
        fn = self._make_perfect_recommender(test_dict)
        metrics = evaluate_model(fn, test_dict, train_dict, k_list=[5])
        assert metrics["ndcg@5"] == pytest.approx(1.0, abs=0.01)

    def test_perfect_recommender_recall_is_1(self):
        test_dict  = {0: {1, 2}, 1: {3, 4}}
        train_dict = {0: set(), 1: set()}
        fn = self._make_perfect_recommender(test_dict)
        metrics = evaluate_model(fn, test_dict, train_dict, k_list=[5])
        assert metrics["recall@5"] == pytest.approx(1.0, abs=0.01)

    def test_empty_test_user_skipped(self):
        """Users with no test items should be skipped, not cause errors."""
        test_dict  = {0: set(), 1: {2, 3}}
        train_dict = {0: {1}, 1: set()}
        fn = self._make_perfect_recommender(test_dict)
        metrics = evaluate_model(fn, test_dict, train_dict, k_list=[5])
        # Should complete without error; user 0 skipped
        assert "ndcg@5" in metrics

    def test_multiple_k_values(self):
        test_dict  = {0: {1, 2, 3}}
        train_dict = {0: set()}
        fn = self._make_perfect_recommender(test_dict)
        metrics = evaluate_model(fn, test_dict, train_dict, k_list=[5, 10, 20])
        for k in [5, 10, 20]:
            assert f"ndcg@{k}" in metrics
            assert f"recall@{k}" in metrics
            assert f"hit@{k}" in metrics

    def test_n_users_eval_limits_users(self):
        test_dict  = {i: {i+100} for i in range(50)}
        train_dict = {i: set() for i in range(50)}
        fn = self._make_random_recommender()
        metrics = evaluate_model(fn, test_dict, train_dict, k_list=[5], n_users_eval=10)
        assert "ndcg@5" in metrics  # completed without error


# ── catalog_coverage ──────────────────────────────────────────────────────

class TestCatalogCoverage:
    def test_full_coverage(self):
        """If every item is recommended to someone, coverage = 1.0."""
        recs = {0: [0, 1, 2], 1: [3, 4, 5]}
        assert catalog_coverage(recs, n_items=6) == pytest.approx(1.0)

    def test_zero_coverage(self):
        """Empty recommendations → 0% coverage."""
        recs = {0: [], 1: []}
        assert catalog_coverage(recs, n_items=100) == pytest.approx(0.0)

    def test_partial_coverage(self):
        recs = {0: [0, 1], 1: [0, 1]}  # same 2 items, 10 total
        assert catalog_coverage(recs, n_items=10) == pytest.approx(0.2)

    def test_single_user(self):
        recs = {0: [5, 6, 7]}
        assert catalog_coverage(recs, n_items=10) == pytest.approx(0.3)


# ── print_metrics ─────────────────────────────────────────────────────────

class TestPrintMetrics:
    def test_runs_without_error(self, capsys):
        metrics = {"ndcg@5": 0.1, "recall@5": 0.2, "hit@5": 0.3,
                   "ndcg@10": 0.15, "recall@10": 0.3, "hit@10": 0.4,
                   "ndcg@20": 0.2, "recall@20": 0.4, "hit@20": 0.5,
                   "coverage": 0.1}
        print_metrics("Test Model", metrics)
        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "NDCG" in captured.out
