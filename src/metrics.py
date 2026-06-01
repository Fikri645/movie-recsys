"""
Ranking evaluation metrics with proper temporal split support.

All metrics use a ranked list approach:
  - top_k_preds: list of item_idx (ranked, best first)
  - relevant:    set of item_idx that are ground-truth positives
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Callable


def precision_at_k(top_k: list[int], relevant: set[int], k: int) -> float:
    top = top_k[:k]
    return sum(1 for i in top if i in relevant) / k


def recall_at_k(top_k: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = top_k[:k]
    return sum(1 for i in top if i in relevant) / len(relevant)


def ndcg_at_k(top_k: list[int], relevant: set[int], k: int) -> float:
    top = top_k[:k]
    dcg  = sum(1 / np.log2(i + 2) for i, item in enumerate(top) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(top_k: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(i in relevant for i in top_k[:k]) else 0.0


def mrr_at_k(top_k: list[int], relevant: set[int], k: int) -> float:
    for rank, item in enumerate(top_k[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_model(
    recommend_fn: Callable[[int, int], list[int]],
    test_dict: dict[int, set[int]],
    train_dict: dict[int, set[int]],
    k_list: list[int] = [5, 10, 20],
    n_users_eval: int | None = None,
) -> dict[str, float]:
    """
    Evaluate a recommendation model.

    Args:
        recommend_fn: fn(user_idx, k) -> list of top-k item_idx (excl. train)
        test_dict:    user_idx -> set of test item_idx (ground truth)
        train_dict:   user_idx -> set of train item_idx (to exclude from preds)
        k_list:       list of K values to evaluate
        n_users_eval: if set, subsample this many test users (faster eval)
    """
    users = list(test_dict.keys())
    if n_users_eval:
        rng = np.random.default_rng(42)
        users = rng.choice(users, min(n_users_eval, len(users)), replace=False).tolist()

    results = defaultdict(list)
    max_k = max(k_list)

    for user in users:
        relevant = test_dict[user]
        if not relevant:
            continue
        top_k = recommend_fn(user, max_k)
        for k in k_list:
            results[f"precision@{k}"].append(precision_at_k(top_k, relevant, k))
            results[f"recall@{k}"].append(recall_at_k(top_k, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(top_k, relevant, k))
            results[f"hit@{k}"].append(hit_at_k(top_k, relevant, k))

    return {k: float(np.mean(v)) for k, v in results.items()}


def catalog_coverage(recommendations: dict[int, list[int]], n_items: int) -> float:
    """Fraction of total catalog recommended to at least one user."""
    recommended = {item for items in recommendations.values() for item in items}
    return len(recommended) / n_items


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    for k_val in [5, 10, 20]:
        p  = metrics.get(f"precision@{k_val}", 0)
        r  = metrics.get(f"recall@{k_val}", 0)
        nd = metrics.get(f"ndcg@{k_val}", 0)
        h  = metrics.get(f"hit@{k_val}", 0)
        print(f"  @{k_val:<3} | Prec={p:.4f}  Rec={r:.4f}  NDCG={nd:.4f}  Hit={h:.4f}")
    if "coverage" in metrics:
        print(f"  Catalog coverage: {metrics['coverage']:.4f}")
