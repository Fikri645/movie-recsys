"""
ALS (Alternating Least Squares) baseline recommender.

Uses the `implicit` library for fast CPU/GPU ALS on implicit feedback.
Ratings >= MIN_RATING are treated as positive interactions (confidence=1).
"""
from __future__ import annotations

import pickle
import time

import mlflow
from implicit.als import AlternatingLeastSquares

from src.config import (
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_MODEL_PATH,
    ALS_REGULARIZE,
    MODELS_DIR,
    SEED,
    TOP_K_LIST,
)
from src.data_loader import build_interaction_matrix, get_user_history, load_data
from src.metrics import catalog_coverage, evaluate_model, print_metrics


class ALSRecommender:
    def __init__(self, factors: int = ALS_FACTORS, iterations: int = ALS_ITERATIONS,
                 regularization: float = ALS_REGULARIZE):
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=SEED,
            use_gpu=False,
        )
        self.user_factors = None
        self.item_factors = None
        self.train_matrix = None

    def fit(self, train_matrix) -> None:
        self.train_matrix = train_matrix
        print("Training ALS ...")
        t0 = time.time()
        self.model.fit(train_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        print(f"  Done in {time.time() - t0:.1f}s")

    def recommend(self, user_idx: int, k: int, train_history: set[int]) -> list[int]:
        ids, _ = self.model.recommend(
            user_idx, self.train_matrix[user_idx],
            N=k, filter_already_liked_items=True,
        )
        return [int(i) for i in ids[:k]]

    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(ALS_MODEL_PATH.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved: {ALS_MODEL_PATH.with_suffix('.pkl')}")

    @classmethod
    def load(cls) -> "ALSRecommender":
        with open(ALS_MODEL_PATH.with_suffix(".pkl"), "rb") as f:
            return pickle.load(f)


def train_als() -> dict[str, float]:
    data = load_data()
    train_df = data["train_df"]
    test_df  = data["test_df"]
    n_users  = data["n_users"]
    n_items  = data["n_items"]

    train_matrix = build_interaction_matrix(train_df, n_users, n_items)
    train_dict   = get_user_history(train_df)
    test_dict    = get_user_history(test_df)

    rec = ALSRecommender()
    rec.fit(train_matrix)

    def recommend_fn(user_idx: int, k: int) -> list[int]:
        return rec.recommend(user_idx, k, train_dict.get(user_idx, set()))

    print("Evaluating ALS (2000 users) ...")
    metrics = evaluate_model(recommend_fn, test_dict, train_dict,
                             k_list=TOP_K_LIST, n_users_eval=2000)

    # Catalog coverage (ALS is known for popularity bias — this exposes it)
    eval_users = list(test_dict.keys())[:500]
    recs_cov = {u: recommend_fn(u, 20) for u in eval_users}
    metrics["coverage@20"] = catalog_coverage(recs_cov, n_items)

    print_metrics("ALS Baseline", metrics)
    rec.save()

    with mlflow.start_run(run_name="ALS"):
        mlflow.set_tag("model", "ALS")
        mlflow.log_params({
            "factors": ALS_FACTORS,
            "iterations": ALS_ITERATIONS,
            "regularization": ALS_REGULARIZE,
        })
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})

    return metrics


if __name__ == "__main__":
    mlflow.set_experiment("movie-recsys")
    train_als()
