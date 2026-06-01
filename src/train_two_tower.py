"""
Train and evaluate the Two-Tower retrieval model.
Builds an item embedding index for fast top-K retrieval.
"""
from __future__ import annotations

import json
import numpy as np
import torch
import mlflow
import time

from src.config import (
    TWO_TOWER_PATH, ITEM_EMBEDDINGS_PATH, USER_EMBEDDINGS_PATH,
    MODELS_DIR, TOP_K_LIST, RETRIEVAL_K, SEED,
)
from src.data_loader import load_data, get_user_history
from src.two_tower import TwoTowerModel, train_two_tower
from src.metrics import evaluate_model, print_metrics, catalog_coverage


def top_k_from_embeddings(
    user_emb: np.ndarray,       # (D,)
    item_embs: np.ndarray,      # (N, D)
    k: int,
    exclude: set[int],
) -> list[int]:
    """Exact dot-product retrieval (fast for N<=10k)."""
    scores = item_embs @ user_emb              # (N,)
    scores[list(exclude)] = -1e9               # mask train items
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top.tolist()


def train() -> tuple[TwoTowerModel, np.ndarray, dict[str, float]]:
    data = load_data()
    train_df    = data["train_df"]
    test_df     = data["test_df"]
    n_users     = data["n_users"]
    n_items     = data["n_items"]

    train_dict  = get_user_history(train_df)
    test_dict   = get_user_history(test_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = train_two_tower(train_df, n_users, n_items, train_dict, device)

    print("\nBuilding item embedding index ...")
    model.eval()
    item_embs = model.get_all_item_embeddings(device)   # (N, D)
    np.save(ITEM_EMBEDDINGS_PATH, item_embs)

    # also cache user embeddings for the ranker feature set
    user_embs = []
    with torch.no_grad():
        for uid in range(n_users):
            user_embs.append(model.get_user_embedding(uid, device))
    user_embs = np.stack(user_embs)
    np.save(USER_EMBEDDINGS_PATH, user_embs)

    def recommend_fn(user_idx: int, k: int) -> list[int]:
        u_emb   = user_embs[user_idx]
        exclude = train_dict.get(user_idx, set())
        return top_k_from_embeddings(u_emb, item_embs, k, exclude)

    print("Evaluating Two-Tower (2000 users) ...")
    metrics = evaluate_model(recommend_fn, test_dict, train_dict,
                             k_list=TOP_K_LIST, n_users_eval=2000)

    # catalog coverage
    recs = {u: recommend_fn(u, 20) for u in list(test_dict.keys())[:500]}
    metrics["coverage@20"] = catalog_coverage(recs, n_items)

    print_metrics("Two-Tower (retrieval only)", metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TWO_TOWER_PATH)
    print(f"  Saved: {TWO_TOWER_PATH}")

    with mlflow.start_run(run_name="Two-Tower"):
        mlflow.set_tag("model", "Two-Tower")
        mlflow.log_params({
            "embedding_dim": model.user_tower[0].in_features,
            "epochs": 15,
            "batch_size": 1024,
            "lr": 1e-3,
            "n_negatives": 4,
        })
        mlflow.log_metrics(metrics)

    return model, item_embs, metrics


if __name__ == "__main__":
    mlflow.set_experiment("movie-recsys")
    train()
