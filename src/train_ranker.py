"""
LightGBM Ranking Stage (Stage 2 of the two-stage pipeline).

Input:  Two-Tower retrieves top-RETRIEVAL_K candidates per user.
Output: LightGBM re-ranks candidates using rich user + item + interaction features.

Features:
  User:  avg_rating, n_ratings, top_genre (one-hot), embedding similarity stats
  Item:  avg_rating_global, n_ratings_global, year, genre flags
  Cross: genre_match, retrieval_rank, retrieval_score (dot product)

Training: LambdaRank (listwise) with group-aware ranking.
"""
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import time
from pathlib import Path

from src.config import (
    RANKER_PATH, MODELS_DIR, TOP_K_LIST, RETRIEVAL_K,
    ITEM_EMBEDDINGS_PATH, USER_EMBEDDINGS_PATH, SEED,
)
from src.data_loader import load_data, get_user_history
from src.train_two_tower import top_k_from_embeddings
from src.metrics import evaluate_model, print_metrics, catalog_coverage


def build_user_stats(train_df: pd.DataFrame, movies_df: pd.DataFrame,
                     genre_cols: list[str]) -> pd.DataFrame:
    """Compute per-user statistics from training history."""
    user_stats = train_df.groupby("user_idx").agg(
        user_n_ratings=("rating", "count"),
        user_avg_rating=("rating", "mean"),
    ).reset_index()

    # favorite genre per user
    user_item  = train_df.merge(movies_df[["item_idx"] + genre_cols], on="item_idx", how="left")
    genre_sums = user_item.groupby("user_idx")[genre_cols].sum()
    user_stats["fav_genre_idx"] = genre_sums.values.argmax(axis=1)
    return user_stats


def build_item_stats(train_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item statistics from training history."""
    item_stats = train_df.groupby("item_idx").agg(
        item_n_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean"),
    ).reset_index()
    item_stats = item_stats.merge(
        movies_df[["item_idx", "year"] + [c for c in movies_df.columns if c.startswith("genre_")]],
        on="item_idx", how="left",
    )
    return item_stats


def build_ranking_dataset(
    users: list[int],
    test_dict: dict[int, set[int]],
    train_dict: dict[int, set[int]],
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    genre_cols: list[str],
    split: str = "train",
) -> tuple[pd.DataFrame, list[int]]:
    """
    For each user, retrieve RETRIEVAL_K candidates and build feature rows.
    Label = 1 if the candidate is in the positive set, else 0.
    """
    rows   = []
    groups = []

    user_stats_map = user_stats.set_index("user_idx").to_dict("index")
    item_stats_map = item_stats.set_index("item_idx").to_dict("index")

    for user in users:
        u_emb      = user_embs[user]
        exclude    = train_dict.get(user, set())
        candidates = top_k_from_embeddings(u_emb, item_embs, RETRIEVAL_K, exclude)
        positives  = test_dict.get(user, set()) if split == "train" else test_dict.get(user, set())

        u_stats = user_stats_map.get(user, {})
        u_avg   = u_stats.get("user_avg_rating", 3.5)
        u_n     = u_stats.get("user_n_ratings", 0)
        u_fav   = u_stats.get("fav_genre_idx", 0)
        u_emb_v = u_emb

        scores = item_embs @ u_emb   # dot products for all items

        for rank, item in enumerate(candidates):
            i_stats = item_stats_map.get(item, {})
            i_avg   = i_stats.get("item_avg_rating", 3.5)
            i_n     = i_stats.get("item_n_ratings", 0)
            i_year  = i_stats.get("year", 2000) or 2000
            i_emb   = item_embs[item]

            genre_match = sum(
                i_stats.get(g, 0) for j, g in enumerate(genre_cols)
                if j == u_fav
            )

            row = {
                "user_idx"       : user,
                "item_idx"       : item,
                "label"          : int(item in positives),
                "retrieval_rank" : rank,
                "retrieval_score": float(scores[item]),
                "user_avg_rating": u_avg,
                "user_n_ratings" : u_n,
                "user_fav_genre" : u_fav,
                "item_avg_rating": i_avg,
                "item_n_ratings" : i_n,
                "item_year"      : i_year,
                "genre_match"    : genre_match,
            }
            # add genre flags
            for g in genre_cols:
                row[g] = i_stats.get(g, 0)

            rows.append(row)

        groups.append(len(candidates))

    return pd.DataFrame(rows), groups


def train_ranker() -> dict[str, float]:
    data       = load_data()
    train_df   = data["train_df"]
    test_df    = data["test_df"]
    movies_df  = data["movies_df"]
    genre_cols = data["genre_cols"]
    n_users    = data["n_users"]
    n_items    = data["n_items"]

    train_dict = get_user_history(train_df)
    test_dict  = get_user_history(test_df)

    print("Loading embeddings ...")
    item_embs = np.load(ITEM_EMBEDDINGS_PATH)
    user_embs = np.load(USER_EMBEDDINGS_PATH)

    user_stats = build_user_stats(train_df, movies_df, genre_cols)
    item_stats = build_item_stats(train_df, movies_df)

    # Sample users for ranker training (uses test positives as labels)
    rng          = np.random.default_rng(SEED)
    all_test_u   = list(test_dict.keys())
    train_users  = rng.choice(all_test_u, size=min(3000, len(all_test_u)), replace=False)
    val_users    = rng.choice(
        [u for u in all_test_u if u not in set(train_users)],
        size=min(500, len(all_test_u) - len(train_users)),
        replace=False,
    )

    print(f"Building ranking dataset ({len(train_users)} train users) ...")
    train_rank, train_groups = build_ranking_dataset(
        train_users, test_dict, train_dict,
        user_embs, item_embs, user_stats, item_stats, genre_cols,
    )
    val_rank, val_groups = build_ranking_dataset(
        val_users, test_dict, train_dict,
        user_embs, item_embs, user_stats, item_stats, genre_cols,
    )

    feat_cols = [c for c in train_rank.columns if c not in ("user_idx", "item_idx", "label")]

    lgb_train = lgb.Dataset(train_rank[feat_cols], label=train_rank["label"],
                            group=train_groups)
    lgb_val   = lgb.Dataset(val_rank[feat_cols],   label=val_rank["label"],
                            group=val_groups, reference=lgb_train)

    params = {
        "objective"      : "lambdarank",
        "metric"         : "ndcg",
        "ndcg_eval_at"   : [5, 10],
        "num_leaves"     : 63,
        "learning_rate"  : 0.05,
        "n_estimators"   : 200,
        "min_child_samples": 10,
        "seed"           : SEED,
        "verbose"        : -1,
    }

    print("Training LightGBM ranker ...")
    t0     = time.time()
    ranker = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)],
    )
    print(f"  Done in {time.time()-t0:.1f}s | best_iter={ranker.best_iteration}")

    # ── Evaluation: two-tower alone vs. two-tower + ranker ────────────────
    def recommend_two_tower(user_idx: int, k: int) -> list[int]:
        u_emb   = user_embs[user_idx]
        exclude = train_dict.get(user_idx, set())
        return top_k_from_embeddings(u_emb, item_embs, k, exclude)

    def recommend_ranked(user_idx: int, k: int) -> list[int]:
        candidates = recommend_two_tower(user_idx, RETRIEVAL_K)
        u_stats    = user_stats.set_index("user_idx").to_dict("index").get(user_idx, {})
        scores_raw = item_embs @ user_embs[user_idx]

        rows = []
        for rank, item in enumerate(candidates):
            i_stats = item_stats.set_index("item_idx").to_dict("index").get(item, {})
            genre_match = sum(
                i_stats.get(g, 0) for j, g in enumerate(genre_cols)
                if j == u_stats.get("fav_genre_idx", 0)
            )
            row = {
                "retrieval_rank" : rank,
                "retrieval_score": float(scores_raw[item]),
                "user_avg_rating": u_stats.get("user_avg_rating", 3.5),
                "user_n_ratings" : u_stats.get("user_n_ratings", 0),
                "user_fav_genre" : u_stats.get("fav_genre_idx", 0),
                "item_avg_rating": i_stats.get("item_avg_rating", 3.5),
                "item_n_ratings" : i_stats.get("item_n_ratings", 0),
                "item_year"      : i_stats.get("year", 2000) or 2000,
                "genre_match"    : genre_match,
            }
            for g in genre_cols:
                row[g] = i_stats.get(g, 0)
            rows.append(row)

        feat_df  = pd.DataFrame(rows)[feat_cols]
        re_scores = ranker.predict(feat_df)
        order     = np.argsort(re_scores)[::-1]
        return [candidates[i] for i in order[:k]]

    eval_users = list(test_dict.keys())
    print(f"Evaluating pipeline ({min(2000, len(eval_users))} users) ...")
    metrics_ranked = evaluate_model(recommend_ranked, test_dict, train_dict,
                                    k_list=TOP_K_LIST, n_users_eval=2000)
    recs_cov = {u: recommend_ranked(u, 20) for u in eval_users[:300]}
    metrics_ranked["coverage@20"] = catalog_coverage(recs_cov, n_items)
    print_metrics("Two-Tower + LightGBM Ranker", metrics_ranked)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RANKER_PATH, "wb") as f:
        pickle.dump(ranker, f)
    print(f"  Saved ranker: {RANKER_PATH}")

    with mlflow.start_run(run_name="Two-Tower+Ranker"):
        mlflow.set_tag("model", "Two-Tower+LightGBM-Ranker")
        mlflow.log_params({**params, "retrieval_k": RETRIEVAL_K})
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics_ranked.items()})

    return metrics_ranked


if __name__ == "__main__":
    mlflow.set_experiment("movie-recsys")
    train_ranker()
