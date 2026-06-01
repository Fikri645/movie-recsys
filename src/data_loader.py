"""
Load and preprocess MovieLens 1M.

Key design decisions:
- Treat ratings >= MIN_RATING as implicit positive interactions
- Temporal split per user (last TEST_FRAC interactions go to test)
- Re-index user/item IDs to 0-based integers for embedding layers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

from src.config import (
    MOVIELENS_DIR, DATA_PROC, MIN_RATING,
    MIN_USER_INTS, MIN_ITEM_INTS, TEST_FRAC, SEED,
)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw .dat files into DataFrames."""
    ratings = pd.read_csv(
        MOVIELENS_DIR / "ratings.dat",
        sep="::", header=None, engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        MOVIELENS_DIR / "movies.dat",
        sep="::", header=None, engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        MOVIELENS_DIR / "users.dat",
        sep="::", header=None, engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    return ratings, movies, users


def load_data() -> dict:
    """
    Full preprocessing pipeline. Returns dict with:
      train_df, test_df  — positive interactions (rating >= MIN_RATING)
      movies_df          — movie metadata (title, genres, year)
      user_map, item_map — original_id → 0-based index
      n_users, n_items
    """
    proc = DATA_PROC
    cache = proc / "dataset.parquet"

    ratings, movies, users = load_raw()

    # ── Implicit positives ────────────────────────────────────────────────
    pos = ratings[ratings["rating"] >= MIN_RATING].copy()

    # ── Filter cold-start ─────────────────────────────────────────────────
    user_counts = pos.groupby("user_id").size()
    item_counts = pos.groupby("movie_id").size()
    active_users = user_counts[user_counts >= MIN_USER_INTS].index
    active_items = item_counts[item_counts >= MIN_ITEM_INTS].index
    pos = pos[pos["user_id"].isin(active_users) & pos["movie_id"].isin(active_items)]

    # ── Re-index ──────────────────────────────────────────────────────────
    user_ids   = sorted(pos["user_id"].unique())
    item_ids   = sorted(pos["movie_id"].unique())
    user_map   = {u: i for i, u in enumerate(user_ids)}
    item_map   = {m: i for i, m in enumerate(item_ids)}
    item_unmap = {i: m for m, i in item_map.items()}

    pos["user_idx"] = pos["user_id"].map(user_map)
    pos["item_idx"] = pos["movie_id"].map(item_map)
    pos = pos.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    # ── Temporal split per user ───────────────────────────────────────────
    def split_user(grp: pd.DataFrame) -> pd.DataFrame:
        n    = len(grp)
        cut  = max(1, int(n * (1 - TEST_FRAC)))
        grp  = grp.sort_values("timestamp")
        grp["split"] = "train"
        grp.iloc[cut:, grp.columns.get_loc("split")] = "test"
        return grp

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pos = pos.groupby("user_idx", group_keys=False).apply(split_user)
    train_df = pos[pos["split"] == "train"].copy()
    test_df  = pos[pos["split"] == "test"].copy()

    # ── Movie metadata ────────────────────────────────────────────────────
    movies = movies[movies["movie_id"].isin(item_ids)].copy()
    movies["item_idx"] = movies["movie_id"].map(item_map)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies = movies.dropna(subset=["item_idx"])
    movies["item_idx"] = movies["item_idx"].astype(int)

    # genre one-hot (for ranker features)
    genre_list = sorted({g for gs in movies["genres"].str.split("|") for g in gs})
    for g in genre_list:
        movies[f"genre_{g}"] = movies["genres"].str.contains(g).astype(int)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(DATA_PROC / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROC / "test.parquet", index=False)
    movies.to_parquet(DATA_PROC / "movies_meta.parquet", index=False)

    n_users = len(user_ids)
    n_items = len(item_ids)

    print(f"Dataset: {n_users:,} users | {n_items:,} items")
    print(f"Train interactions: {len(train_df):,}")
    print(f"Test  interactions: {len(test_df):,}")
    print(f"Sparsity: {len(pos) / (n_users * n_items) * 100:.3f}%")

    return dict(
        train_df=train_df,
        test_df=test_df,
        movies_df=movies,
        user_map=user_map,
        item_map=item_map,
        item_unmap=item_unmap,
        n_users=n_users,
        n_items=n_items,
        genre_cols=[c for c in movies.columns if c.startswith("genre_")],
    )


def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """Build sparse user-item interaction matrix (for ALS)."""
    rows = df["user_idx"].values
    cols = df["item_idx"].values
    data = np.ones(len(df), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def get_user_history(train_df: pd.DataFrame) -> dict[int, set[int]]:
    """Map user_idx -> set of item_idxs seen in training."""
    return (
        train_df.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )
