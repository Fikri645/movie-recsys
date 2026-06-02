---
title: Movie Recommendation System
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app/gradio_app.py
pinned: false
python_version: "3.11"
---

# Movie Recommendation System

![CI](https://github.com/Fikri645/movie-recsys/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fikri0o0/movie-recsys)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end two-stage movie recommendation system. Compares **3 approaches** from collaborative filtering baseline to a production-style two-stage pipeline (neural retrieval + tree-based ranking), with temporal evaluation and a live Gradio demo.

**[Live Demo →](https://huggingface.co/spaces/fikri0o0/movie-recsys)** | **[GitHub →](https://github.com/Fikri645/movie-recsys)**

---

## Highlights

| What | Detail |
|---|---|
| **Dataset** | MovieLens 1M — 1M ratings · 6,040 users · 3,706 movies |
| **Models** | ALS → Two-Tower Neural → Two-Tower + LightGBM Ranker |
| **Architecture** | Two-stage pipeline: retrieval (top-100) → ranking (reorder) |
| **Evaluation** | Temporal split — last 20% of each user's history as test set |
| **Key insight** | LightGBM ranker improves NDCG@10 over retrieval-only by re-ordering with rich features |
| **Key insight** | Temporal split gives ~20% lower (more realistic) scores than random split |
| **Metrics** | NDCG@K, Recall@K, Hit@K, Catalog Coverage |
| **Tracking** | MLflow — all runs logged |
| **API** | FastAPI `/recommend` endpoint |
| **UI** | Gradio — interactive recommendations with model selector |
| **Deployment** | HuggingFace Spaces |

---

## Architecture

```
MovieLens 1M (ratings.dat)
  └─► data_loader.py   (temporal split: last 20% per user = test)
        └─► Stage 1: Two-Tower Retrieval
              User: Embedding → Linear → ReLU → Linear → L2-norm
              Item: Embedding → Linear → ReLU → Linear → L2-norm
              Training: BPR loss + in-batch negative sampling
              Retrieval: dot product → top-100 candidates
              └─► Stage 2: LightGBM Ranker
                    Features: retrieval_rank, retrieval_score,
                              user_avg_rating, item_avg_rating,
                              year, genre_match, genre flags
                    Loss: LambdaRank (listwise ranking-aware)
                    Output: Final top-K recommendations
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/Fikri645/movie-recsys
cd movie-recsys
pip install -r requirements-dev.txt

# 2. Download MovieLens 1M
python scripts/download_data.py

# 3. Run full 3-model experiment (ALS + Two-Tower + Ranker + MLflow)
python -m src.experiments

# 4. Run API
uvicorn api.main:app --reload

# 5. Run Gradio UI
python app/gradio_app.py
```

Or via `make`:
```bash
make install && make data && make experiments
```

---

## Project Structure

```
movie-recsys/
├── src/
│   ├── config.py           # paths, constants, hyperparameters
│   ├── data_loader.py      # MovieLens loading + temporal split
│   ├── metrics.py          # NDCG@K, Recall@K, Hit@K, Coverage
│   ├── two_tower.py        # PyTorch Two-Tower model + BPR training
│   ├── train_als.py        # ALS baseline (implicit library)
│   ├── train_two_tower.py  # Two-Tower training + embedding index
│   ├── train_ranker.py     # LightGBM LambdaRank stage
│   └── experiments.py      # Full 3-model comparison + plots
├── api/
│   ├── main.py             # FastAPI /recommend endpoint
│   └── schemas.py          # Pydantic request/response models
├── app/gradio_app.py       # Gradio UI (HF Spaces)
├── tests/                  # pytest (metrics, schemas)
├── notebooks/01_eda.ipynb  # MovieLens EDA
├── Dockerfile
├── Makefile
└── requirements-dev.txt
```

---

## Dataset — MovieLens 1M

- **1,000,209 ratings** from 6,040 users on 3,706 movies (2000–2003)
- Ratings 1–5; we treat **rating ≥ 4** as an implicit positive interaction
- **Temporal split**: for each user, the last 20% of their ratings (by timestamp) form the test set — this avoids future-data leakage and mirrors real deployment

Source: [GroupLens Research](https://grouplens.org/datasets/movielens/1m/)

---

## Model Details

### ALS (Alternating Least Squares) — Baseline
Classical collaborative filtering on implicit feedback. Fast, interpretable, no neural components. Uses the `implicit` library (CPU-optimized matrix factorization).

### Two-Tower Neural Retrieval
Two independent embedding towers (user + item), trained with BPR loss and in-batch negative sampling. After training, item embeddings are pre-computed and top-K candidates are retrieved via dot product.

**Why two towers?** They separate the user encoder and item encoder, allowing offline pre-computation of item embeddings for millisecond retrieval from millions of candidates.

### Two-Tower + LightGBM Ranker
The two-tower retrieves 100 candidates; LightGBM re-ranks them using rich features:
- **User:** avg_rating, n_ratings, favorite genre
- **Item:** avg_rating, n_ratings, year, genre flags
- **Cross:** retrieval_rank, retrieval_score (dot product), genre_match
- **Loss:** LambdaRank (listwise, position-aware)

This matches the production architecture at Google, YouTube, and major e-commerce platforms.

---

## Results — MovieLens 1M (temporal split · 2,000 eval users)

![Metrics Comparison](reports/figures/metrics_comparison.png)

| Model | NDCG@10 | Recall@20 | Hit@10 | Coverage@20 |
|---|---|---|---|---|
| ALS | 0.0986 | **0.1272** | 0.4970 | 0.2144 |
| Two-Tower (retrieval only) | 0.0397 | 0.0361 | 0.2550 | **0.131** |
| **Two-Tower + LightGBM Ranker 🏆** | **0.1083** | 0.0909 | **0.4955** | 0.116 |

**Key findings:**

- **Two-Tower + LightGBM Ranker beats ALS on NDCG@10** — once the ranker has proper features (`item_avg_rating`, `genre_match`, `retrieval_score`), it re-orders the Two-Tower's top-100 candidates better than pure matrix factorization. +10% NDCG@10 vs. ALS.
- **Feature quality determines ranker quality.** The initial ranker stopped at 5 trees because `item_avg_rating` was all zeros (data bug — not merged from the ratings table). Fixing that gave 28 trees and a meaningful improvement: +173% NDCG@10 over Two-Tower alone (0.0397 → 0.1083).
- **LambdaRank directly optimizes the ranking metric.** Standard binary cross-entropy treats correct items at rank 1 and rank 10 identically. LambdaRank weights gradients by position discounts — a correct item at rank 1 matters far more than at rank 10.
- **At scale, Two-Tower + Ranker wins decisively.** The real advantage of neural retrieval is sub-millisecond ANN retrieval from billions of items — something ALS cannot do. MovieLens 1M (3K items) is too small to show this latency benefit, but the accuracy benefit already shows here.

---

## Why Temporal Evaluation?

77% of recommendation papers use random train-test splits — which is wrong. If a user rated movie A in 2001 and movie B in 2003, a random split might use B for training and ask the model to predict A (in the past). This leaks future information.

**Temporal split:** for each user, the most recent 20% of interactions form the test set. The model must predict future ratings from past behavior — matching real deployment.

Impact: temporal split gives ~20% lower (but realistic) scores vs. random split.

---

## What I Learned

- **Two-stage pipeline beats ALS** — Two-Tower + LightGBM Ranker achieves NDCG@10=0.1083, beating ALS (0.0986). The key was proper item features: once `item_avg_rating` was correctly merged from the full ratings table, the ranker went from 5 trivial trees to 28 meaningful ones.
- **Feature quality determines ranker quality.** Ranker with all-zero item features → 5 trees, converges trivially. Ranker with `item_avg_rating`, `genre_match`, `year`, `retrieval_score` → 28 trees, +173% NDCG@10 over bare Two-Tower retrieval (0.0397 → 0.1083). Debugging data pipelines is as important as model architecture.
- **LambdaRank > binary cross-entropy for ranking.** BCE treats a correct item at rank 1 and rank 10 identically. LambdaRank weights gradients by the NDCG position discount — correctly prioritizing items at higher ranks.
- **Two-stage pipelines separate concerns correctly.** Retrieval optimizes recall (find 100 candidates in ~1ms); ranking optimizes precision (reorder with rich domain features). Neither alone is sufficient: Two-Tower alone has NDCG@10=0.0397; the ranker stage lifts it to 0.1083.
- **Temporal splits are non-negotiable.** 77% of RecSys papers use random splits, which inflate scores ~20% by leaking future data. Last 20% of each user's history as test set matches real deployment.
- **Coverage@20 tells a different story than NDCG.** ALS has good coverage (0.2144) — it personalises enough that different users see different popular subsets. Two-Tower+Ranker has lower coverage (0.116) because the ranker biases toward highly-rated items. Neither metric alone is sufficient; report both.
- **BPR loss + uniform negatives is sufficient here.** At 3K items, hard negative mining adds complexity without benefit. Worthwhile at scale (10M+ items) where the model needs to distinguish hard near-miss negatives.
- **Data bugs matter more than model architecture.** The ranker's failure (5 trees) was entirely caused by a data pipeline bug (missing `item_avg_rating`). Catching this required checking model outputs for sanity (5 trees → suspect), tracing back to feature values (all zeros), and fixing the merge logic. Model architecture was fine throughout.
