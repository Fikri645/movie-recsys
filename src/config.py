"""Central config — paths, constants, hyperparameters."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Dataset ────────────────────────────────────────────────────────────────
# MovieLens 1M: 1M ratings · 6,040 users · 3,706 movies
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_DIR = DATA_RAW / "ml-1m"

MIN_RATING    = 4      # threshold: rating >= MIN_RATING → implicit positive
MIN_USER_INTS = 5      # drop users with fewer interactions (cold-start floor)
MIN_ITEM_INTS = 5      # drop items with fewer interactions

# ── Train / test split ─────────────────────────────────────────────────────
# Temporal split: last TEST_FRAC of each user's interactions go to test
TEST_FRAC     = 0.2    # 20% of each user's rated movies held out

# ── Model hyperparameters ──────────────────────────────────────────────────
EMBEDDING_DIM   = 64   # shared across ALS and two-tower
N_NEGATIVES     = 4    # negative samples per positive (two-tower training)
EPOCHS          = 15   # two-tower training epochs
BATCH_SIZE      = 1024
LEARNING_RATE   = 1e-3

ALS_FACTORS     = 64   # latent factors for ALS
ALS_ITERATIONS  = 20
ALS_REGULARIZE  = 0.01

# ── Retrieval + ranking ────────────────────────────────────────────────────
RETRIEVAL_K     = 100  # candidates retrieved by two-tower before ranking
TOP_K_LIST      = [5, 10, 20]  # evaluation horizons

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "movie-recsys"

# ── Random seed ────────────────────────────────────────────────────────────
SEED = 42

# ── Model save paths ───────────────────────────────────────────────────────
ALS_MODEL_PATH        = MODELS_DIR / "als_model.npz"
TWO_TOWER_PATH        = MODELS_DIR / "two_tower.pt"
RANKER_PATH           = MODELS_DIR / "lgbm_ranker.pkl"
MODEL_META_PATH       = MODELS_DIR / "model_meta.json"
ITEM_EMBEDDINGS_PATH  = MODELS_DIR / "item_embeddings.npy"
USER_EMBEDDINGS_PATH  = MODELS_DIR / "user_embeddings.npy"
MOVIES_META_PATH      = DATA_PROC / "movies_meta.parquet"
