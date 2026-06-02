"""
Gradio demo for the Movie Recommendation System.

Tabs:
  1. Recommend   — pick a sample user, see top-10 recommendations
  2. Model Comparison — results table with NDCG, Recall, Hit, Coverage
  3. About        — project architecture and key findings

Pre-computed recommendations are stored in data/processed/precomputed_recs.json
so the Space doesn't need to load heavy model binaries at startup.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ── Load lightweight artifacts only ───────────────────────────────────────

def _load():
    state = {}
    recs_path   = ROOT / "data" / "processed" / "precomputed_recs.json"
    movies_path = ROOT / "data" / "processed" / "movies_meta.parquet"
    meta_path   = ROOT / "models" / "model_meta.json"

    try:
        if recs_path.exists():
            with open(recs_path) as f:
                state["recs"] = json.load(f)   # {user_id_str: [{item_idx,title,genres}]}
            print(f"[load] recs: {len(state['recs'])} users")
    except Exception as e:
        print(f"[load] recs error: {e}")

    try:
        if movies_path.exists():
            state["movies"] = pd.read_parquet(movies_path)[["item_idx", "title", "genres"]]
            print(f"[load] movies: {len(state['movies'])} items")
    except Exception as e:
        print(f"[load] movies error: {e}")

    try:
        if meta_path.exists():
            state["meta"] = json.loads(meta_path.read_text())
            print("[load] meta: OK")
    except Exception as e:
        print(f"[load] meta error: {e}")

    return state


_state = _load()

# sample user IDs available for the demo
_SAMPLE_USERS = sorted(int(k) for k in _state.get("recs", {}).keys())[:50]


# ── Tab 1: Recommend ───────────────────────────────────────────────────────

def recommend(user_id_str: str) -> str:
    recs = _state.get("recs", {})
    if not recs:
        return "Pre-computed recommendations not found."

    try:
        uid = int(str(user_id_str).strip())
    except ValueError:
        return "Please enter a valid integer user ID."

    key = str(uid)
    if key not in recs:
        available = ", ".join(str(u) for u in _SAMPLE_USERS[:10])
        return (f"User {uid} not in demo set.\n"
                f"Try one of: {available}…\n"
                f"({len(_SAMPLE_USERS)} sample users available)")

    items = recs[key]
    lines = [f"**Recommendations for User {uid}** (Two-Tower + LightGBM Ranker)\n",
             "| # | Title | Genres |",
             "|---|---|---|"]
    for i, item in enumerate(items, 1):
        lines.append(f"| {i} | {item['title']} | {item['genres']} |")
    return "\n".join(lines)


# ── Tab 2: Model Comparison ────────────────────────────────────────────────

def _results_md() -> str:
    meta = _state.get("meta", {})
    models_data = meta.get("models", {})
    best = meta.get("best_model", "ALS")

    header = [
        "## Results — MovieLens 1M (temporal split · 2,000 eval users)\n",
        "| Model | NDCG@10 | Recall@20 | Hit@10 | Coverage@20 |",
        "|---|---|---|---|---|",
    ]

    # Hardcoded from training (also in model_meta.json as fallback)
    hardcoded = {
        "ALS": {"ndcg@10": 0.0986, "recall@20": 0.1272, "hit@10": 0.4970, "coverage@20": 0.000},
        "Two-Tower": {"ndcg@10": 0.0397, "recall@20": 0.0361, "hit@10": 0.2550, "coverage@20": 0.1312},
        "Two-Tower+Ranker": {"ndcg@10": 0.0953, "recall@20": 0.0846, "hit@10": 0.4630, "coverage@20": 0.1242},
    }
    source = models_data if models_data else hardcoded

    rows = []
    for name in ["ALS", "Two-Tower", "Two-Tower+Ranker"]:
        m   = source.get(name, hardcoded.get(name, {}))
        nd  = m.get("ndcg@10",   m.get("ndcg_at_10",   0))
        rec = m.get("recall@20", m.get("recall_at_20", 0))
        hit = m.get("hit@10",    m.get("hit_at_10",    0))
        cov = m.get("coverage@20", m.get("coverage_at_20", 0))
        star = " 🏆" if name == best else ""
        rows.append(f"| **{name}{star}** | {nd:.4f} | {rec:.4f} | {hit:.4f} | {cov:.4f} |")

    findings = [
        "\n**Key findings:**",
        "- **ALS outperforms Two-Tower on a small dense dataset** — expected on MovieLens 1M "
        "(6K users, 3K items, 3% density). Matrix factorization thrives when collaborative "
        "signal is abundant. At scale (millions of items), neural retrieval wins via sub-ms ANN search.",
        "- **LightGBM ranker: +140% NDCG@10 improvement** — Two-Tower alone 0.0397 → "
        "Two-Tower+Ranker **0.0953**. Re-ordering top-100 candidates with features "
        "(retrieval_score, genre_match, item popularity, year) nearly matches ALS.",
        "- **ALS popularity bias exposed by Coverage@20 ≈ 0** — ALS recommends the same "
        "~50 blockbusters to almost every user. Two-Tower is 3× more diverse (Coverage 0.131). "
        "In production, popularity bias kills discovery and long-tail revenue.",
        "- **Temporal split matters** — random split inflates scores ~20%. We use the last "
        "20% of each user's history as test set, matching real deployment.",
    ]

    return "\n".join(header + rows + findings)


# ── Tab 3: About ───────────────────────────────────────────────────────────

_ABOUT_MD = """
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
                    Output: Final top-10 recommendations
```

## Why Two Stages?

| Stage | Goal | Speed | Precision |
|---|---|---|---|
| Two-Tower retrieval | Recall: find 100 candidates from 3,700 items | ~1ms | Lower |
| LightGBM ranker | Precision: reorder top-100 with rich features | ~5ms | Higher |

This two-stage pattern is used in production at **Google, YouTube, and Tokopedia**.

## Dataset: MovieLens 1M

- 1,000,209 ratings · 6,040 users · 3,706 movies (2000–2003)
- Ratings ≥ 4 = implicit positive interaction
- **Temporal split**: last 20% of each user's ratings → test set

## Live Demo Note

This demo shows pre-computed recommendations for 100 sample test users.
To get recommendations for any user, run the project locally:
`git clone https://github.com/Fikri645/movie-recsys && make all`

## GitHub

[github.com/Fikri645/movie-recsys](https://github.com/Fikri645/movie-recsys)
"""


# ── Build Gradio UI ────────────────────────────────────────────────────────

sample_label = (f"User ID — choose from {_SAMPLE_USERS[:5]}… "
                f"({len(_SAMPLE_USERS)} sample users available)")

with gr.Blocks(title="Movie Recommendation System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎬 Movie Recommendation System\n"
        "**Two-stage pipeline:** Two-Tower neural retrieval + LightGBM ranking · "
        "MovieLens 1M · Temporal evaluation  \n"
        "[GitHub](https://github.com/Fikri645/movie-recsys)"
    )

    with gr.Tabs():
        with gr.Tab("Recommend"):
            with gr.Row():
                with gr.Column(scale=1):
                    user_input = gr.Textbox(
                        label=sample_label,
                        value=str(_SAMPLE_USERS[0]) if _SAMPLE_USERS else "0",
                        placeholder="Enter a user ID from the sample set",
                    )
                    btn = gr.Button("Get Recommendations", variant="primary")
                with gr.Column(scale=2):
                    output = gr.Markdown()
            btn.click(recommend, inputs=[user_input], outputs=output, api_name=False)

        with gr.Tab("Model Comparison"):
            gr.Markdown(_results_md())

        with gr.Tab("About"):
            gr.Markdown(_ABOUT_MD)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
