"""Gradio demo — Movie Recommendation System (portfolio)."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]

# ── Load pre-computed recommendations (tiny JSON, <100 KB) ────────────────
_RECS: dict = {}
_META: dict = {}

try:
    p = ROOT / "data" / "processed" / "precomputed_recs.json"
    if p.exists():
        _RECS = json.loads(p.read_text())
except Exception as e:
    print(f"[load] recs: {e}")

try:
    p = ROOT / "models" / "model_meta.json"
    if p.exists():
        _META = json.loads(p.read_text())
except Exception as e:
    print(f"[load] meta: {e}")

_SAMPLE_USERS = sorted(int(k) for k in _RECS.keys())[:50]
print(f"[startup] {len(_RECS)} users loaded, meta: {bool(_META)}")


# ── Recommend function ─────────────────────────────────────────────────────

def recommend(user_id_str: str) -> str:
    if not _RECS:
        return "Pre-computed recommendations not available."
    try:
        uid = int(str(user_id_str).strip())
    except ValueError:
        return "Please enter a valid integer."

    items = _RECS.get(str(uid))
    if items is None:
        sample = ", ".join(str(u) for u in _SAMPLE_USERS[:10])
        return f"User {uid} not in demo set.\nTry: {sample}"

    lines = [f"**Top 10 for User {uid}** · Two-Tower + LightGBM Ranker\n",
             "| # | Title | Genres |", "|---|---|---|"]
    for i, item in enumerate(items, 1):
        lines.append(f"| {i} | {item['title']} | {item['genres']} |")
    return "\n".join(lines)


# ── Results table ──────────────────────────────────────────────────────────

_RESULTS_MD = """
## Results — MovieLens 1M (temporal split · 2,000 eval users)

| Model | NDCG@10 | Recall@20 | Hit@10 | Coverage@20 |
|---|---|---|---|---|
| ALS 🏆 | **0.0986** | **0.1272** | **0.4970** | ~0 (popularity bias) |
| Two-Tower | 0.0397 | 0.0361 | 0.2550 | **0.131** |
| Two-Tower + LightGBM Ranker | 0.0953 | 0.0846 | 0.4630 | 0.124 |

**Key findings:**
- **ALS beats Two-Tower on small dense datasets** — expected on MovieLens 1M (6K users, 3K items).
  At scale (millions of items), neural retrieval wins via sub-millisecond ANN search.
- **LightGBM ranker: +140% NDCG@10** — Two-Tower 0.0397 → Two-Tower+Ranker 0.0953.
  Re-ordering top-100 candidates with rich features nearly matches ALS.
- **ALS Coverage@20 ≈ 0 reveals popularity bias** — same ~50 blockbusters recommended to everyone.
  Two-Tower is 3× more diverse (Coverage 0.131). In production this kills long-tail revenue.
- **Temporal split** — last 20% of each user's history = test set. Random split inflates scores ~20%.
"""

_ABOUT_MD = """
## Architecture

```
MovieLens 1M (1M ratings · 6K users · 3K movies)
  └─► Temporal split: last 20% per user = test
        └─► Stage 1 — Two-Tower Neural Retrieval (PyTorch + BPR loss)
              User tower: Embedding → Linear → ReLU → L2-norm
              Item tower: Embedding → Linear → ReLU → L2-norm
              Retrieval: dot product → top-100 candidates (~1 ms)
              └─► Stage 2 — LightGBM Ranker (LambdaRank)
                    Features: retrieval_rank, retrieval_score,
                              user_avg_rating, item_avg_rating,
                              year, genre_match, genre flags
                    Output: reordered top-10 (~5 ms)
```

## Why Two Stages?

| Stage | Goal | Latency |
|---|---|---|
| Two-Tower retrieval | Recall — find 100 candidates from millions | ~1 ms |
| LightGBM ranker | Precision — reorder with domain features | ~5 ms |

Pattern used in production at Google, YouTube, and Tokopedia.

## GitHub
[github.com/Fikri645/movie-recsys](https://github.com/Fikri645/movie-recsys)
"""


# ── Gradio UI ──────────────────────────────────────────────────────────────

_hint = (f"Sample user IDs: {', '.join(str(u) for u in _SAMPLE_USERS[:6])}…"
         if _SAMPLE_USERS else "No sample users loaded")

with gr.Blocks(title="Movie RecSys", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎬 Movie Recommendation System\n"
        "**Two-stage pipeline:** Two-Tower neural retrieval + LightGBM ranking · "
        "MovieLens 1M · Temporal evaluation"
    )
    with gr.Tabs():
        with gr.Tab("Recommend"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"*{_hint}*")
                    uid_box = gr.Textbox(
                        label="User ID",
                        value=str(_SAMPLE_USERS[0]) if _SAMPLE_USERS else "0",
                    )
                    btn = gr.Button("Recommend", variant="primary")
                with gr.Column(scale=2):
                    out = gr.Markdown()
            btn.click(recommend, inputs=[uid_box], outputs=out, api_name=False)

        with gr.Tab("Model Comparison"):
            gr.Markdown(_RESULTS_MD)

        with gr.Tab("About"):
            gr.Markdown(_ABOUT_MD)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
