"""
Full 3-model comparison experiment.

  Model 1: ALS (collaborative filtering baseline)
  Model 2: Two-Tower Neural (retrieval only)
  Model 3: Two-Tower + LightGBM Ranker (full pipeline)

Saves results to models/model_meta.json and reports/figures/.
"""
from __future__ import annotations

import json
import time
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.config import MODELS_DIR, FIGURES_DIR, MODEL_META_PATH, TOP_K_LIST
from src.train_als import train_als
from src.train_two_tower import train as train_two_tower
from src.train_ranker import train_ranker


def run_experiments() -> None:
    mlflow.set_experiment("movie-recsys")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Movie RecSys — Full Experiment")
    print("="*60)

    results = {}

    print("\n[1/3] ALS Baseline")
    results["ALS"] = train_als()

    print("\n[2/3] Two-Tower Neural Retrieval")
    _, _, tt_metrics = train_two_tower()
    results["Two-Tower"] = tt_metrics

    print("\n[3/3] Two-Tower + LightGBM Ranker")
    results["Two-Tower+Ranker"] = train_ranker()

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  {'Model':<25} {'NDCG@10':>10} {'Recall@20':>10} {'Hit@10':>10} {'Cov@20':>8}")
    print("-"*70)
    for name, m in results.items():
        nd  = m.get("ndcg@10",   0)
        rec = m.get("recall@20", 0)
        hit = m.get("hit@10",    0)
        cov = m.get("coverage@20", 0)
        print(f"  {name:<25} {nd:>10.4f} {rec:>10.4f} {hit:>10.4f} {cov:>8.4f}")
    print("="*70)

    # ── Save meta ─────────────────────────────────────────────────────────
    best = max(results, key=lambda k: results[k].get("ndcg@10", 0))
    meta = {"models": results, "best_model": best}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nBest model: {best} (NDCG@10={results[best]['ndcg@10']:.4f})")
    print(f"Saved: {MODEL_META_PATH}")

    _make_plots(results)


def _make_plots(results: dict) -> None:
    models = list(results.keys())
    colors = ["#6c757d", "#0d6efd", "#198754"]

    # ── NDCG@K bar chart ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Movie RecSys — Model Comparison", fontsize=14, fontweight="bold")

    k_vals = [5, 10, 20]
    x      = np.arange(len(k_vals))
    width  = 0.25

    ax = axes[0]
    for i, (name, color) in enumerate(zip(models, colors)):
        ndcg_vals = [results[name].get(f"ndcg@{k}", 0) for k in k_vals]
        ax.bar(x + i * width, ndcg_vals, width, label=name, color=color, alpha=0.85)
    ax.set_xlabel("K"); ax.set_ylabel("NDCG@K")
    ax.set_title("NDCG@K — all models")
    ax.set_xticks(x + width); ax.set_xticklabels([f"@{k}" for k in k_vals])
    ax.legend(); ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for i, (name, color) in enumerate(zip(models, colors)):
        rec_vals = [results[name].get(f"recall@{k}", 0) for k in k_vals]
        ax.bar(x + i * width, rec_vals, width, label=name, color=color, alpha=0.85)
    ax.set_xlabel("K"); ax.set_ylabel("Recall@K")
    ax.set_title("Recall@K — all models")
    ax.set_xticks(x + width); ax.set_xticklabels([f"@{k}" for k in k_vals])
    ax.legend(); ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Coverage vs NDCG scatter ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in zip(models, colors):
        nd  = results[name].get("ndcg@10", 0)
        cov = results[name].get("coverage@20", 0)
        ax.scatter(cov, nd, s=200, color=color, label=name, zorder=5)
        ax.annotate(name, (cov, nd), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Catalog Coverage @20"); ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. Diversity trade-off")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path2 = FIGURES_DIR / "accuracy_diversity.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")


if __name__ == "__main__":
    run_experiments()
