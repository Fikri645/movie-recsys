"""
Data drift monitoring for the Movie Recommendation System.

Detects distribution shifts in:
  - Rating volume (interaction rate)
  - Item popularity distribution (are users watching different movies?)
  - Rating value distribution (are users rating higher/lower over time?)

Compares a recent window (last N days) against the historical baseline.
Generates an HTML report saved to reports/drift_report.html.

Usage:
    python -m monitoring.drift_report
    python -m monitoring.drift_report --window 60
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_interactions() -> pd.DataFrame:
    """Load processed training data with timestamps."""
    path = ROOT / "data" / "processed" / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No training data at {path}. Run scripts/download_data.py first.")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily feature matrix for drift detection."""
    daily = df.groupby(df["timestamp"].dt.date).agg(
        n_interactions   = ("item_idx", "count"),
        mean_rating      = ("rating",   "mean"),
        unique_users     = ("user_idx", "nunique"),
        unique_items     = ("item_idx", "nunique"),
        pct_high_ratings = ("rating",   lambda x: (x >= 4).mean()),
    ).reset_index()
    daily = daily.rename(columns={"timestamp": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


def compute_psi(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index. >0.2 = drift, 0.1-0.2 = moderate, <0.1 = stable."""
    bins = np.percentile(ref.dropna(), np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    ref_pct = np.maximum(np.histogram(ref, bins=bins)[0] / len(ref), 1e-6)
    cur_pct = np.maximum(np.histogram(cur, bins=bins)[0] / max(len(cur), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_stats(reference: pd.DataFrame, current: pd.DataFrame,
                         features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ref = reference[feat].dropna()
        cur = current[feat].dropna()
        if len(cur) < 2:
            continue
        psi = compute_psi(ref, cur)
        rows.append({
            "feature"      : feat,
            "ref_mean"     : round(float(ref.mean()), 4),
            "cur_mean"     : round(float(cur.mean()), 4),
            "mean_shift_%" : round((float(cur.mean()) - float(ref.mean()))
                                   / (abs(float(ref.mean())) + 1e-9) * 100, 1),
            "psi"          : round(psi, 4),
            "drift_status" : "DRIFT" if psi > 0.2 else "MODERATE" if psi > 0.1 else "STABLE",
        })
    return pd.DataFrame(rows)


def generate_html_report(drift_df: pd.DataFrame, ref_period: str,
                          cur_period: str, output_path: Path) -> None:
    rows_html = ""
    for _, row in drift_df.iterrows():
        color = {"DRIFT": "#ffd7d7", "MODERATE": "#fff4cc", "STABLE": "#d4edda"}.get(
            row["drift_status"], "white"
        )
        rows_html += (
            f"<tr style='background:{color}'>"
            f"<td><b>{row['feature']}</b></td>"
            f"<td>{row['ref_mean']}</td><td>{row['cur_mean']}</td>"
            f"<td>{row['mean_shift_%']:+.1f}%</td>"
            f"<td>{row['psi']:.4f}</td>"
            f"<td><b>{row['drift_status']}</b></td>"
            f"</tr>"
        )

    n_drift = (drift_df["drift_status"] == "DRIFT").sum()
    overall = "DRIFT DETECTED" if n_drift > 0 else "STABLE"
    color   = "#dc3545" if n_drift > 0 else "#28a745"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Movie RecSys — Drift Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #2c3e50; }}
  .badge {{ display: inline-block; padding: 6px 14px; border-radius: 4px; color: white;
            background: {color}; font-size: 16px; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #dee2e6; }}
</style></head><body>
<h1>Movie RecSys — Data Drift Report</h1>
<p><b>Reference:</b> {ref_period} &nbsp;|&nbsp; <b>Current:</b> {cur_period}</p>
<p>Overall: <span class="badge">{overall}</span> ({n_drift}/{len(drift_df)} features drifted)</p>
<table>
<tr><th>Feature</th><th>Ref Mean</th><th>Curr Mean</th>
    <th>Shift</th><th>PSI</th><th>Status</th></tr>
{rows_html}
</table>
<p style="margin-top:20px; color:#666; font-size:13px;">
  PSI: &lt;0.1 = STABLE &nbsp;|&nbsp; 0.1-0.2 = MODERATE &nbsp;|&nbsp; &gt;0.2 = DRIFT (retrain recommended)
</p>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Saved: {output_path}")


def run(recent_window: int = 30) -> pd.DataFrame:
    print("=" * 55)
    print("  Movie RecSys — Drift Report")
    print("=" * 55)

    df    = load_interactions()
    daily = extract_features(df)

    cutoff    = daily["date"].max() - pd.Timedelta(days=recent_window)
    reference = daily[daily["date"] <= cutoff]
    current   = daily[daily["date"] >  cutoff]

    ref_p = f"{reference['date'].min().date()} - {reference['date'].max().date()}"
    cur_p = f"{current['date'].min().date()} - {current['date'].max().date()}"
    print(f"\n  Reference: {ref_p} ({len(reference)} days)")
    print(f"  Current  : {cur_p} ({len(current)} days)")

    features = ["n_interactions", "mean_rating", "unique_users",
                "unique_items", "pct_high_ratings"]
    drift_df = compute_drift_stats(reference, current, features)

    print(f"\n{'Feature':25s} {'Ref':>8} {'Cur':>8} {'Shift':>8} {'PSI':>8}  Status")
    print("-" * 68)
    for _, row in drift_df.iterrows():
        sym = "DRIFT!" if row["drift_status"] == "DRIFT" else row["drift_status"]
        print(f"  {row['feature']:23s} {row['ref_mean']:>8} {row['cur_mean']:>8} "
              f"{row['mean_shift_%']:>7.1f}% {row['psi']:>8.4f}  {sym}")

    output = ROOT / "reports" / "drift_report.html"
    generate_html_report(drift_df, ref_p, cur_p, output)

    n_drift = (drift_df["drift_status"] == "DRIFT").sum()
    if n_drift:
        drifted = drift_df[drift_df["drift_status"] == "DRIFT"]["feature"].tolist()
        print(f"\n  WARNING: {n_drift} feature(s) drifted: {drifted}")
        print("  Consider retraining the Two-Tower model.")
    else:
        print("\n  All features stable.")
    return drift_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()
    run(recent_window=args.window)
