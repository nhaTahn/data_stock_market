from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether a daily high-error filter reduces active prediction tail error.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--filter-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="lstm_high_error_probability")
    parser.add_argument("--train-risk-quantile", type=float, default=0.80)
    parser.add_argument("--spike-threshold", type=float, default=0.035)
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def read_predictions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(
            handle,
            usecols=["code", "split", "actual_date", "actual_aligned", "base_prediction"],
        )
    frame["code"] = frame["code"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["actual_date"])
    frame["actual"] = frame["actual_aligned"].astype(float)
    frame["prediction"] = frame["base_prediction"].astype(float)
    frame["error"] = frame["actual"] - frame["prediction"]
    frame["abs_error"] = frame["error"].abs()
    return frame.dropna(subset=["Date", "actual", "prediction"])


def read_filter_scores(path: Path, score_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Date", "split", score_column])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.drop_duplicates(["Date", "split"], keep="last")


def summarize(frame: pd.DataFrame, spike_threshold: float) -> dict[str, float | int]:
    actual = frame["actual"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=float)
    err = actual - pred
    abs_err = np.abs(err)
    base = robust_loss(actual)
    loss = robust_loss(err)
    daily = (
        frame.groupby("Date", sort=True)
        .agg(daily_q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))))
        .reset_index()
    )
    daily["is_spike"] = daily["daily_q90_abs_error"].ge(spike_threshold)
    return {
        "n_obs": int(len(frame)),
        "n_days": int(daily.shape[0]),
        "q90_abs_error": float(np.quantile(abs_err, 0.90)) if len(abs_err) else float("nan"),
        "daily_q90_abs_error_median": float(daily["daily_q90_abs_error"].median()) if not daily.empty else float("nan"),
        "daily_q90_abs_error_q90": float(daily["daily_q90_abs_error"].quantile(0.90)) if not daily.empty else float("nan"),
        "spike_days": int(daily["is_spike"].sum()) if not daily.empty else 0,
        "spike_rate": float(daily["is_spike"].mean()) if not daily.empty else float("nan"),
        "rel_score": float(1.0 - loss / base) if np.isfinite(base) and base > 0.0 else float("nan"),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(args.predictions)
    scores = read_filter_scores(args.filter_scores, args.score_column)
    merged = predictions.merge(scores, on=["Date", "split"], how="left")
    train_scores = merged.loc[merged["split"].eq("train"), args.score_column].dropna()
    threshold = float(train_scores.quantile(args.train_risk_quantile))
    merged["filter_active"] = merged[args.score_column].le(threshold)

    rows: list[dict[str, object]] = []
    for split, group in merged.groupby("split", sort=True):
        row = {"split": split, "policy": "base_all", "coverage": 1.0, "risk_threshold": threshold}
        row.update(summarize(group, args.spike_threshold))
        rows.append(row)

        active = group[group["filter_active"]].copy()
        row = {
            "split": split,
            "policy": f"drop_top_{int((1.0 - args.train_risk_quantile) * 100)}pct_high_error_risk_days",
            "coverage": float(active["Date"].nunique() / max(group["Date"].nunique(), 1)),
            "risk_threshold": threshold,
        }
        row.update(summarize(active, args.spike_threshold))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "filter_effect_summary.csv", index=False)
    merged.to_csv(args.output_dir / "predictions_with_filter_score.csv", index=False)

    display = summary.copy()
    for col in ["coverage", "q90_abs_error", "daily_q90_abs_error_median", "daily_q90_abs_error_q90", "spike_rate", "rel_score", "directional_accuracy"]:
        display[col] = display[col].map(lambda value: f"{value:.5f}" if isinstance(value, float) and np.isfinite(value) else str(value))
    lines = [
        "# High-Error Filter Effect",
        "",
        f"Risk score: `{args.score_column}`.",
        f"Threshold selected on train q{args.train_risk_quantile:.2f}: `{threshold:.5f}`.",
        "",
        display.to_markdown(index=False),
        "",
        "Interpretation: this evaluates active predictions after dropping days where the daily LSTM predicts high forecast-error risk. It is a filter/risk layer, not a changed base return forecast.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "filter_scores": str(args.filter_scores),
                "score_column": args.score_column,
                "train_risk_quantile": args.train_risk_quantile,
                "risk_threshold": threshold,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "risk_threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
