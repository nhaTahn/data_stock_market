from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_quantiles(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid daily high-error filter thresholds on frozen LSTM predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--filter-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="lstm_high_error_probability")
    parser.add_argument("--risk-quantiles", type=parse_quantiles, default=parse_quantiles("0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95"))
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
            usecols=[
                "code",
                "split",
                "actual_date",
                "actual_aligned",
                "base_prediction",
                "filter_probability",
            ],
        )
    frame["code"] = frame["code"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["actual_date"])
    frame["actual"] = frame["actual_aligned"].astype(float)
    frame["prediction"] = frame["base_prediction"].astype(float)
    frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
    return frame.dropna(subset=["Date", "actual", "prediction"])


def read_scores(path: Path, score_column: str) -> pd.DataFrame:
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
    sign_ok = np.sign(actual) == np.sign(pred)
    tail_cut = float(np.quantile(np.abs(actual), 0.80)) if len(actual) else float("nan")
    tail_mask = np.abs(actual) >= tail_cut if np.isfinite(tail_cut) else np.zeros(len(actual), dtype=bool)
    return {
        "n_obs": int(len(frame)),
        "n_days": int(daily.shape[0]),
        "q90_abs_error": float(np.quantile(abs_err, 0.90)) if len(abs_err) else float("nan"),
        "daily_q90_abs_error_median": float(daily["daily_q90_abs_error"].median()) if not daily.empty else float("nan"),
        "daily_q90_abs_error_q90": float(daily["daily_q90_abs_error"].quantile(0.90)) if not daily.empty else float("nan"),
        "spike_days": int(daily["is_spike"].sum()) if not daily.empty else 0,
        "spike_rate": float(daily["is_spike"].mean()) if not daily.empty else float("nan"),
        "rel_score": float(1.0 - loss / base) if np.isfinite(base) and base > 0.0 else float("nan"),
        "directional_accuracy": float(np.mean(sign_ok)) if len(sign_ok) else float("nan"),
        "tail_directional_accuracy": float(np.mean(sign_ok[tail_mask])) if np.any(tail_mask) else float("nan"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(args.predictions)
    scores = read_scores(args.filter_scores, args.score_column)
    merged = predictions.merge(scores, on=["Date", "split"], how="left")
    train_scores = merged.loc[merged["split"].eq("train"), args.score_column].dropna()
    rows: list[dict[str, object]] = []
    for quantile in args.risk_quantiles:
        threshold = float(train_scores.quantile(quantile))
        for split, group in merged.groupby("split", sort=True):
            active = group[group[args.score_column].le(threshold)].copy()
            row = {
                "split": split,
                "risk_quantile": quantile,
                "risk_threshold": threshold,
                "day_coverage": float(active["Date"].nunique() / max(group["Date"].nunique(), 1)),
                "obs_coverage": float(len(active) / max(len(group), 1)),
            }
            row.update(summarize(active, args.spike_threshold))
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "high_error_filter_grid.csv", index=False)
    val = summary[summary["split"].eq("val")].copy()
    recommended = val[
        (val["day_coverage"] >= 0.45)
        & (val["q90_abs_error"] <= 0.035)
    ].sort_values(["q90_abs_error", "spike_rate"], kind="stable")
    if recommended.empty:
        recommended = val.sort_values(["q90_abs_error", "spike_rate"], kind="stable").head(5)
    recommended.to_csv(args.output_dir / "recommended_filter_grid.csv", index=False)

    display = summary.copy()
    for column in [
        "day_coverage",
        "obs_coverage",
        "q90_abs_error",
        "daily_q90_abs_error_median",
        "daily_q90_abs_error_q90",
        "spike_rate",
        "rel_score",
        "directional_accuracy",
        "tail_directional_accuracy",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.5f}" if np.isfinite(float(value)) else "n/a")
    lines = [
        "# High-Error Filter Grid",
        "",
        f"Risk score: `{args.score_column}`.",
        f"Thresholds are train quantiles: `{','.join(str(q) for q in args.risk_quantiles)}`.",
        "",
        "## Grid",
        "",
        display.to_markdown(index=False),
        "",
        "## Recommended Rows",
        "",
        recommended.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "filter_scores": str(args.filter_scores),
                "score_column": args.score_column,
                "risk_quantiles": list(args.risk_quantiles),
                "spike_threshold": args.spike_threshold,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": int(summary.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
