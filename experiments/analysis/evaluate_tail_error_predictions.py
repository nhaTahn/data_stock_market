from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate q90 absolute-error and spike-rate metrics for prediction artifacts.")
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--prediction-format", choices=["core_predictions", "filter_predictions"], default="core_predictions")
    parser.add_argument("--spike-threshold", type=float, default=0.035)
    parser.add_argument("--segment-year", type=int, default=2017)
    parser.add_argument("--segment-start-day", type=int, default=200)
    parser.add_argument("--segment-end-day", type=int, default=250)
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def read_csv_maybe_gzip(path: Path, **kwargs: object) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle, **kwargs)
    return pd.read_csv(path, **kwargs)


def normalize_predictions(path: Path, prediction_format: str) -> pd.DataFrame:
    frame = read_csv_maybe_gzip(path)
    if prediction_format == "core_predictions":
        required = {"code", "Date", "split", "model", "prediction", "actual"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing core prediction columns: {sorted(missing)}")
        out = frame.loc[:, ["code", "Date", "split", "model", "prediction", "actual"]].copy()
        out["eval_date"] = pd.to_datetime(out["Date"])
    else:
        required = {"code", "Date", "actual_date", "split", "base_prediction", "actual_aligned"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing filter prediction columns: {sorted(missing)}")
        out = frame.loc[:, ["code", "actual_date", "split", "base_prediction", "actual_aligned"]].copy()
        out = out.rename(columns={"actual_date": "Date", "base_prediction": "prediction", "actual_aligned": "actual"})
        out["model"] = "base_prediction"
        out["eval_date"] = pd.to_datetime(out["Date"])
    out["code"] = out["code"].astype(str).str.upper()
    out["split"] = out["split"].astype(str)
    out["model"] = out["model"].astype(str)
    out["prediction"] = out["prediction"].astype(float)
    out["actual"] = out["actual"].astype(float)
    out["error"] = out["actual"] - out["prediction"]
    out["abs_error"] = out["error"].abs()
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["eval_date", "prediction", "actual"])


def summarize_group(group: pd.DataFrame, spike_threshold: float) -> dict[str, float | int]:
    actual = group["actual"].to_numpy(dtype=float)
    pred = group["prediction"].to_numpy(dtype=float)
    error = actual - pred
    abs_error = np.abs(error)
    base_loss = robust_loss(actual)
    abs_loss = robust_loss(error)
    daily = (
        group.groupby("eval_date", sort=True)
        .agg(
            n_names=("code", "nunique"),
            daily_q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
        )
        .reset_index()
    )
    daily["is_spike"] = daily["daily_q90_abs_error"].ge(spike_threshold)
    sign_mask = (np.sign(actual) == np.sign(pred)) & (np.abs(actual) > 0.0)
    tail_cut = float(np.quantile(np.abs(actual), 0.80)) if len(actual) else float("nan")
    tail = np.abs(actual) >= tail_cut if np.isfinite(tail_cut) else np.zeros(len(actual), dtype=bool)
    return {
        "n_obs": int(len(group)),
        "n_days": int(daily.shape[0]),
        "n_codes": int(group["code"].nunique()),
        "mae": float(np.mean(abs_error)) if len(abs_error) else float("nan"),
        "median_abs_error": float(np.quantile(abs_error, 0.50)) if len(abs_error) else float("nan"),
        "q80_abs_error": float(np.quantile(abs_error, 0.80)) if len(abs_error) else float("nan"),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)) if len(abs_error) else float("nan"),
        "actual_abs_q90": float(np.quantile(np.abs(actual), 0.90)) if len(actual) else float("nan"),
        "prediction_abs_q90": float(np.quantile(np.abs(pred), 0.90)) if len(pred) else float("nan"),
        "prediction_actual_abs_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.90) / np.quantile(np.abs(actual), 0.90))
            if len(actual) and np.quantile(np.abs(actual), 0.90) > 0.0
            else float("nan")
        ),
        "actual_abs_mean": float(np.mean(np.abs(actual))) if len(actual) else float("nan"),
        "prediction_abs_mean": float(np.mean(np.abs(pred))) if len(pred) else float("nan"),
        "base_loss": base_loss,
        "abs_loss": abs_loss,
        "rel_score": float(1.0 - abs_loss / base_loss) if np.isfinite(base_loss) and base_loss > 0.0 else float("nan"),
        "directional_accuracy": float(np.mean(sign_mask)) if len(sign_mask) else float("nan"),
        "tail_directional_accuracy": float(np.mean(sign_mask[tail])) if np.any(tail) else float("nan"),
        "daily_q90_abs_error_mean": float(daily["daily_q90_abs_error"].mean()) if not daily.empty else float("nan"),
        "daily_q90_abs_error_median": float(daily["daily_q90_abs_error"].median()) if not daily.empty else float("nan"),
        "daily_q90_abs_error_q90": float(daily["daily_q90_abs_error"].quantile(0.90)) if not daily.empty else float("nan"),
        "spike_threshold": float(spike_threshold),
        "spike_days": int(daily["is_spike"].sum()) if not daily.empty else 0,
        "spike_rate": float(daily["is_spike"].mean()) if not daily.empty else float("nan"),
    }


def segment_mask(frame: pd.DataFrame, year: int, start_day: int, end_day: int) -> pd.Series:
    dates = pd.DataFrame({"eval_date": sorted(frame["eval_date"].dropna().unique())})
    dates["eval_date"] = pd.to_datetime(dates["eval_date"])
    dates = dates[dates["eval_date"].dt.year.eq(year)].reset_index(drop=True)
    dates["trading_day_in_year"] = np.arange(len(dates))
    selected = dates.loc[
        dates["trading_day_in_year"].between(start_day, end_day, inclusive="both"),
        "eval_date",
    ]
    return frame["eval_date"].isin(set(selected))


def evaluate(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []
    for (model, split), group in frame.groupby(["model", "split"], sort=True):
        row = {"artifact": args.artifact_name, "model": model, "split": split, "scope": "full_split"}
        row.update(summarize_group(group, args.spike_threshold))
        rows.append(row)

        daily = (
            group.groupby("eval_date", sort=True)
            .agg(
                n_names=("code", "nunique"),
                daily_q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
                daily_mean_abs_error=("abs_error", "mean"),
            )
            .reset_index()
        )
        daily["artifact"] = args.artifact_name
        daily["model"] = model
        daily["split"] = split
        daily["is_spike"] = daily["daily_q90_abs_error"].ge(args.spike_threshold)
        daily_rows.append(daily)

    mask = segment_mask(frame, args.segment_year, args.segment_start_day, args.segment_end_day)
    segment = frame[mask].copy()
    for (model, split), group in segment.groupby(["model", "split"], sort=True):
        row = {
            "artifact": args.artifact_name,
            "model": model,
            "split": split,
            "scope": f"segment_{args.segment_year}_d{args.segment_start_day}_{args.segment_end_day}",
        }
        row.update(summarize_group(group, args.spike_threshold))
        rows.append(row)
    daily_out = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return pd.DataFrame(rows), daily_out


def fmt(value: object) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.5f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.to_numpy()]
        widths = [
            max(len(column), *(len(row[idx]) for row in rows)) if rows else len(column)
            for idx, column in enumerate(columns)
        ]
        header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
        separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
        body = [
            "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
            for row in rows
        ]
        return "\n".join([header, separator, *body])


def write_summary(output_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    display = summary.copy()
    for col in [
        "mae",
        "median_abs_error",
        "q80_abs_error",
        "q90_abs_error",
        "actual_abs_q90",
        "prediction_abs_q90",
        "prediction_actual_abs_q90_ratio",
        "actual_abs_mean",
        "prediction_abs_mean",
        "rel_score",
        "daily_q90_abs_error_median",
        "daily_q90_abs_error_q90",
        "spike_rate",
        "directional_accuracy",
        "tail_directional_accuracy",
    ]:
        if col in display.columns:
            display[col] = display[col].map(fmt)
    lines = [
        "# Tail Error Evaluation",
        "",
        f"Artifact: `{args.artifact_name}`.",
        f"Prediction file: `{args.prediction_file}`.",
        f"Spike rule: daily `q90(|actual - prediction|) >= {args.spike_threshold:.1%}`.",
        "",
        markdown_table(
            display[
            [
                "artifact",
                "model",
                "split",
                "scope",
                "n_obs",
                "n_days",
                "q90_abs_error",
                "actual_abs_q90",
                "prediction_abs_q90",
                "prediction_actual_abs_q90_ratio",
                "daily_q90_abs_error_median",
                "daily_q90_abs_error_q90",
                "spike_rate",
                "rel_score",
                "directional_accuracy",
                "tail_directional_accuracy",
            ]
            ]
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = normalize_predictions(args.prediction_file, args.prediction_format)
    summary, daily = evaluate(frame, args)
    summary.to_csv(args.output_dir / "tail_error_summary.csv", index=False)
    daily.to_csv(args.output_dir / "daily_tail_error.csv", index=False)
    write_summary(args.output_dir, summary, args)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prediction_file": str(args.prediction_file),
                "prediction_format": args.prediction_format,
                "artifact_name": args.artifact_name,
                "spike_threshold": args.spike_threshold,
                "segment_year": args.segment_year,
                "segment_start_day": args.segment_start_day,
                "segment_end_day": args.segment_end_day,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": int(summary.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
