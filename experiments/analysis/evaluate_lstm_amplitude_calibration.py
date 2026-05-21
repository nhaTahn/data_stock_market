from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = (
    ROOT
    / "gold"
    / "vn_transition_pressure_20260512"
    / "plots"
    / "tail_confidence_lstm_ablation"
    / "tail_confidence_seed_sweep_comparison.csv"
)
DEFAULT_RUN_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "tail_confidence_lstm_ablation"
)
DEFAULT_OUTPUT = (
    ROOT
    / "gold"
    / "vn_transition_pressure_20260512"
    / "plots"
    / "tail_confidence_lstm_ablation"
    / "amplitude_calibration"
)


@dataclass(frozen=True)
class CalibrationResult:
    name: str
    train_score: float
    scales: dict[str, float]
    prediction: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate post-LSTM amplitude calibration on train/val split.")
    parser.add_argument("--sweep-file", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--max-scale", type=float, default=3.0)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--segment-year", type=int, default=2017)
    parser.add_argument("--segment-start-day", type=int, default=200)
    parser.add_argument("--segment-end-day", type=int, default=250)
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base_loss = robust_loss(actual)
    error_loss = robust_loss(actual - prediction)
    if not np.isfinite(base_loss) or base_loss <= 0.0:
        return float("nan")
    return float(1.0 - error_loss / base_loss)


def q90_abs_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.quantile(np.abs(actual - prediction), 0.90))


def pred_actual_abs_q90_ratio(actual: np.ndarray, prediction: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    actual_q90 = float(np.quantile(np.abs(actual), 0.90))
    if actual_q90 <= 0.0:
        return float("nan")
    return float(np.quantile(np.abs(prediction), 0.90) / actual_q90)


def directional_accuracy(actual: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.abs(actual) > 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(actual[mask]) == np.sign(prediction[mask])))


def choose_lstm_rows(frame: pd.DataFrame, preferred_model: str) -> pd.DataFrame:
    models = set(frame["model"].astype(str).unique())
    if preferred_model in models:
        chosen = preferred_model
    elif "lstm_best_by_val" in models:
        chosen = "lstm_best_by_val"
    else:
        lstm_models = sorted(model for model in models if model.startswith("lstm"))
        if not lstm_models:
            raise ValueError(f"No LSTM model rows found. Available models: {sorted(models)}")
        chosen = lstm_models[0]
    out = frame[frame["model"].astype(str).eq(chosen)].copy()
    out["chosen_model"] = chosen
    return out


def fit_best_scale(actual: np.ndarray, prediction: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    best_scale = 1.0
    best_score = -np.inf
    for scale in grid:
        score = rel_score(actual, prediction * scale)
        if score > best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale, float(best_score)


def apply_bucket_scales(values: np.ndarray, prediction: np.ndarray, thresholds: np.ndarray, scales: list[float]) -> np.ndarray:
    bucket = np.digitize(values, thresholds, right=False)
    adjusted = prediction.copy()
    for idx, scale in enumerate(scales):
        adjusted[bucket == idx] = prediction[bucket == idx] * scale
    return adjusted


def fit_bucket_scales(
    train_actual: np.ndarray,
    train_prediction: np.ndarray,
    train_values: np.ndarray,
    val_prediction: np.ndarray,
    val_values: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float]]:
    thresholds = np.quantile(train_values[np.isfinite(train_values)], [1.0 / 3.0, 2.0 / 3.0])
    train_bucket = np.digitize(train_values, thresholds, right=False)
    scales: list[float] = []
    for idx in range(3):
        mask = train_bucket == idx
        if np.sum(mask) < 50:
            scale = 1.0
        else:
            scale, _ = fit_best_scale(train_actual[mask], train_prediction[mask], grid)
        scales.append(scale)
    train_adjusted = apply_bucket_scales(train_values, train_prediction, thresholds, scales)
    val_adjusted = apply_bucket_scales(val_values, val_prediction, thresholds, scales)
    return (
        val_adjusted,
        rel_score(train_actual, train_adjusted),
        {
            "threshold_1": float(thresholds[0]),
            "threshold_2": float(thresholds[1]),
            "scale_low": float(scales[0]),
            "scale_mid": float(scales[1]),
            "scale_high": float(scales[2]),
        },
    )


def fit_sign_split(
    train_actual: np.ndarray,
    train_prediction: np.ndarray,
    val_prediction: np.ndarray,
    grid: np.ndarray,
) -> CalibrationResult:
    positive = train_prediction >= 0.0
    negative = ~positive
    pos_scale, _ = fit_best_scale(train_actual[positive], train_prediction[positive], grid) if np.any(positive) else (1.0, float("nan"))
    neg_scale, _ = fit_best_scale(train_actual[negative], train_prediction[negative], grid) if np.any(negative) else (1.0, float("nan"))
    train_adjusted = np.where(positive, train_prediction * pos_scale, train_prediction * neg_scale)
    val_adjusted = np.where(val_prediction >= 0.0, val_prediction * pos_scale, val_prediction * neg_scale)
    return CalibrationResult(
        name="sign_split_grid",
        train_score=rel_score(train_actual, train_adjusted),
        scales={"positive_scale": float(pos_scale), "negative_scale": float(neg_scale)},
        prediction=val_adjusted,
    )


def segment_mask(frame: pd.DataFrame, year: int, start_day: int, end_day: int) -> pd.Series:
    dates = pd.DataFrame({"Date": sorted(frame["Date"].dropna().unique())})
    dates["Date"] = pd.to_datetime(dates["Date"])
    dates = dates[dates["Date"].dt.year.eq(year)].reset_index(drop=True)
    dates["trading_day_in_year"] = np.arange(len(dates))
    selected = dates.loc[dates["trading_day_in_year"].between(start_day, end_day), "Date"]
    return pd.to_datetime(frame["Date"]).isin(set(selected))


def summarize(frame: pd.DataFrame, prediction: np.ndarray, scope: str) -> dict[str, float | int | str]:
    actual = frame["actual"].to_numpy(dtype=float)
    if len(frame) == 0:
        return {
            "scope": scope,
            "n_obs": 0,
            "n_days": 0,
            "rel_score": float("nan"),
            "q90_abs_error": float("nan"),
            "daily_q90_median": float("nan"),
            "daily_q90_q90": float("nan"),
            "spike_rate": float("nan"),
            "prediction_actual_abs_q90_ratio": float("nan"),
            "directional_accuracy": float("nan"),
        }
    daily = (
        pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "abs_error": np.abs(actual - prediction)})
        .groupby("Date", sort=True)["abs_error"]
        .quantile(0.90)
    )
    return {
        "scope": scope,
        "n_obs": int(len(frame)),
        "n_days": int(daily.shape[0]),
        "rel_score": rel_score(actual, prediction),
        "q90_abs_error": q90_abs_error(actual, prediction),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_q90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "spike_rate": float((daily >= 0.035).mean()) if not daily.empty else float("nan"),
        "prediction_actual_abs_q90_ratio": pred_actual_abs_q90_ratio(actual, prediction),
        "directional_accuracy": directional_accuracy(actual, prediction),
    }


def evaluate_run(row: pd.Series, args: argparse.Namespace, grid: np.ndarray) -> list[dict[str, object]]:
    run_name = str(row["run_name"])
    prediction_file = args.run_root / run_name / "reports" / "core" / "predictions.csv"
    frame = pd.read_csv(prediction_file)
    frame = choose_lstm_rows(frame, args.model)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", "actual", "Date"])
    train = frame[frame["split"].astype(str).eq("train")].copy()
    val = frame[frame["split"].astype(str).eq("val")].copy()
    train_actual = train["actual"].to_numpy(dtype=float)
    train_prediction = train["prediction"].to_numpy(dtype=float)
    val_prediction = val["prediction"].to_numpy(dtype=float)
    calibrations: list[CalibrationResult] = [
        CalibrationResult(
            name="identity",
            train_score=rel_score(train_actual, train_prediction),
            scales={"scale": 1.0},
            prediction=val_prediction,
        )
    ]
    global_scale, global_train = fit_best_scale(train_actual, train_prediction, grid)
    calibrations.append(
        CalibrationResult(
            name="global_grid",
            train_score=global_train,
            scales={"scale": global_scale},
            prediction=val_prediction * global_scale,
        )
    )
    up_grid = grid[grid >= 1.0]
    up_scale, up_train = fit_best_scale(train_actual, train_prediction, up_grid)
    calibrations.append(
        CalibrationResult(
            name="global_up_only",
            train_score=up_train,
            scales={"scale": up_scale},
            prediction=val_prediction * up_scale,
        )
    )
    calibrations.append(fit_sign_split(train_actual, train_prediction, val_prediction, grid))
    val_bucket_prediction, train_score, scale_info = fit_bucket_scales(
        train_actual,
        train_prediction,
        np.abs(train_prediction),
        val_prediction,
        np.abs(val_prediction),
        grid,
    )
    calibrations.append(
        CalibrationResult(
            name="predmag_bucket_grid",
            train_score=train_score,
            scales=scale_info,
            prediction=val_bucket_prediction,
        )
    )
    normalizer_col = "__target_normalizer__volatility_20"
    if normalizer_col in train.columns:
        val_bucket_prediction, train_score, scale_info = fit_bucket_scales(
            train_actual,
            train_prediction,
            train[normalizer_col].to_numpy(dtype=float),
            val_prediction,
            val[normalizer_col].to_numpy(dtype=float),
            grid,
        )
        calibrations.append(
            CalibrationResult(
                name="vol_bucket_grid",
                train_score=train_score,
                scales=scale_info,
                prediction=val_bucket_prediction,
            )
        )

    rows: list[dict[str, object]] = []
    mask = segment_mask(val, args.segment_year, args.segment_start_day, args.segment_end_day)
    for calibration in calibrations:
        val_summary = summarize(val, calibration.prediction, "val_full")
        summaries = [val_summary]
        if bool(mask.any()):
            summaries.append(
                summarize(
                    val[mask],
                    calibration.prediction[mask.to_numpy()],
                    f"segment_{args.segment_year}_d{args.segment_start_day}_{args.segment_end_day}",
                )
            )
        for summary in summaries:
            rows.append(
                {
                    "candidate": row["candidate"],
                    "seed": int(row["seed"]),
                    "run_name": run_name,
                    "chosen_model": frame["chosen_model"].iloc[0],
                    "calibration": calibration.name,
                    "train_rel_score": calibration.train_score,
                    "scales_json": json.dumps(calibration.scales, sort_keys=True),
                    **summary,
                }
            )
    return rows


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


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "amplitude_calibration_by_run.csv", index=False)
    val = frame[frame["scope"].eq("val_full")].copy()
    aggregate = (
        val.groupby(["candidate", "calibration"], sort=True)
        .agg(
            seeds=("seed", "count"),
            val_rel_score_mean=("rel_score", "mean"),
            val_rel_score_min=("rel_score", "min"),
            val_q90_abs_error_mean=("q90_abs_error", "mean"),
            val_spike_rate_mean=("spike_rate", "mean"),
            val_pred_actual_abs_q90_ratio_mean=("prediction_actual_abs_q90_ratio", "mean"),
            val_directional_accuracy_mean=("directional_accuracy", "mean"),
            train_rel_score_mean=("train_rel_score", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "amplitude_calibration_aggregate.csv", index=False)
    segment = frame[frame["scope"].str.startswith("segment_")].copy()
    segment_aggregate = (
        segment.groupby(["candidate", "calibration"], sort=True)
        .agg(
            seeds=("seed", "count"),
            segment_rel_score_mean=("rel_score", "mean"),
            segment_q90_abs_error_mean=("q90_abs_error", "mean"),
            segment_pred_actual_abs_q90_ratio_mean=("prediction_actual_abs_q90_ratio", "mean"),
        )
        .reset_index()
    )
    segment_aggregate.to_csv(output_dir / "amplitude_calibration_segment_aggregate.csv", index=False)
    display = aggregate.copy()
    for column in display.columns:
        if column not in {"candidate", "calibration"}:
            display[column] = display[column].map(lambda value: f"{value:.5f}" if isinstance(value, float) else value)
    lines = [
        "# LSTM Amplitude Calibration",
        "",
        "Scope: fit calibration scales on train split only, then evaluate on validation split.",
        "",
        "## Validation Aggregate",
        "",
        markdown_table(display),
        "",
        "## Read",
        "",
        "- `global_up_only` checks whether simply forcing larger LSTM predictions helps.",
        "- `global_grid`, `sign_split_grid`, `predmag_bucket_grid`, and `vol_bucket_grid` are post-hoc diagnostics, not deployable improvements unless they survive walk-forward.",
        "- If scale-up does not help, the next model improvement should be a learned confidence/regime head rather than manual amplitude scaling.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep = pd.read_csv(args.sweep_file)
    grid = np.round(np.arange(0.0, args.max_scale + args.grid_step / 2.0, args.grid_step), 10)
    rows: list[dict[str, object]] = []
    for _, row in sweep.iterrows():
        rows.extend(evaluate_run(row, args, grid))
    write_outputs(rows, args.output_dir)
    print(json.dumps({"rows": len(rows), "summary": str(args.output_dir / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
