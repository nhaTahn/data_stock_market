from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "tail_aware_lstm_multiseed_20260519"
)
DEFAULT_GOLD_DIR = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tail_aware_lstm_multiseed_20260519"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train-fitted calibration for multiseed LSTM predictions.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--shrink-max", type=float, default=1.25)
    return parser.parse_args(argv)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def daily_q90_penalized_loss(frame: pd.DataFrame, prediction: np.ndarray, tail_penalty: float) -> float:
    actual = frame["actual"].to_numpy(dtype=float)
    error = actual - prediction
    daily_q90 = (
        pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "abs_error": np.abs(error)})
        .groupby("Date", sort=True)["abs_error"]
        .quantile(0.90)
    )
    tail_term = float(daily_q90.quantile(0.90)) if not daily_q90.empty else 0.0
    return robust_loss(error) + tail_penalty * tail_term


def summarize_prediction(frame: pd.DataFrame, prediction: np.ndarray, name: str, seed: int, split: str) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    error = actual - prediction
    abs_error = np.abs(error)
    daily = (
        pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "abs_error": abs_error})
        .groupby("Date", sort=True)["abs_error"]
        .quantile(0.90)
    )
    actual_abs_q90 = float(np.quantile(np.abs(actual), 0.90)) if len(actual) else float("nan")
    prediction_abs_q90 = float(np.quantile(np.abs(prediction), 0.90)) if len(prediction) else float("nan")
    return {
        "seed": seed,
        "candidate": name,
        "split": split,
        "n_obs": int(len(frame)),
        "n_days": int(daily.shape[0]),
        "rel_score": rel_score(actual, prediction),
        "median_abs_error": float(np.quantile(abs_error, 0.50)) if len(abs_error) else float("nan"),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)) if len(abs_error) else float("nan"),
        "daily_q90_abs_error_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_abs_error_q90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_abs_error_max": float(daily.max()) if not daily.empty else float("nan"),
        "spike_days_ge_7pct": int(daily.ge(0.07).sum()) if not daily.empty else 0,
        "spike_days_ge_8pct": int(daily.ge(0.08).sum()) if not daily.empty else 0,
        "prediction_actual_abs_q90_ratio": (
            prediction_abs_q90 / actual_abs_q90 if np.isfinite(actual_abs_q90) and actual_abs_q90 > 0.0 else float("nan")
        ),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))) if len(actual) else float("nan"),
    }


def read_seed_predictions(run_root: Path, seed: int) -> pd.DataFrame:
    names = {
        "base": "plain_global_rel",
        "weighted": "plain_global_weighted_mild",
        "instance": "plain_global_instance_rel",
    }
    merged: pd.DataFrame | None = None
    for short_name, variant in names.items():
        path = run_root / f"seed_{seed}" / f"predictions_{variant}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["Date"])
        cols = frame.loc[:, ["code", "Date", "split", "actual", "prediction"]].copy()
        cols = cols.rename(columns={"prediction": f"pred_{short_name}"})
        if merged is None:
            merged = cols
        else:
            merged = merged.merge(cols, on=["code", "Date", "split", "actual"], how="inner")
    if merged is None:
        raise ValueError(f"No predictions found for seed {seed}")
    return merged


def fit_best_scale(actual: np.ndarray, prediction: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    best_scale = 1.0
    best_score = -np.inf
    for scale in grid:
        score = rel_score(actual, prediction * float(scale))
        if score > best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale, best_score


def candidate_predictions(frame: pd.DataFrame, params: dict[str, float]) -> dict[str, np.ndarray]:
    base = frame["pred_base"].to_numpy(dtype=float)
    weighted = frame["pred_weighted"].to_numpy(dtype=float)
    instance = frame["pred_instance"].to_numpy(dtype=float)
    out = {
        "base": base,
        "weighted": weighted,
        "instance": instance,
        "weighted_scale": weighted * params["weighted_scale"],
        "base_weighted_ensemble": params["bw_base"] * base + params["bw_weighted"] * weighted,
        "simplex_ensemble": (
            params["simplex_base"] * base
            + params["simplex_weighted"] * weighted
            + params["simplex_instance"] * instance
        ),
        "bw_tail_penalty_0p25": params["bw_tail_0p25_base"] * base + params["bw_tail_0p25_weighted"] * weighted,
        "simplex_tail_penalty_0p25": (
            params["simplex_tail_0p25_base"] * base
            + params["simplex_tail_0p25_weighted"] * weighted
            + params["simplex_tail_0p25_instance"] * instance
        ),
        "simplex_tail_penalty_0p50": (
            params["simplex_tail_0p50_base"] * base
            + params["simplex_tail_0p50_weighted"] * weighted
            + params["simplex_tail_0p50_instance"] * instance
        ),
    }
    switch_threshold = params["switch_threshold"]
    out["weighted_instance_tail_switch"] = np.where(np.abs(weighted) >= switch_threshold, instance, weighted)
    return out


def fit_best_bw_tail(train: pd.DataFrame, grid: np.ndarray, tail_penalty: float) -> tuple[float, float]:
    base = train["pred_base"].to_numpy(dtype=float)
    weighted = train["pred_weighted"].to_numpy(dtype=float)
    best = (1.0, 0.0, np.inf)
    for w_weighted in grid:
        w_base = 1.0 - float(w_weighted)
        pred = w_base * base + float(w_weighted) * weighted
        score = daily_q90_penalized_loss(train, pred, tail_penalty)
        if score < best[2]:
            best = (w_base, float(w_weighted), score)
    return best[0], best[1]


def fit_best_simplex_tail(train: pd.DataFrame, grid: np.ndarray, tail_penalty: float) -> tuple[float, float, float]:
    base = train["pred_base"].to_numpy(dtype=float)
    weighted = train["pred_weighted"].to_numpy(dtype=float)
    instance = train["pred_instance"].to_numpy(dtype=float)
    best = (1.0, 0.0, 0.0, np.inf)
    for w_base in grid:
        for w_weighted in grid:
            w_instance = 1.0 - float(w_base) - float(w_weighted)
            if w_instance < -1e-9:
                continue
            pred = float(w_base) * base + float(w_weighted) * weighted + float(w_instance) * instance
            score = daily_q90_penalized_loss(train, pred, tail_penalty)
            if score < best[3]:
                best = (float(w_base), float(w_weighted), float(w_instance), score)
    return best[0], best[1], best[2]


def fit_calibration_params(train: pd.DataFrame, grid_step: float, shrink_max: float) -> dict[str, float]:
    actual = train["actual"].to_numpy(dtype=float)
    base = train["pred_base"].to_numpy(dtype=float)
    weighted = train["pred_weighted"].to_numpy(dtype=float)
    instance = train["pred_instance"].to_numpy(dtype=float)
    scale_grid = np.arange(0.0, shrink_max + 1e-9, grid_step)
    weighted_scale, _ = fit_best_scale(actual, weighted, scale_grid)

    best_bw = (1.0, 0.0, -np.inf)
    weight_grid = np.arange(0.0, 1.0 + 1e-9, grid_step)
    for w_weighted in weight_grid:
        w_base = 1.0 - float(w_weighted)
        pred = w_base * base + float(w_weighted) * weighted
        score = rel_score(actual, pred)
        if score > best_bw[2]:
            best_bw = (w_base, float(w_weighted), score)

    best_simplex = (1.0, 0.0, 0.0, -np.inf)
    simplex_grid = np.arange(0.0, 1.0 + 1e-9, max(grid_step, 0.10))
    for w_base in simplex_grid:
        for w_weighted in simplex_grid:
            w_instance = 1.0 - float(w_base) - float(w_weighted)
            if w_instance < -1e-9:
                continue
            pred = float(w_base) * base + float(w_weighted) * weighted + float(w_instance) * instance
            score = rel_score(actual, pred)
            if score > best_simplex[3]:
                best_simplex = (float(w_base), float(w_weighted), float(w_instance), score)

    best_switch = (float(np.quantile(np.abs(weighted), 0.90)), -np.inf)
    for quantile in (0.70, 0.80, 0.90, 0.95):
        threshold = float(np.quantile(np.abs(weighted), quantile))
        pred = np.where(np.abs(weighted) >= threshold, instance, weighted)
        score = rel_score(actual, pred)
        if score > best_switch[1]:
            best_switch = (threshold, score)
    bw_tail_0p25 = fit_best_bw_tail(train, weight_grid, tail_penalty=0.25)
    simplex_tail_0p25 = fit_best_simplex_tail(train, simplex_grid, tail_penalty=0.25)
    simplex_tail_0p50 = fit_best_simplex_tail(train, simplex_grid, tail_penalty=0.50)

    return {
        "weighted_scale": weighted_scale,
        "bw_base": best_bw[0],
        "bw_weighted": best_bw[1],
        "simplex_base": best_simplex[0],
        "simplex_weighted": best_simplex[1],
        "simplex_instance": best_simplex[2],
        "switch_threshold": best_switch[0],
        "bw_tail_0p25_base": bw_tail_0p25[0],
        "bw_tail_0p25_weighted": bw_tail_0p25[1],
        "simplex_tail_0p25_base": simplex_tail_0p25[0],
        "simplex_tail_0p25_weighted": simplex_tail_0p25[1],
        "simplex_tail_0p25_instance": simplex_tail_0p25[2],
        "simplex_tail_0p50_base": simplex_tail_0p50[0],
        "simplex_tail_0p50_weighted": simplex_tail_0p50[1],
        "simplex_tail_0p50_instance": simplex_tail_0p50[2],
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame.loc[:, columns].to_markdown(index=False)


def write_report(output_dir: Path, by_seed: pd.DataFrame, aggregate: pd.DataFrame, params: pd.DataFrame) -> None:
    display_seed = by_seed[by_seed["split"].eq("val")].copy()
    display_seed = display_seed.sort_values(["seed", "rel_score"], ascending=[True, False])
    display_agg = aggregate.copy().sort_values("rel_score_mean", ascending=False)
    for frame in (display_seed, display_agg):
        for col in frame.columns:
            if col.endswith("error") or col.endswith("q90") or col in {
                "q90_abs_error",
                "daily_q90_abs_error_q90",
                "daily_q90_abs_error_max",
                "q90_abs_error_mean",
                "daily_q90_abs_error_q90_mean",
                "daily_q90_abs_error_max_mean",
            }:
                frame[col] = frame[col].map(lambda x: f"{100 * float(x):.3f}%" if pd.notna(x) else "n/a")
            elif col in {"rel_score", "rel_score_mean", "rel_score_std", "prediction_actual_abs_q90_ratio", "prediction_actual_abs_q90_ratio_mean"}:
                frame[col] = frame[col].map(lambda x: f"{float(x):.5f}" if pd.notna(x) else "n/a")
    lines = [
        "# LSTM Calibration Multiseed Readout",
        "",
        "Calibration is fitted on the train split and evaluated on validation. Holdout/test is not used.",
        "",
        "## Validation By Seed",
        "",
        markdown_table(
            display_seed,
            [
                "seed",
                "candidate",
                "rel_score",
                "q90_abs_error",
                "daily_q90_abs_error_q90",
                "daily_q90_abs_error_max",
                "spike_days_ge_7pct",
                "spike_days_ge_8pct",
                "prediction_actual_abs_q90_ratio",
            ],
        ),
        "",
        "## Validation Aggregate",
        "",
        markdown_table(
            display_agg,
            [
                "candidate",
                "rel_score_mean",
                "rel_score_std",
                "q90_abs_error_mean",
                "daily_q90_abs_error_q90_mean",
                "daily_q90_abs_error_max_mean",
                "spike_days_ge_7pct_mean",
                "spike_days_ge_8pct_mean",
                "prediction_actual_abs_q90_ratio_mean",
            ],
        ),
        "",
        "## Fitted Parameters",
        "",
        params.to_markdown(index=False),
        "",
        "## Read",
        "",
        "- A calibrated/ensemble prediction is useful only if it improves `rel_score` without increasing daily tail spikes.",
        "- If the best train-fitted calibration is close to `weighted`, the model improvement comes from mild imbalance weighting.",
        "- If the best train-fitted calibration adds `instance`, the spike-control signal is useful but should be promoted carefully because it can under-amplify returns.",
    ]
    (output_dir / "calibration_readout.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = parse_ints(args.seeds)
    output_dir = args.gold_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    for seed in seeds:
        frame = read_seed_predictions(args.run_root, seed)
        train = frame[frame["split"].eq("train")].copy()
        params = fit_calibration_params(train, args.grid_step, args.shrink_max)
        param_rows.append({"seed": seed, **params})
        for split, split_frame in frame.groupby("split", sort=True):
            candidates = candidate_predictions(split_frame, params)
            for name, prediction in candidates.items():
                all_rows.append(summarize_prediction(split_frame, prediction, name, seed, str(split)))
    by_seed = pd.DataFrame(all_rows)
    val = by_seed[by_seed["split"].eq("val")].copy()
    aggregate = (
        val.groupby("candidate", sort=True)
        .agg(
            rel_score_mean=("rel_score", "mean"),
            rel_score_std=("rel_score", "std"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_abs_error_q90_mean=("daily_q90_abs_error_q90", "mean"),
            daily_q90_abs_error_max_mean=("daily_q90_abs_error_max", "mean"),
            spike_days_ge_7pct_mean=("spike_days_ge_7pct", "mean"),
            spike_days_ge_8pct_mean=("spike_days_ge_8pct", "mean"),
            prediction_actual_abs_q90_ratio_mean=("prediction_actual_abs_q90_ratio", "mean"),
        )
        .reset_index()
    )
    params_df = pd.DataFrame(param_rows)
    by_seed.to_csv(output_dir / "calibration_by_seed.csv", index=False)
    aggregate.to_csv(output_dir / "calibration_aggregate.csv", index=False)
    params_df.to_csv(output_dir / "calibration_params.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "seeds": seeds,
                "grid_step": args.grid_step,
                "shrink_max": args.shrink_max,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir, by_seed, aggregate, params_df)
    print((output_dir / "calibration_readout.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
