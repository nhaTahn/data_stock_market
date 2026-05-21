from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analysis.evaluate_selective_error_control import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_REPORT_ROOT,
    RAW_FEATURES,
    RISK_FEATURES,
    add_risk_features,
    daily_q90_for_selection,
    load_predictions,
    parse_seeds,
    rel_score,
    split_train_calibration,
)


DEFAULT_OUTPUT = DEFAULT_REPORT_ROOT / "distributional_lstm_calibration_20260520"
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "distributional_lstm_calibration_20260520"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"

MODEL_FEATURES = (
    "prediction",
    "pred_stressaux",
    "pred_tail_loss",
    "pred_tailstress",
    "pred_base",
    "abs_prediction",
    "prediction_disagreement",
    "tailstress_gap",
    "base_gap",
    "input_noise_score",
    "tail5_probability",
    "tail7_probability",
    "past_return_1",
    "past_abs_return_1",
    "past_return_5",
    "past_return_20",
    "market_return_1",
    "market_abs_return_q90",
    "market_dispersion_q90",
    "market_volume_ratio_q90",
    *RAW_FEATURES,
    *RISK_FEATURES,
)


def safe_nanquantile(values: pd.Series | np.ndarray, quantile: float) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    return float(np.quantile(clean, quantile))


def safe_dispersion_q90(values: pd.Series | np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean - np.median(clean)), 0.90))


@dataclass(frozen=True)
class Candidate:
    name: str
    pred_calibration: np.ndarray
    pred_validation: np.ndarray
    params: dict[str, float | str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate distributional, conformal, and volatility-aware calibration on frozen LSTM forecasts.",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--target-error", type=float, default=0.035)
    return parser.parse_args(argv)


def load_extended_raw_features(data_path: Path) -> pd.DataFrame:
    available = pd.read_csv(data_path, nrows=0).columns
    usecols = [
        "Date",
        "code",
        "adjust",
        *[column for column in RAW_FEATURES if column in available],
    ]
    raw = pd.read_csv(data_path, usecols=usecols, parse_dates=["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw.sort_values(["code", "Date"], kind="stable")
    raw["past_return_1"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    raw["past_abs_return_1"] = raw["past_return_1"].abs()
    raw["past_return_5"] = raw.groupby("code", sort=False)["adjust"].pct_change(5)
    raw["past_return_20"] = raw.groupby("code", sort=False)["adjust"].pct_change(20)

    daily = (
        raw.groupby("Date", sort=True)
        .agg(
            market_return_1=("past_return_1", "mean"),
            market_abs_return_q90=("past_abs_return_1", lambda values: safe_nanquantile(values, 0.90)),
            market_dispersion_q90=("past_return_1", safe_dispersion_q90),
            market_volume_ratio_q90=("volume_ratio_20", lambda values: safe_nanquantile(values, 0.90)),
        )
        .reset_index()
    )
    raw = raw.merge(daily, on="Date", how="left")
    for column in RAW_FEATURES:
        if column not in raw.columns:
            raw[column] = np.nan
    return raw.drop(columns=["adjust"])


def available_features(frame: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    columns: list[str] = []
    for column in MODEL_FEATURES:
        if column in frame.columns and column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def hgb_regressor(seed: int, *, loss: str = "squared_error", quantile: float | None = None) -> HistGradientBoostingRegressor:
    kwargs: dict[str, object] = {
        "loss": loss,
        "max_iter": 180,
        "learning_rate": 0.04,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 35,
        "l2_regularization": 0.20,
        "random_state": seed,
    }
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def fit_quantile_regressor(x: pd.DataFrame, y: pd.Series, seed: int, quantile: float) -> HistGradientBoostingRegressor:
    try:
        model = hgb_regressor(seed, loss="quantile", quantile=quantile)
        return model.fit(x, y)
    except TypeError:
        model = hgb_regressor(seed, loss="squared_error")
        return model.fit(x, y)


def finite_prediction(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    return np.where(np.isfinite(out), out, fallback)


def qrank_from_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    clean = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if clean.size == 0:
        return np.full(len(values), 0.5, dtype=float)
    idx = np.searchsorted(clean, np.asarray(values, dtype=float), side="right")
    return np.clip(idx / clean.size, 0.0, 1.0)


def day_rescale_prediction(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    abs_return_qhat: np.ndarray,
    *,
    multiplier: float,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    work = frame.loc[:, ["Date"]].copy()
    work["base_abs"] = np.abs(base_prediction)
    work["abs_return_qhat"] = np.maximum(abs_return_qhat, 0.0)
    daily = (
        work.groupby("Date", sort=True)
        .agg(
            pred_abs_q90=("base_abs", lambda values: float(np.nanquantile(values, 0.90))),
            target_abs_q90=("abs_return_qhat", lambda values: float(np.nanquantile(values, 0.90))),
        )
        .reset_index()
    )
    daily["scale"] = multiplier * daily["target_abs_q90"] / daily["pred_abs_q90"].clip(lower=1e-4)
    daily["scale"] = daily["scale"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(min_scale, max_scale)
    scale_by_date = dict(zip(daily["Date"], daily["scale"]))
    scales = frame["Date"].map(scale_by_date).fillna(1.0).to_numpy(dtype=float)
    return np.asarray(base_prediction, dtype=float) * scales


def summarize_prediction(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    seed: int,
    variant: str,
    min_daily_n: int,
) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    pred = finite_prediction(prediction, frame["prediction"].to_numpy(dtype=float))
    work = frame.loc[:, ["Date"]].copy()
    work["abs_error"] = np.abs(actual - pred)
    daily = daily_q90_for_selection(work, min_daily_n)
    full_days = max(frame["Date"].nunique(), 1)
    return {
        "seed": seed,
        "variant": variant,
        "n_obs": int(len(frame)),
        "n_days": int(frame["Date"].nunique()),
        "day_coverage": float(frame["Date"].nunique() / full_days),
        "rel_score": rel_score(actual, pred),
        "median_abs_error": float(np.quantile(np.abs(actual - pred), 0.50)),
        "q90_abs_error": float(np.quantile(np.abs(actual - pred), 0.90)),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_p90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_max": float(daily.max()) if not daily.empty else float("nan"),
        "days_ge_3p5": int(daily.ge(0.035).sum()) if not daily.empty else 0,
        "days_ge_5": int(daily.ge(0.050).sum()) if not daily.empty else 0,
        "days_ge_7": int(daily.ge(0.070).sum()) if not daily.empty else 0,
        "days_ge_8": int(daily.ge(0.080).sum()) if not daily.empty else 0,
    }


def calibration_objective(row: dict[str, object], target_error: float) -> float:
    rel = float(row["rel_score"])
    daily_p90 = float(row["daily_q90_p90"])
    daily_max = float(row["daily_q90_max"])
    n_days = max(float(row["n_days"]), 1.0)
    days_ge_5 = float(row["days_ge_5"]) / n_days
    days_ge_7 = float(row["days_ge_7"]) / n_days
    return (
        rel
        - 3.0 * max(0.0, daily_p90 - target_error)
        - 1.2 * max(0.0, daily_max - 0.060)
        - 0.08 * days_ge_5
        - 0.16 * days_ge_7
    )


def choose_grid_candidate(
    name: str,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    min_daily_n: int,
    target_error: float,
    grid_fn: Callable[[], list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]],
) -> tuple[Candidate, dict[str, object]]:
    best: tuple[float, Candidate, dict[str, object]] | None = None
    for params, pred_cal, pred_val in grid_fn():
        row = summarize_prediction(calibration, pred_cal, seed=seed, variant=name, min_daily_n=min_daily_n)
        objective = calibration_objective(row, target_error)
        candidate = Candidate(name=name, pred_calibration=pred_cal, pred_validation=pred_val, params=params)
        if best is None or objective > best[0]:
            best = (objective, candidate, row)
    if best is None:
        raise ValueError(f"No grid candidates generated for {name}.")
    objective, candidate, row = best
    selected_row = dict(row)
    selected_row["calibration_objective"] = objective
    selected_row.update({f"param_{key}": value for key, value in candidate.params.items()})
    return candidate, selected_row


def fit_seed_candidates(
    seed: int,
    args: argparse.Namespace,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_predictions(args.report_root, seed)
    frame = add_risk_features(frame, raw)
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("val")].copy()
    fit, calibration = split_train_calibration(train, args.calibration_fraction)

    features = available_features(frame)
    x_fit = fit.loc[:, features]
    y_fit = fit["actual"].astype(float)
    base_fit = fit["prediction"].astype(float)
    residual_fit = y_fit - base_fit
    abs_error_fit = residual_fit.abs()
    abs_actual_fit = y_fit.abs()

    ridge_residual = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    ).fit(x_fit, residual_fit)
    hgb_residual = hgb_regressor(seed + 101).fit(x_fit, residual_fit)
    hgb_return = hgb_regressor(seed + 202).fit(x_fit, y_fit)
    q90_error = fit_quantile_regressor(x_fit, abs_error_fit, seed + 303, 0.90)
    q90_abs_return = fit_quantile_regressor(x_fit, abs_actual_fit, seed + 404, 0.90)

    x_cal = calibration.loc[:, features]
    x_val = validation.loc[:, features]
    base_cal = calibration["prediction"].to_numpy(dtype=float)
    base_val = validation["prediction"].to_numpy(dtype=float)
    actual_cal = calibration["actual"].to_numpy(dtype=float)

    ridge_resid_cal = finite_prediction(ridge_residual.predict(x_cal), np.zeros_like(base_cal))
    ridge_resid_val = finite_prediction(ridge_residual.predict(x_val), np.zeros_like(base_val))
    hgb_resid_cal = finite_prediction(hgb_residual.predict(x_cal), np.zeros_like(base_cal))
    hgb_resid_val = finite_prediction(hgb_residual.predict(x_val), np.zeros_like(base_val))
    hgb_return_cal = finite_prediction(hgb_return.predict(x_cal), base_cal)
    hgb_return_val = finite_prediction(hgb_return.predict(x_val), base_val)
    q90_error_cal_raw = np.maximum(finite_prediction(q90_error.predict(x_cal), np.abs(actual_cal - base_cal)), 1e-5)
    nonconformity = np.maximum(np.abs(actual_cal - base_cal) - q90_error_cal_raw, 0.0)
    conformal_add = float(np.quantile(nonconformity[np.isfinite(nonconformity)], 0.90)) if np.isfinite(nonconformity).any() else 0.0
    q90_error_cal = q90_error_cal_raw + conformal_add
    q90_error_val = np.maximum(finite_prediction(q90_error.predict(x_val), np.full_like(base_val, np.nanmedian(q90_error_cal_raw))), 1e-5) + conformal_add
    q90_abs_return_cal = np.maximum(finite_prediction(q90_abs_return.predict(x_cal), np.abs(base_cal)), 1e-5)
    q90_abs_return_val = np.maximum(finite_prediction(q90_abs_return.predict(x_val), np.abs(base_val)), 1e-5)
    q90_error_rank_cal = qrank_from_reference(q90_error_cal, q90_error_cal)
    q90_error_rank_val = qrank_from_reference(q90_error_cal, q90_error_val)

    candidates: list[Candidate] = [
        Candidate("baseline_stressaux_w20", base_cal, base_val, {"type": "base"}),
    ]
    selection_rows: list[dict[str, object]] = []

    def residual_grid(resid_cal: np.ndarray, resid_val: np.ndarray) -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        return [
            ({"alpha": alpha}, base_cal + alpha * resid_cal, base_val + alpha * resid_val)
            for alpha in (-0.25, 0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
        ]

    def return_blend_grid() -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        return [
            ({"alpha": alpha}, (1.0 - alpha) * base_cal + alpha * hgb_return_cal, (1.0 - alpha) * base_val + alpha * hgb_return_val)
            for alpha in (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
        ]

    def conformal_shrink_grid() -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        rows: list[tuple[dict[str, float | str], np.ndarray, np.ndarray]] = []
        for target in (0.025, 0.030, 0.035, 0.040, 0.050):
            for min_scale in (0.0, 0.25, 0.50, 0.75):
                scale_cal = np.clip(target / q90_error_cal, min_scale, 1.0)
                scale_val = np.clip(target / q90_error_val, min_scale, 1.0)
                rows.append(({"target": target, "min_scale": min_scale}, base_cal * scale_cal, base_val * scale_val))
        return rows

    def uncertainty_damped_residual_grid() -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        rows: list[tuple[dict[str, float | str], np.ndarray, np.ndarray]] = []
        for alpha in (0.15, 0.30, 0.50, 0.75, 1.0):
            for damping in (0.50, 0.75, 1.0):
                weight_cal = alpha * np.power(1.0 - q90_error_rank_cal, damping)
                weight_val = alpha * np.power(1.0 - q90_error_rank_val, damping)
                rows.append(
                    (
                        {"alpha": alpha, "damping": damping},
                        base_cal + weight_cal * hgb_resid_cal,
                        base_val + weight_val * hgb_resid_val,
                    )
                )
        return rows

    def ridge_conformal_grid() -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        rows: list[tuple[dict[str, float | str], np.ndarray, np.ndarray]] = []
        for alpha in (0.15, 0.30, 0.50, 0.75, 1.0):
            raw_cal = base_cal + alpha * ridge_resid_cal
            raw_val = base_val + alpha * ridge_resid_val
            for target in (0.025, 0.030, 0.035, 0.040, 0.050):
                for min_scale in (0.0, 0.25, 0.50, 0.75):
                    scale_cal = np.clip(target / q90_error_cal, min_scale, 1.0)
                    scale_val = np.clip(target / q90_error_val, min_scale, 1.0)
                    rows.append(
                        (
                            {"alpha": alpha, "target": target, "min_scale": min_scale},
                            raw_cal * scale_cal,
                            raw_val * scale_val,
                        )
                    )
        return rows

    def return_conformal_grid() -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        rows: list[tuple[dict[str, float | str], np.ndarray, np.ndarray]] = []
        for alpha in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
            raw_cal = (1.0 - alpha) * base_cal + alpha * hgb_return_cal
            raw_val = (1.0 - alpha) * base_val + alpha * hgb_return_val
            for target in (0.025, 0.030, 0.035, 0.040, 0.050):
                for min_scale in (0.0, 0.25, 0.50, 0.75):
                    scale_cal = np.clip(target / q90_error_cal, min_scale, 1.0)
                    scale_val = np.clip(target / q90_error_val, min_scale, 1.0)
                    rows.append(
                        (
                            {"alpha": alpha, "target": target, "min_scale": min_scale},
                            raw_cal * scale_cal,
                            raw_val * scale_val,
                        )
                    )
        return rows

    def day_vol_rescale_grid() -> list[tuple[dict[str, float | str], np.ndarray, np.ndarray]]:
        rows: list[tuple[dict[str, float | str], np.ndarray, np.ndarray]] = []
        for multiplier in (0.25, 0.35, 0.50, 0.75, 1.0):
            for min_scale in (0.50, 0.75, 1.0):
                for max_scale in (1.25, 1.50, 2.00, 2.50):
                    rows.append(
                        (
                            {"multiplier": multiplier, "min_scale": min_scale, "max_scale": max_scale},
                            day_rescale_prediction(calibration, base_cal, q90_abs_return_cal, multiplier=multiplier, min_scale=min_scale, max_scale=max_scale),
                            day_rescale_prediction(validation, base_val, q90_abs_return_val, multiplier=multiplier, min_scale=min_scale, max_scale=max_scale),
                        )
                    )
        return rows

    for name, grid_fn in [
        ("ridge_residual_blend", lambda: residual_grid(ridge_resid_cal, ridge_resid_val)),
        ("hgb_residual_blend", lambda: residual_grid(hgb_resid_cal, hgb_resid_val)),
        ("hgb_return_blend", return_blend_grid),
        ("conformal_q90_shrink", conformal_shrink_grid),
        ("uncertainty_damped_residual", uncertainty_damped_residual_grid),
        ("ridge_conformal_shrink", ridge_conformal_grid),
        ("hgb_return_conformal_shrink", return_conformal_grid),
        ("distributional_day_vol_rescale", day_vol_rescale_grid),
    ]:
        candidate, selected = choose_grid_candidate(
            name,
            calibration,
            validation,
            seed,
            args.min_daily_n,
            args.target_error,
            grid_fn,
        )
        candidates.append(candidate)
        selected["seed"] = seed
        selected["variant"] = name
        selection_rows.append(selected)

    result_rows: list[dict[str, object]] = []
    val_output = validation.loc[:, ["code", "Date", "split", "actual", "prediction"]].copy()
    val_output = val_output.rename(columns={"prediction": "pred_baseline_stressaux_w20"})
    val_output["seed"] = seed
    val_output["q90_error_conformal"] = q90_error_val
    val_output["q90_abs_return"] = q90_abs_return_val
    for candidate in candidates:
        result_rows.append(
            summarize_prediction(
                validation,
                candidate.pred_validation,
                seed=seed,
                variant=candidate.name,
                min_daily_n=args.min_daily_n,
            )
        )
        val_output[f"pred_{candidate.name}"] = candidate.pred_validation
    return pd.DataFrame(result_rows), pd.DataFrame(selection_rows), val_output


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("variant", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            rel_score_mean=("rel_score", "mean"),
            rel_score_std=("rel_score", "std"),
            median_abs_error_mean=("median_abs_error", "mean"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_median_mean=("daily_q90_median", "mean"),
            daily_q90_p90_mean=("daily_q90_p90", "mean"),
            daily_q90_max_mean=("daily_q90_max", "mean"),
            days_ge_3p5_mean=("days_ge_3p5", "mean"),
            days_ge_5_mean=("days_ge_5", "mean"),
            days_ge_7_mean=("days_ge_7", "mean"),
            days_ge_8_mean=("days_ge_8", "mean"),
        )
        .reset_index()
    )


def read_symbols(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def build_index_proxy(data_path: Path, symbols: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].copy().sort_values(["code", "Date"], kind="stable")
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = raw.groupby("Date", sort=True)["stock_return"].mean().rename("index_proxy_return").reset_index()
    daily["index_proxy"] = (1.0 + daily["index_proxy_return"].fillna(0.0)).cumprod()
    return daily


def rebase_to_100(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return clean / clean.dropna().iloc[0] * 100.0


def build_daily_variant_errors(predictions: pd.DataFrame, variant: str, min_daily_n: int) -> pd.DataFrame:
    pred_col = f"pred_{variant}"
    work = predictions.loc[:, ["seed", "Date", "code", "actual", pred_col]].copy()
    work["abs_error"] = (work["actual"].astype(float) - work[pred_col].astype(float)).abs()
    seed_daily = (
        work.groupby(["seed", "Date"], sort=True)
        .agg(
            n_stocks=("code", "nunique"),
            q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
        )
        .reset_index()
    )
    seed_daily = seed_daily[seed_daily["n_stocks"].ge(min_daily_n)].copy()
    return (
        seed_daily.groupby("Date", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            n_stocks=("n_stocks", "mean"),
            q90_abs_error=("q90_abs_error", "mean"),
        )
        .reset_index()
    )


def write_frontier_plot(gold_dir: Path, aggregate: pd.DataFrame) -> None:
    plot = aggregate.copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.scatter(100 * plot["daily_q90_p90_mean"], 100 * plot["daily_q90_max_mean"], s=80, alpha=0.85)
    for _, row in plot.iterrows():
        ax.annotate(str(row["variant"]), (100 * row["daily_q90_p90_mean"], 100 * row["daily_q90_max_mean"]), fontsize=8)
    ax.axvline(3.5, color="#dc2626", linestyle="--", linewidth=1.0, label="3.5% p90 target")
    ax.axhline(5.0, color="#f97316", linestyle="--", linewidth=1.0, label="5.0% max reference")
    ax.set_xlabel("Daily q90(|E|) p90 (%)")
    ax.set_ylabel("Daily q90(|E|) max (%)")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gold_dir / "distributional_calibration_frontier.png", dpi=180)
    plt.close(fig)


def write_teacher_style_plot(
    gold_dir: Path,
    predictions: pd.DataFrame,
    data_path: Path,
    best_variant: str,
    min_daily_n: int,
) -> None:
    symbols = read_symbols(DEFAULT_VN100_SYMBOLS)
    index = build_index_proxy(data_path, symbols)
    baseline = build_daily_variant_errors(predictions, "baseline_stressaux_w20", min_daily_n).rename(
        columns={"q90_abs_error": "baseline_q90_abs_error"}
    )
    best = build_daily_variant_errors(predictions, best_variant, min_daily_n).rename(
        columns={"q90_abs_error": "best_q90_abs_error"}
    )
    frame = (
        index.merge(baseline[["Date", "baseline_q90_abs_error"]], on="Date", how="inner")
        .merge(best[["Date", "best_q90_abs_error"]], on="Date", how="inner")
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    frame["index_proxy_rebased"] = rebase_to_100(frame["index_proxy"])
    frame.to_csv(gold_dir / "teacher_style_baseline_vs_best_abs_error.csv", index=False)

    x = np.arange(len(frame))
    fig, ax1 = plt.subplots(figsize=(12.5, 5.2))
    ax1.plot(x, frame["index_proxy_rebased"], color="#1f77b4", linewidth=1.5, label="VN100")
    ax1.set_ylabel("VN100, rebased to 100")
    ax1.grid(True, alpha=0.22)
    years = frame["Date"].dt.year.to_numpy()
    ticks: list[int] = []
    labels: list[str] = []
    last_year: int | None = None
    for idx, year in enumerate(years):
        if last_year != int(year):
            ticks.append(idx)
            labels.append(str(year))
            last_year = int(year)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(x, frame["baseline_q90_abs_error"], color="#ef4444", linestyle="--", linewidth=1.0, alpha=0.75, label="baseline q90(|E|)")
    ax2.plot(x, frame["best_q90_abs_error"], color="#111827", linestyle="-", linewidth=1.05, label=f"{best_variant} q90(|E|)")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("q90 absolute return error")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("VN100 vs q90(|actual return - predicted return|), validation")
    ax1.set_xlabel("Trading days in validation period")
    fig.tight_layout()
    fig.savefig(gold_dir / "teacher_style_baseline_vs_best_abs_error.png", dpi=180)
    plt.close(fig)

    work = frame.copy()
    work["year"] = work["Date"].dt.year
    years_unique = sorted(work["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years_unique) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 3.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax1, year in zip(axes, years_unique):
        part = work[work["year"].eq(year)].reset_index(drop=True)
        xx = np.arange(len(part))
        ax1.plot(xx, part["index_proxy_rebased"], color="#1f77b4", linewidth=1.15)
        ax1.set_title(str(year), loc="left", fontsize=10, fontweight="bold")
        ax1.grid(True, alpha=0.18)
        ax2 = ax1.twinx()
        ax2.plot(xx, part["baseline_q90_abs_error"], color="#ef4444", linestyle="--", linewidth=0.95, alpha=0.7)
        ax2.plot(xx, part["best_q90_abs_error"], color="#111827", linewidth=1.0)
        ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax1.set_xlabel("Trading day")
    for ax in axes[len(years_unique) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.4, label="VN100"),
        plt.Line2D([0], [0], color="#ef4444", lw=1.1, linestyle="--", label="baseline q90(|E|)"),
        plt.Line2D([0], [0], color="#111827", lw=1.2, label=f"{best_variant} q90(|E|)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3)
    fig.suptitle("VN100 vs q90 prediction error by year, validation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(gold_dir / "teacher_style_baseline_vs_best_abs_error_by_year.png", dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def choose_report_variant(aggregate: pd.DataFrame) -> str:
    base = aggregate[aggregate["variant"].eq("baseline_stressaux_w20")]
    if not base.empty:
        base_row = base.iloc[0]
        stable = aggregate[
            aggregate["rel_score_mean"].ge(0.0)
            & aggregate["daily_q90_max_mean"].lt(float(base_row["daily_q90_max_mean"]))
            & aggregate["daily_q90_p90_mean"].le(float(base_row["daily_q90_p90_mean"]) + 0.0005)
        ].copy()
        if not stable.empty:
            stable["rank_score"] = (
                stable["rel_score_mean"]
                - 2.0 * stable["daily_q90_p90_mean"]
                - 1.5 * stable["daily_q90_max_mean"]
                - 0.004 * stable["days_ge_5_mean"]
                - 0.006 * stable["days_ge_7_mean"]
            )
            return str(stable.sort_values("rank_score", ascending=False).iloc[0]["variant"])
    usable = aggregate[aggregate["rel_score_mean"].ge(0.0)].copy()
    if usable.empty:
        usable = aggregate.copy()
    usable["rank_score"] = (
        usable["rel_score_mean"]
        - 2.5 * usable["daily_q90_p90_mean"]
        - 1.0 * usable["daily_q90_max_mean"]
        - 0.003 * usable["days_ge_5_mean"]
    )
    return str(usable.sort_values("rank_score", ascending=False).iloc[0]["variant"])


def write_summary(gold_dir: Path, aggregate: pd.DataFrame, selections: pd.DataFrame, best_variant: str) -> None:
    display = aggregate.sort_values(["daily_q90_p90_mean", "daily_q90_max_mean"], ascending=[True, True]).copy()
    base = aggregate[aggregate["variant"].eq("baseline_stressaux_w20")].iloc[0]
    best = aggregate[aggregate["variant"].eq(best_variant)].iloc[0]
    best_rel = aggregate.sort_values("rel_score_mean", ascending=False).iloc[0]
    low_spike = aggregate[aggregate["rel_score_mean"].ge(0.0)].sort_values("daily_q90_max_mean", ascending=True).iloc[0]
    lines = [
        "# Distributional LSTM Calibration Readout",
        "",
        "Base forecast: `stressaux_w20`. Additional heads are trained only on early-train, tuned on late-train, and evaluated on validation.",
        "",
        "## Validation Frontier",
        "",
        "| variant | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=7% |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| `{row.variant}` | {float(row.rel_score_mean):.5f} | {pct(row.q90_abs_error_mean)} | "
            f"{pct(row.daily_q90_median_mean)} | {pct(row.daily_q90_p90_mean)} | {pct(row.daily_q90_max_mean)} | "
            f"{float(row.days_ge_3p5_mean):.1f} | {float(row.days_ge_5_mean):.1f} | {float(row.days_ge_7_mean):.1f} |"
        )
    lines += [
        "",
        "## Selected Read",
        "",
        f"- Baseline: rel_score `{float(base.rel_score_mean):.5f}`, daily p90 `{pct(base.daily_q90_p90_mean)}`, daily max `{pct(base.daily_q90_max_mean)}`.",
        f"- Best rel_score variant: `{best_rel.variant}` with rel_score `{float(best_rel.rel_score_mean):.5f}`, but daily max `{pct(best_rel.daily_q90_max_mean)}`.",
        f"- Lowest-spike positive-rel variant: `{low_spike.variant}` with rel_score `{float(low_spike.rel_score_mean):.5f}`, daily max `{pct(low_spike.daily_q90_max_mean)}`.",
        f"- Best report/stability variant: `{best_variant}` with rel_score `{float(best.rel_score_mean):.5f}`, daily p90 `{pct(best.daily_q90_p90_mean)}`, daily max `{pct(best.daily_q90_max_mean)}`.",
        "- If this improves daily p90 but leaves max spikes high, the literature-consistent interpretation is: uncertainty is partially learnable, but point-return direction remains the bottleneck on stress days.",
        "- If the best variant is a shrink/calibration method rather than residual correction, the next proper LSTM change should be a multi-head model: return head + tail/scale head, not a larger plain LSTM.",
        "",
        "## Tuned Parameters",
        "",
        selections.to_markdown(index=False),
        "",
        "Files:",
        "",
        "- `distributional_calibration_frontier.png`",
        "- `teacher_style_baseline_vs_best_abs_error.png`",
        "- `teacher_style_baseline_vs_best_abs_error_by_year.png`",
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    raw = load_extended_raw_features(args.data)

    result_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for seed in seeds:
        results, selections, predictions = fit_seed_candidates(seed, args, raw)
        result_parts.append(results)
        selection_parts.append(selections)
        prediction_parts.append(predictions)
        predictions.to_csv(args.output_dir / f"val_distributional_predictions_seed_{seed}.csv", index=False)

    results = pd.concat(result_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    aggregate = aggregate_results(results)
    best_variant = choose_report_variant(aggregate)

    results.to_csv(args.output_dir / "distributional_calibration_by_seed.csv", index=False)
    selections.to_csv(args.output_dir / "distributional_calibration_selected_params.csv", index=False)
    aggregate.to_csv(args.output_dir / "distributional_calibration_aggregate.csv", index=False)
    predictions.to_csv(args.output_dir / "val_distributional_predictions_all.csv", index=False)
    manifest = {
        "seeds": seeds,
        "calibration_fraction": args.calibration_fraction,
        "min_daily_n": args.min_daily_n,
        "target_error": args.target_error,
        "base_prediction": "stressaux_w20",
        "best_variant": best_variant,
        "features": available_features(add_risk_features(load_predictions(args.report_root, seeds[0]), raw)),
        "holdout_test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for file_name in [
        "distributional_calibration_by_seed.csv",
        "distributional_calibration_selected_params.csv",
        "distributional_calibration_aggregate.csv",
        "manifest.json",
    ]:
        (args.gold_dir / file_name).write_bytes((args.output_dir / file_name).read_bytes())
    write_frontier_plot(args.gold_dir, aggregate)
    write_teacher_style_plot(args.gold_dir, predictions, args.data, best_variant, args.min_daily_n)
    write_summary(args.gold_dir, aggregate, selections, best_variant)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
