from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "residual_target_probe_20260519"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "residual_market_component_processing_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "residual_market_component_processing_20260520"


MARKET_FEATURES = (
    "mkt_ret_lag1",
    "mkt_ret_lag2",
    "mkt_ret_lag3",
    "mkt_ret_ewm3_lag1",
    "mkt_ret_ewm5_lag1",
    "mkt_abs_q90_lag1",
    "mkt_abs_q90_ewm5_lag1",
    "mkt_neg_ratio_lag1",
    "mkt_neg_ratio_ewm5_lag1",
    "mkt_dispersion_lag1",
    "mkt_dispersion_ewm5_lag1",
    "mkt_breadth_lag1",
    "mkt_breadth_ewm5_lag1",
)


@dataclass(frozen=True)
class Policy:
    name: str
    prediction: pd.Series
    params: dict[str, float | str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual target reconstruction using lagged market components.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--target-error", type=float, default=0.035)
    return parser.parse_args(argv)


def parse_seeds(value: str) -> list[int]:
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


def safe_quantile(values: pd.Series | np.ndarray, quantile: float) -> float:
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


def build_market_frame(data_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust", "target_next_return"], parse_dates=["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw.sort_values(["code", "Date"], kind="stable")
    raw["ret_today"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = (
        raw.groupby("Date", sort=True)
        .agg(
            market_oracle=("target_next_return", "mean"),
            mkt_ret_today=("ret_today", "mean"),
            mkt_abs_q90=("ret_today", lambda values: safe_quantile(np.abs(values), 0.90)),
            mkt_neg_ratio=("ret_today", lambda values: float(np.nanmean(np.asarray(values, dtype=float) < 0.0))),
            mkt_dispersion=("ret_today", safe_dispersion_q90),
            mkt_breadth=("ret_today", lambda values: float(np.nanmean(np.asarray(values, dtype=float) > 0.0))),
        )
        .reset_index()
        .sort_values("Date", kind="stable")
    )
    daily["mkt_ret_lag1"] = daily["mkt_ret_today"].shift(1)
    daily["mkt_ret_lag2"] = daily["mkt_ret_today"].shift(2)
    daily["mkt_ret_lag3"] = daily["mkt_ret_today"].shift(3)
    daily["mkt_ret_ewm3_lag1"] = daily["mkt_ret_today"].shift(1).ewm(span=3, adjust=False).mean()
    daily["mkt_ret_ewm5_lag1"] = daily["mkt_ret_today"].shift(1).ewm(span=5, adjust=False).mean()
    for column in ("mkt_abs_q90", "mkt_neg_ratio", "mkt_dispersion", "mkt_breadth"):
        daily[f"{column}_lag1"] = daily[column].shift(1)
        daily[f"{column}_ewm5_lag1"] = daily[column].shift(1).ewm(span=5, adjust=False).mean()
    return daily


def split_fit_calibration(train: pd.DataFrame, calibration_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(pd.to_datetime(sorted(train["Date"].dropna().unique())))
    cutoff_idx = max(1, int(len(dates) * (1.0 - calibration_fraction)))
    cutoff = dates.iloc[cutoff_idx - 1]
    return train[train["Date"].le(cutoff)].copy(), train[train["Date"].gt(cutoff)].copy()


def fit_market_models(fit_daily: pd.DataFrame) -> dict[str, Pipeline]:
    x = fit_daily.loc[:, MARKET_FEATURES]
    y = fit_daily["market_oracle"].astype(float)
    models: dict[str, Pipeline] = {}
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        models[f"ridge_a{int(alpha)}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        ).fit(x, y)
    return models


def add_daily_market_predictions(daily: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
    out = daily.copy()
    out["zero"] = 0.0
    out["lag1"] = out["mkt_ret_lag1"].fillna(0.0)
    out["ewm3"] = out["mkt_ret_ewm3_lag1"].fillna(0.0)
    out["ewm5"] = out["mkt_ret_ewm5_lag1"].fillna(0.0)
    for name, model in models.items():
        out[name] = model.predict(out.loc[:, MARKET_FEATURES])
    return out


def prediction_columns(daily: pd.DataFrame) -> list[str]:
    return [column for column in daily.columns if column in {"zero", "lag1", "ewm3", "ewm5"} or column.startswith("ridge_")]


def daily_q90(frame: pd.DataFrame, pred_col: str, min_daily_n: int) -> pd.Series:
    work = frame.loc[:, ["Date", "code", "actual", pred_col]].copy()
    work["abs_error"] = (work["actual"].astype(float) - work[pred_col].astype(float)).abs()
    counts = work.groupby("Date", sort=True)["code"].nunique()
    daily = work.groupby("Date", sort=True)["abs_error"].quantile(0.90)
    return daily[counts >= min_daily_n]


def summarize(frame: pd.DataFrame, pred_col: str, *, seed: int, policy: str, min_daily_n: int) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    daily = daily_q90(frame, pred_col, min_daily_n)
    return {
        "seed": seed,
        "policy": policy,
        "n_obs": int(len(frame)),
        "n_days": int(frame["Date"].nunique()),
        "rel_score": rel_score(actual, pred),
        "q90_abs_error": float(np.quantile(np.abs(actual - pred), 0.90)),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_p90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_max": float(daily.max()) if not daily.empty else float("nan"),
        "days_ge_3p5": int(daily.ge(0.035).sum()) if not daily.empty else 0,
        "days_ge_5": int(daily.ge(0.050).sum()) if not daily.empty else 0,
        "days_ge_7": int(daily.ge(0.070).sum()) if not daily.empty else 0,
        "days_ge_8": int(daily.ge(0.080).sum()) if not daily.empty else 0,
        "prediction_abs_q90": float(np.quantile(np.abs(pred), 0.90)),
        "actual_abs_q90": float(np.quantile(np.abs(actual), 0.90)),
    }


def objective(row: dict[str, object], target_error: float) -> float:
    return (
        float(row["rel_score"])
        - 2.0 * max(0.0, float(row["daily_q90_p90"]) - target_error)
        - 1.0 * max(0.0, float(row["daily_q90_max"]) - 0.060)
        - 0.003 * float(row["days_ge_5"])
        - 0.006 * float(row["days_ge_7"])
    )


def load_residual_predictions(source_dir: Path, seed: int) -> pd.DataFrame:
    path = source_dir / f"predictions_residual_lagged_ar1_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    return frame


def evaluate_seed(seed: int, args: argparse.Namespace, market_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = load_residual_predictions(args.source_dir, seed)
    fit_rows = residual[residual["split"].eq("train")].loc[:, ["Date"]].drop_duplicates()
    fit_daily_all = fit_rows.merge(market_daily, on="Date", how="left")
    fit_part, calibration_daily = split_fit_calibration(fit_daily_all, args.calibration_fraction)
    models = fit_market_models(fit_part)
    scored_daily = add_daily_market_predictions(market_daily, models)

    frame = residual.merge(scored_daily, on="Date", how="left")
    market_cols = prediction_columns(scored_daily)
    policy_rows: list[dict[str, object]] = []
    policy_specs: list[tuple[str, str, float]] = []
    for market_col in market_cols:
        for scale in (-0.25, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
            policy = f"{market_col}_scale_{scale:g}"
            frame[policy] = frame["prediction_target_space"].astype(float) + scale * frame[market_col].astype(float).fillna(0.0)
            policy_specs.append((policy, market_col, scale))

    calibration = frame[frame["Date"].isin(set(calibration_daily["Date"]))].copy()
    best_policy = "zero_scale_0"
    best_score = -np.inf
    for policy, market_col, scale in policy_specs:
        row = summarize(calibration, policy, seed=seed, policy=policy, min_daily_n=args.min_daily_n)
        row["market_column"] = market_col
        row["scale"] = scale
        row["selection_split"] = "late_train"
        row["objective"] = objective(row, args.target_error)
        policy_rows.append(row)
        if float(row["objective"]) > best_score:
            best_score = float(row["objective"])
            best_policy = policy

    val = frame[frame["split"].eq("val")].copy()
    selected = summarize(val, best_policy, seed=seed, policy=f"selected:{best_policy}", min_daily_n=args.min_daily_n)
    selected["selection_split"] = "val"
    policy_rows.append(selected)

    baseline = summarize(val, "prediction_raw", seed=seed, policy="existing_residual_lagged_ar1", min_daily_n=args.min_daily_n)
    baseline["selection_split"] = "val"
    policy_rows.append(baseline)

    raw = pd.read_csv(args.source_dir / f"predictions_raw_baseline_seed_{seed}.csv", parse_dates=["Date"])
    raw_val = raw[raw["split"].eq("val")].copy()
    raw_baseline = summarize(raw_val, "prediction_raw", seed=seed, policy="raw_baseline", min_daily_n=args.min_daily_n)
    raw_baseline["selection_split"] = "val"
    policy_rows.append(raw_baseline)

    val_keep = val.loc[:, ["seed", "code", "Date", "actual", "prediction_target_space", "prediction_raw", best_policy]].copy()
    val_keep = val_keep.rename(columns={best_policy: "prediction_selected"})
    val_keep["selected_policy"] = best_policy
    return pd.DataFrame(policy_rows), val_keep


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    val = rows[rows["selection_split"].eq("val")].copy()
    return (
        val.groupby("policy", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            rel_score_mean=("rel_score", "mean"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_median_mean=("daily_q90_median", "mean"),
            daily_q90_p90_mean=("daily_q90_p90", "mean"),
            daily_q90_max_mean=("daily_q90_max", "mean"),
            days_ge_3p5_mean=("days_ge_3p5", "mean"),
            days_ge_5_mean=("days_ge_5", "mean"),
            days_ge_7_mean=("days_ge_7", "mean"),
            days_ge_8_mean=("days_ge_8", "mean"),
            prediction_abs_q90_mean=("prediction_abs_q90", "mean"),
            actual_abs_q90_mean=("actual_abs_q90", "mean"),
        )
        .reset_index()
    )


def aggregate_modes(rows: pd.DataFrame) -> pd.DataFrame:
    val = rows[rows["selection_split"].eq("val")].copy()
    val["mode"] = val["policy"].map(lambda value: "selected_lagged_market" if str(value).startswith("selected:") else str(value))
    return (
        val.groupby("mode", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            policies=("policy", lambda values: ", ".join(sorted(set(map(str, values))))),
            rel_score_mean=("rel_score", "mean"),
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


def write_frontier(gold_dir: Path, agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(100 * agg["daily_q90_p90_mean"], 100 * agg["daily_q90_max_mean"], s=85, alpha=0.85)
    for _, row in agg.iterrows():
        ax.annotate(str(row["policy"]), (100 * row["daily_q90_p90_mean"], 100 * row["daily_q90_max_mean"]), fontsize=8)
    ax.axvline(3.5, color="#dc2626", linestyle="--", linewidth=1.0)
    ax.axhline(5.0, color="#f97316", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Daily q90(|E|) p90 (%)")
    ax.set_ylabel("Daily q90(|E|) max (%)")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(gold_dir / "residual_market_component_frontier.png", dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def write_summary(gold_dir: Path, agg: pd.DataFrame, args: argparse.Namespace) -> None:
    display = agg.sort_values(["daily_q90_p90_mean", "daily_q90_max_mean"], ascending=[True, True]).copy()
    lines = [
        "# Residual Market Component Processing",
        "",
        "Scope: reuse residual-target LSTM predictions, tune market reconstruction component on late-train, evaluate on validation.",
        "",
        "| policy | seeds | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=5% | days >=7% | days >=8% |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| `{row.policy}` | {int(row.seeds)} | {float(row.rel_score_mean):.5f} | {pct(row.q90_abs_error_mean)} | "
            f"{pct(row.daily_q90_median_mean)} | {pct(row.daily_q90_p90_mean)} | {pct(row.daily_q90_max_mean)} | "
            f"{float(row.days_ge_5_mean):.1f} | {float(row.days_ge_7_mean):.1f} | {float(row.days_ge_8_mean):.1f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- If selected lagged-market reconstruction does not beat `raw_baseline`, residual target is only useful with an oracle or a much stronger market nowcast.",
        "- If `scale_0` wins, the best practical use is alpha/residual-only target, not adding noisy market prediction back.",
        "- This is target processing, not a new architecture: it tests whether the target decomposition is operationally useful without future market information.",
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_mode_summary(gold_dir: Path, mode_agg: pd.DataFrame) -> None:
    display = mode_agg.sort_values(["daily_q90_p90_mean", "daily_q90_max_mean"], ascending=[True, True]).copy()
    lines = [
        "# Residual Market Component Mode Summary",
        "",
        "| mode | policies | seeds | rel_score | daily p90 | daily max | days >=7% | days >=8% |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| `{row['mode']}` | `{row.policies}` | {int(row.seeds)} | {float(row.rel_score_mean):.5f} | "
            f"{pct(row.daily_q90_p90_mean)} | {pct(row.daily_q90_max_mean)} | "
            f"{float(row.days_ge_7_mean):.1f} | {float(row.days_ge_8_mean):.1f} |"
        )
    (gold_dir / "mode_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    market_daily = build_market_frame(args.data)
    row_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for seed in seeds:
        rows, predictions = evaluate_seed(seed, args, market_daily)
        row_parts.append(rows)
        prediction_parts.append(predictions)
    rows = pd.concat(row_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    agg = aggregate(rows)
    mode_agg = aggregate_modes(rows)
    rows.to_csv(args.output_dir / "residual_market_component_by_seed.csv", index=False)
    predictions.to_csv(args.output_dir / "residual_market_component_val_predictions.csv", index=False)
    agg.to_csv(args.output_dir / "residual_market_component_aggregate.csv", index=False)
    mode_agg.to_csv(args.output_dir / "residual_market_component_mode_aggregate.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir),
                "data": str(args.data),
                "seeds": seeds,
                "market_features": list(MARKET_FEATURES),
                "calibration_fraction": args.calibration_fraction,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in [
        "residual_market_component_by_seed.csv",
        "residual_market_component_aggregate.csv",
        "residual_market_component_mode_aggregate.csv",
        "manifest.json",
    ]:
        (args.gold_dir / name).write_bytes((args.output_dir / name).read_bytes())
    write_frontier(args.gold_dir, agg)
    write_summary(args.gold_dir, agg, args)
    write_mode_summary(args.gold_dir, mode_agg)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
