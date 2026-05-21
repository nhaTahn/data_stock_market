from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs" / "reports"
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = DEFAULT_REPORT_ROOT / "selective_error_control_20260520"
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "selective_error_control_20260520"

RAW_FEATURES = (
    "volume_ratio_20",
    "intraday_return",
    "gap_open",
    "close_position",
    "momentum_20",
    "volatility_20",
    "bb_width",
    "macd_hist",
    "effort_result_ratio",
    "buying_pressure",
    "selling_pressure",
    "wyckoff_phase_60d",
    "rsi_14",
    "sector_momentum_rank",
    "alpha_sector",
)

RISK_FEATURES = (
    "abs_prediction",
    "prediction_disagreement",
    "tailstress_gap",
    "base_gap",
    "input_noise_score",
    "volatility_20",
    "bb_width",
    "volume_ratio_20",
    "abs_intraday_return",
    "abs_gap_open",
    "abs_momentum_20",
    "abs_macd_hist",
    "abs_alpha_sector",
    "sector_momentum_rank",
    "daily_abs_prediction_q90",
    "daily_disagreement_q90",
    "daily_input_noise_q90",
    "daily_volatility_q90",
    "daily_bb_width_q90",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated selective prediction/error-control on frozen VN LSTM predictions.",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--target-error", type=float, default=0.035)
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
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


def prediction_path(report_root: Path, seed: int, family: str, variant: str) -> Path:
    return report_root / family / f"seed_{seed}" / f"predictions_{variant}.csv"


def load_predictions(report_root: Path, seed: int) -> pd.DataFrame:
    specs = {
        "stressaux": ("stressaux_lstm_probe_20260519", "plain_global_weighted_mild_tail35_stressaux_w20"),
        "tail_loss": ("tail_loss_lstm_probe_20260519", "plain_global_weighted_mild_tail35_p05"),
        "tailstress": ("tailstress_lstm_probe_20260519", "plain_global_weighted_mild_tail35_p05_tailstress"),
        "base": ("tail_aware_lstm_multiseed_20260519", "plain_global_rel"),
    }
    merged: pd.DataFrame | None = None
    for name, (family, variant) in specs.items():
        path = prediction_path(report_root, seed, family, variant)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["Date"])
        keep = frame.loc[:, ["code", "Date", "split", "actual", "prediction"]].copy()
        keep = keep.rename(columns={"prediction": f"pred_{name}"})
        if merged is None:
            merged = keep
        else:
            merged = merged.merge(keep, on=["code", "Date", "split", "actual"], how="inner")
    if merged is None:
        raise ValueError(f"No predictions loaded for seed {seed}.")
    merged["prediction"] = merged["pred_stressaux"].astype(float)
    merged["abs_error"] = (merged["actual"].astype(float) - merged["prediction"]).abs()
    return merged


def load_raw_features(data_path: Path) -> pd.DataFrame:
    available = pd.read_csv(data_path, nrows=0).columns
    usecols = ["Date", "code", *[column for column in RAW_FEATURES if column in available]]
    raw = pd.read_csv(data_path, usecols=usecols, parse_dates=["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    for column in RAW_FEATURES:
        if column not in raw.columns:
            raw[column] = np.nan
    return raw


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).fillna(0.5)


def add_risk_features(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    work = frame.merge(raw, on=["code", "Date"], how="left")
    work["abs_prediction"] = work["prediction"].abs()
    work["prediction_disagreement"] = (
        work[["pred_stressaux", "pred_tail_loss", "pred_tailstress", "pred_base"]]
        .astype(float)
        .std(axis=1)
        .fillna(0.0)
    )
    work["tailstress_gap"] = (work["pred_tailstress"] - work["pred_stressaux"]).abs()
    work["base_gap"] = (work["pred_base"] - work["pred_stressaux"]).abs()
    work["abs_intraday_return"] = work["intraday_return"].abs()
    work["abs_gap_open"] = work["gap_open"].abs()
    work["abs_momentum_20"] = work["momentum_20"].abs()
    work["abs_macd_hist"] = work["macd_hist"].abs()
    work["abs_alpha_sector"] = work["alpha_sector"].abs()
    noise_parts = [
        percentile_rank(work["volatility_20"].astype(float)),
        percentile_rank(work["bb_width"].astype(float)),
        percentile_rank(work["volume_ratio_20"].astype(float)),
        percentile_rank(work["abs_intraday_return"].astype(float)),
        percentile_rank(work["abs_gap_open"].astype(float)),
        percentile_rank(work["abs_momentum_20"].astype(float)),
        percentile_rank(work["prediction_disagreement"].astype(float)),
    ]
    work["input_noise_score"] = pd.concat(noise_parts, axis=1).mean(axis=1)
    daily = (
        work.groupby(["split", "Date"], sort=True)
        .agg(
            daily_abs_prediction_q90=("abs_prediction", lambda values: float(np.nanquantile(values, 0.90))),
            daily_disagreement_q90=("prediction_disagreement", lambda values: float(np.nanquantile(values, 0.90))),
            daily_input_noise_q90=("input_noise_score", lambda values: float(np.nanquantile(values, 0.90))),
            daily_volatility_q90=("volatility_20", lambda values: float(np.nanquantile(values, 0.90))),
            daily_bb_width_q90=("bb_width", lambda values: float(np.nanquantile(values, 0.90))),
        )
        .reset_index()
    )
    work = work.merge(daily, on=["split", "Date"], how="left")
    return work.replace([np.inf, -np.inf], np.nan)


def split_train_calibration(train: pd.DataFrame, calibration_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(pd.to_datetime(sorted(train["Date"].dropna().unique())))
    if dates.empty:
        raise ValueError("No train dates available.")
    cutoff_idx = max(1, int(len(dates) * (1.0 - calibration_fraction)))
    cutoff = dates.iloc[cutoff_idx - 1]
    fit = train[train["Date"].le(cutoff)].copy()
    calibration = train[train["Date"].gt(cutoff)].copy()
    if calibration.empty:
        calibration = train.tail(max(1, len(train) // 4)).copy()
    return fit, calibration


def fit_risk_models(fit: pd.DataFrame, target_error: float) -> dict[str, Pipeline]:
    x = fit.loc[:, RISK_FEATURES]
    y = fit["abs_error"].ge(target_error).astype(int)
    models: dict[str, Pipeline] = {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.05,
                        l2_regularization=0.10,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    return {name: model.fit(x, y) for name, model in models.items()}


def add_model_scores(frame: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
    out = frame.copy()
    for name, model in models.items():
        out[f"risk_{name}"] = model.predict_proba(out.loc[:, RISK_FEATURES])[:, 1]
    out["risk_input_noise"] = out["input_noise_score"].astype(float)
    out["risk_disagreement"] = percentile_rank(out["prediction_disagreement"].astype(float)).to_numpy(dtype=float)
    out["risk_hybrid"] = (
        out[["risk_hgb", "risk_logistic", "risk_input_noise", "risk_disagreement"]].astype(float).rank(pct=True).mean(axis=1)
    )
    return out


def daily_q90_for_selection(frame: pd.DataFrame, min_daily_n: int) -> pd.Series:
    counts = frame.groupby("Date", sort=True)["abs_error"].count()
    daily = frame.groupby("Date", sort=True)["abs_error"].quantile(0.90)
    return daily[counts >= min_daily_n]


def summarize_selection(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    seed: int,
    score_name: str,
    policy: str,
    threshold: float,
    min_daily_n: int,
) -> dict[str, object]:
    selected_frame = frame.loc[selected].copy()
    full_days = max(frame["Date"].nunique(), 1)
    actual = selected_frame["actual"].to_numpy(dtype=float)
    pred = selected_frame["prediction"].to_numpy(dtype=float)
    daily = daily_q90_for_selection(selected_frame, min_daily_n)
    return {
        "seed": seed,
        "score": score_name,
        "policy": policy,
        "threshold": threshold,
        "n_obs": int(len(selected_frame)),
        "obs_coverage": float(len(selected_frame) / max(len(frame), 1)),
        "n_days": int(selected_frame["Date"].nunique()),
        "day_coverage": float(selected_frame["Date"].nunique() / full_days),
        "rel_score": rel_score(actual, pred) if len(selected_frame) else float("nan"),
        "q90_abs_error": float(np.quantile(np.abs(actual - pred), 0.90)) if len(selected_frame) else float("nan"),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_p90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_max": float(daily.max()) if not daily.empty else float("nan"),
        "days_ge_3p5": int(daily.ge(0.035).sum()) if not daily.empty else 0,
        "days_ge_5": int(daily.ge(0.05).sum()) if not daily.empty else 0,
        "days_ge_7": int(daily.ge(0.07).sum()) if not daily.empty else 0,
        "days_ge_8": int(daily.ge(0.08).sum()) if not daily.empty else 0,
    }


def summarize_abstain_zero(
    frame: pd.DataFrame,
    selected: pd.Series,
    *,
    seed: int,
    score_name: str,
    policy: str,
    threshold: float,
    min_daily_n: int,
) -> dict[str, object]:
    modified = frame.copy()
    modified["prediction"] = np.where(selected.to_numpy(dtype=bool), modified["prediction"].to_numpy(dtype=float), 0.0)
    modified["abs_error"] = (modified["actual"].astype(float) - modified["prediction"].astype(float)).abs()
    row = summarize_selection(
        modified,
        pd.Series(True, index=modified.index),
        seed=seed,
        score_name=score_name,
        policy=policy,
        threshold=threshold,
        min_daily_n=min_daily_n,
    )
    row["obs_coverage"] = float(selected.mean())
    row["day_coverage"] = 1.0
    return row


def choose_calibrated_threshold(
    calibration: pd.DataFrame,
    score_column: str,
    target_error: float,
    min_daily_n: int,
) -> tuple[float, str]:
    scores = calibration[score_column].dropna().to_numpy(dtype=float)
    if scores.size == 0:
        return float("nan"), "missing_score"
    quantiles = np.array([0.10, 0.15, 0.20, 0.25, *np.linspace(0.30, 1.00, 15)])
    best_threshold = float(np.quantile(scores, 0.10))
    best_policy = "lowest_risk_available"
    best_coverage = -np.inf
    for quantile in quantiles:
        threshold = float(np.quantile(scores, quantile))
        selected = calibration[score_column].le(threshold)
        daily = daily_q90_for_selection(calibration.loc[selected], min_daily_n)
        if daily.empty:
            continue
        daily_p90 = float(daily.quantile(0.90))
        coverage = float(selected.mean())
        if daily_p90 <= target_error and coverage > best_coverage:
            best_threshold = threshold
            best_policy = "calibrated_p90_3p5"
            best_coverage = coverage
    return best_threshold, best_policy


def evaluate_seed(seed: int, args: argparse.Namespace, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_predictions(args.report_root, seed)
    frame = add_risk_features(frame, raw)
    train = frame[frame["split"].eq("train")].copy()
    val = frame[frame["split"].eq("val")].copy()
    fit, calibration = split_train_calibration(train, args.calibration_fraction)
    models = fit_risk_models(fit, args.target_error)
    calibration = add_model_scores(calibration, models)
    val = add_model_scores(val, models)
    score_columns = ["risk_hgb", "risk_logistic", "risk_hybrid", "risk_input_noise", "risk_disagreement"]

    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for score_column in score_columns:
        threshold, policy_name = choose_calibrated_threshold(
            calibration,
            score_column,
            args.target_error,
            args.min_daily_n,
        )
        threshold_rows.append({"seed": seed, "score": score_column, "threshold": threshold, "policy": policy_name})
        for quantile in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
            q_threshold = float(np.quantile(calibration[score_column].dropna().to_numpy(dtype=float), quantile))
            selected = val[score_column].le(q_threshold)
            rows.append(
                summarize_selection(
                    val,
                    selected,
                    seed=seed,
                    score_name=score_column,
                    policy=f"coverage_q{int(quantile * 100)}",
                    threshold=q_threshold,
                    min_daily_n=args.min_daily_n,
                )
            )
            rows.append(
                summarize_abstain_zero(
                    val,
                    selected,
                    seed=seed,
                    score_name=score_column,
                    policy=f"abstain_zero_q{int(quantile * 100)}",
                    threshold=q_threshold,
                    min_daily_n=args.min_daily_n,
                )
            )
        calibrated_selected = val[score_column].le(threshold)
        rows.append(
            summarize_selection(
                val,
                calibrated_selected,
                seed=seed,
                score_name=score_column,
                policy=policy_name,
                threshold=threshold,
                min_daily_n=args.min_daily_n,
            )
        )
        rows.append(
            summarize_abstain_zero(
                val,
                calibrated_selected,
                seed=seed,
                score_name=score_column,
                policy=f"abstain_zero_{policy_name}",
                threshold=threshold,
                min_daily_n=args.min_daily_n,
            )
        )

    rows.append(
        summarize_selection(
            val,
            pd.Series(True, index=val.index),
            seed=seed,
            score_name="none",
            policy="full_coverage",
            threshold=float("inf"),
            min_daily_n=args.min_daily_n,
        )
    )
    scored_keep = val.loc[:, ["code", "Date", "split", "actual", "prediction", "abs_error", *score_columns, *RISK_FEATURES]].copy()
    scored_keep["seed"] = seed
    scored_keep.to_csv(args.output_dir / f"val_selective_scores_seed_{seed}.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["score", "policy"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            obs_coverage_mean=("obs_coverage", "mean"),
            day_coverage_mean=("day_coverage", "mean"),
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


def write_plot(gold_dir: Path, aggregate: pd.DataFrame) -> None:
    selected = aggregate[
        aggregate["policy"].isin(
            [
                "full_coverage",
                "calibrated_p90_3p5",
                "lowest_risk_available",
                "coverage_q50",
                "coverage_q70",
                "abstain_zero_calibrated_p90_3p5",
                "abstain_zero_q50",
                "abstain_zero_q70",
            ]
        )
    ].copy()
    selected["label"] = selected["score"] + " / " + selected["policy"]
    selected = selected.sort_values(["obs_coverage_mean", "daily_q90_p90_mean"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(100 * selected["obs_coverage_mean"], 100 * selected["daily_q90_p90_mean"], s=70, alpha=0.85)
    for _, row in selected.iterrows():
        ax.annotate(str(row["label"]), (100 * row["obs_coverage_mean"], 100 * row["daily_q90_p90_mean"]), fontsize=8)
    ax.axhline(3.5, color="#dc2626", linestyle="--", linewidth=1, label="3.5% target")
    ax.set_xlabel("Observation coverage (%)")
    ax.set_ylabel("Validation daily q90(|E|) p90 (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(gold_dir / "selective_error_frontier.png", dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100 * float(value):.2f}%"


def write_summary(gold_dir: Path, aggregate: pd.DataFrame, thresholds: pd.DataFrame, args: argparse.Namespace) -> None:
    display = aggregate.sort_values(["daily_q90_p90_mean", "obs_coverage_mean"], ascending=[True, False]).copy()
    lines = [
        "# Selective Error-Control Readout",
        "",
        "Base prediction: `stressaux_w20`. Risk scores are fitted on early train, calibrated on late train, evaluated on validation.",
        f"Target: validation-style accepted daily `q90(|E|)` p90 near `{args.target_error:.1%}`.",
        "",
        "## Best Frontier Rows",
        "",
        "| score | policy | obs coverage | day coverage | rel_score | q90 error | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=8% |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.head(24).iterrows():
        lines.append(
            f"| `{row.score}` | `{row.policy}` | {pct(row.obs_coverage_mean)} | {pct(row.day_coverage_mean)} | "
            f"{float(row.rel_score_mean):.5f} | {pct(row.q90_abs_error_mean)} | {pct(row.daily_q90_median_mean)} | "
            f"{pct(row.daily_q90_p90_mean)} | {pct(row.daily_q90_max_mean)} | {float(row.days_ge_3p5_mean):.1f} | "
            f"{float(row.days_ge_5_mean):.1f} | {float(row.days_ge_8_mean):.1f} |"
        )
    full = aggregate[(aggregate["score"].eq("none")) & (aggregate["policy"].eq("full_coverage"))]
    calibrated = aggregate[aggregate["policy"].isin(["calibrated_p90_3p5", "lowest_risk_available"])].sort_values(
        ["daily_q90_p90_mean", "obs_coverage_mean"],
        ascending=[True, False],
    )
    lines += [
        "",
        "## Calibration Thresholds",
        "",
        thresholds.to_markdown(index=False),
        "",
        "## Read",
        "",
    ]
    if not full.empty:
        f = full.iloc[0]
        lines.append(
            f"- Full coverage stressaux_w20: rel_score `{f.rel_score_mean:.5f}`, daily p90 `{100*f.daily_q90_p90_mean:.2f}%`, days >=3.5% `{f.days_ge_3p5_mean:.1f}`."
        )
    if not calibrated.empty:
        c = calibrated.iloc[0]
        lines.append(
            f"- Best selective row: `{c.score}/{c.policy}` with obs coverage `{100*c.obs_coverage_mean:.1f}%`, rel_score `{c.rel_score_mean:.5f}`, daily p90 `{100*c.daily_q90_p90_mean:.2f}%`."
        )
    lines += [
        "- If the 3.5% target is only reached at low coverage, the right paper framing is selective prediction/error-control, not full-coverage forecasting.",
        "- If input-noise or disagreement scores rank well, the next model improvement should move those features into a learned confidence head and cleaner input normalization.",
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    raw = load_raw_features(args.data)
    result_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    for seed in seeds:
        result, thresholds = evaluate_seed(seed, args, raw)
        result_parts.append(result)
        threshold_parts.append(thresholds)
    results = pd.concat(result_parts, ignore_index=True)
    thresholds = pd.concat(threshold_parts, ignore_index=True)
    aggregate = aggregate_results(results)
    results.to_csv(args.output_dir / "selective_error_by_seed.csv", index=False)
    thresholds.to_csv(args.output_dir / "selective_error_thresholds.csv", index=False)
    aggregate.to_csv(args.output_dir / "selective_error_aggregate.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "target_error": args.target_error,
                "min_daily_n": args.min_daily_n,
                "calibration_fraction": args.calibration_fraction,
                "base_prediction": "stressaux_w20",
                "risk_features": list(RISK_FEATURES),
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for file_name in ["selective_error_by_seed.csv", "selective_error_thresholds.csv", "selective_error_aggregate.csv", "manifest.json"]:
        source = args.output_dir / file_name
        target = args.gold_dir / file_name
        target.write_bytes(source.read_bytes())
    write_plot(args.gold_dir, aggregate)
    write_summary(args.gold_dir, aggregate, thresholds, args)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
