from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLD_DIR = ROOT / "gold" / "vn_transition_pressure_20260512"
DEFAULT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "filter_signal"
    / "portable_lstm_filter_signal_20260512_r02_no_leader_seed43"
    / "filter_predictions.csv.gz"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_VN30_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn30_symbols.csv"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"


RAW_FEATURE_COLUMNS = [
    "sector",
    "close",
    "adjust",
    "volume_match",
    "volume_ratio_20",
    "intraday_return",
    "gap_open",
    "close_position",
    "momentum_20",
    "volatility_20",
    "bb_width",
    "macd_hist",
    "sector_momentum_rank",
    "alpha_sector",
]


DAILY_FEATURE_COLUMNS = [
    "n_predictions",
    "base_abs_mean",
    "base_abs_q90",
    "base_prediction_std",
    "past_return_mean",
    "past_return_q10",
    "past_return_q25",
    "past_return_std",
    "past_abs_return_q90",
    "past_breadth",
    "past_negative_ratio",
    "past_left_tail_4pct_ratio",
    "past_left_tail_6pct_ratio",
    "leader_return_k10_w60",
    "leader_abs_return_k10_w60",
    "leader_return_excess_k10_w60",
    "feature_volatility_20_mean",
    "feature_volatility_20_q90",
    "feature_volume_ratio_20_mean",
    "feature_volume_ratio_20_q90",
    "feature_momentum_20_mean",
    "feature_momentum_20_std",
    "feature_bb_width_mean",
    "feature_bb_width_q90",
    "feature_abs_intraday_return_q90",
    "feature_alpha_sector_std",
    "feature_sector_momentum_rank_std",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe high-error regime filters on frozen LSTM predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="high_error_regime_filter_probe")
    parser.add_argument("--universe", default="vn100", choices=["vn30", "vn100", "all"])
    parser.add_argument("--spike-threshold", type=float, default=0.035)
    parser.add_argument("--leader-top-k", type=int, default=10)
    parser.add_argument("--leader-window", type=int, default=60)
    parser.add_argument("--leader-min-periods", type=int, default=20)
    parser.add_argument("--segment-year", type=int, default=2017)
    parser.add_argument("--segment-start-day", type=int, default=200)
    parser.add_argument("--segment-end-day", type=int, default=250)
    return parser.parse_args(argv)


def read_symbol_file(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def read_universe(name: str) -> tuple[str, set[str] | None]:
    if name == "vn30":
        return "VN30", read_symbol_file(DEFAULT_VN30_SYMBOLS)
    if name == "vn100":
        return "VN100", read_symbol_file(DEFAULT_VN100_SYMBOLS)
    return "ALL", None


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return 1.0 - robust_loss(actual - predicted) / base


def read_predictions(path: Path, symbols: set[str] | None) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(
            handle,
            usecols=["code", "split", "Date", "actual_date", "actual_aligned", "base_prediction"],
        )
    frame["code"] = frame["code"].astype(str).str.upper()
    if symbols is not None:
        frame = frame[frame["code"].isin(symbols)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    frame["actual_return"] = frame["actual_aligned"].astype(float)
    frame["predicted_return"] = frame["base_prediction"].astype(float)
    frame["error"] = frame["actual_return"] - frame["predicted_return"]
    frame["abs_error"] = frame["error"].abs()
    return frame


def read_raw(path: Path, symbols: set[str] | None) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = ["Date", "code"] + [column for column in RAW_FEATURE_COLUMNS if column in available]
    frame = pd.read_csv(path, usecols=usecols)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    if symbols is not None:
        frame = frame[frame["code"].isin(symbols)].copy()
    frame = frame.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
    frame["past_return_1"] = frame.groupby("code", sort=False)["adjust"].pct_change()
    frame["traded_value"] = frame["close"].abs() * frame["volume_match"].astype(float)
    return frame


def add_leader_signal(
    raw: pd.DataFrame,
    *,
    top_k: int,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    work = raw.loc[:, ["Date", "code", "past_return_1", "traded_value"]].copy()
    work["liquidity_score"] = work.groupby("code", sort=False)["traded_value"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    work["leader_rank"] = work.groupby("Date", sort=False)["liquidity_score"].rank(
        ascending=False,
        method="first",
    )
    daily_market = (
        work.groupby("Date", sort=False)
        .agg(past_return_mean=("past_return_1", "mean"))
        .reset_index()
    )
    leaders = work[(work["leader_rank"] <= top_k) & work["liquidity_score"].notna()].copy()
    if leaders.empty:
        daily_market[f"leader_return_k{top_k}_w{window}"] = np.nan
        daily_market[f"leader_abs_return_k{top_k}_w{window}"] = np.nan
        daily_market[f"leader_return_excess_k{top_k}_w{window}"] = np.nan
        return daily_market
    leaders["weighted_return"] = leaders["past_return_1"].fillna(0.0) * leaders["liquidity_score"].fillna(0.0)
    leader_daily = (
        leaders.groupby("Date", sort=False)
        .agg(weighted_return=("weighted_return", "sum"), weight=("liquidity_score", "sum"))
        .reset_index()
    )
    leader_col = f"leader_return_k{top_k}_w{window}"
    leader_daily[leader_col] = leader_daily["weighted_return"] / leader_daily["weight"].replace(0.0, np.nan)
    leader_daily[f"leader_abs_return_k{top_k}_w{window}"] = leader_daily[leader_col].abs()
    out = daily_market.merge(
        leader_daily[["Date", leader_col, f"leader_abs_return_k{top_k}_w{window}"]],
        on="Date",
        how="left",
    )
    out[f"leader_return_excess_k{top_k}_w{window}"] = out[leader_col] - out["past_return_mean"]
    return out


def build_daily_regime_frame(predictions: pd.DataFrame, raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    raw_features = raw.rename(columns={column: f"feature_{column}" for column in RAW_FEATURE_COLUMNS if column in raw.columns})
    merged = predictions.merge(
        raw_features.drop(columns=[column for column in ["feature_sector"] if column in raw_features.columns]),
        on=["Date", "code"],
        how="left",
    )
    merged["feature_abs_intraday_return"] = merged["feature_intraday_return"].abs()
    daily = (
        merged.groupby(["split", "actual_date"], sort=True)
        .agg(
            feature_date=("Date", "max"),
            n_predictions=("code", "nunique"),
            q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
            base_abs_mean=("predicted_return", lambda values: float(np.mean(np.abs(values)))),
            base_abs_q90=("predicted_return", lambda values: float(np.quantile(np.abs(values), 0.90))),
            base_prediction_std=("predicted_return", "std"),
            feature_volatility_20_mean=("feature_volatility_20", "mean"),
            feature_volatility_20_q90=("feature_volatility_20", lambda values: float(np.nanquantile(values, 0.90))),
            feature_volume_ratio_20_mean=("feature_volume_ratio_20", "mean"),
            feature_volume_ratio_20_q90=("feature_volume_ratio_20", lambda values: float(np.nanquantile(values, 0.90))),
            feature_momentum_20_mean=("feature_momentum_20", "mean"),
            feature_momentum_20_std=("feature_momentum_20", "std"),
            feature_bb_width_mean=("feature_bb_width", "mean"),
            feature_bb_width_q90=("feature_bb_width", lambda values: float(np.nanquantile(values, 0.90))),
            feature_abs_intraday_return_q90=(
                "feature_abs_intraday_return",
                lambda values: float(np.nanquantile(values, 0.90)),
            ),
            feature_alpha_sector_std=("feature_alpha_sector", "std"),
            feature_sector_momentum_rank_std=("feature_sector_momentum_rank", "std"),
        )
        .reset_index()
        .rename(columns={"actual_date": "Date"})
    )
    raw_daily = (
        raw.groupby("Date", sort=True)
        .agg(
            past_return_mean=("past_return_1", "mean"),
            past_return_q10=("past_return_1", lambda values: float(np.nanquantile(values, 0.10))),
            past_return_q25=("past_return_1", lambda values: float(np.nanquantile(values, 0.25))),
            past_return_std=("past_return_1", "std"),
            past_abs_return_q90=("past_return_1", lambda values: float(np.nanquantile(np.abs(values), 0.90))),
            past_breadth=("past_return_1", lambda values: float(np.nanmean(values > 0.0))),
            past_negative_ratio=("past_return_1", lambda values: float(np.nanmean(values < 0.0))),
            past_left_tail_4pct_ratio=("past_return_1", lambda values: float(np.nanmean(values <= -0.04))),
            past_left_tail_6pct_ratio=("past_return_1", lambda values: float(np.nanmean(values <= -0.06))),
        )
        .reset_index()
    )
    leader = add_leader_signal(
        raw,
        top_k=args.leader_top_k,
        window=args.leader_window,
        min_periods=args.leader_min_periods,
    )
    daily = daily.merge(raw_daily.rename(columns={"Date": "feature_date"}), on="feature_date", how="left")
    daily = daily.merge(leader.drop(columns=["past_return_mean"]).rename(columns={"Date": "feature_date"}), on="feature_date", how="left")
    daily["high_error_label"] = daily["q90_abs_error"].ge(args.spike_threshold).astype(int)
    return daily.sort_values(["split", "Date"], kind="stable").reset_index(drop=True)


def fit_models(train: pd.DataFrame) -> dict[str, Pipeline]:
    x_train = train.loc[:, DAILY_FEATURE_COLUMNS]
    y_train = train["high_error_label"].to_numpy(dtype=int)
    models: dict[str, Pipeline] = {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=160,
                        learning_rate=0.045,
                        l2_regularization=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    return {name: model.fit(x_train, y_train) for name, model in models.items()}


def predict_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame.loc[:, DAILY_FEATURE_COLUMNS])[:, 1]


def score_classifier(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    clean = frame.dropna(subset=[score_column, "high_error_label", "q90_abs_error"]).copy()
    labels = clean["high_error_label"].to_numpy(dtype=int)
    scores = clean[score_column].to_numpy(dtype=float)
    out: dict[str, float] = {
        "n_days": int(len(clean)),
        "spike_rate": float(np.mean(labels)) if len(labels) else float("nan"),
        "auc": float("nan"),
        "average_precision": float("nan"),
    }
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, scores))
        out["average_precision"] = float(average_precision_score(labels, scores))
    cutoff = float(np.quantile(scores, 0.80)) if len(scores) else float("nan")
    top = clean[scores >= cutoff] if np.isfinite(cutoff) else clean.iloc[0:0]
    rest = clean[scores < cutoff] if np.isfinite(cutoff) else clean.iloc[0:0]
    out.update(
        {
            "top20_days": int(len(top)),
            "top20_spike_rate": float(top["high_error_label"].mean()) if len(top) else float("nan"),
            "rest_spike_rate": float(rest["high_error_label"].mean()) if len(rest) else float("nan"),
            "top20_median_q90_abs_error": float(top["q90_abs_error"].median()) if len(top) else float("nan"),
            "rest_median_q90_abs_error": float(rest["q90_abs_error"].median()) if len(rest) else float("nan"),
        }
    )
    return out


def score_models(daily: pd.DataFrame, models: dict[str, Pipeline]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = daily.copy()
    for name, model in models.items():
        scored[f"{name}_high_error_probability"] = predict_probability(model, scored)

    rows: list[dict[str, object]] = []
    for name in models:
        score_column = f"{name}_high_error_probability"
        for split, group in scored.groupby("split", sort=True):
            row = {"model": name, "split": split}
            row.update(score_classifier(group, score_column))
            rows.append(row)
    return scored, pd.DataFrame(rows)


def calibration_scale_probe(predictions: pd.DataFrame, segment_dates: pd.Series) -> pd.DataFrame:
    train = predictions[predictions["split"].eq("train")].dropna(subset=["actual_return", "predicted_return"])
    candidates = np.r_[np.linspace(0.0, 3.0, 61), np.linspace(3.25, 8.0, 20)]
    losses = [
        robust_loss(train["actual_return"].to_numpy(dtype=float) - scale * train["predicted_return"].to_numpy(dtype=float))
        for scale in candidates
    ]
    best_scale = float(candidates[int(np.nanargmin(losses))])
    rows: list[dict[str, object]] = []
    for split, group in predictions.groupby("split", sort=True):
        clean = group.dropna(subset=["actual_return", "predicted_return"])
        actual = clean["actual_return"].to_numpy(dtype=float)
        base_pred = clean["predicted_return"].to_numpy(dtype=float)
        for name, scale in [("base_scale_1", 1.0), ("train_best_scale", best_scale)]:
            rows.append(
                {
                    "scope": split,
                    "candidate": name,
                    "scale": scale,
                    "rel_score": rel_score(actual, scale * base_pred),
                    "robust_loss": robust_loss(actual - scale * base_pred),
                    "median_abs_error": float(np.median(np.abs(actual - scale * base_pred))),
                    "q90_abs_error": float(np.quantile(np.abs(actual - scale * base_pred), 0.90)),
                }
            )
    segment = predictions[predictions["actual_date"].isin(segment_dates)].dropna(subset=["actual_return", "predicted_return"])
    actual = segment["actual_return"].to_numpy(dtype=float)
    base_pred = segment["predicted_return"].to_numpy(dtype=float)
    for name, scale in [("base_scale_1", 1.0), ("train_best_scale", best_scale)]:
        rows.append(
            {
                "scope": "segment_2017_d200_250",
                "candidate": name,
                "scale": scale,
                "rel_score": rel_score(actual, scale * base_pred),
                "robust_loss": robust_loss(actual - scale * base_pred),
                "median_abs_error": float(np.median(np.abs(actual - scale * base_pred))),
                "q90_abs_error": float(np.quantile(np.abs(actual - scale * base_pred), 0.90)),
            }
        )
    return pd.DataFrame(rows)


def segment_frame(scored_daily: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    work = scored_daily[scored_daily["Date"].dt.year.eq(args.segment_year)].sort_values("Date", kind="stable").reset_index(drop=True)
    work["trading_day_in_year"] = np.arange(len(work))
    return work[
        work["trading_day_in_year"].between(args.segment_start_day, args.segment_end_day, inclusive="both")
    ].copy()


def classifier_feature_importance(model: Pipeline, name: str) -> pd.DataFrame:
    if name == "logistic":
        coef = model.named_steps["model"].coef_[0]
        return pd.DataFrame({"feature": DAILY_FEATURE_COLUMNS, "importance": coef}).sort_values(
            "importance",
            key=lambda values: values.abs(),
            ascending=False,
            kind="stable",
        )
    if name == "hist_gradient_boosting":
        return pd.DataFrame({"feature": DAILY_FEATURE_COLUMNS, "importance": np.nan})
    return pd.DataFrame({"feature": DAILY_FEATURE_COLUMNS, "importance": np.nan})


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def write_summary(
    output_dir: Path,
    label: str,
    metrics: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    scale_probe: pd.DataFrame,
    segment_scored: pd.DataFrame,
    logistic_importance: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    val_metrics = metrics[metrics["split"].eq("val")].sort_values("auc", ascending=False, kind="stable")
    best_val = val_metrics.iloc[0]
    best_segment = segment_metrics.sort_values("top20_spike_rate", ascending=False, kind="stable").iloc[0]
    segment_spike_rate = float(segment_scored["high_error_label"].mean())
    segment_median_error = float(segment_scored["q90_abs_error"].median())

    display_metrics = metrics.copy()
    display_segment_metrics = segment_metrics.copy()
    for column in [
        "spike_rate",
        "top20_spike_rate",
        "rest_spike_rate",
        "top20_median_q90_abs_error",
        "rest_median_q90_abs_error",
    ]:
        display_metrics[column] = display_metrics[column].map(pct)
        display_segment_metrics[column] = display_segment_metrics[column].map(pct)
    display_scale = scale_probe.copy()
    for column in ["rel_score", "robust_loss", "median_abs_error", "q90_abs_error"]:
        display_scale[column] = display_scale[column].map(lambda value: f"{value:+.5f}" if np.isfinite(value) else "n/a")

    top_importance = logistic_importance.head(12).copy()
    top_importance["importance"] = top_importance["importance"].map(lambda value: f"{value:+.4f}" if np.isfinite(value) else "n/a")

    lines = [
        "# High-Error Regime Filter Probe",
        "",
        f"Scope: `{label}`. Target label: daily `q90(|actual_return - predicted_return|) > {args.spike_threshold:.1%}`.",
        "This is a train/validation probe on frozen base LSTM predictions. Holdout/test is not used.",
        "",
        "## Result",
        "",
        f"- Best validation AUC: `{best_val['auc']:.3f}` from `{best_val['model']}`.",
        f"- Validation baseline spike rate: `{pct(float(best_val['spike_rate']))}`.",
        f"- Validation top-20% risk spike rate: `{pct(float(best_val['top20_spike_rate']))}`.",
        f"- 2017 segment spike rate: `{pct(segment_spike_rate)}`.",
        f"- Best 2017 segment top-20% risk spike rate: `{pct(float(best_segment['top20_spike_rate']))}` from `{best_segment['model']}`.",
        f"- 2017 segment median q90(|E|): `{pct(segment_median_error)}`; best top-risk median: `{pct(float(best_segment['top20_median_q90_abs_error']))}`.",
        "",
        "Read: if validation AUC is above random and top-risk buckets contain materially more spike days, the high-error timing has regime signal. That supports adding a sidecar LSTM/filter/regime layer instead of only enlarging the base LSTM.",
        "The 2017 segment ranking is diagnostic because that segment is inside train; use validation metrics for model-selection claims.",
        "",
        "## Classifier Metrics",
        "",
        display_metrics.to_markdown(index=False),
        "",
        "## 2017 Segment Risk Ranking",
        "",
        display_segment_metrics.to_markdown(index=False),
        "",
        "## Output-Scale Probe",
        "",
        display_scale.to_markdown(index=False),
        "",
        "Read: if `train_best_scale` does not materially reduce q90 error, the problem is not just prediction magnitude calibration. The model is missing tail timing/cross-sectional selection.",
        "",
        "## Logistic Feature Weights",
        "",
        top_importance.to_markdown(index=False),
        "",
        "## Next Train Test",
        "",
        "Train the next LSTM filter with explicit high-error/tail-risk supervision and these regime features, then compare validation high-risk buckets and the 2017 uptrend segment against the frozen base output.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    label, symbols = read_universe(args.universe)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(args.predictions, symbols)
    raw = read_raw(args.data, symbols)
    daily = build_daily_regime_frame(predictions, raw, args)
    train = daily[daily["split"].eq("train")].dropna(subset=["high_error_label"]).copy()
    if train["high_error_label"].nunique() < 2:
        raise ValueError("Training split has only one high-error class; cannot fit classifier.")
    models = fit_models(train)
    scored_daily, metrics = score_models(daily, models)
    segment_scored = segment_frame(scored_daily, args)
    segment_metrics = pd.DataFrame(
        [
            {
                "model": name,
                "split": f"segment_{args.segment_year}_d{args.segment_start_day}_{args.segment_end_day}",
                **score_classifier(segment_scored, f"{name}_high_error_probability"),
            }
            for name in models
        ]
    )
    scale_probe = calibration_scale_probe(predictions, segment_scored["Date"])
    logistic_importance = classifier_feature_importance(models["logistic"], "logistic")

    scored_daily.to_csv(output_dir / "daily_regime_filter_scores.csv", index=False)
    metrics.to_csv(output_dir / "regime_filter_metrics.csv", index=False)
    segment_metrics.to_csv(output_dir / "segment_regime_filter_metrics.csv", index=False)
    segment_scored.to_csv(output_dir / "segment_2017_regime_filter_scores.csv", index=False)
    scale_probe.to_csv(output_dir / "output_scale_probe.csv", index=False)
    logistic_importance.to_csv(output_dir / "logistic_feature_weights.csv", index=False)
    write_summary(output_dir, label, metrics, segment_metrics, scale_probe, segment_scored, logistic_importance, args)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "data": str(args.data),
                "universe": args.universe,
                "spike_threshold": args.spike_threshold,
                "leader_top_k": args.leader_top_k,
                "leader_window": args.leader_window,
                "leader_min_periods": args.leader_min_periods,
                "segment_year": args.segment_year,
                "segment_start_day": args.segment_start_day,
                "segment_end_day": args.segment_end_day,
                "feature_columns": DAILY_FEATURE_COLUMNS,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "daily_rows": int(scored_daily.shape[0]),
                "metrics_rows": int(metrics.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
