from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.config import DEFAULT_DATA_PATH  # noqa: E402
from src.models.selection.holding_period import simulate_rebalance, summarize_rebalance  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
DEFAULT_RUNS = (
    "portable_lstm_filter_signal_20260509_r06_selector_module",
    "portable_lstm_filter_signal_20260508_r05_signmag",
)
DEFAULT_OUTPUT = FILTER_ROOT / "market_leader_gate_20260512_r01"
BASE_CANDIDATES = (
    "prediction_base",
    "prediction_gate",
    "prediction_move_top_train_ic_selected",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate market-leader return as a post-model filter gate on filter-signal artifacts."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k-grid", default="3,5,8")
    parser.add_argument("--liquidity-window-grid", default="60,90")
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-positions", type=int, default=10)
    parser.add_argument("--rebalance-days", default="1,2,3,5,10")
    return parser.parse_args(argv)


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def robust_loss(values: pd.Series | np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: pd.Series, prediction: pd.Series) -> float:
    base_loss = robust_loss(actual)
    if not np.isfinite(base_loss) or base_loss == 0.0:
        return float("nan")
    return 1.0 - robust_loss(actual.to_numpy(dtype=float) - prediction.to_numpy(dtype=float)) / base_loss


def daily_ic_t_stat(frame: pd.DataFrame, prediction_column: str) -> tuple[float, float, int]:
    rows: list[float] = []
    for _, day in frame.groupby("actual_date", sort=False):
        active = day.loc[day[prediction_column].astype(float) != 0.0]
        if len(active) < 5:
            continue
        corr = active[prediction_column].corr(active["actual_aligned"], method="spearman")
        if np.isfinite(corr):
            rows.append(float(corr))
    if not rows:
        return float("nan"), float("nan"), 0
    values = np.asarray(rows, dtype=float)
    std = values.std(ddof=1)
    t_stat = float(values.mean() / std * np.sqrt(len(values))) if len(values) > 1 and std > 0.0 else float("nan")
    return float(values.mean()), t_stat, int(len(values))


def evaluate_prediction(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    prediction = frame[prediction_column].fillna(0.0).astype(float)
    actual = frame["actual_aligned"].astype(float)
    active = prediction != 0.0
    mean_ic, ic_t, ic_days = daily_ic_t_stat(frame, prediction_column)
    return {
        "n_obs": int(len(frame)),
        "coverage": float(active.mean()),
        "rel_score": rel_score(actual, prediction),
        "directional_accuracy": float((np.sign(prediction) == np.sign(actual)).mean()),
        "active_hit_rate": float((np.sign(prediction[active]) == np.sign(actual[active])).mean()) if active.any() else float("nan"),
        "mean_daily_ic": mean_ic,
        "daily_ic_t": ic_t,
        "ic_days": ic_days,
    }


def build_market_leader_signals(
    data_path: Path,
    top_k_values: tuple[int, ...],
    liquidity_windows: tuple[int, ...],
    min_periods: int,
) -> pd.DataFrame:
    usecols = {"Date", "code", "close", "adjust", "volume_match"}
    raw = pd.read_csv(data_path, usecols=lambda column: column in usecols)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
    raw["stock_return_1"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    raw["traded_value"] = raw["close"].abs() * raw["volume_match"].astype(float)

    out = pd.DataFrame({"Date": sorted(raw["Date"].dropna().unique())})
    daily_market = raw.groupby("Date", sort=False)["stock_return_1"].mean().rename("market_return_1").reset_index()
    out = out.merge(daily_market, on="Date", how="left")

    for window in liquidity_windows:
        local_min_periods = min(min_periods, window)
        work = raw.loc[:, ["Date", "code", "stock_return_1", "traded_value"]].copy()
        work["liquidity_score"] = work.groupby("code", sort=False)["traded_value"].transform(
            lambda series: series.shift(1).rolling(window, min_periods=local_min_periods).mean()
        )
        work["leader_rank"] = work.groupby("Date", sort=False)["liquidity_score"].rank(
            ascending=False,
            method="first",
        )
        for top_k in top_k_values:
            leaders = work.loc[
                (work["leader_rank"] <= top_k) & work["liquidity_score"].notna()
            ].copy()
            leaders["weighted_return"] = leaders["stock_return_1"].fillna(0.0) * leaders["liquidity_score"].fillna(0.0)
            signal = (
                leaders.groupby("Date", sort=False)
                .agg(weighted_return=("weighted_return", "sum"), weight=("liquidity_score", "sum"))
                .reset_index()
            )
            column = f"leader_return_k{top_k}_w{window}"
            signal[column] = signal["weighted_return"] / signal["weight"].replace(0.0, np.nan)
            out = out.merge(signal[["Date", column]], on="Date", how="left")
            out[f"{column}_excess"] = out[column] - out["market_return_1"]
    return out


def add_leader_gate_candidates(
    frame: pd.DataFrame,
    leader_columns: list[str],
    base_candidates: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    new_columns: list[str] = []
    train = out.loc[out["split"] == "train"]
    thresholds = {
        column: float(train[column].abs().median()) if column in train else float("nan")
        for column in leader_columns
    }
    for leader_column in leader_columns:
        threshold = thresholds[leader_column]
        for candidate in base_candidates:
            if candidate not in out.columns:
                continue
            suffix = leader_column.replace("leader_return_", "leader_")
            agree_column = f"{candidate}_{suffix}_agree"
            strong_agree_column = f"{candidate}_{suffix}_strong_agree"
            riskoff_column = f"{candidate}_{suffix}_riskoff"
            prediction = out[candidate].fillna(0.0).astype(float)
            leader = out[leader_column].fillna(0.0).astype(float)
            agrees = (np.sign(prediction) == np.sign(leader)) | (prediction == 0.0) | (leader == 0.0)
            strong = leader.abs() >= threshold if np.isfinite(threshold) else pd.Series(False, index=out.index)
            out[agree_column] = np.where(agrees, prediction, 0.0)
            out[strong_agree_column] = np.where(strong & ~agrees, 0.0, prediction)
            out[riskoff_column] = np.where((leader < 0.0) & (prediction > 0.0), 0.0, prediction)
            new_columns.extend([agree_column, strong_agree_column, riskoff_column])
    return out, new_columns


def evaluate_run(
    run_name: str,
    predictions: pd.DataFrame,
    leader_signals: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    frame = frame.merge(leader_signals, on="Date", how="left")
    leader_columns = [column for column in leader_signals.columns if column.startswith("leader_return_") and not column.endswith("_excess")]
    frame, gate_columns = add_leader_gate_candidates(frame, leader_columns, BASE_CANDIDATES)

    candidates = [column for column in BASE_CANDIDATES if column in frame.columns] + gate_columns
    metric_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []
    for split, split_frame in frame.groupby("split", sort=False):
        split_frame = split_frame.dropna(subset=["actual_aligned", "actual_date"]).copy()
        for candidate in candidates:
            metric_rows.append(
                {
                    "run": run_name,
                    "split": split,
                    "candidate": candidate,
                    **evaluate_prediction(split_frame, candidate),
                }
            )
            for rebalance_every in parse_int_list(args.rebalance_days):
                daily = simulate_rebalance(
                    split_frame,
                    candidate,
                    rebalance_every=rebalance_every,
                    cost_bps=args.cost_bps,
                    min_positions=args.min_positions,
                )
                rebalance_rows.append(
                    {
                        "run": run_name,
                        **summarize_rebalance(daily, split, candidate, rebalance_every, args.cost_bps),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(rebalance_rows)


def summarize_cross_artifact(metrics: pd.DataFrame, rebalances: pd.DataFrame) -> pd.DataFrame:
    val_metrics = metrics.loc[metrics["split"] == "val"].copy()
    val_rebalance = rebalances.loc[rebalances["split"] == "val"].copy()
    best_rebalance = (
        val_rebalance.sort_values(["run", "candidate", "net_equity"], ascending=[True, True, False], kind="stable")
        .groupby(["run", "candidate"], sort=False)
        .head(1)
    )
    merged = val_metrics.merge(
        best_rebalance[
            [
                "run",
                "candidate",
                "rebalance_every",
                "net_equity",
                "net_sharpe",
                "net_max_drawdown",
                "avg_turnover",
            ]
        ],
        on=["run", "candidate"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    for candidate, group in merged.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                "n_runs": int(group["run"].nunique()),
                "mean_val_rel_score": float(group["rel_score"].mean()),
                "min_val_rel_score": float(group["rel_score"].min()),
                "mean_val_ic": float(group["mean_daily_ic"].mean()),
                "min_val_ic": float(group["mean_daily_ic"].min()),
                "mean_net_equity": float(group["net_equity"].mean()),
                "min_net_equity": float(group["net_equity"].min()),
                "mean_net_sharpe": float(group["net_sharpe"].mean()),
                "min_net_sharpe": float(group["net_sharpe"].min()),
                "worst_max_drawdown": float(group["net_max_drawdown"].min()),
                "max_turnover": float(group["avg_turnover"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["min_val_rel_score", "min_net_equity", "min_val_ic"],
        ascending=False,
        kind="stable",
    )


def write_summary(output_dir: Path, cross_summary: pd.DataFrame, metrics: pd.DataFrame, rebalances: pd.DataFrame) -> None:
    top_columns = [
        "candidate",
        "n_runs",
        "mean_val_rel_score",
        "min_val_rel_score",
        "mean_val_ic",
        "min_val_ic",
        "mean_net_equity",
        "min_net_equity",
        "mean_net_sharpe",
        "min_net_sharpe",
        "worst_max_drawdown",
        "max_turnover",
    ]
    val_metric_columns = [
        "run",
        "candidate",
        "rel_score",
        "coverage",
        "active_hit_rate",
        "mean_daily_ic",
        "daily_ic_t",
    ]
    val_rebalance_columns = [
        "run",
        "candidate",
        "rebalance_every",
        "net_equity",
        "net_sharpe",
        "net_max_drawdown",
        "avg_turnover",
    ]
    lines = [
        "# Market Leader Filter Gate",
        "",
        "Purpose: test whether compact market-leader signals improve the post-model filter/selector layer.",
        "",
        "Rules: leader baskets are selected by lagged rolling traded value; metrics are train/validation only; costs are included in rebalance summaries.",
        "",
        "## Cross-Artifact Summary",
        "",
        cross_summary.loc[:, top_columns].head(20).to_markdown(index=False),
        "",
        "## Validation Metrics",
        "",
        metrics.loc[metrics["split"] == "val", val_metric_columns]
        .sort_values(["run", "rel_score"], ascending=[True, False], kind="stable")
        .head(40)
        .to_markdown(index=False),
        "",
        "## Best Validation Rebalance Per Candidate",
        "",
        rebalances.loc[rebalances["split"] == "val", val_rebalance_columns]
        .sort_values(["run", "candidate", "net_equity"], ascending=[True, True, False], kind="stable")
        .groupby(["run", "candidate"], sort=False)
        .head(1)
        .sort_values(["run", "net_equity"], ascending=[True, False], kind="stable")
        .head(40)
        .to_markdown(index=False),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = tuple(item.strip() for item in args.runs.split(",") if item.strip())
    leader_signals = build_market_leader_signals(
        args.data_path,
        parse_int_list(args.top_k_grid),
        parse_int_list(args.liquidity_window_grid),
        args.min_periods,
    )
    metric_frames: list[pd.DataFrame] = []
    rebalance_frames: list[pd.DataFrame] = []
    for run_name in runs:
        predictions_path = FILTER_ROOT / run_name / "filter_predictions.csv.gz"
        predictions = pd.read_csv(predictions_path)
        metrics, rebalances = evaluate_run(run_name, predictions, leader_signals, args)
        metric_frames.append(metrics)
        rebalance_frames.append(rebalances)

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    rebalances_all = pd.concat(rebalance_frames, ignore_index=True)
    cross_summary = summarize_cross_artifact(metrics_all, rebalances_all)

    metrics_all.to_csv(output_dir / "market_leader_gate_metrics.csv", index=False)
    rebalances_all.to_csv(output_dir / "market_leader_gate_rebalance.csv", index=False)
    cross_summary.to_csv(output_dir / "market_leader_gate_cross_artifact_summary.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    write_summary(output_dir, cross_summary, metrics_all, rebalances_all)
    print(json.dumps(cross_summary.head(10).to_dict(orient="records"), indent=2, default=str))


if __name__ == "__main__":
    main()
