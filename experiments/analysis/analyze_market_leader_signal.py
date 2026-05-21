from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.config import DEFAULT_DATA_PATH, TRAIN_END_DATE, VAL_END_DATE  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "market_leader_signal"


@dataclass(frozen=True)
class SignalConfig:
    top_k: int
    liquidity_window: int
    min_periods: int
    signal_name: str


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search market-leader basket parameters as a causal market-context signal."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT / "market_leader_signal_grid_latest")
    parser.add_argument("--train-end-date", default=TRAIN_END_DATE)
    parser.add_argument("--val-end-date", default=VAL_END_DATE)
    parser.add_argument("--top-k-grid", default="3,5,8,10,15,20,30")
    parser.add_argument("--liquidity-window-grid", default="20,40,60,90,120")
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--include-test", action="store_true")
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    abs_values = np.abs(np.asarray(values, dtype=float))
    if len(abs_values) == 0:
        return float("nan")
    return float(np.nanquantile(abs_values, 0.5) + 0.5 * np.nanquantile(abs_values, 0.9))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base_loss = robust_loss(actual)
    if not np.isfinite(base_loss) or base_loss == 0.0:
        return float("nan")
    return 1.0 - robust_loss(actual - prediction) / base_loss


def t_stat(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return float("nan")
    std = float(np.std(clean, ddof=1))
    if std == 0.0:
        return float("nan")
    return float(np.mean(clean) / std * np.sqrt(len(clean)))


def sharpe(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return float("nan")
    std = float(np.std(clean, ddof=1))
    if std == 0.0:
        return float("nan")
    return float(np.mean(clean) / std * np.sqrt(252.0))


def fit_linear(train: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    clean = train.dropna(subset=[*feature_columns, "market_next_return"])
    if len(clean) < len(feature_columns) + 3:
        return np.full(len(feature_columns) + 1, np.nan, dtype=float)
    x = clean.loc[:, feature_columns].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = clean["market_next_return"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta.astype(float)


def predict_linear(frame: pd.DataFrame, feature_columns: list[str], beta: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(beta)):
        return np.full(len(frame), np.nan, dtype=float)
    x = frame.loc[:, feature_columns].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    return x @ beta


def build_base_frame(data_path: Path) -> pd.DataFrame:
    usecols = [
        "Date",
        "code",
        "close",
        "adjust",
        "volume_match",
        "target_next_return",
    ]
    raw = pd.read_csv(data_path, usecols=lambda column: column in set(usecols))
    missing = sorted(set(usecols).difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)
    raw["stock_return_1"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    raw["traded_value"] = raw["close"].abs() * raw["volume_match"].astype(float)
    return raw


def build_daily_signal(raw: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    work = raw.loc[:, ["Date", "code", "stock_return_1", "target_next_return", "traded_value"]].copy()
    work["liquidity_score"] = work.groupby("code", sort=False)["traded_value"].transform(
        lambda series: series.shift(1).rolling(config.liquidity_window, min_periods=config.min_periods).mean()
    )
    work["leader_rank"] = work.groupby("Date", sort=False)["liquidity_score"].rank(
        ascending=False,
        method="first",
    )
    market_daily = (
        work.groupby("Date", sort=False)
        .agg(
            market_return_1=("stock_return_1", "mean"),
            market_next_return=("target_next_return", "mean"),
            breadth_next=("target_next_return", lambda values: float((values > 0.0).mean())),
            n_names=("code", "nunique"),
        )
        .reset_index()
    )
    leaders = work[(work["leader_rank"] <= config.top_k) & work["liquidity_score"].notna()].copy()
    if leaders.empty:
        market_daily[config.signal_name] = np.nan
        market_daily[f"{config.signal_name}_excess"] = np.nan
        return market_daily
    leaders["weighted_return"] = leaders["stock_return_1"].fillna(0.0) * leaders["liquidity_score"].fillna(0.0)
    leader_daily = (
        leaders.groupby("Date", sort=False)
        .agg(
            weighted_return=("weighted_return", "sum"),
            weight=("liquidity_score", "sum"),
            leader_count=("code", "nunique"),
        )
        .reset_index()
    )
    leader_daily[config.signal_name] = leader_daily["weighted_return"] / leader_daily["weight"].replace(0.0, np.nan)
    daily = market_daily.merge(
        leader_daily[["Date", config.signal_name, "leader_count"]],
        on="Date",
        how="left",
    )
    daily[f"{config.signal_name}_excess"] = daily[config.signal_name] - daily["market_return_1"]
    return daily


def split_frame(daily: pd.DataFrame, train_end_date: str, val_end_date: str, include_test: bool) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)
    splits = {
        "train": daily[daily["Date"] <= train_end].copy(),
        "val": daily[(daily["Date"] > train_end) & (daily["Date"] <= val_end)].copy(),
    }
    if include_test:
        splits["test"] = daily[daily["Date"] > val_end].copy()
    return splits


def evaluate_signal_split(frame: pd.DataFrame, signal_column: str, cost_bps: float) -> dict[str, float]:
    clean = frame.dropna(subset=[signal_column, "market_next_return"]).copy()
    if clean.empty:
        return {
            "n_days": 0,
            "spearman_ic": float("nan"),
            "pearson_ic": float("nan"),
            "sign_accuracy": float("nan"),
            "timing_equity_net": float("nan"),
            "timing_sharpe_net": float("nan"),
            "timing_t_stat_net": float("nan"),
            "turnover": float("nan"),
        }
    signal = clean[signal_column].to_numpy(dtype=float)
    target = clean["market_next_return"].to_numpy(dtype=float)
    position = np.sign(signal)
    position[np.abs(signal) < 1e-12] = 0.0
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    cost = (cost_bps / 10000.0) * turnover
    net_return = position * target - cost
    return {
        "n_days": int(len(clean)),
        "spearman_ic": float(clean[signal_column].corr(clean["market_next_return"], method="spearman")),
        "pearson_ic": float(clean[signal_column].corr(clean["market_next_return"], method="pearson")),
        "sign_accuracy": float(np.mean(np.sign(signal) == np.sign(target))),
        "timing_equity_net": float(np.prod(1.0 + np.nan_to_num(net_return, nan=0.0))),
        "timing_sharpe_net": sharpe(net_return),
        "timing_t_stat_net": t_stat(net_return),
        "turnover": float(np.mean(turnover)),
    }


def evaluate_config(
    raw: pd.DataFrame,
    config: SignalConfig,
    train_end_date: str,
    val_end_date: str,
    include_test: bool,
    cost_bps: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    daily = build_daily_signal(raw, config)
    signal_columns = [config.signal_name, f"{config.signal_name}_excess"]
    splits = split_frame(daily, train_end_date, val_end_date, include_test)
    baseline_beta = fit_linear(splits["train"], ["market_return_1"])

    row: dict[str, object] = asdict(config)
    row["baseline_train_rel_score"] = rel_score(
        splits["train"]["market_next_return"].to_numpy(dtype=float),
        predict_linear(splits["train"], ["market_return_1"], baseline_beta),
    )
    row["baseline_val_rel_score"] = rel_score(
        splits["val"]["market_next_return"].to_numpy(dtype=float),
        predict_linear(splits["val"], ["market_return_1"], baseline_beta),
    )
    if include_test and "test" in splits:
        row["baseline_test_rel_score"] = rel_score(
            splits["test"]["market_next_return"].to_numpy(dtype=float),
            predict_linear(splits["test"], ["market_return_1"], baseline_beta),
        )

    best_train_score = float("-inf")
    best_signal_column = signal_columns[0]
    best_metrics: dict[str, float] = {}
    for signal_column in signal_columns:
        feature_columns = ["market_return_1", signal_column]
        beta = fit_linear(splits["train"], feature_columns)
        candidate_metrics: dict[str, float] = {}
        for split_name, split_df in splits.items():
            prediction = predict_linear(split_df, feature_columns, beta)
            actual = split_df["market_next_return"].to_numpy(dtype=float)
            split_metrics = evaluate_signal_split(split_df, signal_column, cost_bps)
            rel = rel_score(actual, prediction)
            candidate_metrics[f"{split_name}_rel_score"] = rel
            candidate_metrics[f"{split_name}_rel_improvement"] = rel - row[f"baseline_{split_name}_rel_score"]
            for metric_name, metric_value in split_metrics.items():
                candidate_metrics[f"{split_name}_{metric_name}"] = metric_value
        train_score = float(candidate_metrics.get("train_rel_improvement", float("nan")))
        if np.isfinite(train_score) and train_score > best_train_score:
            best_train_score = train_score
            best_signal_column = signal_column
            best_metrics = candidate_metrics

    row["selected_by_train_signal_column"] = best_signal_column
    row["selected_signal_variant"] = "excess" if best_signal_column.endswith("_excess") else "raw"
    for metric_name, metric_value in best_metrics.items():
        row[f"selected_{metric_name}"] = metric_value
    return row, daily


def build_leader_snapshot(raw: pd.DataFrame, top_k: int, liquidity_window: int, min_periods: int) -> pd.DataFrame:
    work = raw.loc[:, ["Date", "code", "stock_return_1", "traded_value"]].copy()
    work["liquidity_score"] = work.groupby("code", sort=False)["traded_value"].transform(
        lambda series: series.shift(1).rolling(liquidity_window, min_periods=min_periods).mean()
    )
    latest_date = work["Date"].max()
    latest = work[work["Date"] == latest_date].copy()
    latest["leader_rank"] = latest["liquidity_score"].rank(ascending=False, method="first")
    return latest[latest["leader_rank"] <= top_k].sort_values("leader_rank")[
        ["Date", "leader_rank", "code", "liquidity_score", "stock_return_1"]
    ]


def write_summary(output_dir: Path, summary: pd.DataFrame, selected: dict[str, object], latest_leaders: pd.DataFrame) -> None:
    lines = [
        "# Market Leader Signal Grid",
        "",
        "Feature hypothesis: replace the old hard-coded `vingroup_momentum` with a causal basket of market leaders.",
        "",
        "Selection rule: choose `top_k`, liquidity window, and raw/excess variant by train rel_score improvement over a one-factor market-return baseline; report validation separately.",
        "",
        "## Selected By Train",
        "",
    ]
    selected_display_keys = [
        "top_k",
        "liquidity_window",
        "min_periods",
        "selected_by_train_signal_column",
        "selected_signal_variant",
        "selected_train_rel_improvement",
        "selected_val_rel_improvement",
        "selected_val_spearman_ic",
        "selected_val_sign_accuracy",
        "selected_val_timing_equity_net",
        "selected_val_timing_t_stat_net",
    ]
    for key in selected_display_keys:
        value = selected.get(key)
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Validation Rows", ""])
    display_columns = [
        "top_k",
        "liquidity_window",
        "selected_by_train_signal_column",
        "selected_signal_variant",
        "selected_train_rel_improvement",
        "selected_val_rel_improvement",
        "selected_val_spearman_ic",
        "selected_val_sign_accuracy",
        "selected_val_timing_equity_net",
        "selected_val_timing_t_stat_net",
    ]
    top_rows = summary.sort_values("selected_val_rel_improvement", ascending=False).head(10)
    lines.append(top_rows.loc[:, display_columns].to_markdown(index=False))
    lines.extend(["", "## Latest Leaders For Selected Config", ""])
    lines.append(latest_leaders.to_markdown(index=False))
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = build_base_frame(args.data_path)
    rows: list[dict[str, object]] = []
    daily_by_key: dict[str, pd.DataFrame] = {}
    for liquidity_window in parse_int_list(args.liquidity_window_grid):
        min_periods = min(args.min_periods, liquidity_window)
        for top_k in parse_int_list(args.top_k_grid):
            config = SignalConfig(
                top_k=top_k,
                liquidity_window=liquidity_window,
                min_periods=min_periods,
                signal_name=f"market_leader_return_k{top_k}_w{liquidity_window}",
            )
            row, daily = evaluate_config(
                raw,
                config,
                args.train_end_date,
                args.val_end_date,
                args.include_test,
                args.cost_bps,
            )
            rows.append(row)
            daily_by_key[f"k{top_k}_w{liquidity_window}"] = daily

    summary = pd.DataFrame(rows).sort_values(
        ["selected_train_rel_improvement", "selected_val_rel_improvement"],
        ascending=False,
        kind="stable",
    )
    selected = summary.iloc[0].to_dict()
    selected_top_k = int(selected["top_k"])
    selected_window = int(selected["liquidity_window"])
    selected_daily = daily_by_key[f"k{selected_top_k}_w{selected_window}"]
    latest_leaders = build_leader_snapshot(raw, selected_top_k, selected_window, int(selected["min_periods"]))

    summary.to_csv(output_dir / "market_leader_signal_grid.csv", index=False)
    selected_daily.to_csv(output_dir / "selected_daily_signal.csv", index=False)
    latest_leaders.to_csv(output_dir / "selected_latest_leaders.csv", index=False)
    (output_dir / "selected_config.json").write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")
    write_summary(output_dir, summary, selected, latest_leaders)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected_top_k": selected_top_k,
                "selected_liquidity_window": selected_window,
                "selected_signal_column": selected["selected_by_train_signal_column"],
                "selected_train_rel_improvement": selected["selected_train_rel_improvement"],
                "selected_val_rel_improvement": selected["selected_val_rel_improvement"],
                "selected_val_spearman_ic": selected["selected_val_spearman_ic"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
