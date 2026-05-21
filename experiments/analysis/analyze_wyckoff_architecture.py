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

from src.models.selection.holding_period import (  # noqa: E402
    annualized_sharpe,
    desired_weights_for_day,
    l1_turnover,
    max_drawdown,
)
from experiments.analysis.analyze_committee_hypothesis_grid import (  # noqa: E402
    DEFAULT_COVERAGE_GRID,
    add_committee_candidate_columns,
)


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
FILTER_ROOT = RUN_ROOT / "reports" / "filter_signal"
QUALITY_DATASET = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_quality_dataset.csv"
DEFAULT_RUN = "portable_lstm_filter_signal_20260509_r06_selector_module"
DEFAULT_COMMITTEE_OUTPUTS = (
    "committee_hypothesis_grid_train_split_t025_dd15",
    "committee_hypothesis_grid_val_split_t025_dd15",
)
PHASE_ORDER = ("accumulation", "markup", "distribution", "markdown", "transition")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate committee architecture by Wyckoff-style market phase.")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--committee-outputs", default=",".join(DEFAULT_COMMITTEE_OUTPUTS))
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--min-positions", type=int, default=5)
    parser.add_argument("--output-name", default="wyckoff_architecture_eval")
    return parser.parse_args(argv)


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_wyckoff_daily() -> pd.DataFrame:
    usecols = [
        "Date",
        "code",
        "close",
        "momentum_20",
        "volatility_20",
        "wyckoff_phase_60d",
        "buying_pressure",
        "selling_pressure",
        "effort_result_ratio",
    ]
    quality = pd.read_csv(QUALITY_DATASET, usecols=usecols, parse_dates=["Date"])
    quality = quality.sort_values(["code", "Date"], kind="stable")
    quality["close_return"] = quality.groupby("code", sort=False)["close"].pct_change()
    quality["pressure_delta"] = quality["buying_pressure"] - quality["selling_pressure"]
    daily = (
        quality.groupby("Date", as_index=False)
        .agg(
            market_return=("close_return", "mean"),
            breadth=("close_return", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0.0))),
            momentum_20=("momentum_20", "mean"),
            volatility_20=("volatility_20", "mean"),
            wyckoff_location_60=("wyckoff_phase_60d", "mean"),
            buying_pressure=("buying_pressure", "mean"),
            selling_pressure=("selling_pressure", "mean"),
            pressure_delta=("pressure_delta", "mean"),
            effort_result_ratio=("effort_result_ratio", "median"),
            stock_count=("code", "nunique"),
        )
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    daily["market_index"] = (1.0 + daily["market_return"].fillna(0.0)).cumprod()
    daily["market_return_20"] = daily["market_return"].rolling(20, min_periods=10).mean()
    daily["market_return_60"] = daily["market_return"].rolling(60, min_periods=30).mean()
    daily["breadth_20"] = daily["breadth"].rolling(20, min_periods=10).mean()
    daily["breadth_20_delta"] = daily["breadth_20"] - daily["breadth_20"].shift(20)
    daily["location_20"] = daily["wyckoff_location_60"].rolling(20, min_periods=10).mean()
    daily["pressure_delta_20"] = daily["pressure_delta"].rolling(20, min_periods=10).mean()
    daily["rolling_peak_60"] = daily["market_index"].rolling(60, min_periods=30).max()
    daily["rolling_trough_60"] = daily["market_index"].rolling(60, min_periods=30).min()
    daily["drawdown_60"] = daily["market_index"] / daily["rolling_peak_60"].replace(0.0, np.nan) - 1.0
    daily["recovery_from_trough_60"] = daily["market_index"] / daily["rolling_trough_60"].replace(0.0, np.nan) - 1.0
    return assign_wyckoff_phase(daily)


def assign_wyckoff_phase(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    location = df["location_20"]
    ret20 = df["market_return_20"]
    ret60 = df["market_return_60"]
    breadth20 = df["breadth_20"]
    pressure = df["pressure_delta_20"]

    accumulation = (
        (location <= 0.40)
        & (df["drawdown_60"] < -0.03)
        & ((ret20 >= 0.0) | (df["breadth_20_delta"] > 0.0) | (pressure > 0.0))
    )
    markup = (
        (location >= 0.45)
        & (ret20 > 0.0)
        & (ret60 > 0.0)
        & (breadth20 >= 0.53)
    )
    distribution = (
        (location >= 0.60)
        & ((breadth20 <= 0.50) | (ret20 < 0.0))
        & ((pressure <= 0.0) | (df["selling_pressure"] >= df["buying_pressure"]))
    )
    markdown = (
        (ret20 < 0.0)
        & (ret60 < 0.0)
        & (breadth20 <= 0.47)
        & ((location <= 0.60) | (pressure < 0.0))
    )

    phase = np.full(len(df), "transition", dtype=object)
    phase[accumulation] = "accumulation"
    phase[markup] = "markup"
    phase[distribution] = "distribution"
    phase[markdown] = "markdown"
    df["wyckoff_phase"] = phase
    df["wyckoff_phase"] = smooth_short_phases(pd.Series(df["wyckoff_phase"], index=df.index), min_days=5)
    return df


def smooth_short_phases(phases: pd.Series, min_days: int) -> pd.Series:
    smoothed = phases.copy()
    episode_id = (smoothed != smoothed.shift()).cumsum()
    episodes = (
        pd.DataFrame({"phase": smoothed, "episode_id": episode_id})
        .groupby("episode_id", sort=True)
        .agg(phase=("phase", "first"), days=("phase", "size"))
        .reset_index()
    )
    for idx, row in episodes.iterrows():
        if int(row["days"]) >= min_days or idx == 0:
            continue
        smoothed.loc[episode_id == row["episode_id"]] = str(episodes.loc[idx - 1, "phase"])
    return smoothed


def simulate_rebalance_with_signal_date(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    rebalance_every: int,
    cost_bps: float,
    min_positions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_weights: dict[str, float] = {}
    daily_groups = list(frame.dropna(subset=["actual_date"]).sort_values("actual_date").groupby("actual_date", sort=True))
    for idx, (actual_date, day) in enumerate(daily_groups):
        returns = dict(zip(day["code"].astype(str), day["actual_aligned"].astype(float)))
        turnover = 0.0
        if idx % rebalance_every == 0:
            target_weights = desired_weights_for_day(day, prediction_column, min_positions)
            turnover = l1_turnover(current_weights, target_weights)
            current_weights = target_weights

        gross_return = float(sum(weight * returns.get(code, 0.0) for code, weight in current_weights.items()))
        cost_return = turnover * cost_bps / 10_000.0
        rows.append(
            {
                "signal_date": pd.Timestamp(day["Date"].max()),
                "actual_date": actual_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost_return": cost_return,
                "net_return": gross_return - cost_return,
                "n_positions": int(sum(abs(weight) > 0.0 for weight in current_weights.values())),
                "gross_exposure": float(sum(abs(weight) for weight in current_weights.values())),
                "is_rebalance_day": bool(idx % rebalance_every == 0),
            }
        )
    return pd.DataFrame(rows)


def summarize_returns(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "n_days": 0,
            "net_equity": float("nan"),
            "gross_equity": float("nan"),
            "net_sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "avg_turnover": float("nan"),
            "avg_positions": float("nan"),
        }
    gross_equity = (1.0 + daily["gross_return"]).cumprod()
    net_equity = (1.0 + daily["net_return"]).cumprod()
    return {
        "n_days": int(len(daily)),
        "net_equity": float(net_equity.iloc[-1]),
        "gross_equity": float(gross_equity.iloc[-1]),
        "net_sharpe": annualized_sharpe(daily["net_return"]),
        "max_drawdown": max_drawdown(net_equity),
        "hit_rate": float((daily["net_return"] > 0.0).mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_positions": float(daily["n_positions"].mean()),
    }


def build_selected_daily(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
    cost_bps: float,
    min_positions: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, row in fold_results.iterrows():
        fold_frame = predictions.loc[
            (predictions["actual_date"] >= pd.Timestamp(row["test_start"]))
            & (predictions["actual_date"] <= pd.Timestamp(row["test_end"]))
        ].copy()
        daily = simulate_rebalance_with_signal_date(
            fold_frame,
            str(row["candidate"]),
            rebalance_every=int(row["rebalance_every"]),
            cost_bps=cost_bps,
            min_positions=min_positions,
        )
        daily.insert(0, "candidate", str(row["candidate"]))
        daily.insert(0, "hypothesis", str(row["hypothesis"]))
        daily.insert(0, "fold", int(row["fold"]))
        parts.append(daily)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_by_phase(selected_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    for hypothesis, group in selected_daily.groupby("hypothesis", sort=True):
        total_rows.append({"hypothesis": hypothesis, "phase": "all", **summarize_returns(group)})
        for phase in PHASE_ORDER:
            phase_group = group.loc[group["wyckoff_phase"] == phase].copy()
            rows.append({"hypothesis": hypothesis, "phase": phase, **summarize_returns(phase_group)})
    summary = pd.DataFrame(rows)
    totals = pd.DataFrame(total_rows)
    return summary, totals


def write_markdown(
    output_dir: Path,
    phase_summary: pd.DataFrame,
    total_summary: pd.DataFrame,
    phase_days: pd.DataFrame,
    output_name: str,
) -> None:
    lines = [
        "# Wyckoff Architecture Evaluation",
        "",
        f"Committee output: `{output_name}`.",
        "Phase is assigned on signal date using point-in-time market breadth, return, pressure, and `wyckoff_phase_60d` from the VN quality dataset.",
        "",
        "## Overall",
        "",
        "| Hypothesis | Net equity | Sharpe | Max DD | Avg turnover | Hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in total_summary.sort_values("net_equity", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['hypothesis']}` | {float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} | {float(row['hit_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Phase Days",
            "",
            "| Phase | Days | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in phase_days.iterrows():
        lines.append(f"| `{row['wyckoff_phase']}` | {int(row['days'])} | {float(row['share']):.1%} |")
    lines.extend(
        [
            "",
            "## By Wyckoff Phase",
            "",
            "| Hypothesis | Phase | Days | Net equity | Sharpe | Max DD | Avg turnover | Hit rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ordered = phase_summary.copy()
    ordered["phase_order"] = ordered["phase"].map({phase: idx for idx, phase in enumerate(PHASE_ORDER)})
    ordered = ordered.sort_values(["hypothesis", "phase_order"], kind="stable")
    for _, row in ordered.iterrows():
        lines.append(
            "| "
            f"`{row['hypothesis']}` | `{row['phase']}` | {int(row['n_days'])} | "
            f"{float(row['net_equity']):.3f} | {float(row['net_sharpe']):+.2f} | "
            f"{float(row['max_drawdown']):.1%} | {float(row['avg_turnover']):.2f} | {float(row['hit_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- A robust architecture should not depend on only one Wyckoff phase.",
            "- Treat phases with few days as diagnostic only.",
            "- If a hypothesis wins overall but fails markdown/distribution, it needs a regime risk gate before promotion.",
        ]
    )
    output_dir.joinpath(f"{output_name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = FILTER_ROOT / args.run
    predictions_path = run_dir / "filter_predictions.csv.gz"
    predictions = pd.read_csv(predictions_path, parse_dates=["Date", "actual_date"])
    add_committee_candidate_columns(predictions, DEFAULT_COVERAGE_GRID)
    wyckoff_daily = load_wyckoff_daily()
    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "run": args.run,
        "cost_bps": args.cost_bps,
        "min_positions": args.min_positions,
        "committee_outputs": [],
    }
    combined_phase_parts: list[pd.DataFrame] = []
    combined_total_parts: list[pd.DataFrame] = []
    for output_name in split_csv(args.committee_outputs):
        committee_dir = run_dir / output_name
        folds_path = committee_dir / "committee_hypothesis_folds.csv"
        if not folds_path.exists():
            raise FileNotFoundError(folds_path)
        fold_results = pd.read_csv(folds_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
        selected_daily = build_selected_daily(predictions, fold_results, args.cost_bps, args.min_positions)
        selected_daily = selected_daily.merge(
            wyckoff_daily[
                [
                    "Date",
                    "wyckoff_phase",
                    "market_return_20",
                    "market_return_60",
                    "breadth_20",
                    "location_20",
                    "pressure_delta_20",
                ]
            ],
            left_on="signal_date",
            right_on="Date",
            how="left",
        ).drop(columns=["Date"])
        selected_daily["wyckoff_phase"] = selected_daily["wyckoff_phase"].fillna("transition")
        phase_summary, total_summary = summarize_by_phase(selected_daily)
        phase_days = (
            selected_daily[["signal_date", "wyckoff_phase"]]
            .drop_duplicates()
            .groupby("wyckoff_phase", as_index=False)
            .agg(days=("signal_date", "nunique"))
        )
        phase_days["share"] = phase_days["days"] / phase_days["days"].sum()
        phase_days["phase_order"] = phase_days["wyckoff_phase"].map({phase: idx for idx, phase in enumerate(PHASE_ORDER)})
        phase_days = phase_days.sort_values("phase_order", kind="stable").drop(columns=["phase_order"])

        selected_daily.to_csv(output_dir / f"{output_name}_selected_daily.csv", index=False)
        phase_summary.to_csv(output_dir / f"{output_name}_phase_summary.csv", index=False)
        total_summary.to_csv(output_dir / f"{output_name}_total_summary.csv", index=False)
        phase_days.to_csv(output_dir / f"{output_name}_phase_days.csv", index=False)
        write_markdown(output_dir, phase_summary, total_summary, phase_days, output_name)

        phase_summary.insert(0, "committee_output", output_name)
        total_summary.insert(0, "committee_output", output_name)
        combined_phase_parts.append(phase_summary)
        combined_total_parts.append(total_summary)
        manifest["committee_outputs"].append(
            {
                "name": output_name,
                "fold_rows": int(len(fold_results)),
                "daily_rows": int(len(selected_daily)),
            }
        )

    combined_phase = pd.concat(combined_phase_parts, ignore_index=True)
    combined_total = pd.concat(combined_total_parts, ignore_index=True)
    combined_phase.to_csv(output_dir / "combined_phase_summary.csv", index=False)
    combined_total.to_csv(output_dir / "combined_total_summary.csv", index=False)
    output_dir.joinpath("summary.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "outputs": len(manifest["committee_outputs"])}, indent=2))


if __name__ == "__main__":
    main()
