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

from experiments.analysis.analyze_regime_performance import rel_score  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "router_validation"
DEFAULT_ROUTER_REPORT = RUN_ROOT / "reports" / "router_analysis" / "anchor_sector19_router_20260425_r01"
DEFAULT_CANDIDATES = (
    "anchor",
    "avg_70_challenger",
    "sector19_down_up_anchor_else",
    "sector19_up_anchor_else",
)
DEFAULT_FILTERS = (
    "all_regimes",
    "skip_downtrend",
    "trade_distribution_sideways",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling validation for router candidates using existing train/validation predictions."
    )
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--stamp", default="20260426_r01")
    parser.add_argument("--output-name", default="anchor_sector19_router_rolling")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--filters", default=",".join(DEFAULT_FILTERS))
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_router_frames(router_report: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(router_report / "candidate_predictions.csv")
    candidates["signal_date"] = pd.to_datetime(candidates["signal_date"])
    candidates["actual_date"] = pd.to_datetime(candidates["actual_date"])
    daily = pd.read_csv(router_report / "daily_quartile_returns.csv")
    daily["actual_date"] = pd.to_datetime(daily["actual_date"])
    return candidates, daily


def build_windows(candidate_df: pd.DataFrame) -> pd.DataFrame:
    val_dates = candidate_df.loc[candidate_df["split"] == "val", "actual_date"]
    min_val = val_dates.min()
    max_val = val_dates.max()
    windows: list[dict[str, object]] = [
        {
            "window": "val_all",
            "start_date": min_val,
            "end_date": max_val,
            "scope": "val",
        }
    ]
    for year in sorted(val_dates.dt.year.unique()):
        start = max(pd.Timestamp(year=int(year), month=1, day=1), min_val)
        end = min(pd.Timestamp(year=int(year), month=12, day=31), max_val)
        windows.append({"window": f"val_{year}", "start_date": start, "end_date": end, "scope": "val_year"})

    periods = sorted(val_dates.dt.to_period("Q").unique())
    for period in periods:
        start = max(period.start_time.normalize(), min_val)
        end = min(period.end_time.normalize(), max_val)
        windows.append({"window": f"val_{period}", "start_date": start, "end_date": end, "scope": "val_quarter"})
    return pd.DataFrame(windows)


def candidate_column(candidate_name: str) -> str:
    return f"candidate__{candidate_name}"


def summarize_prediction_windows(
    candidate_df: pd.DataFrame,
    windows: pd.DataFrame,
    candidate_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    val_df = candidate_df[candidate_df["split"] == "val"].copy()
    for _, window in windows.iterrows():
        window_df = val_df[
            (val_df["actual_date"] >= pd.Timestamp(window["start_date"]))
            & (val_df["actual_date"] <= pd.Timestamp(window["end_date"]))
        ].copy()
        for candidate in candidate_names:
            column = candidate_column(candidate)
            if column not in window_df.columns or window_df.empty:
                continue
            error = window_df["actual"] - window_df[column]
            rows.append(
                {
                    "window": window["window"],
                    "scope": window["scope"],
                    "start_date": window["start_date"],
                    "end_date": window["end_date"],
                    "candidate": candidate,
                    "n_obs": int(len(window_df)),
                    "n_days": int(window_df["actual_date"].nunique()),
                    "rel_score": rel_score(error, window_df["actual"]) if len(window_df) >= 3 else float("nan"),
                    "directional_accuracy": float((np.sign(window_df[column]) == np.sign(window_df["actual"])).mean()),
                    "error_q2": float(error.quantile(0.2)),
                    "error_q8": float(error.quantile(0.8)),
                }
            )
    return pd.DataFrame(rows)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min())


def filter_daily_returns(daily: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    if filter_name == "all_regimes":
        return daily.copy()
    if filter_name == "skip_downtrend":
        return daily[daily["regime"] != "downtrend"].copy()
    if filter_name == "trade_distribution_sideways":
        return daily[daily["regime"].isin(["distribution", "sideways"])].copy()
    if filter_name == "trade_distribution_sideways_recovery":
        return daily[daily["regime"].isin(["distribution", "sideways", "recovery"])].copy()
    if filter_name == "trade_distribution_only":
        return daily[daily["regime"] == "distribution"].copy()
    if filter_name == "trade_sideways_only":
        return daily[daily["regime"] == "sideways"].copy()
    raise ValueError(f"Unsupported filter: {filter_name}")


def summarize_trade_windows(
    daily: pd.DataFrame,
    windows: pd.DataFrame,
    candidate_names: list[str],
    filter_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    val_daily = daily[daily["split"] == "val"].copy()
    for _, window in windows.iterrows():
        window_daily = val_daily[
            (val_daily["actual_date"] >= pd.Timestamp(window["start_date"]))
            & (val_daily["actual_date"] <= pd.Timestamp(window["end_date"]))
        ].copy()
        for candidate in candidate_names:
            candidate_daily = window_daily[window_daily["run_name"] == candidate].copy()
            if candidate_daily.empty:
                continue
            for filter_name in filter_names:
                filtered = filter_daily_returns(candidate_daily, filter_name)
                returns = filtered["long_short_return"].to_numpy(dtype=float)
                equity = np.cumprod(1.0 + returns) if len(returns) else np.asarray([], dtype=float)
                rows.append(
                    {
                        "window": window["window"],
                        "scope": window["scope"],
                        "start_date": window["start_date"],
                        "end_date": window["end_date"],
                        "candidate": candidate,
                        "filter_name": filter_name,
                        "trade_days": int(len(filtered)),
                        "final_equity": float(equity[-1]) if len(equity) else float("nan"),
                        "hit_rate": float(np.mean(returns > 0.0)) if len(returns) else float("nan"),
                        "mean_return": float(np.mean(returns)) if len(returns) else float("nan"),
                        "max_drawdown": max_drawdown(equity),
                    }
                )
    return pd.DataFrame(rows)


def summarize_stability(prediction_summary: pd.DataFrame, trade_summary: pd.DataFrame) -> pd.DataFrame:
    val_year_pred = prediction_summary[prediction_summary["scope"] == "val_year"].copy()
    val_year_trade = trade_summary[(trade_summary["scope"] == "val_year") & (trade_summary["filter_name"] == "all_regimes")].copy()
    rows: list[dict[str, object]] = []
    for candidate, group in val_year_pred.groupby("candidate", sort=True):
        trade_group = val_year_trade[val_year_trade["candidate"] == candidate]
        rows.append(
            {
                "candidate": candidate,
                "years": int(group["window"].nunique()),
                "avg_year_rel_score": float(group["rel_score"].mean()),
                "worst_year_rel_score": float(group["rel_score"].min()),
                "positive_rel_score_years": int((group["rel_score"] > 0.0).sum()),
                "avg_year_equity_all": float(trade_group["final_equity"].mean()) if not trade_group.empty else float("nan"),
                "worst_year_equity_all": float(trade_group["final_equity"].min()) if not trade_group.empty else float("nan"),
                "profitable_years_all": int((trade_group["final_equity"] > 1.0).sum()) if not trade_group.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["profitable_years_all", "avg_year_equity_all"], ascending=[False, False], kind="stable")


def write_markdown(
    output_dir: Path,
    prediction_summary: pd.DataFrame,
    trade_summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    val_all_pred = prediction_summary[prediction_summary["window"] == "val_all"].copy()
    val_all_trade = trade_summary[trade_summary["window"] == "val_all"].copy()
    best_filters = (
        val_all_trade.sort_values(["candidate", "final_equity"], ascending=[True, False], kind="stable")
        .groupby("candidate", as_index=False)
        .head(1)
    )
    year_trade = trade_summary[(trade_summary["scope"] == "val_year") & (trade_summary["filter_name"] == "all_regimes")].copy()

    lines = [
        "# Router Rolling Validation",
        "",
        "Scope: validation/in-sample only. No test/out-sample data is used.",
        "",
        "## Full Validation Prediction",
        "",
        "| Candidate | rel_score | Direction | Obs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in val_all_pred.sort_values("rel_score", ascending=False, kind="stable").iterrows():
        lines.append(
            f"| `{row['candidate']}` | {float(row['rel_score']):+.4f} | {float(row['directional_accuracy']):.1%} | {int(row['n_obs'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Full Validation Trade Filter",
            "",
            "| Candidate | Filter | Trade days | Equity | Hit rate | Mean return | Max DD |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_filters.sort_values("final_equity", ascending=False, kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | `{row['filter_name']}` | {int(row['trade_days'])} | "
            f"{float(row['final_equity']):.3f} | {float(row['hit_rate']):.1%} | "
            f"{float(row['mean_return']):+.4f} | {float(row['max_drawdown']):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Year Stability",
            "",
            "| Candidate | Years | Positive rel years | Profitable years | Avg rel | Worst rel | Avg equity | Worst equity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in stability.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {int(row['years'])} | {int(row['positive_rel_score_years'])} | "
            f"{int(row['profitable_years_all'])} | {float(row['avg_year_rel_score']):+.4f} | "
            f"{float(row['worst_year_rel_score']):+.4f} | {float(row['avg_year_equity_all']):.3f} | "
            f"{float(row['worst_year_equity_all']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Year Trade Equity: All Regimes",
            "",
            "| Year | Candidate | Equity | Hit rate | Trade days | Max DD |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in year_trade.sort_values(["window", "final_equity"], ascending=[True, False], kind="stable").iterrows():
        lines.append(
            "| "
            f"`{row['window']}` | `{row['candidate']}` | {float(row['final_equity']):.3f} | "
            f"{float(row['hit_rate']):.1%} | {int(row['trade_days'])} | {float(row['max_drawdown']):.1%} |"
        )

    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_names = split_csv(args.candidates)
    filter_names = split_csv(args.filters)
    output_dir = REPORT_ROOT / f"{args.output_name}_{args.stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_df, daily = load_router_frames(args.router_report)
    windows = build_windows(candidate_df)
    prediction_summary = summarize_prediction_windows(candidate_df, windows, candidate_names)
    trade_summary = summarize_trade_windows(daily, windows, candidate_names, filter_names)
    stability = summarize_stability(prediction_summary, trade_summary)

    windows.to_csv(output_dir / "windows.csv", index=False)
    prediction_summary.to_csv(output_dir / "prediction_by_window.csv", index=False)
    trade_summary.to_csv(output_dir / "trade_by_window.csv", index=False)
    stability.to_csv(output_dir / "candidate_stability.csv", index=False)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "router_report": str(args.router_report),
                "candidates": candidate_names,
                "filters": filter_names,
                "stability": stability.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(output_dir, prediction_summary, trade_summary, stability)
    print(json.dumps({"output_dir": str(output_dir), "candidates": candidate_names}, indent=2))


if __name__ == "__main__":
    main()
